"""
TEM nanoparticle size analysis.

Detects large circular particles (vesicles) in TEM images using Cellpose.
Measures outer diameter and wall thickness (via radial intensity profile).
Auto-detects scale bar from image when possible.

Usage:
    uv run python measure.py images/sample.jpeg --scale-nm 200
    uv run python measure.py images/*.jpeg --scale-nm 200
    uv run python measure.py images/ --scale-nm 200
    uv run python measure.py images/ --scale-nm 100 --scale-px 150
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellpose.models import CellposeModel
from scipy.ndimage import uniform_filter1d
from skimage import measure

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
CELLPOSE_CELLPROB = 0.0
CELLPOSE_FLOW = 0.8

# Reject elongated/non-circular detections (lines, debris, image edges).
# 1.0 is a perfect circle; vesicles are typically > 0.7.
MIN_CIRCULARITY = 0.5

# Reject detections whose bounding box reaches into a margin near any image
# edge — these are almost always artifacts from where the image meets a
# label strip, dark border, or grid bar. Expressed as a fraction of the
# smaller image dimension (~40 px on a 2048² image).
EDGE_MARGIN_FRAC = 0.02

# Real TEM particles are markedly darker than their immediate surroundings
# (heavy-metal staining + projection density). Reject detections where the
# mean intensity inside the mask isn't at least this many uint8 units below
# the mean intensity of an annular ring around it.
MIN_CONTRAST = 10.0

# Physical sanity check: the technician picks the magnification to match the
# particle size, so real particles cluster within ~0.5-3x the scale bar.
# Anything an order of magnitude smaller (grain, stain artifact) or larger
# (image features that aren't single particles) is rejected.
MIN_DIAM_VS_BAR = 0.4
MAX_DIAM_VS_BAR = 4.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """
    Stretch intensity to [0, 255] using 1st/99th percentile clipping.

    Some TIF acquisitions store data with a tiny dynamic range (e.g. 17-26
    out of 0-255); without this stretch, both OCR and Cellpose see a uniform
    gray and find nothing. Percentile clipping (vs. plain min/max) keeps
    bright outliers like the scale bar from compressing the bulk image.
    """
    p1, p99 = np.percentile(img, [1, 99])
    if p99 - p1 < 1:
        return img
    stretched = (img.astype(np.float32) - p1) * (255.0 / (p99 - p1))
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scale bar detection
# ---------------------------------------------------------------------------
@dataclass
class ScaleBarLocation:
    """Pixel coordinates of a scale bar in the original (full) image."""

    width_px: int
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int


def detect_scale_bar(img: np.ndarray) -> ScaleBarLocation | None:
    """
    Auto-detect scale bar in the bottom of a TEM image.

    Uses morphological filtering to find horizontal line structures.
    Returns location (in full-image coordinates) or None if not found.
    """
    h, w = img.shape[:2]
    strip_y0 = int(h * 0.85)
    bottom = img[strip_y0:, :]

    _, binary = cv2.threshold(bottom, 20, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz_lines)

    best = None
    for i in range(1, n_labels):
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if comp_w <= w * 0.05 or comp_h >= h * 0.05:
            continue
        if best is not None and comp_w <= best.width_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        best = ScaleBarLocation(
            width_px=comp_w,
            x_left=x,
            x_right=x + comp_w,
            y_top=strip_y0 + y,
            y_bottom=strip_y0 + y + comp_h,
        )

    return best


_ocr_reader = None

# Pattern to match scale text like "200.0nm", "100nm", "1.5um", "500 nm"
# Handles common OCR mistakes: O->0, l->1
_SCALE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(nm|um|μm)",
    re.IGNORECASE,
)


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        # Try MPS first (Apple Silicon); fall back to CPU if EasyOCR's
        # device selection can't handle it on this version/platform.
        try:
            _ocr_reader = easyocr.Reader(["en"], gpu="mps", verbose=False)
        except (ValueError, RuntimeError, AssertionError):
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


_MIN_OCR_CONFIDENCE = 0.2

# TEM scale bars are always one of these (n*10^k for n in {1,2,5}).
# OCR readings are snapped to the nearest within ±15%; anything else is junk.
_STANDARD_SCALES_NM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
_SCALE_SNAP_TOLERANCE = 0.15


def _snap_to_standard_scale(value: float) -> float | None:
    """Return the nearest standard scale bar value if within tolerance, else None."""
    for standard in _STANDARD_SCALES_NM:
        if abs(value - standard) / standard <= _SCALE_SNAP_TOLERANCE:
            return float(standard)
    return None


# If any single read crosses this confidence, accept its value immediately
# and skip the remaining (slower) OCR strategies.
_OCR_HIGH_CONFIDENCE = 0.7


def _ocr_one(strip: np.ndarray) -> dict[float, float]:
    """Run OCR on a single preprocessed strip; return {snapped_nm: cumulative_confidence}."""
    reader = _get_ocr_reader()
    upscaled = cv2.resize(
        strip, (strip.shape[1] * 2, strip.shape[0] * 2), interpolation=cv2.INTER_CUBIC
    )
    out: dict[float, float] = {}
    for _bbox, text, conf in reader.readtext(upscaled):
        if conf < _MIN_OCR_CONFIDENCE:
            continue
        # Fix common OCR mistakes: O->0, l->1, , -> .
        # Also collapse digit-space-digit (e.g. "100 0nm") into "100.0nm"
        # since the decimal point is often misread as whitespace.
        cleaned = text.replace("O", "0").replace("l", "1").replace(",", ".")
        cleaned = re.sub(r"(\d)\s+(\d)", r"\1.\2", cleaned)
        match = _SCALE_PATTERN.search(cleaned)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in ("um", "μm"):
            value *= 1000
        snapped = _snap_to_standard_scale(value)
        if snapped is None:
            continue
        out[snapped] = out.get(snapped, 0.0) + conf
    return out


def _ocr_strategies(img: np.ndarray, bar: ScaleBarLocation | None) -> list[np.ndarray]:
    """
    Yield preprocessed strips to try in order, cheapest-likely-to-work first.

    Most images succeed on the first (bright-thresholded bottom strip);
    the rest are escalated through Otsu, normalization, and — if the bar
    location is known — a tight crop next to the bar.
    """
    h, w = img.shape[:2]
    bottom = img[int(h * 0.85) :, :]
    bottom_right = img[int(h * 0.85) :, int(w * 0.4) :]

    strategies: list[np.ndarray] = []
    # Cheap fast path: bright white text on dark background
    _, fixed = cv2.threshold(bottom, 200, 255, cv2.THRESH_BINARY)
    strategies.append(fixed)
    # Tight crop next to the detected bar — usually cleanest input
    if bar is not None:
        bar_h = max(bar.y_bottom - bar.y_top, 8)
        y0 = max(0, bar.y_top - bar_h * 6)
        y1 = min(h, bar.y_bottom + bar_h * 6)
        x0 = max(0, bar.x_right - bar.width_px // 4)
        x1 = min(w, bar.x_right + bar.width_px * 2)
        if y1 > y0 and x1 > x0:
            tight = img[y0:y1, x0:x1]
            strategies.append(normalize_intensity(tight))
            _, tight_otsu = cv2.threshold(tight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(tight_otsu)
    # Adaptive: Otsu on bottom-right (good for mid-gray-on-mid-gray text)
    _, otsu = cv2.threshold(bottom_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(otsu)
    strategies.append(cv2.bitwise_not(otsu))
    # Last resort: normalized bottom (helps low-dynamic-range tifs)
    strategies.append(normalize_intensity(bottom))
    return strategies


def detect_scale_text(img: np.ndarray, bar: ScaleBarLocation | None = None) -> float | None:
    """
    Read scale bar text (e.g. '200.0nm') from the bottom of the image using OCR.

    Returns the scale value in nm, or None if not found.

    Strategies are tried in order from cheapest-likely-to-work to slowest;
    accumulated candidates are kept across attempts and OCR stops as soon
    as a confident standard-value read appears. Parsed values are snapped
    to the nearest standard TEM scale (1, 2, 5, ..., 5000 nm); garbled
    reads like "350nm" or "7100nm" are discarded.
    """
    candidates: dict[float, float] = {}
    for strip in _ocr_strategies(img, bar):
        for k, v in _ocr_one(strip).items():
            candidates[k] = candidates.get(k, 0.0) + v
        if candidates:
            best, best_conf = max(candidates.items(), key=lambda kv: kv[1])
            if best_conf >= _OCR_HIGH_CONFIDENCE:
                return best

    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def determine_scale(
    img: np.ndarray, scale_nm: float | None
) -> tuple[float | None, int | None, float | None]:
    """
    Determine nm/pixel scale. Returns (nm_per_pixel, bar_pixels, scale_nm).

    Strategy:
    1. Auto-detect bar pixel length from the image
    2. Use provided scale_nm, or try OCR to read the nm value from the image
       (passing bar location for a tight crop next to the bar)
    3. Return (None, ...) if either piece is missing
    """
    bar = detect_scale_bar(img)
    bar_px = bar.width_px if bar is not None else None

    # If scale_nm not provided, try OCR
    if scale_nm is None:
        ocr_nm = detect_scale_text(img, bar)
        if ocr_nm is not None:
            print(f"  OCR detected scale text: {ocr_nm}nm")
            scale_nm = ocr_nm

    if bar_px is not None and scale_nm is not None:
        return scale_nm / bar_px, bar_px, scale_nm

    if bar_px is not None:
        print(f"  WARNING: detected scale bar = {bar_px}px but could not read the nm value.")
        print("  Use --scale-nm to specify (e.g. --scale-nm 200)")
        return None, bar_px, None

    if scale_nm is not None:
        print("  WARNING: could not auto-detect scale bar pixels.")
        print("  Use --scale-px to specify, or provide both --scale-nm and --scale-px.")
        return None, None, scale_nm

    print("  WARNING: could not detect scale bar or read scale text.")
    print("  Use --scale-nm and --scale-px to specify manually.")
    return None, None, None


# ---------------------------------------------------------------------------
# TIFF metadata scale extraction
# ---------------------------------------------------------------------------
# JEOL TemReporter XML (embedded in ImageDescription) stores the on-image
# scale bar value explicitly in these two tags — no OCR required.
_JEOL_MICRONBAR_VALUE = re.compile(r"<MicronbarValue>([^<]+)</MicronbarValue>")
_JEOL_MICRONBAR_UNIT = re.compile(r"<MicronbarMeasureUint>([^<]+)</MicronbarMeasureUint>")

_UNIT_TO_NM = {
    "nanometer": 1.0,
    "nm": 1.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
    "micron": 1000.0,
    "um": 1000.0,
    "μm": 1000.0,
    "millimeter": 1_000_000.0,
    "mm": 1_000_000.0,
}


def read_tiff_scale_nm(path: Path) -> float | None:
    """
    Extract the on-image scale bar value (in nm) from TIFF metadata, if present.

    Currently understands JEOL TemReporter XML (used by JEM-1400 and similar);
    returns None for other TIFF flavors and non-TIFF files. Far more reliable
    than OCR when the metadata is available.
    """
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(path) as t:
            desc_tag = t.pages[0].tags.get("ImageDescription")
            if desc_tag is None:
                return None
            desc = str(desc_tag.value)
    except (OSError, ValueError, KeyError):
        return None

    m_val = _JEOL_MICRONBAR_VALUE.search(desc)
    m_unit = _JEOL_MICRONBAR_UNIT.search(desc)
    if not m_val or not m_unit:
        return None
    try:
        value = float(m_val.group(1))
    except ValueError:
        return None
    factor = _UNIT_TO_NM.get(m_unit.group(1).strip().lower())
    if factor is None or value <= 0:
        return None
    return value * factor


# ---------------------------------------------------------------------------
# Detection and measurement
# ---------------------------------------------------------------------------
_cellpose_model = None


def get_cellpose_model() -> CellposeModel:
    """Lazy-load Cellpose model (shared across images)."""
    global _cellpose_model
    if _cellpose_model is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  Loading Cellpose model (device: {device})...")
        _cellpose_model = CellposeModel(gpu=True, device=device)
    return _cellpose_model


def run_cellpose(roi: np.ndarray, diameter_px: float) -> np.ndarray:
    """
    Cellpose with an explicit diameter hint (the scale-bar pixel length).

    Auto-sizing (diameter=None) was tried but in TEM images it estimates
    much too small — segmenting noise and missing real particles — and
    runs ~15x slower because the model does a pre-pass to estimate size.
    """
    model = get_cellpose_model()
    masks, _, _ = model.eval(
        roi,
        diameter=diameter_px,
        channels=[0, 0],
        flow_threshold=CELLPOSE_FLOW,
        cellprob_threshold=CELLPOSE_CELLPROB,
    )
    return masks


def compute_radial_profile(
    roi: np.ndarray, cy: int, cx: int, max_r: int, n_angles: int = 360
) -> np.ndarray:
    """Compute average radial intensity profile using vectorized sampling."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.arange(max_r)

    # Build (n_angles, max_r) grids of x,y coordinates
    cos_a = np.cos(angles)[:, np.newaxis]  # (n_angles, 1)
    sin_a = np.sin(angles)[:, np.newaxis]
    r = radii[np.newaxis, :]  # (1, max_r)

    xs = (cx + r * cos_a).astype(int)  # (n_angles, max_r)
    ys = (cy + r * sin_a).astype(int)

    # Mask out-of-bounds coordinates
    h, w = roi.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    # Sample intensities (use 0 for out-of-bounds, then average only valid)
    xs_safe = np.clip(xs, 0, w - 1)
    ys_safe = np.clip(ys, 0, h - 1)
    intensities = roi[ys_safe, xs_safe].astype(np.float64)
    intensities[~valid] = 0.0

    counts = valid.sum(axis=0).astype(np.float64)
    profile = np.where(counts > 0, intensities.sum(axis=0) / counts, 0.0)
    return profile


def measure_wall_thickness(
    roi: np.ndarray, cy: int, cx: int, radius_px: float, nm_per_pixel: float
) -> dict:
    """
    Measure wall thickness via radial intensity profile.

    Returns dict with wall_px, radii_nm, profile, threshold, inner/outer edges, is_vesicle.
    """
    max_r = int(radius_px * 1.8)
    profile = compute_radial_profile(roi, cy, cx, max_r)
    profile_smooth = uniform_filter1d(profile, size=7)

    radii_nm = np.arange(max_r) * nm_per_pixel

    # Wall = region darker than midpoint between background and darkest point
    bg = np.mean(profile_smooth[max(0, max_r - 30) : max_r])
    darkest = np.min(profile_smooth[5:])
    threshold = float((bg + darkest) / 2)

    below = profile_smooth < threshold
    if not np.any(below):
        return {
            "wall_px": None,
            "radii_nm": radii_nm,
            "profile": profile_smooth,
            "threshold": threshold,
            "inner_nm": None,
            "outer_nm": None,
            "is_vesicle": False,
        }

    indices = np.where(below)[0]
    inner_px = indices[0]
    outer_px = indices[-1]
    wall_px = float(outer_px - inner_px)

    # Vesicle validation: a real vesicle (donut) has:
    # 1. A brighter center (hollow interior) — center intensity above the threshold
    # 2. Dark wall ring away from center — inner edge not at r=0
    # A solid dark blob has its darkest point at/near the center.
    center_intensity = float(np.mean(profile_smooth[: max(5, int(radius_px * 0.3))]))
    is_vesicle = center_intensity > threshold and inner_px > int(radius_px * 0.15)

    return {
        "wall_px": wall_px,
        "radii_nm": radii_nm,
        "profile": profile_smooth,
        "threshold": threshold,
        "inner_nm": float(inner_px * nm_per_pixel),
        "outer_nm": float(outer_px * nm_per_pixel),
        "is_vesicle": is_vesicle,
    }


def _is_near_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """True if a region's bounding box reaches into the EDGE_MARGIN_FRAC margin."""
    h, w = shape
    margin = int(min(h, w) * EDGE_MARGIN_FRAC)
    min_row, min_col, max_row, max_col = bbox
    return (
        min_row < margin
        or min_col < margin
        or max_row > h - margin
        or max_col > w - margin
    )


def _contrast_vs_ring(roi: np.ndarray, mask: np.ndarray, radius_px: float) -> float:
    """
    Mean intensity in a ring just outside the mask minus mean intensity inside it.

    Positive values mean the detection is darker than its surroundings (real
    particle); ~0 means the detection blends with background (noise/artifact).
    Ring width scales with particle size so it samples a representative
    background patch and not random distant pixels.
    """
    ring_width = max(3, int(radius_px * 0.5))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1)
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    ring = dilated & ~mask
    if not ring.any() or not mask.any():
        return 0.0
    return float(roi[ring].mean() - roi[mask].mean())


def measure_particles(
    roi: np.ndarray,
    masks: np.ndarray,
    nm_per_pixel: float,
    bar_nm: float,
) -> pd.DataFrame:
    min_diam_nm = MIN_DIAM_VS_BAR * bar_nm
    max_diam_nm = MAX_DIAM_VS_BAR * bar_nm

    props = measure.regionprops(masks)
    rows = []
    for p in props:
        if _is_near_edge(p.bbox, roi.shape):
            continue

        circ = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
        if circ < MIN_CIRCULARITY:
            continue

        area_nm2 = p.area * nm_per_pixel**2
        diam_nm = 2 * np.sqrt(area_nm2 / np.pi)
        if diam_nm < min_diam_nm or diam_nm > max_diam_nm:
            continue

        radius_px = np.sqrt(p.area / np.pi)
        mask_i = masks == p.label
        contrast = _contrast_vs_ring(roi, mask_i, radius_px)
        if contrast < MIN_CONTRAST:
            continue

        cy, cx = int(p.centroid[0]), int(p.centroid[1])

        wall_info = measure_wall_thickness(roi, cy, cx, radius_px, nm_per_pixel)

        wall_nm = wall_info["wall_px"] * nm_per_pixel if wall_info["wall_px"] is not None else None

        rows.append(
            {
                "id": p.label,
                "cx": cx,
                "cy": cy,
                "radius_px": round(radius_px, 1),
                "diam_nm": round(diam_nm, 1),
                "wall_nm": round(wall_nm, 1) if wall_nm is not None else None,
                "circularity": round(circ, 3),
                "is_vesicle": wall_info["is_vesicle"],
                "_wall_info": wall_info,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_scale_bar(vis: np.ndarray, nm_per_pixel: float, length_nm: float):
    """Draw a reference scale bar on the image for visual verification."""
    bar_px = int(length_nm / nm_per_pixel)
    margin = 20
    y = vis.shape[0] - margin
    x_start = margin
    x_end = x_start + bar_px

    cv2.line(vis, (x_start, y), (x_end, y), (255, 255, 255), 3)
    cv2.line(vis, (x_start, y - 5), (x_start, y + 5), (255, 255, 255), 2)
    cv2.line(vis, (x_end, y - 5), (x_end, y + 5), (255, 255, 255), 2)
    cv2.putText(
        vis,
        f"{length_nm:.0f} nm",
        (x_start, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def draw_detections(
    roi: np.ndarray,
    masks: np.ndarray,
    df: pd.DataFrame,
    nm_per_pixel: float,
    scale_bar_nm: float,
) -> np.ndarray:
    vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for _, row in df.iterrows():
        mask_i = (masks == row["id"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        cx, cy = int(row["cx"]), int(row["cy"])
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        r = int(row["radius_px"])
        cv2.circle(vis, (cx, cy), r, (255, 200, 0), 1)
        cv2.line(vis, (cx - r, cy), (cx + r, cy), (0, 150, 255), 1)

        label = f"#{row['id']} d={row['diam_nm']:.0f}nm"
        if pd.notna(row.get("wall_nm")):
            label += f" w={row['wall_nm']:.0f}nm"
        cv2.putText(vis, label, (cx + r + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    draw_scale_bar(vis, nm_per_pixel, scale_bar_nm)
    return vis


def save_profiles(df: pd.DataFrame, out_dir: Path, nm_per_pixel: float):
    """Save radial intensity profile with wall measurement visualization."""
    for _, row in df.iterrows():
        w = row["_wall_info"]
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(w["radii_nm"], w["profile"], color="black", lw=1.5, label="Radial profile")

        ax.axhline(
            w["threshold"],
            color="gray",
            ls=":",
            lw=1,
            label=f"Threshold = {w['threshold']:.1f}",
        )

        r_nm = row["radius_px"] * nm_per_pixel
        ax.axvline(r_nm, color="green", ls="--", alpha=0.5, label=f"Eq. radius = {r_nm:.0f} nm")

        if w["inner_nm"] is not None:
            inner, outer = w["inner_nm"], w["outer_nm"]
            ax.axvline(inner, color="red", ls="-", lw=1.5, label=f"Wall inner = {inner:.0f} nm")
            ax.axvline(outer, color="blue", ls="-", lw=1.5, label=f"Wall outer = {outer:.0f} nm")
            wall = row["wall_nm"]
            ax.axvspan(inner, outer, alpha=0.15, color="red", label=f"Wall = {wall:.0f} nm")

        if pd.notna(row.get("wall_nm")):
            ax.set_title(
                f"Particle #{row['id']} — d={row['diam_nm']:.0f}nm, wall={row['wall_nm']:.0f}nm"
            )
        else:
            ax.set_title(f"Particle #{row['id']} — d={row['diam_nm']:.0f}nm")

        ax.set_xlabel("Distance from center (nm)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"profile_{row['id']}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------
def make_unique_dir(output_dir: Path, image_path: Path, subfolder: str | None = None) -> Path:
    """Create a unique output subdirectory for an image, handling name collisions."""
    base = image_path.stem
    parent = output_dir / subfolder if subfolder else output_dir
    candidate = parent / base
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = parent / f"{base}_{counter}"
    candidate.mkdir(parents=True)
    return candidate


def process_image(
    image_path: Path,
    output_dir: Path,
    scale_nm: float | None,
    scale_px: int | None,
    image_label: str,
    subfolder: str | None,
) -> pd.DataFrame:
    """Process a single TEM image. Returns DataFrame of particle measurements."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ERROR: could not load {image_path}")
        return pd.DataFrame()

    # Per-image scale calibration
    if scale_px is not None and scale_nm is not None:
        nm_per_pixel = scale_nm / scale_px
        bar_px = scale_px
        bar_nm: float | None = scale_nm
    else:
        # If CLI didn't override and the file is a TIFF with embedded scale
        # metadata, prefer that over OCR — it's deterministic and exact.
        if scale_nm is None:
            tiff_nm = read_tiff_scale_nm(image_path)
            if tiff_nm is not None:
                print(f"  TIFF metadata scale: {tiff_nm}nm")
                scale_nm = tiff_nm
        nm_per_pixel, bar_px, bar_nm = determine_scale(img, scale_nm)

    if nm_per_pixel is None or bar_px is None or bar_nm is None or bar_nm <= 0 or bar_px <= 0:
        print(f"  SKIPPING {image_label}: could not determine scale.")
        print("  Provide --scale-nm and --scale-px, or ensure image has a visible scale bar.")
        return pd.DataFrame()

    # The scale bar's pixel length is Cellpose's diameter hint (the model
    # rescales internally to find features at that scale). We no longer
    # impose a min/max diameter filter on the output — every Cellpose
    # detection is accepted subject to shape, edge, and contrast checks.
    diameter_px = float(bar_px)

    t0 = time.time()
    masks = run_cellpose(img, diameter_px)
    elapsed = time.time() - t0
    raw_count = masks.max()

    df = measure_particles(img, masks, nm_per_pixel, bar_nm)
    print(
        f"  {image_label}: {len(df)} particles ({raw_count} raw)"
        f" [{elapsed:.1f}s] scale={nm_per_pixel:.3f} nm/px"
        f" bar={bar_nm:.0f}nm/{bar_px}px"
    )

    if len(df) == 0:
        return pd.DataFrame()

    # Print per-image results
    for _, row in df.iterrows():
        wall_str = f" w={row['wall_nm']:.0f}nm" if pd.notna(row.get("wall_nm")) else ""
        ves_str = "" if row["is_vesicle"] else " [solid]"
        print(f"    #{row['id']} d={row['diam_nm']:.0f}nm{wall_str}{ves_str}")

    # Save per-image outputs
    img_dir = make_unique_dir(output_dir, image_path, subfolder)

    cv2.imwrite(str(img_dir / "roi.png"), img)

    vis = draw_detections(img, masks, df, nm_per_pixel, bar_nm)
    cv2.imwrite(str(img_dir / "detections.png"), vis)

    save_profiles(df, img_dir, nm_per_pixel)

    export_cols = [c for c in df.columns if not c.startswith("_")]
    df[export_cols].to_csv(img_dir / "particles.csv", index=False)

    # Add source image and scale columns for aggregate
    df_export = df[export_cols].copy()
    df_export.insert(0, "image", image_label)
    df_export["nm_per_pixel"] = round(nm_per_pixel, 4)
    df_export["scale_bar_nm"] = bar_nm
    return df_export


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_images(paths: list[str]) -> list[tuple[Path, Path]]:
    """
    Resolve CLI paths into (image_path, root_dir) tuples.

    root_dir is the user-given directory the image was discovered under (or the
    image's parent if a file was passed directly); used to derive the output
    subfolder so multi-folder batches stay organized.
    """
    images: list[tuple[Path, Path]] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            found = sorted(
                f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            images.extend((f, path) for f in found)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((path, path.parent))
        else:
            print(f"  Skipping: {path}")
    return images


def derive_labels(image_path: Path, root_dir: Path) -> tuple[str, str | None]:
    """
    Compute (image_label, output_subfolder) for an image.

    - image_label: human-readable identifier used in CSVs and logs. Includes
      the relative path under root_dir so multi-folder batches stay distinct.
    - output_subfolder: subdirectory under output/ that mirrors the input's
      folder name(s). None when the image is at root_dir's top level.
    """
    try:
        rel = image_path.relative_to(root_dir)
    except ValueError:
        return image_path.name, None
    if rel.parent == Path("."):
        # Image is at the top level of the passed directory. Use the
        # directory's own name as the subfolder when meaningful.
        if root_dir.name and root_dir.name != "images":
            return f"{root_dir.name}/{image_path.name}", root_dir.name
        return image_path.name, None
    sub = f"{root_dir.name}/{rel.parent}" if root_dir.name != "images" else str(rel.parent)
    return f"{sub}/{image_path.name}", sub


def main():
    parser = argparse.ArgumentParser(description="TEM nanoparticle size analysis")
    parser.add_argument(
        "images",
        nargs="*",
        default=["images/"],
        help="Image files or directories (default: images/)",
    )
    parser.add_argument("--scale-nm", type=float, default=None, help="Scale bar value in nm")
    parser.add_argument(
        "--scale-px", type=int, default=None, help="Scale bar length in pixels (skip auto-detect)"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory (default: output)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect images
    image_entries = collect_images(args.images)
    if not image_entries:
        print("No images found.")
        sys.exit(1)

    print("\n=== TEM particle analysis ===")
    print(f"Images: {len(image_entries)}")
    if args.scale_nm:
        print(f"Scale bar value: {args.scale_nm} nm")
    if args.scale_px:
        print(f"Scale bar pixels: {args.scale_px} px")
    if not args.scale_nm and not args.scale_px:
        print("Scale: auto-detect per image (provide --scale-nm for accuracy)")

    # Process each image (scale detected per-image)
    print(f"\n--- Processing {len(image_entries)} image(s) ---")
    all_results = []
    for image_path, root_dir in image_entries:
        image_label, subfolder = derive_labels(image_path, root_dir)
        df = process_image(
            image_path, output_dir, args.scale_nm, args.scale_px, image_label, subfolder
        )
        if len(df) > 0:
            all_results.append(df)

    # Aggregate results
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(output_dir / "all_particles.csv", index=False)

        n = len(df_all)
        n_images = len(all_results)
        print(f"\n--- Summary ({n} particles across {n_images} images) ---")
        d = df_all["diam_nm"]
        if n > 1:
            print(f"  Diameter: {d.mean():.1f} +/- {d.std():.1f} nm")
        else:
            print(f"  Diameter: {d.mean():.1f} nm")
        print(f"  Median:   {d.median():.1f} nm")
        print(f"  Range:    {d.min():.1f} - {d.max():.1f} nm")
        walls = df_all["wall_nm"].dropna()
        if len(walls) > 1:
            print(f"  Wall:     {walls.mean():.1f} +/- {walls.std():.1f} nm")
        elif len(walls) == 1:
            print(f"  Wall:     {walls.mean():.1f} nm")

        print("\n--- Files ---")
        print(f"  {output_dir}/all_particles.csv")
        print(f"  {output_dir}/<image_name>/detections.png")
        print(f"  {output_dir}/<image_name>/profile_*.png")
        print(f"  {output_dir}/<image_name>/particles.csv")
    else:
        print("\nNo particles found in any image.")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
