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
CELLPOSE_DIAMETER = 120
CELLPOSE_CELLPROB = 0.0
CELLPOSE_FLOW = 0.8

MIN_DIAM_NM = 80
MAX_DIAM_NM = 500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Scale bar detection
# ---------------------------------------------------------------------------
def detect_scale_bar(img: np.ndarray) -> int | None:
    """
    Auto-detect scale bar length in pixels from the bottom of a TEM image.

    Uses morphological filtering to find horizontal line structures.
    Returns the bar length in pixels, or None if not found.
    """
    h, w = img.shape[:2]
    bottom = img[int(h * 0.85) :, :]

    _, binary = cv2.threshold(bottom, 20, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz_lines)

    best_width = 0
    for i in range(1, n_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        if comp_w > w * 0.05 and comp_h < h * 0.05 and comp_w > best_width:
            best_width = comp_w

    return best_width if best_width > 0 else None


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

        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def detect_scale_text(img: np.ndarray) -> float | None:
    """
    Read scale bar text (e.g. '200.0nm') from the bottom of the image using OCR.

    Returns the scale value in nm, or None if not found.
    """
    h, _w = img.shape[:2]
    bottom = img[int(h * 0.85) :, :]

    reader = _get_ocr_reader()
    results = reader.readtext(bottom)

    for _bbox, text, _conf in results:
        # Fix common OCR mistakes
        cleaned = text.replace("O", "0").replace("l", "1").replace(",", ".")
        match = _SCALE_PATTERN.search(cleaned)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit in ("um", "μm"):
                value *= 1000  # convert um to nm
            return value

    return None


def determine_scale(img: np.ndarray, scale_nm: float | None) -> tuple[float | None, int | None]:
    """
    Determine nm/pixel scale. Returns (nm_per_pixel, bar_pixels).

    Strategy:
    1. Auto-detect bar pixel length from the image
    2. Use provided scale_nm, or try OCR to read the nm value from the image
    3. Return (None, ...) if either piece is missing
    """
    bar_px = detect_scale_bar(img)

    # If scale_nm not provided, try OCR
    if scale_nm is None:
        ocr_nm = detect_scale_text(img)
        if ocr_nm is not None:
            print(f"  OCR detected scale text: {ocr_nm}nm")
            scale_nm = ocr_nm

    if bar_px is not None and scale_nm is not None:
        return scale_nm / bar_px, bar_px

    if bar_px is not None:
        print(f"  WARNING: detected scale bar = {bar_px}px but could not read the nm value.")
        print("  Use --scale-nm to specify (e.g. --scale-nm 200)")
        return None, bar_px

    if scale_nm is not None:
        print("  WARNING: could not auto-detect scale bar pixels.")
        print("  Use --scale-px to specify, or provide both --scale-nm and --scale-px.")
        return None, None

    print("  WARNING: could not detect scale bar or read scale text.")
    print("  Use --scale-nm and --scale-px to specify manually.")
    return None, None


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


def run_cellpose(roi: np.ndarray) -> np.ndarray:
    model = get_cellpose_model()
    masks, _, _ = model.eval(
        roi,
        diameter=CELLPOSE_DIAMETER,
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


def measure_particles(roi: np.ndarray, masks: np.ndarray, nm_per_pixel: float) -> pd.DataFrame:
    props = measure.regionprops(masks)
    rows = []
    for p in props:
        area_nm2 = p.area * nm_per_pixel**2
        diam_nm = 2 * np.sqrt(area_nm2 / np.pi)

        if diam_nm < MIN_DIAM_NM or diam_nm > MAX_DIAM_NM:
            continue

        circ = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        radius_px = np.sqrt(p.area / np.pi)

        wall_info = measure_wall_thickness(roi, cy, cx, radius_px, nm_per_pixel)

        if not wall_info["is_vesicle"]:
            continue

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
                "_wall_info": wall_info,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_scale_bar(vis: np.ndarray, nm_per_pixel: float, length_nm: float = 200.0):
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
    roi: np.ndarray, masks: np.ndarray, df: pd.DataFrame, nm_per_pixel: float
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

    draw_scale_bar(vis, nm_per_pixel)
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
def make_unique_dir(output_dir: Path, image_path: Path) -> Path:
    """Create a unique output subdirectory for an image, handling name collisions."""
    base = image_path.stem
    candidate = output_dir / base
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    # Append parent directory name to disambiguate
    parent_name = image_path.parent.name
    candidate = output_dir / f"{parent_name}_{base}"
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = output_dir / f"{parent_name}_{base}_{counter}"
    candidate.mkdir(parents=True)
    return candidate


def process_image(
    image_path: Path,
    output_dir: Path,
    scale_nm: float | None,
    scale_px: int | None,
    roi_fraction: float,
) -> pd.DataFrame:
    """Process a single TEM image. Returns DataFrame of particle measurements."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ERROR: could not load {image_path}")
        return pd.DataFrame()

    # Per-image scale calibration
    if scale_px is not None and scale_nm is not None:
        nm_per_pixel = scale_nm / scale_px
    else:
        nm_per_pixel, _auto_bar_px = determine_scale(img, scale_nm)

    if nm_per_pixel is None:
        print(f"  SKIPPING {image_path.name}: could not determine scale.")
        print("  Provide --scale-nm and --scale-px, or ensure image has a visible scale bar.")
        return pd.DataFrame()

    roi_y = int(img.shape[0] * roi_fraction)
    roi = img[:roi_y, :]

    # Detect
    t0 = time.time()
    masks = run_cellpose(roi)
    elapsed = time.time() - t0
    raw_count = masks.max()

    # Measure
    df = measure_particles(roi, masks, nm_per_pixel)
    print(
        f"  {image_path.name}: {len(df)} particles ({raw_count} raw)"
        f" [{elapsed:.1f}s] scale={nm_per_pixel:.3f} nm/px"
    )

    if len(df) == 0:
        return pd.DataFrame()

    # Print per-image results
    for _, row in df.iterrows():
        wall_str = f" w={row['wall_nm']:.0f}nm" if pd.notna(row.get("wall_nm")) else ""
        print(f"    #{row['id']} d={row['diam_nm']:.0f}nm{wall_str}")

    # Save per-image outputs
    img_dir = make_unique_dir(output_dir, image_path)

    cv2.imwrite(str(img_dir / "roi.png"), roi)

    vis = draw_detections(roi, masks, df, nm_per_pixel)
    cv2.imwrite(str(img_dir / "detections.png"), vis)

    save_profiles(df, img_dir, nm_per_pixel)

    export_cols = [c for c in df.columns if not c.startswith("_")]
    df[export_cols].to_csv(img_dir / "particles.csv", index=False)

    # Add source image and scale columns for aggregate
    df_export = df[export_cols].copy()
    df_export.insert(0, "image", image_path.name)
    df_export["nm_per_pixel"] = round(nm_per_pixel, 4)
    return df_export


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_images(paths: list[str]) -> list[Path]:
    """Resolve CLI paths into a list of image files."""
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in IMAGE_EXTENSIONS:
                images.extend(sorted(path.glob(f"*{ext}")))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            print(f"  Skipping: {path}")
    return images


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
        "--roi-fraction",
        type=float,
        default=0.72,
        help="Fraction of image height to use as ROI from the top (default: 0.72)",
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory (default: output)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect images
    image_paths = collect_images(args.images)
    if not image_paths:
        print("No images found.")
        sys.exit(1)

    print("\n=== TEM particle analysis ===")
    print(f"Images: {len(image_paths)}")
    print(f"Diameter range: {MIN_DIAM_NM}-{MAX_DIAM_NM} nm")
    print(f"ROI fraction: {args.roi_fraction}")
    if args.scale_nm:
        print(f"Scale bar value: {args.scale_nm} nm")
    if args.scale_px:
        print(f"Scale bar pixels: {args.scale_px} px")
    if not args.scale_nm and not args.scale_px:
        print("Scale: auto-detect per image (provide --scale-nm for accuracy)")

    # Process each image (scale detected per-image)
    print(f"\n--- Processing {len(image_paths)} image(s) ---")
    all_results = []
    for image_path in image_paths:
        df = process_image(image_path, output_dir, args.scale_nm, args.scale_px, args.roi_fraction)
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
