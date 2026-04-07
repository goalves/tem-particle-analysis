"""
TEM nanoparticle size analysis.

Detects large circular particles in TEM images using Cellpose.
Measures outer diameter and wall thickness (via radial intensity profile).
Auto-detects scale bar from image when possible.

Usage:
    uv run python analise.py images/sample.jpeg
    uv run python analise.py images/*.jpeg
    uv run python analise.py images/ --scale-nm 200
"""

import argparse
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
ROI_TOP_FRACTION = 0.72  # crop grid hole at the bottom

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

    # Threshold to find dark pixels
    _, binary = cv2.threshold(bottom, 20, 255, cv2.THRESH_BINARY_INV)

    # Morphological opening with horizontal kernel to isolate horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz_lines)

    best_width = 0
    for i in range(1, n_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        # Scale bar: wide horizontal structure, not too tall
        if comp_w > w * 0.05 and comp_h < h * 0.05 and comp_w > best_width:
            best_width = comp_w

    return best_width if best_width > 0 else None


def parse_scale_text(img: np.ndarray) -> float | None:
    """
    Try to read scale bar text (e.g. '200.0nm') from the bottom of the image.

    Uses template matching on common patterns. Returns nm value or None.
    """
    # This is hard to do reliably without OCR.
    # For now, return None and let the user provide --scale-nm.
    return None


def determine_scale(img: np.ndarray, scale_nm: float | None) -> tuple[float, int | None]:
    """
    Determine nm/pixel scale. Returns (nm_per_pixel, bar_pixels).

    Priority:
    1. Auto-detect bar pixels from image + user-provided scale_nm
    2. User-provided scale_nm + auto-detected bar pixels
    3. Fall back to user-provided values
    """
    bar_px = detect_scale_bar(img)

    if bar_px is not None and scale_nm is not None:
        nm_per_pixel = scale_nm / bar_px
        return nm_per_pixel, bar_px

    if bar_px is not None and scale_nm is None:
        print(f"  WARNING: detected scale bar = {bar_px}px but don't know the nm value.")
        print("  Use --scale-nm to specify (e.g. --scale-nm 200)")
        print("  Assuming 200nm for now.")
        nm_per_pixel = 200.0 / bar_px
        return nm_per_pixel, bar_px

    if scale_nm is not None:
        print("  WARNING: could not auto-detect scale bar. Using --scale-nm with default bar.")
        # Fall back to a reasonable default
        nm_per_pixel = scale_nm / 208
        return nm_per_pixel, None

    print("  WARNING: no scale info. Use --scale-nm and/or --scale-px.")
    print("  Assuming 200nm / 208px for now.")
    return 200.0 / 208, None


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


def measure_wall_thickness(
    roi: np.ndarray, cy: int, cx: int, radius_px: float, nm_per_pixel: float
) -> dict:
    """
    Measure wall thickness via radial intensity profile.

    Returns dict with wall_px, radii_nm, profile, threshold, inner/outer edges.
    """
    n_angles = 360
    max_r = int(radius_px * 1.8)
    profile = np.zeros(max_r)
    counts = np.zeros(max_r)

    for angle in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for r in range(max_r):
            x = int(cx + r * cos_a)
            y = int(cy + r * sin_a)
            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                profile[r] += roi[y, x]
                counts[r] += 1

    valid = counts > 0
    profile[valid] /= counts[valid]
    profile_smooth = uniform_filter1d(profile, size=7)

    radii_nm = np.arange(max_r) * nm_per_pixel

    # Wall = region darker than midpoint between background and darkest point
    bg = np.mean(profile_smooth[max(0, max_r - 30) : max_r])
    darkest = np.min(profile_smooth[5:])
    threshold = (bg + darkest) / 2

    below = profile_smooth < threshold
    if not np.any(below):
        return {
            "wall_px": None,
            "radii_nm": radii_nm,
            "profile": profile_smooth,
            "threshold": threshold,
            "inner_nm": None,
            "outer_nm": None,
        }

    indices = np.where(below)[0]
    inner_px = indices[0]
    outer_px = indices[-1]
    wall_px = float(outer_px - inner_px)

    return {
        "wall_px": wall_px,
        "radii_nm": radii_nm,
        "profile": profile_smooth,
        "threshold": threshold,
        "inner_nm": inner_px * nm_per_pixel,
        "outer_nm": outer_px * nm_per_pixel,
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
def process_image(
    image_path: Path, output_dir: Path, nm_per_pixel: float, scale_bar_px: int | None
) -> pd.DataFrame:
    """Process a single TEM image. Returns DataFrame of particle measurements."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ERROR: could not load {image_path}")
        return pd.DataFrame()

    roi_y = int(img.shape[0] * ROI_TOP_FRACTION)
    roi = img[:roi_y, :]

    # Detect
    t0 = time.time()
    masks = run_cellpose(roi)
    elapsed = time.time() - t0
    raw_count = masks.max()

    # Measure
    df = measure_particles(roi, masks, nm_per_pixel)
    print(f"  {image_path.name}: {len(df)} particles ({raw_count} raw) [{elapsed:.1f}s]")

    if len(df) == 0:
        return pd.DataFrame()

    # Print per-image results
    for _, row in df.iterrows():
        wall_str = f" w={row['wall_nm']:.0f}nm" if pd.notna(row.get("wall_nm")) else ""
        print(f"    #{row['id']} d={row['diam_nm']:.0f}nm{wall_str}")

    # Save per-image outputs
    img_dir = output_dir / image_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_dir / "roi.png"), roi)

    vis = draw_detections(roi, masks, df, nm_per_pixel)
    cv2.imwrite(str(img_dir / "detections.png"), vis)

    save_profiles(df, img_dir, nm_per_pixel)

    export_cols = [c for c in df.columns if not c.startswith("_")]
    df[export_cols].to_csv(img_dir / "particles.csv", index=False)

    # Add source image column for aggregate
    df_export = df[export_cols].copy()
    df_export.insert(0, "image", image_path.name)
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
    parser.add_argument("--scale-px", type=int, default=None, help="Scale bar length in pixels")
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

    # Determine scale from first image
    print("\n--- Scale calibration ---")
    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f"ERROR: could not load {image_paths[0]}")
        sys.exit(1)

    if args.scale_px is not None and args.scale_nm is not None:
        nm_per_pixel = args.scale_nm / args.scale_px
        bar_px = args.scale_px
        print(f"  Scale (from args): {bar_px}px = {args.scale_nm}nm -> {nm_per_pixel:.3f} nm/px")
    else:
        nm_per_pixel, bar_px = determine_scale(first_img, args.scale_nm)
        if bar_px is not None:
            print(f"  Scale bar detected: {bar_px}px -> {nm_per_pixel:.3f} nm/px")
        else:
            print(f"  Scale: {nm_per_pixel:.3f} nm/px (default/assumed)")

    print(f"  Diameter range: {MIN_DIAM_NM}-{MAX_DIAM_NM} nm")
    cp_nm = CELLPOSE_DIAMETER * nm_per_pixel
    print(f"  Cellpose diameter: {CELLPOSE_DIAMETER} px ({cp_nm:.0f} nm)")

    # Process each image
    print(f"\n--- Processing {len(image_paths)} image(s) ---")
    all_results = []
    for image_path in image_paths:
        df = process_image(image_path, output_dir, nm_per_pixel, bar_px)
        if len(df) > 0:
            all_results.append(df)

    # Aggregate results
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(output_dir / "all_particles.csv", index=False)

        print(f"\n--- Summary ({len(df_all)} particles across {len(all_results)} images) ---")
        d = df_all["diam_nm"]
        print(f"  Diameter: {d.mean():.1f} +/- {d.std():.1f} nm")
        print(f"  Median:   {d.median():.1f} nm")
        print(f"  Range:    {d.min():.1f} - {d.max():.1f} nm")
        walls = df_all["wall_nm"].dropna()
        if len(walls) > 0:
            print(f"  Wall:     {walls.mean():.1f} +/- {walls.std():.1f} nm")

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
