"""
Interactive particle annotation with SAM (Segment Anything for Microscopy).

Click each particle once; SAM segments it precisely from that point prompt.
This combines human recall (you find the faint particles the automatic
detector misses) with SAM's boundary precision (accurate size/wall). It is
the recommended path when fully-automatic detection isn't reliable enough.

Why prompted SAM and not automatic SAM: in automatic mode SAM misses faint,
low-edge-contrast vesicles entirely. Prompted at a point it traces the
boundary cleanly (validated on these images). The light-microscopy weights
(vit_b_lm) match round vesicle-like blobs better than the EM-organelle ones.

Works on a single image, several images, or a whole folder (recursing). The
SAM model loads once and is reused across every image. Each saved image gets
the same per-image outputs as the batch pipeline (detections.png,
particles.csv, profiles) plus a combined output_annotated/all_particles.csv.

Usage:
    uv run python annotate.py "images/AMOSTRA ANA/MP Ana_SA-MAG_X50k_026.bmp"
    uv run python annotate.py images/                 # whole tree
    uv run python annotate.py "images/AMOSTRA ANA/"   # one folder
    uv run python annotate.py IMAGE --scale-nm 100

Controls (matplotlib window):
    left click   segment a particle at that point and add it
    right click  remove the particle under the cursor (or use 'u')
    u            undo last added particle
    s            save this image's particles and advance to the next
    n            skip this image (no particles) and advance
    q            quit the whole session
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

import measure as m

# measure.py selects the headless "Agg" backend on import. Override it *after*
# that import with an interactive one so the annotation window can open. Try a
# few in order: QtAgg (PyQt5 ships with micro_sam's napari), then the macOS
# native backend, then Tk.
for _backend in ("QtAgg", "macosx", "TkAgg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)

# SAM works at 1024 internally; downsample large micrographs to this for the
# embedding and prompting, then measurements are done at this work resolution.
WORK_DIM = 1024
DEFAULT_MODEL = "vit_b_lm"


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_scale(image_path: Path, img: np.ndarray, scale_nm: float | None):
    """Return (bar_nm, bar_px) at full resolution, or (None, None)."""
    bar = m.detect_scale_bar(img)
    bar_px = bar.width_px if bar is not None else None
    if scale_nm is None:
        tiff_nm = m.read_tiff_scale_nm(image_path)
        scale_nm = tiff_nm if tiff_nm is not None else m.detect_scale_text(img, bar)
    return scale_nm, bar_px


class ImageAnnotator:
    """One interactive window for a single image, using a shared SAM predictor."""

    def __init__(
        self, image_path, image_label, subfolder, output_dir, predictor, scale_nm, progress
    ):
        self.image_path = image_path
        self.image_label = image_label
        self.subfolder = subfolder
        self.output_dir = output_dir
        self.predictor = predictor
        self.progress = progress  # e.g. "3/19"

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"could not load {image_path}")

        bar_nm, bar_px = _resolve_scale(image_path, img, scale_nm)
        if bar_nm is None or bar_px is None or bar_nm <= 0 or bar_px <= 0:
            raise ValueError("could not determine scale (no readable scale bar)")

        h, w = img.shape
        self.sf = min(1.0, WORK_DIM / max(h, w))
        if self.sf < 1.0:
            self.work = cv2.resize(
                img, (int(w * self.sf), int(h * self.sf)), interpolation=cv2.INTER_AREA
            )
        else:
            self.work = img
        self.nm_per_pixel = bar_nm / (bar_px * self.sf)  # at work resolution
        self.bar_nm = bar_nm

        self.predictor.set_image(cv2.cvtColor(m.normalize_intensity(self.work), cv2.COLOR_GRAY2RGB))

        self.masks: list[np.ndarray] = []  # bool masks at work resolution
        self.action: str | None = None  # 'save' | 'skip' | 'quit'
        self.saved_df: pd.DataFrame | None = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.work, cmap="gray")
        self.overlay = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self._redraw()

    def _labels(self) -> np.ndarray:
        labels = np.zeros(self.work.shape, dtype=np.int32)
        for i, mk in enumerate(self.masks, start=1):
            labels[mk] = i
        return labels

    def _redraw(self):
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None
        if self.masks:
            rgba = np.zeros((*self.work.shape, 4), dtype=np.float32)
            rng = np.random.default_rng(0)
            for mk in self.masks:
                c = rng.random(3)
                rgba[mk, :3] = c
                rgba[mk, 3] = 0.45
            self.overlay = self.ax.imshow(rgba)
        self.ax.set_title(
            f"[{self.progress}] {self.image_path.name} — {len(self.masks)} particle(s)\n"
            "click=add  right-click/u=undo   s=save+next   n=skip   q=quit"
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = round(event.xdata), round(event.ydata)
        if event.button == 3:  # right click = remove particle under cursor
            for i in range(len(self.masks) - 1, -1, -1):
                if self.masks[i][y, x]:
                    del self.masks[i]
                    self._redraw()
                    return
            return
        if event.button != 1:
            return
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(bool)
        frac = best.mean()
        if frac > 0.5:  # selected most of the frame — almost certainly background
            print(f"  ignored click ({x},{y}): mask covers {frac * 100:.0f}% of image (background)")
            return
        self.masks.append(best)
        print(f"  + particle {len(self.masks)} (score {scores.max():.2f})")
        self._redraw()

    def on_key(self, event):
        if event.key == "u":
            if self.masks:
                self.masks.pop()
                self._redraw()
        elif event.key == "s":
            self.action = "save"
            self._save()
            plt.close(self.fig)
        elif event.key == "n":
            self.action = "skip"
            plt.close(self.fig)
        elif event.key == "q":
            self.action = "quit"
            self._save()
            plt.close(self.fig)

    def on_close(self, event):
        # Window closed via the OS button: treat as quit if no explicit action.
        if self.action is None:
            self.action = "quit"
            self._save()

    def _save(self):
        if not self.masks:
            return
        labels = self._labels()
        df = m.measure_regions(self.work, labels, self.nm_per_pixel)
        if df.empty:
            return

        img_dir = m.make_unique_dir(self.output_dir, self.image_path, self.subfolder)
        cv2.imwrite(str(img_dir / "roi.png"), self.work)
        vis = m.draw_detections(self.work, labels, df, self.nm_per_pixel, self.bar_nm)
        cv2.imwrite(str(img_dir / "detections.png"), vis)
        m.save_profiles(df, img_dir, self.nm_per_pixel)

        export = df[[c for c in df.columns if not c.startswith("_")]].copy()
        export.to_csv(img_dir / "particles.csv", index=False)
        export.insert(0, "image", self.image_label)
        export["nm_per_pixel"] = round(self.nm_per_pixel, 4)
        export["scale_bar_nm"] = self.bar_nm
        self.saved_df = export

        print(f"  saved {len(df)} particle(s) -> {img_dir}")
        for _, r in df.iterrows():
            wall = f" wall={r['wall_nm']:.0f}nm" if r["wall_nm"] is not None else ""
            ves = "vesicle" if r["is_vesicle"] else "solid"
            print(f"    #{r['id']} d={r['diam_nm']:.0f}nm{wall} [{ves}]")


def main():
    ap = argparse.ArgumentParser(description="Interactive SAM particle annotation")
    ap.add_argument("images", nargs="+", help="Image file(s) or folder(s)")
    ap.add_argument(
        "--scale-nm", type=float, default=None, help="Scale bar value in nm (else auto)"
    )
    ap.add_argument("--output", default="output_annotated", help="Output dir")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"micro_sam model (default {DEFAULT_MODEL})"
    )
    ap.add_argument("--device", default=None, help="cpu/mps/cuda (default auto)")
    args = ap.parse_args()

    entries = m.collect_images(args.images)
    if not entries:
        print("No images found.")
        sys.exit(1)

    device = args.device or _pick_device()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f"\n=== Interactive annotation: {len(entries)} image(s) ===")
    print(f"  loading SAM model '{args.model}' on {device} (first run downloads weights)...")
    from micro_sam.util import get_sam_model

    predictor = get_sam_model(model_type=args.model, device=device)
    print("  ready.\n  controls: click=add  right-click/u=undo  s=save+next  n=skip  q=quit\n")

    results: list[pd.DataFrame] = []
    for idx, (image_path, root_dir) in enumerate(entries, start=1):
        image_label, subfolder = m.derive_labels(image_path, root_dir)
        progress = f"{idx}/{len(entries)}"
        print(f"[{progress}] {image_label}")
        try:
            ann = ImageAnnotator(
                image_path, image_label, subfolder, output_dir, predictor, args.scale_nm, progress
            )
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        plt.show()  # blocks until this image's window is closed

        if ann.saved_df is not None:
            results.append(ann.saved_df)
        if ann.action == "quit":
            print("  quitting session.")
            break

    if results:
        df_all = pd.concat(results, ignore_index=True)
        agg = output_dir / "all_particles.csv"
        df_all.to_csv(agg, index=False)
        n, n_imgs = len(df_all), len(results)
        print(f"\n--- Saved {n} particle(s) across {n_imgs} image(s) ---")
        d = df_all["diam_nm"]
        print(
            f"  Diameter: {d.mean():.1f} +/- {d.std():.1f} nm"
            if n > 1
            else f"  Diameter: {d.mean():.1f} nm"
        )
        print(f"  Median:   {d.median():.1f} nm")
        print(f"  Range:    {d.min():.1f} - {d.max():.1f} nm")
        print(f"  Aggregate CSV: {agg}")
    else:
        print("\nNo particles saved.")
    print("\n=== Done ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
