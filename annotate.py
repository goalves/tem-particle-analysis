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

Usage:
    uv run python annotate.py "images/AMOSTRA ANA/MP Ana_SA-MAG_X50k_026.bmp"
    uv run python annotate.py IMAGE --scale-nm 100

Controls (matplotlib window):
    left click   segment a particle at that point and add it
    right click  remove the particle under the cursor (or use 'u')
    u            undo last added particle
    s            save measurements (CSV + detections.png + profiles)
    q            quit (prompts nothing; save first with 's')
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

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


class Annotator:
    def __init__(self, image_path: Path, output_dir: Path, scale_nm, model, device):
        self.image_path = image_path
        self.output_dir = output_dir

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Could not load {image_path}")

        bar_nm, bar_px = _resolve_scale(image_path, img, scale_nm)
        if bar_nm is None or bar_px is None or bar_nm <= 0 or bar_px <= 0:
            raise SystemExit(
                f"Could not determine scale for {image_path.name}. "
                "Pass --scale-nm (and ensure a scale bar is visible)."
            )

        h, w = img.shape
        self.sf = min(1.0, WORK_DIM / max(h, w))
        if self.sf < 1.0:
            self.work = cv2.resize(
                img, (int(w * self.sf), int(h * self.sf)), interpolation=cv2.INTER_AREA
            )
        else:
            self.work = img
        # nm per pixel at the work resolution
        self.nm_per_pixel = bar_nm / (bar_px * self.sf)
        self.bar_nm = bar_nm

        print(
            f"  scale: {bar_nm:.0f} nm / {bar_px} px (full) -> "
            f"{self.nm_per_pixel:.3f} nm/px at work res"
        )
        print(f"  loading SAM model '{model}' on {device} (first run downloads weights)...")
        from micro_sam.util import get_sam_model

        self.predictor = get_sam_model(model_type=model, device=device)
        norm = m.normalize_intensity(self.work)
        self.predictor.set_image(cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB))
        print("  ready. Click particles; press 's' to save, 'u' to undo, 'q' to quit.")

        self.masks: list[np.ndarray] = []  # list of bool masks at work res

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.work, cmap="gray")
        self.overlay = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._redraw()

    def _labels(self) -> np.ndarray:
        """Compose stored masks into a label image (later clicks drawn on top)."""
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
            f"{self.image_path.name} — {len(self.masks)} particle(s)   "
            "[click=add  right-click/u=undo  s=save  q=quit]"
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
            self.save()
        elif event.key == "q":
            plt.close(self.fig)

    def save(self):
        if not self.masks:
            print("  nothing to save (no particles).")
            return
        labels = self._labels()
        df = m.measure_regions(self.work, labels, self.nm_per_pixel)
        if df.empty:
            print("  no measurable particles.")
            return

        img_dir = m.make_unique_dir(self.output_dir, self.image_path)
        cv2.imwrite(str(img_dir / "roi.png"), self.work)
        vis = m.draw_detections(self.work, labels, df, self.nm_per_pixel, self.bar_nm)
        cv2.imwrite(str(img_dir / "detections.png"), vis)
        m.save_profiles(df, img_dir, self.nm_per_pixel)
        export = df[[c for c in df.columns if not c.startswith("_")]].copy()
        export.insert(0, "image", self.image_path.name)
        export["nm_per_pixel"] = round(self.nm_per_pixel, 4)
        export["scale_bar_nm"] = self.bar_nm
        export.to_csv(img_dir / "particles.csv", index=False)
        print(f"  saved {len(df)} particle(s) -> {img_dir}")
        for _, r in df.iterrows():
            wall = f" wall={r['wall_nm']:.0f}nm" if r["wall_nm"] is not None else ""
            print(f"    #{r['id']} d={r['diam_nm']:.0f}nm{wall}")


def main():
    ap = argparse.ArgumentParser(description="Interactive SAM particle annotation")
    ap.add_argument("image", help="TEM image to annotate")
    ap.add_argument(
        "--scale-nm", type=float, default=None, help="Scale bar value in nm (else auto)"
    )
    ap.add_argument("--output", default="output_annotated", help="Output dir")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"micro_sam model (default {DEFAULT_MODEL})"
    )
    ap.add_argument("--device", default=None, help="cpu/mps/cuda (default auto)")
    args = ap.parse_args()

    device = args.device or _pick_device()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f"\n=== Interactive annotation: {Path(args.image).name} ===")
    # Keep a reference so the figure's callbacks stay alive during plt.show().
    _annotator = Annotator(Path(args.image), output_dir, args.scale_nm, args.model, device)
    plt.show()
    del _annotator
    print("=== Done ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
