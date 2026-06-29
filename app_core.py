"""
Core logic for the particle measurement web app, kept UI-free so it can be
tested headlessly and reused by app.py (Streamlit).

A "particle" is stored as a polygon (contour) in WORK-resolution coordinates
plus its source ("auto" or "manual"). Polygons are compact, JSON-serialisable,
editable, and rasterisable back to a mask for (re-)measurement — so a saved
session fully round-trips: reopen, verify, edit, re-measure, re-export.

The annotation store on disk (annotations.json) maps each image's label to its
scale and particle polygons. Measurements are always recomputed from polygons
+ scale, never trusted from a stale cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import measure as m

WORK_DIM = 1024  # SAM's native size; large micrographs are downsampled to this


# ---------------------------------------------------------------------------
# Image loading + scale
# ---------------------------------------------------------------------------
def load_work_image(image_path: Path) -> tuple[np.ndarray, float]:
    """Load grayscale image downsampled to <=WORK_DIM. Returns (work_img, scale_factor)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not load {image_path}")
    h, w = img.shape
    sf = min(1.0, WORK_DIM / max(h, w))
    if sf < 1.0:
        work = cv2.resize(img, (int(w * sf), int(h * sf)), interpolation=cv2.INTER_AREA)
    else:
        work = img
    return work, sf


def auto_scale(
    image_path: Path, full_img: np.ndarray | None = None
) -> tuple[float | None, int | None]:
    """Best-effort (scale_nm, scale_px) at full resolution: TIFF metadata, then OCR."""
    img = full_img if full_img is not None else cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    bar = m.detect_scale_bar(img)
    bar_px = bar.width_px if bar is not None else None
    scale_nm = m.read_tiff_scale_nm(image_path)
    if scale_nm is None:
        scale_nm = m.detect_scale_text(img, bar)
    return scale_nm, bar_px


# ---------------------------------------------------------------------------
# Particle model
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    contour: list[list[int]]  # [[x, y], ...] in work-resolution coords
    source: str = "auto"  # "auto" | "manual"

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        m_ = np.zeros(shape, dtype=np.uint8)
        pts = np.array(self.contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m_, [pts], 1)
        return m_.astype(bool)


@dataclass
class ImageState:
    label: str  # display/CSV label, e.g. "AMOSTRA ANA/x.bmp"
    scale_nm: float | None = None  # full-res scale bar value
    scale_px: int | None = None  # full-res scale bar pixel length
    sf: float = 1.0  # work/full resolution ratio
    particles: list[Particle] = field(default_factory=list)
    reviewed: bool = False

    @property
    def nm_per_pixel(self) -> float | None:
        """nm per pixel at WORK resolution (where contours live)."""
        if not self.scale_nm or not self.scale_px:
            return None
        return self.scale_nm / (self.scale_px * self.sf)


def mask_to_contour(mask: np.ndarray) -> list[list[int]] | None:
    """Largest external contour of a boolean mask as [[x, y], ...]."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt.reshape(-1, 2).tolist()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def auto_detect(work: np.ndarray, state: ImageState) -> list[Particle]:
    """Run the LoG detector + filters on the work image; return particle polygons."""
    nmpp = state.nm_per_pixel
    if nmpp is None:
        return []
    # run_log_detector/measure_particles operate in the given image's own
    # pixel space, so passing the work image with work-res nm/px and a work-res
    # scale-bar length keeps the size-vs-bar filter consistent.
    bar_px_work = max(1, round(state.scale_px * state.sf))
    masks = m.run_log_detector(work, nmpp, state.scale_nm)
    df = m.measure_particles(work, masks, nmpp, state.scale_nm, bar_px_work)
    particles = []
    for _, row in df.iterrows():
        mask = masks == row["id"]
        contour = mask_to_contour(mask)
        if contour is not None:
            particles.append(Particle(contour=contour, source="auto"))
    return particles


def sam_predict(predictor, points: list[tuple[int, int, int]], scale_idx: int = 0):
    """
    Run SAM from accumulated points. `points` is [(x, y, label)] with label
    1=positive/0=negative. Returns (list_of_candidate_masks, best_index).
    scale_idx is unused here (caller selects); kept for API symmetry.
    """
    coords = np.array([[x, y] for x, y, _ in points])
    labels = np.array([lab for _, _, lab in points])
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels, multimask_output=True
    )
    return [mk.astype(bool) for mk in masks], int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_state(work: np.ndarray, state: ImageState) -> pd.DataFrame:
    """Measure all particles in an image state. Empty frame if no scale/particles."""
    nmpp = state.nm_per_pixel
    if nmpp is None or not state.particles:
        return pd.DataFrame()
    labels = np.zeros(work.shape, dtype=np.int32)
    for i, p in enumerate(state.particles, start=1):
        labels[p.mask(work.shape)] = i
    df = m.measure_regions(work, labels, nmpp)
    if not df.empty:
        df["source"] = [state.particles[i - 1].source for i in df["id"]]
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_annotations(path: Path, states: dict[str, ImageState]) -> None:
    data = {}
    for key, st in states.items():
        d = asdict(st)
        d["particles"] = [asdict(p) for p in st.particles]
        data[key] = d
    path.write_text(json.dumps(data, indent=1))


def load_annotations(path: Path) -> dict[str, ImageState]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    states: dict[str, ImageState] = {}
    for key, d in raw.items():
        parts = [Particle(**p) for p in d.get("particles", [])]
        states[key] = ImageState(
            label=d["label"],
            scale_nm=d.get("scale_nm"),
            scale_px=d.get("scale_px"),
            sf=d.get("sf", 1.0),
            particles=parts,
            reviewed=d.get("reviewed", False),
        )
    return states


def export_csv(
    path: Path, work_images: dict[str, np.ndarray], states: dict[str, ImageState]
) -> pd.DataFrame:
    """Recompute every image's measurements and write the corpus CSV. Returns the frame."""
    frames = []
    for key, st in states.items():
        work = work_images.get(key)
        if work is None:
            continue
        df = measure_state(work, st)
        if df.empty:
            continue
        export = df[[c for c in df.columns if not c.startswith("_")]].copy()
        export.insert(0, "image", st.label)
        export["nm_per_pixel"] = round(st.nm_per_pixel, 4)
        export["scale_bar_nm"] = st.scale_nm
        frames.append(export)
    if not frames:
        path.write_text("")
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(path, index=False)
    return df_all
