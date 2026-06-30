"""
Particle measurement web app.

One coherent application to measure particle sizes across an image corpus:

- A file explorer (left) lists every image grouped by folder with a review
  status: ○ unreviewed, ✓ reviewed, ⚑ flagged for re-review.
- Opening an image is fast — nothing is computed until you ask. Run auto-detect
  (LoG or Cellpose) when you want it, or add particles by hand with SAM.
- Correct freely: add (SAM point-prompt), remove, re-segment, fix the scale bar.
- Zoom and pan the image to click precisely.
- Everything persists to annotations.json beside the images, so you can close,
  reopen, verify and re-measure later.

Output:
- measurements.csv : the corpus result, one row per particle (same columns as
  the batch pipeline, plus a 'source' = auto|manual).
- annotations.json : the resumable session (scale + particle polygons + status).

Run:  uv run --group app streamlit run app.py     (or: just app)
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import app_core as c
import measure as m

st.set_page_config(page_title="Particle measurement", layout="wide")

AUTO_COLOR = (60, 220, 60)
MANUAL_COLOR = (0, 200, 255)
DISPLAY_W = 760  # width of the viewer in px


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SAM model (first time only)...")
def get_predictor():
    import torch
    from micro_sam.util import get_sam_model

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return get_sam_model(model_type="vit_b_lm", device=dev)


@st.cache_data(show_spinner=False)
def load_work_cached(path_str: str):
    return c.load_work_image(Path(path_str))


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
def init_corpus(folder: str):
    ss = st.session_state
    ss.entries = [(p, m.derive_labels(p, root)[0]) for p, root in m.collect_images([folder])]
    ss.ann_path = Path(folder) / "annotations.json"
    ss.states = c.load_annotations(ss.ann_path)
    ss.idx = 0
    ss.corpus_folder = folder
    reset_view()
    reset_drawing()
    ss.embedded_label = None


def reset_drawing():
    ss = st.session_state
    ss.cur_points = []
    ss.cur_masks = None
    ss.cur_idx = 0
    ss.last_click = None


def reset_view():
    ss = st.session_state
    ss.zoom = 1.0
    ss.center = None  # set to image middle on first render


def current_entry():
    return st.session_state.entries[st.session_state.idx]


def go_to(idx: int):
    ss = st.session_state
    ss.idx = idx
    reset_view()
    reset_drawing()


def get_state(path: Path, label: str, sf: float) -> c.ImageState:
    """Fetch or create the state. Does NOT run OCR or detection (fast open)."""
    ss = st.session_state
    if label not in ss.states:
        ss.states[label] = c.ImageState(label=label, sf=sf)
    return ss.states[label]


def ensure_scale(state: c.ImageState, path: Path) -> bool:
    """Resolve the scale on demand (OCR / TIFF metadata). Returns True if known."""
    if state.nm_per_pixel is not None:
        return True
    nm, px = c.auto_scale(path)
    state.scale_nm, state.scale_px = nm, px
    persist()
    return state.nm_per_pixel is not None


def persist():
    c.save_annotations(st.session_state.ann_path, st.session_state.states)


def ensure_embedding(label: str, work):
    ss = st.session_state
    if ss.embedded_label != label:
        rgb = cv2.cvtColor(m.normalize_intensity(work), cv2.COLOR_GRAY2RGB)
        get_predictor().set_image(rgb)
        ss.embedded_label = label


def cur_mask():
    ss = st.session_state
    return None if ss.cur_masks is None else ss.cur_masks[ss.cur_idx]


# ---------------------------------------------------------------------------
# Rendering (full-res overlays, then crop+resize for the zoom/pan view)
# ---------------------------------------------------------------------------
def build_overlay(work, state: c.ImageState) -> np.ndarray:
    ss = st.session_state
    vis = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
    for i, p in enumerate(state.particles):
        pts = np.array(p.contour, dtype=np.int32).reshape(-1, 1, 2)
        color = MANUAL_COLOR if p.source == "manual" else AUTO_COLOR
        cv2.polylines(vis, [pts], True, color, 2)
        cxy = pts.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(vis, str(i + 1), tuple(cxy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cm = cur_mask()
    if cm is not None:
        ov = vis.copy()
        ov[cm] = (255, 255, 0)
        vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
        cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 0), 2)
    for x, y, lab in ss.cur_points:
        cv2.circle(vis, (x, y), 6, (0, 255, 0) if lab == 1 else (255, 0, 0), -1)
    return vis


def render_view(work, state: c.ImageState):
    """Return (display_img, window, disp_w, disp_h) for the current zoom/pan."""
    ss = st.session_state
    h, w = work.shape
    if ss.center is None:
        ss.center = (w // 2, h // 2)
    vis = build_overlay(work, state)
    window = c.view_window(work.shape, ss.zoom, ss.center)
    x0, y0, vw, vh = window
    crop = vis[y0 : y0 + vh, x0 : x0 + vw]
    disp_w = DISPLAY_W
    disp_h = max(1, round(vh * disp_w / vw))
    disp = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    return disp, window, disp_w, disp_h


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def handle_click(x, y, mode, work, state: c.ImageState):
    ss = st.session_state
    if mode == "remove":
        for i in range(len(state.particles) - 1, -1, -1):
            if state.particles[i].mask(work.shape)[y, x]:
                del state.particles[i]
                persist()
                return
        return
    ss.cur_points.append((x, y, 1 if mode == "+ point" else 0))
    ensure_embedding(state.label, work)
    masks, best = c.sam_predict(get_predictor(), ss.cur_points)
    ss.cur_masks, ss.cur_idx = masks, best


def undo_point(state: c.ImageState, work):
    """Remove the last prompt point of the in-progress particle and re-segment."""
    ss = st.session_state
    if not ss.cur_points:
        return
    ss.cur_points.pop()
    if ss.cur_points:
        ensure_embedding(state.label, work)
        ss.cur_masks, ss.cur_idx = c.sam_predict(get_predictor(), ss.cur_points)
    else:
        ss.cur_masks, ss.cur_idx = None, 0


def commit_current(state: c.ImageState):
    cm = cur_mask()
    if cm is None:
        return
    if cm.mean() > 0.5:
        st.warning("Mask covers most of the frame — looks like background, not committed.")
        return
    contour = c.mask_to_contour(cm)
    if contour is not None:
        state.particles.append(c.Particle(contour=contour, source="manual"))
        persist()
    reset_drawing()


def run_auto_detect(work, state: c.ImageState, path: Path, params: c.DetectParams):
    if not ensure_scale(state, path):
        st.warning("No scale for this image — set it in the Scale panel first.")
        return
    manual = [p for p in state.particles if p.source == "manual"]
    with st.spinner(f"Detecting with {params.detector}..."):
        state.particles = c.auto_detect(work, state, params) + manual
    state.detected = True
    persist()


def detection_settings_ui() -> c.DetectParams:
    """Expose every detector knob + filter; returns the chosen DetectParams."""
    d = c.DetectParams()  # defaults
    with st.expander("Detection settings", expanded=False):
        detector = st.selectbox("Detector", ["log", "cellpose"], key="p_detector")
        if detector == "log":
            log_threshold = st.slider(
                "LoG threshold (lower = more, noisier)",
                0.0,
                0.5,
                d.log_threshold,
                0.01,
                key="p_log_thr",
            )
            log_num_sigma = st.slider("LoG num_sigma", 3, 12, d.log_num_sigma, key="p_log_ns")
            log_overlap = st.slider("LoG overlap", 0.0, 1.0, d.log_overlap, 0.05, key="p_log_ov")
            cp_flow, cp_cellprob = d.cp_flow, d.cp_cellprob
        else:
            cp_flow = st.slider(
                "Cellpose flow_threshold", 0.0, 3.0, d.cp_flow, 0.1, key="p_cp_flow"
            )
            cp_cellprob = st.slider(
                "Cellpose cellprob_threshold (lower = more)",
                -6.0,
                6.0,
                d.cp_cellprob,
                0.5,
                key="p_cp_cp",
            )
            log_threshold, log_num_sigma, log_overlap = (
                d.log_threshold,
                d.log_num_sigma,
                d.log_overlap,
            )

        st.markdown("**Filters** — set to 0 / off to disable")
        col1, col2 = st.columns(2)
        min_diam = col1.slider("min diam ÷ bar", 0.0, 2.0, d.min_diam_vs_bar, 0.05, key="p_mind")
        max_diam = col2.slider("max diam ÷ bar", 1.0, 8.0, d.max_diam_vs_bar, 0.5, key="p_maxd")
        min_circ = col1.slider("min circularity", 0.0, 1.0, d.min_circularity, 0.05, key="p_circ")
        min_contrast = col2.slider("min contrast", 0.0, 60.0, d.min_contrast, 1.0, key="p_con")
        edge = col1.slider("edge margin frac", 0.0, 0.1, d.edge_margin_frac, 0.005, key="p_edge")
        grid = col2.checkbox("reject grid holes", d.reject_grid_holes, key="p_grid")

    return c.DetectParams(
        detector=detector,
        log_threshold=log_threshold,
        log_num_sigma=log_num_sigma,
        log_overlap=log_overlap,
        cp_flow=cp_flow,
        cp_cellprob=cp_cellprob,
        min_diam_vs_bar=min_diam,
        max_diam_vs_bar=max_diam,
        min_circularity=min_circ,
        min_contrast=min_contrast,
        edge_margin_frac=edge,
        reject_grid_holes=grid,
    )


# ---------------------------------------------------------------------------
# Sidebar: file explorer
# ---------------------------------------------------------------------------
def status_of(label: str) -> str:
    st_ = st.session_state.states.get(label)
    return st_.status if st_ else c.STATUS_UNREVIEWED


def sidebar_explorer():
    ss = st.session_state
    st.title("Particle measurement")
    folder = st.text_input("Image folder", value=ss.get("corpus_folder", "images"))
    if st.button("Load / reload corpus") or "entries" not in ss:
        init_corpus(folder)
    if not ss.entries:
        st.warning("No images found.")
        st.stop()

    # progress summary
    statuses = [status_of(lbl) for _, lbl in ss.entries]
    rev = statuses.count(c.STATUS_REVIEWED)
    flg = statuses.count(c.STATUS_FLAGGED)
    total = len(statuses)
    st.caption(f"✓ {rev} reviewed · ⚑ {flg} flagged · ○ {total - rev - flg} to do · {total} total")

    st.divider()
    # grouped file tree
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (_, label) in enumerate(ss.entries):
        folder_name = label.split("/")[0] if "/" in label else "."
        groups[folder_name].append(i)
    for folder_name, idxs in groups.items():
        n_rev = sum(status_of(ss.entries[i][1]) == c.STATUS_REVIEWED for i in idxs)
        with st.expander(f"{folder_name}  ({n_rev}/{len(idxs)})", expanded=ss.idx in idxs):
            for i in idxs:
                label = ss.entries[i][1]
                name = label.split("/")[-1]
                icon = c.STATUS_ICON[status_of(label)]
                marker = "➤ " if i == ss.idx else ""
                if st.button(f"{marker}{icon} {name}", key=f"nav_{i}", use_container_width=True):
                    go_to(i)
                    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def viewer_controls(work):
    ss = st.session_state
    h, w = work.shape
    z1, z2, z3, z4, z5, z6 = st.columns(6)
    ss.zoom = z1.slider("Zoom", 1.0, 8.0, ss.zoom, 0.5, label_visibility="collapsed")
    _, _, vw, vh = c.view_window(work.shape, ss.zoom, ss.center)
    step_x, step_y = int(vw * 0.3), int(vh * 0.3)
    cx, cy = ss.center
    if z2.button("⬅", use_container_width=True):
        ss.center = (max(0, cx - step_x), cy)
        ss.last_click = None
    if z3.button("➡", use_container_width=True):
        ss.center = (min(w, cx + step_x), cy)
        ss.last_click = None
    if z4.button("⬆", use_container_width=True):
        ss.center = (cx, max(0, cy - step_y))
        ss.last_click = None
    if z5.button("⬇", use_container_width=True):
        ss.center = (cx, min(h, cy + step_y))
        ss.last_click = None
    if z6.button("Reset", use_container_width=True):
        reset_view()


def main():
    ss = st.session_state
    with st.sidebar:
        sidebar_explorer()

    path, label = current_entry()
    work, sf = load_work_cached(str(path))
    state = get_state(path, label, sf)

    ss.setdefault("mode", "+ point")
    if ss.get("center") is None:
        ss.center = (work.shape[1] // 2, work.shape[0] // 2)

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"**{label}**  ·  {c.STATUS_ICON[state.status]} {state.status}")
        viewer_controls(work)
        disp, window, disp_w, disp_h = render_view(work, state)
        coords = streamlit_image_coordinates(disp, key=f"view_{ss.idx}")
        if coords is not None and (coords["x"], coords["y"]) != ss.last_click:
            ss.last_click = (coords["x"], coords["y"])
            wx, wy = c.display_to_work(coords["x"], coords["y"], window, disp_w, disp_h)
            wx = int(np.clip(wx, 0, work.shape[1] - 1))
            wy = int(np.clip(wy, 0, work.shape[0] - 1))
            handle_click(wx, wy, ss.mode, work, state)
            st.rerun()

    with right:
        ss.mode = st.radio("Click action", ["+ point", "- point", "remove"], horizontal=True)
        npts = len(ss.cur_points)
        st.caption(f"current particle: {npts} point(s)" if npts else "no particle in progress")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Commit", use_container_width=True):
            commit_current(state)
            st.rerun()
        if b2.button("Cycle scale", use_container_width=True) and ss.cur_masks is not None:
            ss.cur_idx = (ss.cur_idx + 1) % len(ss.cur_masks)
            st.rerun()
        if b3.button("Undo point", use_container_width=True):
            undo_point(state, work)
            st.rerun()
        if b4.button("Clear", use_container_width=True):
            reset_drawing()
            st.rerun()

        params = detection_settings_ui()
        if st.button(
            f"Run auto-detect ({params.detector})", type="primary", use_container_width=True
        ):
            run_auto_detect(work, state, path, params)
            st.rerun()

        with st.expander("Scale bar", expanded=state.nm_per_pixel is None):
            if st.button("Auto-read scale (OCR / TIFF)"):
                ensure_scale(state, path)
                st.rerun()
            sc_nm = st.number_input(
                "Scale value (nm)", value=float(state.scale_nm or 0.0), step=10.0
            )
            sc_px = st.number_input(
                "Scale length (px, full-res)", value=int(state.scale_px or 0), step=1
            )
            if st.button("Apply scale"):
                state.scale_nm = sc_nm or None
                state.scale_px = int(sc_px) or None
                persist()
                st.rerun()
            if state.nm_per_pixel:
                st.caption(f"{state.nm_per_pixel:.3f} nm/px (work)")

        # measurements
        df = c.measure_state(work, state)
        st.markdown(f"**{len(state.particles)} particle(s)**")
        if not df.empty:
            st.dataframe(
                df[["id", "diam_nm", "wall_nm", "circularity", "is_vesicle", "source"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"median {df['diam_nm'].median():.0f} nm | "
                f"range {df['diam_nm'].min():.0f}-{df['diam_nm'].max():.0f} nm"
            )
            # delete a listed particle by its # (matches the 'id' column / overlay label)
            d1, d2 = st.columns([3, 2])
            del_id = d1.selectbox("Delete particle #", list(df["id"]), key="del_id")
            if d2.button("Delete", use_container_width=True):
                # measure_state numbers particles 1..N in list order, so id-1 is the index
                del state.particles[int(del_id) - 1]
                persist()
                st.rerun()
            if st.button("Delete ALL particles on this image"):
                state.particles = []
                persist()
                st.rerun()

        # review status
        s1, s2, s3 = st.columns(3)
        if s1.button("✓ Reviewed", use_container_width=True):
            state.status = c.STATUS_REVIEWED
            persist()
            st.rerun()
        if s2.button("⚑ Flag", use_container_width=True):
            state.status = c.STATUS_FLAGGED
            persist()
            st.rerun()
        if s3.button("○ Clear", use_container_width=True):
            state.status = c.STATUS_UNREVIEWED
            persist()
            st.rerun()

        st.divider()
        if st.button("Export corpus CSV", use_container_width=True):
            work_images = {
                lbl: load_work_cached(str(p))[0] for p, lbl in ss.entries if lbl in ss.states
            }
            out = Path(ss.corpus_folder) / "measurements.csv"
            df_all = c.export_csv(out, work_images, ss.states)
            n_imgs = df_all["image"].nunique() if len(df_all) else 0
            st.success(f"Wrote {len(df_all)} particles across {n_imgs} images to {out}")
            if len(df_all):
                st.download_button("Download CSV", df_all.to_csv(index=False), "measurements.csv")


if __name__ == "__main__":
    main()
