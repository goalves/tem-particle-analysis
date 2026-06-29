"""
Particle measurement web app.

One coherent application to measure particle sizes across a whole image corpus:
auto-detect on load, correct by hand (add / remove / re-segment), fix the scale
bar when OCR misreads, navigate the corpus, and export one CSV. All edits are
persisted to annotations.json next to the images, so you can close, reopen,
verify and re-measure any image later.

Run:
    uv run --group app streamlit run app.py
then set the image folder in the sidebar (default: images/).

Click actions (radio in the sidebar):
    + point   build a particle: click on it (add more clicks to grow it)
    - point   exclude a region SAM wrongly grabbed
    remove    click a particle to delete it
Use "Commit particle" to finalise the one you're building; "Cycle scale" takes
SAM's next mask size (small part -> whole object).
"""

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
SELECTED_COLOR = (255, 60, 60)


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SAM model...")
def get_predictor():
    import torch
    from micro_sam.util import get_sam_model

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return get_sam_model(model_type="vit_b_lm", device=device)


@st.cache_data(show_spinner=False)
def load_work_cached(path_str: str):
    work, sf = c.load_work_image(Path(path_str))
    return work, sf


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
def init_corpus(folder: str):
    entries = m.collect_images([folder])
    ss = st.session_state
    ss.entries = []  # list of (path, label)
    for p, root in entries:
        label, _ = m.derive_labels(p, root)
        ss.entries.append((p, label))
    ss.ann_path = Path(folder) / "annotations.json"
    ss.states = c.load_annotations(ss.ann_path)
    ss.idx = 0
    ss.cur_points = []
    ss.cur_masks = None
    ss.cur_idx = 0
    ss.selected = None
    ss.last_click = None
    ss.embedded_label = None
    ss.corpus_folder = folder


def current_entry():
    return st.session_state.entries[st.session_state.idx]


def get_state(path: Path, label: str, work, sf) -> c.ImageState:
    """Fetch or create the ImageState for an image, auto-detecting on first open."""
    ss = st.session_state
    if label in ss.states:
        return ss.states[label]
    nm, px = c.auto_scale(path)
    state = c.ImageState(label=label, scale_nm=nm, scale_px=px, sf=sf)
    if state.nm_per_pixel is not None:
        with st.spinner("Auto-detecting particles..."):
            state.particles = c.auto_detect(work, state)
    ss.states[label] = state
    persist()
    return state


def persist():
    ss = st.session_state
    c.save_annotations(ss.ann_path, ss.states)


def reset_current_drawing():
    ss = st.session_state
    ss.cur_points = []
    ss.cur_masks = None
    ss.cur_idx = 0


def ensure_embedding(label: str, work):
    ss = st.session_state
    if ss.embedded_label != label:
        predictor = get_predictor()
        rgb = cv2.cvtColor(m.normalize_intensity(work), cv2.COLOR_GRAY2RGB)
        predictor.set_image(rgb)
        ss.embedded_label = label


def cur_mask():
    ss = st.session_state
    if ss.cur_masks is None:
        return None
    return ss.cur_masks[ss.cur_idx]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(work, state: c.ImageState) -> np.ndarray:
    ss = st.session_state
    vis = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
    for i, p in enumerate(state.particles):
        pts = np.array(p.contour, dtype=np.int32).reshape(-1, 1, 2)
        color = (
            SELECTED_COLOR
            if ss.selected == i
            else (MANUAL_COLOR if p.source == "manual" else AUTO_COLOR)
        )
        cv2.polylines(vis, [pts], True, color, 2)
        cxy = pts.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(vis, str(i + 1), tuple(cxy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cm = cur_mask()
    if cm is not None:
        overlay = vis.copy()
        overlay[cm] = (255, 255, 0)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 0), 2)
    for x, y, lab in ss.cur_points:
        cv2.circle(vis, (x, y), 5, (0, 255, 0) if lab == 1 else (255, 0, 0), -1)
    return vis


# ---------------------------------------------------------------------------
# Click handling
# ---------------------------------------------------------------------------
def handle_click(x, y, mode, work, state: c.ImageState):
    ss = st.session_state
    if mode == "remove":
        for i in range(len(state.particles) - 1, -1, -1):
            if state.particles[i].mask(work.shape)[y, x]:
                del state.particles[i]
                ss.selected = None
                persist()
                return
        # nothing under click: select/deselect
        ss.selected = None
        return
    # + / - point: build current particle
    label = 1 if mode == "+ point" else 0
    ss.cur_points.append((x, y, label))
    ensure_embedding(state.label, work)
    predictor = get_predictor()
    masks, best = c.sam_predict(predictor, ss.cur_points)
    ss.cur_masks = masks
    ss.cur_idx = best


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
    reset_current_drawing()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    ss = st.session_state

    with st.sidebar:
        st.title("Particle measurement")
        folder = st.text_input(
            "Image folder", value=st.session_state.get("corpus_folder", "images")
        )
        if st.button("Load / reload corpus") or "entries" not in ss:
            init_corpus(folder)

        if not ss.entries:
            st.warning("No images found in that folder.")
            st.stop()

        # Navigation
        n = len(ss.entries)
        col_a, col_b = st.columns(2)
        if col_a.button("◀ Prev", use_container_width=True) and ss.idx > 0:
            ss.idx -= 1
            reset_current_drawing()
            ss.selected = None
        if col_b.button("Next ▶", use_container_width=True) and ss.idx < n - 1:
            ss.idx += 1
            reset_current_drawing()
            ss.selected = None
        labels = [lbl for _, lbl in ss.entries]
        picked = st.selectbox("Image", labels, index=ss.idx)
        if labels.index(picked) != ss.idx:
            ss.idx = labels.index(picked)
            reset_current_drawing()
            ss.selected = None
        st.caption(f"{ss.idx + 1} / {n}")

        mode = st.radio("Click action", ["+ point", "- point", "remove"], horizontal=True)

        c1, c2 = st.columns(2)
        if c1.button("Commit particle", use_container_width=True):
            commit_current(ss.states[current_entry()[1]])
        if c2.button("Cycle scale (m)", use_container_width=True) and ss.cur_masks is not None:
            ss.cur_idx = (ss.cur_idx + 1) % len(ss.cur_masks)
        if st.button("Clear current drawing", use_container_width=True):
            reset_current_drawing()

    # ---- current image ----
    path, label = current_entry()
    work, sf = load_work_cached(str(path))
    state = get_state(path, label, work, sf)

    left, right = st.columns([3, 2])

    with left:
        st.subheader(label)
        if state.nm_per_pixel is None:
            st.error(
                "No scale for this image — set the scale bar on the right to enable detection."
            )
        vis = render(work, state)
        coords = streamlit_image_coordinates(vis, key=f"img_{ss.idx}")
        if coords is not None:
            click = (coords["x"], coords["y"])
            if click != ss.last_click:
                ss.last_click = click
                x = int(np.clip(click[0], 0, work.shape[1] - 1))
                y = int(np.clip(click[1], 0, work.shape[0] - 1))
                handle_click(x, y, mode, work, state)
                st.rerun()

    with right:
        # Scale-bar fix
        with st.expander("Scale bar", expanded=state.nm_per_pixel is None):
            sc_nm = st.number_input(
                "Scale value (nm)", value=float(state.scale_nm or 0.0), step=10.0
            )
            sc_px = st.number_input(
                "Scale length (px, full-res)", value=int(state.scale_px or 0), step=1
            )
            if st.button("Apply scale + re-detect"):
                state.scale_nm = sc_nm or None
                state.scale_px = int(sc_px) or None
                if state.nm_per_pixel is not None:
                    # replace only auto particles; keep manual edits
                    state.particles = [p for p in state.particles if p.source == "manual"]
                    state.particles = c.auto_detect(work, state) + state.particles
                persist()
                st.rerun()

        if st.button("Re-run auto-detect (replace auto)") and state.nm_per_pixel is not None:
            manual = [p for p in state.particles if p.source == "manual"]
            state.particles = c.auto_detect(work, state) + manual
            persist()
            st.rerun()

        # Measurements
        df = c.measure_state(work, state)
        st.markdown(f"**{len(state.particles)} particle(s)**")
        if not df.empty:
            show = df[["id", "diam_nm", "wall_nm", "circularity", "is_vesicle", "source"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(
                f"median {df['diam_nm'].median():.0f} nm | "
                f"range {df['diam_nm'].min():.0f}-{df['diam_nm'].max():.0f} nm"
            )

        # Remove a specific particle by number
        if state.particles:
            rm = st.number_input(
                "Remove particle #", min_value=0, max_value=len(state.particles), value=0
            )
            if rm and st.button(f"Delete particle {rm}"):
                del state.particles[int(rm) - 1]
                persist()
                st.rerun()

        st.divider()
        if st.button("Export corpus CSV", type="primary"):
            work_images = {}
            for p, lbl in ss.entries:
                if lbl in ss.states:
                    work_images[lbl], _ = load_work_cached(str(p))
            out = Path(ss.corpus_folder) / "measurements.csv"
            df_all = c.export_csv(out, work_images, ss.states)
            n_imgs = df_all["image"].nunique() if len(df_all) else 0
            st.success(f"Wrote {len(df_all)} particles across {n_imgs} images to {out}")
            if len(df_all):
                st.download_button("Download CSV", df_all.to_csv(index=False), "measurements.csv")


if __name__ == "__main__":
    main()
