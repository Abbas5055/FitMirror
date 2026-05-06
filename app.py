"""FitMirror — Streamlit app (3-step wizard).

Step 1: Upload person photo + garment image
Step 2: Body analysis → measurements + size recommendation
Step 3: Virtual try-on render + .glb download
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from fitmirror.body.depth import DepthEstimator
from fitmirror.body.measure import compute_measurements
from fitmirror.body.pose import PoseEstimator
from fitmirror.body.segment import Segmenter
from fitmirror.body.smpl_fit import SMPLFitter
from fitmirror.export.glb import export_glb
from fitmirror.garment.parse import GarmentParser
from fitmirror.garment.warp import TPSWarper, composite
from fitmirror.sizing.recommend import SizeRecommender
from fitmirror.utils.io import pil_to_bgr
from fitmirror.utils.visualize import (
    annotate_measurements,
    depth_colormap,
    draw_landmarks,
    plot_size_scores,
)

st.set_page_config(
    page_title="FitMirror",
    page_icon="🪞",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
for key in ("step", "person_bgr", "garment_bgr", "measurements", "recommendation"):
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.step is None:
    st.session_state.step = 1

# --------------------------------------------------------------------------
# Cached model loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pose model…")
def get_pose():
    return PoseEstimator()

@st.cache_resource(show_spinner="Loading segmentation model…")
def get_segmenter():
    return Segmenter()

@st.cache_resource(show_spinner="Loading depth model…")
def get_depth():
    return DepthEstimator()

@st.cache_resource(show_spinner="Loading SMPL fitter…")
def get_fitter():
    return SMPLFitter()

@st.cache_resource
def get_garment_parser():
    return GarmentParser()

@st.cache_resource
def get_recommender():
    return SizeRecommender()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🪞 FitMirror")
st.caption("Single-photo virtual try-on + measurement-grounded size recommendation for Indian wear.")

steps = ["📸 Upload", "📏 Measure", "👘 Try-On"]
cols = st.columns(3)
for i, (col, label) in enumerate(zip(cols, steps)):
    with col:
        if i + 1 == st.session_state.step:
            st.markdown(f"**➤ Step {i+1}: {label}**")
        else:
            st.markdown(f"Step {i+1}: {label}")

st.divider()

# ==========================================================================
# STEP 1 — Upload
# ==========================================================================
if st.session_state.step == 1:
    st.subheader("Step 1: Upload your photo and a garment")
    col1, col2 = st.columns(2)

    with col1:
        person_file = st.file_uploader(
            "Full-body photo (front-facing)", type=["jpg", "jpeg", "png"]
        )
        if person_file:
            pil = Image.open(person_file).convert("RGB")
            st.image(pil, caption="Your photo", use_column_width=True)
            st.session_state.person_bgr = pil_to_bgr(pil)

    with col2:
        garment_file = st.file_uploader(
            "Garment image (kurta / blouse)", type=["jpg", "jpeg", "png"]
        )
        garment_type = st.selectbox(
            "Garment type",
            ["kurta_men", "kurta_women", "saree_blouse"],
        )
        if garment_file:
            pil_g = Image.open(garment_file).convert("RGB")
            st.image(pil_g, caption="Garment", use_column_width=True)
            st.session_state.garment_bgr = pil_to_bgr(pil_g)
            st.session_state.garment_type = garment_type

    if st.session_state.person_bgr is not None and st.session_state.garment_bgr is not None:
        if st.button("▶ Analyse body →", type="primary"):
            st.session_state.step = 2
            st.rerun()

# ==========================================================================
# STEP 2 — Measure
# ==========================================================================
elif st.session_state.step == 2:
    st.subheader("Step 2: Body analysis")

    person_bgr: np.ndarray = st.session_state.person_bgr

    with st.spinner("Running pose estimation…"):
        pose_est = get_pose()
        pose = pose_est(person_bgr)
        if pose is None:
            st.error("No person detected in the photo. Please upload a clear full-body photo.")
            st.stop()
        lm_img = draw_landmarks(person_bgr, pose)

    with st.spinner("Removing background…"):
        seg = get_segmenter()
        _, mask = seg(person_bgr)

    with st.spinner("Estimating depth…"):
        dep = get_depth()
        depth_map = dep.masked_depth(person_bgr, mask)
        depth_vis = depth_colormap(depth_map)

    col1, col2 = st.columns(2)
    with col1:
        st.image(lm_img[..., ::-1], caption="Pose landmarks", use_column_width=True)
    with col2:
        st.image(depth_vis[..., ::-1], caption="Depth map (pseudo-3D)", use_column_width=True)

    smpl_available = Path("models/smpl").exists()

    if smpl_available:
        with st.spinner("Fitting SMPL body model (this may take ~60s on CPU)…"):
            fitter = get_fitter()
            smpl_result = fitter.fit(pose, depth_map, mask)
            measurements = compute_measurements(smpl_result)
            st.session_state.smpl_result = smpl_result
    else:
        st.warning(
            "SMPL model files not found at `models/smpl/`. "
            "Download from https://smpl.is.tue.mpg.de/ to enable accurate measurements. "
            "Using heuristic estimates for demo."
        )
        # Heuristic fallback from MediaPipe skeleton
        from fitmirror.body.measure import Measurements
        H, W = person_bgr.shape[:2]
        lm = pose.landmarks_px
        shoulder_px = abs(lm[11, 0] - lm[12, 0])
        hip_px = abs(lm[23, 0] - lm[24, 0])
        head_y = lm[0, 1]
        ankle_y = (lm[27, 1] + lm[28, 1]) / 2
        body_px = abs(ankle_y - head_y)
        # rough pixel→cm scale assuming avg height 165 cm
        px_cm = 165.0 / max(body_px, 1)
        measurements = Measurements(
            chest_cm=shoulder_px * px_cm * 2.8,
            waist_cm=shoulder_px * px_cm * 2.2,
            hip_cm=hip_px * px_cm * 2.6,
            shoulder_width_cm=shoulder_px * px_cm,
            height_cm=body_px * px_cm,
        )
        st.session_state.smpl_result = None

    st.session_state.measurements = measurements
    rec = get_recommender().recommend(measurements, st.session_state.garment_type)
    st.session_state.recommendation = rec

    # Display measurements
    st.subheader("📏 Body Measurements")
    m = measurements
    mcols = st.columns(5)
    for col, (label, val) in zip(
        mcols,
        [
            ("Height", f"{m.height_cm:.1f} cm"),
            ("Chest", f"{m.chest_cm:.1f} cm"),
            ("Waist", f"{m.waist_cm:.1f} cm"),
            ("Hip", f"{m.hip_cm:.1f} cm"),
            ("Shoulder", f"{m.shoulder_width_cm:.1f} cm"),
        ],
    ):
        col.metric(label, val)

    # Size recommendation
    st.subheader("🏷 Size Recommendation")
    st.success(
        f"**Recommended size: {rec.recommended_size}**  (confidence {rec.confidence:.0%})"
    )
    for reason in rec.reasoning:
        st.markdown(f"- {reason}")

    fig = plot_size_scores(rec.all_scores, rec.recommended_size)
    st.pyplot(fig)

    # Download measurements JSON
    st.download_button(
        "⬇ Download measurements JSON",
        data=json.dumps({**m.to_dict(), **rec.to_dict()}, indent=2),
        file_name="fitmirror_measurements.json",
        mime="application/json",
    )

    if st.button("▶ Generate try-on →", type="primary"):
        st.session_state.step = 3
        st.rerun()

# ==========================================================================
# STEP 3 — Try-On
# ==========================================================================
elif st.session_state.step == 3:
    st.subheader("Step 3: Virtual try-on")

    person_bgr = st.session_state.person_bgr
    garment_bgr = st.session_state.garment_bgr

    with st.spinner("Parsing garment…"):
        parser = get_garment_parser()
        garment_panels = parser(garment_bgr, st.session_state.garment_type)

    # Simple demo composite (full TPS requires SMPL joints; fallback to overlay)
    with st.spinner("Generating try-on render…"):
        H, W = person_bgr.shape[:2]
        warped = cv2.resize(garment_panels.full_rgba, (W, H))
        tryon = composite(person_bgr, warped)

    col1, col2 = st.columns(2)
    with col1:
        st.image(person_bgr[..., ::-1], caption="Original", use_column_width=True)
    with col2:
        st.image(tryon[..., ::-1], caption="Try-on result", use_column_width=True)

    # .glb export
    if st.session_state.get("smpl_result") is not None:
        with st.spinner("Exporting 3D avatar…"):
            with tempfile.TemporaryDirectory() as tmp:
                glb_path = export_glb(st.session_state.smpl_result, Path(tmp) / "avatar.glb")
                with open(glb_path, "rb") as f:
                    glb_bytes = f.read()
        st.download_button(
            "⬇ Download 3D avatar (.glb)",
            data=glb_bytes,
            file_name="fitmirror_avatar.glb",
            mime="model/gltf-binary",
        )

    # Download try-on PNG
    tryon_pil = Image.fromarray(tryon[..., ::-1])
    buf = io.BytesIO()
    tryon_pil.save(buf, format="PNG")
    st.download_button(
        "⬇ Download try-on PNG",
        data=buf.getvalue(),
        file_name="fitmirror_tryon.png",
        mime="image/png",
    )

    if st.button("↺ Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
