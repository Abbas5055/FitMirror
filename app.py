"""
FitMirror — Gradio entry point.

Run locally:    python app.py
HF Spaces:      this file is auto-launched by the Spaces runtime.

UI:
  - Image upload (front-facing full-body photo)
  - Height in cm (number)
  - Gender (radio) — drives anthropometric depth ratios
  - Garment type (dropdown) — drives the size chart used
  - Outputs: annotated image, measurement table, size recommendation,
             friendly error message on failure.
"""

from __future__ import annotations

# --- Monkey-patch gradio_client schema introspection BEFORE importing gradio.
# Some gradio_client versions crash on JSON schemas where `additionalProperties`
# is a bool (which is valid). Wrap _json_schema_to_python_type to short-circuit
# bool schemas to "Any". Safe no-op on versions that already handle this.
import gradio_client.utils as _gcu  # noqa: E402

_orig_json_to_python = _gcu._json_schema_to_python_type


def _safe_json_to_python(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_to_python(schema, defs)


_gcu._json_schema_to_python_type = _safe_json_to_python
# --- end monkey-patch

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fitmirror import pose as P
from fitmirror.measure import measure_all
from fitmirror.pose import PoseError
from fitmirror.sizing import recommend


# Garment dropdown values map to (label, internal_id, gender_constraint or None).
GARMENT_OPTIONS = [
    ("Men's Kurta",         "mens_kurta",      "male"),
    ("Women's Kurta",       "womens_kurta",    "female"),
    ("Women's Anarkali",    "womens_anarkali", "female"),
    ("Saree Blouse",        "saree_blouse",    "female"),
]
GARMENT_LABELS = [g[0] for g in GARMENT_OPTIONS]
GARMENT_BY_LABEL = {g[0]: g for g in GARMENT_OPTIONS}


# --- Image annotation ---------------------------------------------------

LANDMARK_DRAW_RADIUS = 5
LINE_WIDTH = 3

# Skeleton connections to draw (subset of MediaPipe Pose connections).
SKELETON_PAIRS = [
    (P.LEFT_SHOULDER, P.RIGHT_SHOULDER),
    (P.LEFT_SHOULDER, P.LEFT_ELBOW),  (P.LEFT_ELBOW, P.LEFT_WRIST),
    (P.RIGHT_SHOULDER, P.RIGHT_ELBOW),(P.RIGHT_ELBOW, P.RIGHT_WRIST),
    (P.LEFT_SHOULDER, P.LEFT_HIP),    (P.RIGHT_SHOULDER, P.RIGHT_HIP),
    (P.LEFT_HIP, P.RIGHT_HIP),
    (P.LEFT_HIP, P.LEFT_KNEE),        (P.LEFT_KNEE, P.LEFT_ANKLE),
    (P.RIGHT_HIP, P.RIGHT_KNEE),      (P.RIGHT_KNEE, P.RIGHT_ANKLE),
]


def _annotate(image_rgb: np.ndarray, landmarks_px: np.ndarray) -> Image.Image:
    """Overlay skeleton + key landmarks on the input image for visual feedback."""
    img = Image.fromarray(image_rgb).convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Skeleton lines
    for a, b in SKELETON_PAIRS:
        if landmarks_px[a, 2] < P.MIN_VISIBILITY or landmarks_px[b, 2] < P.MIN_VISIBILITY:
            continue
        draw.line(
            [
                (float(landmarks_px[a, 0]), float(landmarks_px[a, 1])),
                (float(landmarks_px[b, 0]), float(landmarks_px[b, 1])),
            ],
            fill=(0, 200, 255),
            width=LINE_WIDTH,
        )

    # Landmark dots
    for idx in P.REQUIRED_LANDMARKS:
        if landmarks_px[idx, 2] < P.MIN_VISIBILITY:
            continue
        x, y = float(landmarks_px[idx, 0]), float(landmarks_px[idx, 1])
        r = LANDMARK_DRAW_RADIUS
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 80, 80))

    return img


def _measurements_table_md(m) -> str:
    """Markdown table of the measurements we computed."""
    L = m.linear
    C = m.circumferences
    return (
        "### Measurements\n\n"
        "| Dimension | Value (cm) |\n"
        "|---|---|\n"
        f"| Shoulder width | {L.shoulder_cm} |\n"
        f"| Sleeve length  | {L.sleeve_cm} |\n"
        f"| Torso length   | {L.torso_cm} |\n"
        f"| Inseam         | {L.inseam_cm} |\n"
        f"| Chest (circumference) | {C.chest_cm} |\n"
        f"| Waist (circumference) | {C.waist_cm} |\n"
        f"| Hip (circumference)   | {C.hip_cm} |\n"
    )


# --- Pipeline runner ----------------------------------------------------

def _error_md(msg: str) -> str:
    return (
        "### Couldn't process this photo\n\n"
        f"{msg}\n\n"
        "**Tips for a good photo:**\n"
        "- Stand upright, facing the camera.\n"
        "- Head to feet must be inside the frame.\n"
        "- Plain background, even lighting.\n"
        "- Arms slightly away from your torso.\n"
    )


def run(image, height_cm, gender_label, garment_label):
    """Top-level Gradio handler. Returns (annotated_image_or_None, message_md)."""
    # --- Input validation ---
    if image is None:
        return None, _error_md("Please upload a photo first.")

    try:
        height_cm = float(height_cm)
    except (TypeError, ValueError):
        return None, _error_md("Please enter your height as a number (cm).")

    if not (100.0 <= height_cm <= 230.0):
        return None, _error_md("Height must be between 100 cm and 230 cm.")

    gender = "male" if gender_label.lower().startswith("m") else "female"

    if garment_label not in GARMENT_BY_LABEL:
        return None, _error_md("Please pick a garment type.")
    _, garment_id, garment_gender = GARMENT_BY_LABEL[garment_label]

    if garment_gender and garment_gender != gender:
        return None, _error_md(
            f"That garment is configured for **{garment_gender}**. "
            "Pick a different garment or update the gender field."
        )

    # --- Pose ---
    try:
        result = P.detect(image)
    except PoseError as e:
        return None, _error_md(str(e))
    except Exception:  # last-resort safety net so the UI never shows a stack trace
        return None, _error_md(
            "Something went wrong reading the image. Try a different photo "
            "(JPEG or PNG, under ~10 MB)."
        )

    if result is None:
        return None, _error_md("No person detected in the photo.")

    if not P.has_required_landmarks(result):
        missing = ", ".join(P.missing_landmark_names(result))
        return None, _error_md(
            f"Couldn't see your full body — missing/occluded: **{missing}**. "
            "Step back so head, hands, and feet are all inside the frame."
        )

    # --- Measure + recommend ---
    try:
        m = measure_all(result, user_height_cm=height_cm, gender=gender)
    except PoseError as e:
        return None, _error_md(str(e))

    rec = recommend(
        garment_id,
        chest_cm=m.circumferences.chest_cm,
        waist_cm=m.circumferences.waist_cm,
        hip_cm=m.circumferences.hip_cm,
    )

    annotated = _annotate(result.image_rgb, result.landmarks_px)

    msg = (
        _measurements_table_md(m)
        + "\n\n"
        + rec.to_markdown()
        + "\n\n"
        + "_Stage 1 demo — single photo, monocular. "
        "Linear measurements ±2-3 cm; circumferences ±4-6 cm. "
        "Use as a starting point for sizing, not as a tailoring spec._"
    )
    return annotated, msg


# --- Gradio UI ----------------------------------------------------------

DESCRIPTION = """
**FitMirror** — Upload one front-facing full-body photo and get body
measurements + an Indian-wear size recommendation.

How it works: MediaPipe detects 33 body landmarks → your height is used to
calibrate cm-per-pixel → linear measurements come from landmark distances →
chest/waist/hip circumferences combine silhouette width with anthropometric
depth ratios via the Ramanujan ellipse-perimeter approximation → the result
is matched against standard Indian-wear size charts.

Stage 1 demo. CPU-only. No data is stored.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FitMirror — Body Measurement & Sizing") as demo:
        gr.Markdown("# FitMirror")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Full-body photo (front-facing)",
                    type="pil",
                    sources=["upload", "webcam"],
                    height=400,
                )
                height_in = gr.Number(
                    label="Your height (cm)",
                    value=170,
                    minimum=100,
                    maximum=230,
                    step=1,
                )
                gender_in = gr.Radio(
                    label="Gender (drives depth ratios)",
                    choices=["Male", "Female"],
                    value="Male",
                )
                garment_in = gr.Dropdown(
                    label="Garment type",
                    choices=GARMENT_LABELS,
                    value=GARMENT_LABELS[0],
                )
                go_btn = gr.Button("Measure & recommend size", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Detected pose", type="pil", height=400)
                msg_out = gr.Markdown("Upload a photo and press the button.")

        with gr.Accordion("How accurate is this?", open=False):
            gr.Markdown(
                "- **Linear measurements** (shoulder, sleeve, torso, inseam): "
                "typically within **±2-3 cm** when the photo is well framed.\n"
                "- **Circumferences** (chest, waist, hip): currently **±4-6 cm**, "
                "limited by the single-camera depth assumption.\n"
                "- Stage 2 (under development): integrate a monocular depth model "
                "and SMPL body fitting to push circumferences to **±2-3 cm**.\n\n"
                "This page shows what's possible with classical anthropometry + a "
                "free CPU instance — no GPU required."
            )

        go_btn.click(
            run,
            inputs=[image_in, height_in, gender_in, garment_in],
            outputs=[image_out, msg_out],
        )

    return demo


if __name__ == "__main__":
    # ssr_mode=False: gradio 5.x SSR can break the API endpoint behind the HF
    # Spaces iframe, surfacing as "No API found" in the browser. Disable it.
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_error=True,
    )
