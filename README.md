---
title: FitMirror
emoji: 📏
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Single-photo body measurement pipeline for Indian garment sizing
---

# FitMirror

Single-photo body measurement pipeline for Indian garment sizing.

**Live demo:** https://huggingface.co/spaces/Abbas0807/FitMirror

---

## How it works

```
photo + height
      |
      v
MediaPipe Pose        33 landmarks + segmentation mask
      |
      v
Pixel Calibration     cm/pixel = height / (nose to ankle px / 0.94)
      |
      v
Linear Measurements   landmark distances x cm/pixel
      |
      v
Circumferences        Ramanujan ellipse with anthropometric depth ratios
      |
      v
Size Recommendation   per-dimension scoring against Indian wear charts
```

---

## Accuracy

Validation on N subjects with manual tape measurement as ground truth.

| Measurement | Mean abs. error | Notes |
|---|---|---|
| Height          | TBD cm | Calibration anchor (user-provided) |
| Shoulder width  | TBD cm | Linear; landmark-driven |
| Sleeve length   | TBD cm | Linear; landmark-driven |
| Torso length    | TBD cm | Linear; landmark-driven |
| Inseam          | TBD cm | Linear; landmark-driven |
| Chest circ.     | TBD cm | Ellipse-approximated |
| Waist circ.     | TBD cm | Ellipse-approximated |
| Hip circ.       | TBD cm | Ellipse-approximated |

Linear measurements (distances between landmarks) are reliable as long as calibration is correct. Circumferences carry the larger error because depth is not directly observed — see Known Limitations.

*Replace TBD with real numbers once tape-measure validation is done.*

---

## Known limitations

The core limitation is depth. Circumferences are estimated using population-average depth-to-width ratios from NHANES and ISI Calcutta anthropometric surveys. This works on average but drifts for body shapes at the extremes. The correct fix is fitting a parametric body model (SMPL or SMPL-X) to the 2D pose observations with a monocular depth prior,which would give subject-specific shape rather than a statistical guess. This is planned as the next stage of the project.

A few secondary limitations worth being explicit about:

- **Loose clothing inflates the silhouette.** Width at each cross-section is read from the body silhouette via Otsu thresholding; if the subject is wearing a kurta or jacket, the measurement is of the garment, not the body.Best results require fitted clothing for the measurement photo.
- 
- **Single-view occlusion of depth.** Even with a perfect silhouette, depth at the cross-section is unobserved from a front view. The ellipse approximation assumes the body cross-section is well-fit by an ellipse, which is reasonable at chest and hip but degrades around the waist for some body shapes.
- 
- **Calibration depends on user-reported height.** Self-reported height has ±1-2 cm noise in adults. This propagates linearly to every other measurement.
- 
- **Pose model sensitivity.** Strong backlighting, partial occlusion, or unusual stance can drop MediaPipe's landmark visibility scores. The pipeline checks visibility and refuses to produce a measurement when key landmarks are unreliable, rather than producing a silent wrong answer.

---

## Stack

Python 3.10 · Gradio 4.44 · MediaPipe 0.10.18 · OpenCV 4.10 · NumPy 1.26 · Pillow 10.4 CPU-only, runs on a free Hugging Face Spaces instance. No GPU dependencies.

---

## How to run

### Locally

```bash
git clone https://github.com/Abbas5055/FitMirror.git
cd FitMirror

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Open `http://localhost:7860` in a browser.

### On Hugging Face Spaces

The deployed Space at https://huggingface.co/spaces/Abbas0807/FitMirror auto-builds from this repository. To deploy your own copy, duplicate the Space or push to a Space repository:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/FitMirror
git push space main
```

The first build takes 3-5 minutes while MediaPipe weights download.

---

## Repository layout

```
fitmirror/
├── app.py                      # Gradio entry point
├── requirements.txt
├── README.md
└── fitmirror/
    ├── __init__.py
    ├── pose.py                 # MediaPipe wrapper
    ├── measure.py              # Calibration + linear measurements + circumferences
    └── sizing.py               # Size chart matching with per-dimension reasoning
```

---

## Author

[Abbas S](https://linkedin.com/in/abbas0807kl) · final-year B.Tech CSE (AI/ML),
SIMATS Engineering College, Chennai.

License: MIT.
