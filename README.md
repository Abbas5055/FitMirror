---
title: FitMirror
emoji: 👗
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# FitMirror

**Single-photo body measurement and Indian-wear size recommendation.**

Upload one front-facing full-body photo, enter your height, pick a garment,
and get back: a pose overlay, body measurements (shoulder, sleeve, torso,
inseam, chest / waist / hip), and a size recommendation matched against
standard Indian-wear charts (kurta, anarkali, saree blouse).

CPU-only. Runs on the free Hugging Face Spaces tier.

## How it works

```
photo + height
     │
     ▼
┌─────────────────────┐
│ MediaPipe Pose      │  → 33 landmarks + body segmentation mask
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Calibration         │  cm/pixel = user_height / (nose→ankle px / 0.94)
└─────────┬───────────┘
          ▼
┌─────────────────────┐  shoulder, sleeve, torso, inseam
│ Linear measurements │  = landmark distance × cm/pixel
└─────────┬───────────┘
          ▼
┌──────────────────────────────────────────────────────┐
│ Circumferences                                       │
│  width_cm = silhouette width at chest/waist/hip line │
│  depth_cm = width_cm × anthropometric depth ratio    │
│  C        = Ramanujan-II ellipse perimeter(width,depth)
└─────────┬────────────────────────────────────────────┘
          ▼
┌─────────────────────┐
│ Size recommendation │  per-dim scoring → rounded up on disagreement
└─────────────────────┘
```

The depth ratios are population averages from anthropometric surveys
(NHANES + ISI Calcutta), gender-adjusted. The Ramanujan II ellipse formula is
accurate to <1% for a known (a, b); the dominant error in the pipeline is the
depth assumption itself.

## Honest accuracy

Because we only see width directly from one photo, circumferences are an
*estimate*, not a tailoring spec.

| Measurement                     | Typical error |
|---------------------------------|---------------|
| Linear (shoulder, sleeve, etc.) | ±2-3 cm       |
| Circumference (chest/waist/hip) | ±4-6 cm       |

Stage 2 work (post-MVP): integrate a monocular depth model
(e.g. Depth Anything v2) and SMPL body fitting to push circumferences into
the ±2-3 cm band.

## Repo layout

```
fitmirror/
├── app.py                 Gradio entry point
├── requirements.txt
├── README.md
├── .gitignore
├── fitmirror/
│   ├── __init__.py
│   ├── pose.py            MediaPipe wrapper
│   ├── measure.py         Calibration + measurements + circumferences
│   └── sizing.py          Size charts + recommendation engine
└── tests/
    └── test_sizing.py     Sizing engine unit tests
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                # opens http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a Gradio Space at https://huggingface.co/new-space
2. Clone it locally (`git clone https://huggingface.co/spaces/<user>/<space>`)
3. Copy these files in and `git push`
4. Spaces builds + serves the app automatically (~3-5 min)

## Stack

Python 3.10 · Gradio 4.44.0 · MediaPipe 0.10.18 · OpenCV 4.10 · NumPy 1.26 ·
Pillow 10.4 · protobuf 4.25.5

## License

MIT.
