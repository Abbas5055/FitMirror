"""
fitmirror.measure
=================

Calibration + body measurements from a single front-facing photo.

Pipeline:
  1) calibrate(): turn user's reported height into cm/pixel using the
     vertical span between the nose and the mid-ankle, corrected by an
     anthropometric ratio (nose -> top-of-head ≈ 0.06 * total height).
  2) linear_measurements(): direct landmark-to-landmark distances scaled by
     cm/pixel — shoulder, sleeve, arm, torso, inseam.
  3) circumferences(): silhouette width (from MediaPipe segmentation mask)
     measured at chest/waist/hip Y-lines, combined with anthropometric
     depth/width ratios into an elliptical perimeter approximation
     (Ramanujan II).

Why ellipse + depth ratios:
  We have one camera, one photo. We only see *width* directly. To recover a
  circumference we have to assume a cross-section. An ellipse is a much better
  approximation than a circle for the human torso (chest depth < width;
  waist depth ≈ width; hips depth < width). The depth/width ratios are
  population averages from anthropometric surveys (NHANES + ISI Calcutta),
  gender-adjusted. This is honest about its limits — the README publishes
  ±4-6cm on circumferences as a result.

Ramanujan II ellipse perimeter (very accurate, error <1% even for elongated ellipses):
    h = ((a - b) / (a + b))**2
    P ≈ π (a + b) (1 + 3h / (10 + sqrt(4 - 3h)))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

from . import pose as P
from .pose import PoseResult, PoseError


Gender = Literal["male", "female"]


# Nose -> top-of-head as a fraction of total height (anthropometric average).
# Standing height = (mid_ankle_y - top_of_head_y).
# We see (mid_ankle_y - nose_y); add 0.06 * standing_height to recover the rest.
# Equivalently: standing_height_px ≈ (mid_ankle_y - nose_y) / (1 - 0.06)
NOSE_TO_HEAD_TOP_FRACTION = 0.06

# Depth-to-width ratios at chest / natural waist / hip.
# Sources: NHANES anthropometric reference data + ISI Calcutta Indian survey,
# averaged across adult population, gender-adjusted.
DEPTH_RATIOS: dict[Gender, dict[str, float]] = {
    "male":   {"chest": 0.69, "waist": 0.84, "hip": 0.76},
    "female": {"chest": 0.70, "waist": 0.80, "hip": 0.75},
}

# Vertical positions of the chest / waist / hip lines, expressed as a fraction
# of the shoulder→hip distance below the shoulder line.
# 0.0 == at shoulder line, 1.0 == at hip line.
CHEST_FRAC = 0.20   # ~ pectoral / bust line
WAIST_FRAC = 0.65   # natural waist
HIP_FRAC   = 1.00   # at hip line


@dataclass
class Calibration:
    cm_per_pixel: float
    standing_height_px: float


@dataclass
class LinearMeasurements:
    shoulder_cm: float
    sleeve_cm: float        # shoulder -> wrist (along upper + forearm)
    arm_cm: float           # same as sleeve in v1; kept separate for future
    torso_cm: float         # mid-shoulder -> mid-hip
    inseam_cm: float        # mid-hip -> mid-ankle


@dataclass
class Circumferences:
    chest_cm: float
    waist_cm: float
    hip_cm: float


@dataclass
class Measurements:
    calibration: Calibration
    linear: LinearMeasurements
    circumferences: Circumferences

    def to_dict(self) -> dict:
        return {
            "calibration": asdict(self.calibration),
            "linear": asdict(self.linear),
            "circumferences": asdict(self.circumferences),
        }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) / 2.0


def calibrate(result: PoseResult, user_height_cm: float) -> Calibration:
    """Return cm/pixel for this image, given the user's reported standing height."""
    if not (100.0 <= user_height_cm <= 230.0):
        raise PoseError("Height must be between 100 cm and 230 cm.")

    nose = result.landmarks_px[P.NOSE]
    mid_ankle_y = (result.landmarks_px[P.LEFT_ANKLE, 1] +
                   result.landmarks_px[P.RIGHT_ANKLE, 1]) / 2.0

    nose_to_ankle_px = mid_ankle_y - nose[1]
    if nose_to_ankle_px <= 0:
        raise PoseError(
            "Couldn't read your full height in the photo. "
            "Make sure the photo is upright and head-to-feet are in frame."
        )

    standing_height_px = nose_to_ankle_px / (1.0 - NOSE_TO_HEAD_TOP_FRACTION)
    cm_per_pixel = user_height_cm / standing_height_px
    return Calibration(cm_per_pixel=cm_per_pixel, standing_height_px=standing_height_px)


# --------------------------------------------------------------------------
# Linear measurements
# --------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def linear_measurements(result: PoseResult, calib: Calibration) -> LinearMeasurements:
    lm = result.landmarks_px
    s = calib.cm_per_pixel

    shoulder_px = _dist(lm[P.LEFT_SHOULDER], lm[P.RIGHT_SHOULDER])

    # Sleeve = shoulder -> elbow + elbow -> wrist, averaged over both arms.
    left_sleeve_px = (
        _dist(lm[P.LEFT_SHOULDER], lm[P.LEFT_ELBOW]) +
        _dist(lm[P.LEFT_ELBOW], lm[P.LEFT_WRIST])
    )
    right_sleeve_px = (
        _dist(lm[P.RIGHT_SHOULDER], lm[P.RIGHT_ELBOW]) +
        _dist(lm[P.RIGHT_ELBOW], lm[P.RIGHT_WRIST])
    )
    sleeve_px = (left_sleeve_px + right_sleeve_px) / 2.0

    mid_shoulder = _midpoint(lm[P.LEFT_SHOULDER], lm[P.RIGHT_SHOULDER])
    mid_hip = _midpoint(lm[P.LEFT_HIP], lm[P.RIGHT_HIP])
    torso_px = float(np.linalg.norm(mid_shoulder - mid_hip))

    mid_ankle = _midpoint(lm[P.LEFT_ANKLE], lm[P.RIGHT_ANKLE])
    inseam_px = float(np.linalg.norm(mid_hip - mid_ankle))

    return LinearMeasurements(
        shoulder_cm=round(shoulder_px * s, 1),
        sleeve_cm=round(sleeve_px * s, 1),
        arm_cm=round(sleeve_px * s, 1),
        torso_cm=round(torso_px * s, 1),
        inseam_cm=round(inseam_px * s, 1),
    )


# --------------------------------------------------------------------------
# Circumferences via silhouette width + ellipse approximation
# --------------------------------------------------------------------------

def _silhouette_width_px(mask: np.ndarray, y: int) -> float:
    """Width (px) of the body mask at a given image-row y. 0 if no body pixels."""
    if y < 0 or y >= mask.shape[0]:
        return 0.0
    row = mask[y]
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return 0.0
    return float(xs[-1] - xs[0])


def _ramanujan_ellipse_perimeter(width_cm: float, depth_cm: float) -> float:
    """Ramanujan II approximation. width_cm and depth_cm are full diameters."""
    a = width_cm / 2.0
    b = depth_cm / 2.0
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _smooth_width_px(mask: np.ndarray, y_center: int, half_window: int = 3) -> float:
    """Median width over a small vertical window — robust to single-row noise."""
    h = mask.shape[0]
    y_lo = max(0, y_center - half_window)
    y_hi = min(h, y_center + half_window + 1)
    widths = [_silhouette_width_px(mask, y) for y in range(y_lo, y_hi)]
    widths = [w for w in widths if w > 0]
    if not widths:
        return 0.0
    return float(np.median(widths))


def circumferences(
    result: PoseResult,
    calib: Calibration,
    gender: Gender,
) -> Circumferences:
    """
    Estimate chest / waist / hip circumferences from pose landmarks.

    Why landmarks (not silhouette): the MediaPipe segmentation mask includes
    arms at the body sides, so a raw silhouette width over-counts by 2 * arm
    thickness. Landmark-based widths sidestep this entirely.

    Anthropometric multipliers convert between landmark distance and the
    relevant body width:
      chest_width_cm = shoulder_landmark_cm * 0.92  (chest a touch narrower
                       than the shoulder-joint span across the upper torso)
      hip_width_cm   = hip_landmark_cm * 1.10       (outer hip is wider than
                       the iliac-crest landmark line)
      waist_width_cm = mean(chest_width, hip_width) * 0.86  (waist narrower
                       than both, by population average)
    """
    if gender not in DEPTH_RATIOS:
        raise PoseError("Gender must be 'male' or 'female'.")

    lm = result.landmarks_px
    s = calib.cm_per_pixel

    shoulder_landmark_cm = _dist(lm[P.LEFT_SHOULDER], lm[P.RIGHT_SHOULDER]) * s
    hip_landmark_cm = _dist(lm[P.LEFT_HIP], lm[P.RIGHT_HIP]) * s

    if shoulder_landmark_cm <= 0 or hip_landmark_cm <= 0:
        raise PoseError(
            "Couldn't locate your shoulders or hips. "
            "Try a more upright, front-facing photo."
        )

    chest_width_cm = shoulder_landmark_cm * 0.90
    hip_width_cm = hip_landmark_cm * 0.95
    waist_width_cm = (chest_width_cm + hip_width_cm) / 2.0 * 0.78

    ratios = DEPTH_RATIOS[gender]
    chest = _ramanujan_ellipse_perimeter(chest_width_cm, chest_width_cm * ratios["chest"])
    waist = _ramanujan_ellipse_perimeter(waist_width_cm, waist_width_cm * ratios["waist"])
    hip = _ramanujan_ellipse_perimeter(hip_width_cm, hip_width_cm * ratios["hip"])

    return Circumferences(
        chest_cm=round(chest, 1),
        waist_cm=round(waist, 1),
        hip_cm=round(hip, 1),
    )


# --------------------------------------------------------------------------
# Top-level convenience
# --------------------------------------------------------------------------

def measure_all(
    result: PoseResult,
    user_height_cm: float,
    gender: Gender,
) -> Measurements:
    calib = calibrate(result, user_height_cm)
    linear = linear_measurements(result, calib)
    circ = circumferences(result, calib, gender)
    return Measurements(calibration=calib, linear=linear, circumferences=circ)
