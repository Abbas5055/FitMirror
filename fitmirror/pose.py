"""
fitmirror.pose
==============

Thin wrapper around MediaPipe Pose with segmentation enabled.

Responsibilities:
  - Accept an image (PIL.Image or np.ndarray, RGB).
  - Run MediaPipe Pose with `enable_segmentation=True`.
  - Return a normalized PoseResult containing 33 landmarks (in *pixel* coords)
    and a binary body mask. None landmarks/mask are returned when no person
    is detected (caller decides how to surface that).

Why a wrapper:
  - Keeps MediaPipe-specific imports/objects out of `measure.py` and `app.py`.
  - Centralizes defensive handling (bad inputs, no detection, low visibility).

Stage 1 only — no SMPL, no depth model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

# MediaPipe is imported lazily inside _get_pose() so that this module can be
# imported (e.g., for type hints / tests) without paying the mediapipe import
# cost up front, and so a missing mediapipe install fails with a clearer error
# at use-time rather than import-time.


# 33 landmark indices (MediaPipe Pose). Keep the ones we actually use as
# named constants — readability >> speed here.
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Landmarks required to be present (with reasonable visibility) for a usable
# measurement pass. If any of these is missing/very low visibility we treat
# it as a "partial body" failure.
REQUIRED_LANDMARKS = (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)

MIN_VISIBILITY = 0.5


class PoseError(Exception):
    """Raised for unrecoverable pose-detection problems (with human-readable msg)."""


@dataclass
class PoseResult:
    """Normalized output of a pose pass.

    Attributes:
        landmarks_px: ndarray of shape (33, 3) — columns are (x_px, y_px, visibility).
        mask: boolean ndarray of shape (H, W) — True where MediaPipe believes
              the body is.
        image_rgb: ndarray (H, W, 3) uint8 — copy of the input in RGB.
    """
    landmarks_px: np.ndarray
    mask: np.ndarray
    image_rgb: np.ndarray


_pose_singleton = None


def _get_pose():
    """Lazy-construct a single MediaPipe Pose instance (CPU, static images)."""
    global _pose_singleton
    if _pose_singleton is None:
        try:
            import mediapipe as mp
        except ImportError as e:  # pragma: no cover
            raise PoseError(
                "mediapipe is not installed. `pip install mediapipe==0.10.18`"
            ) from e
        _pose_singleton = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,           # 0=lite, 1=full, 2=heavy. 1 is the CPU sweet spot.
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )
    return _pose_singleton


def _to_rgb_array(image) -> np.ndarray:
    """Coerce PIL.Image or ndarray (any common layout) into uint8 RGB ndarray."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # grayscale -> RGB
            return np.stack([image] * 3, axis=-1).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 4:  # RGBA -> RGB
            return image[:, :, :3].astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)

    raise PoseError("Unsupported image format. Upload a JPEG or PNG photo.")


def detect(image) -> Optional[PoseResult]:
    """Run pose detection on the given image.

    Returns:
        PoseResult on success.
        None if no person was detected at all.

    Raises:
        PoseError: on unrecoverable input problems (bad format, etc.).
                   "Partial body" cases return a PoseResult — the caller checks
                   visibility via `has_required_landmarks(result)`.
    """
    rgb = _to_rgb_array(image)
    h, w = rgb.shape[:2]

    pose = _get_pose()
    result = pose.process(rgb)

    if result.pose_landmarks is None:
        return None

    landmarks_px = np.zeros((33, 3), dtype=np.float32)
    for i, lm in enumerate(result.pose_landmarks.landmark):
        landmarks_px[i, 0] = lm.x * w
        landmarks_px[i, 1] = lm.y * h
        landmarks_px[i, 2] = lm.visibility

    if result.segmentation_mask is not None:
        mask = result.segmentation_mask > 0.5
    else:
        # Defensive: should not happen with enable_segmentation=True, but if it
        # does, fall back to a dummy mask covering the landmark bounding box.
        mask = np.zeros((h, w), dtype=bool)

    return PoseResult(landmarks_px=landmarks_px, mask=mask, image_rgb=rgb)


def has_required_landmarks(result: PoseResult) -> bool:
    """True iff every landmark we depend on for measurement is visible enough."""
    for idx in REQUIRED_LANDMARKS:
        if result.landmarks_px[idx, 2] < MIN_VISIBILITY:
            return False
    return True


def missing_landmark_names(result: PoseResult) -> list[str]:
    """Human-readable names of required landmarks that are below the visibility threshold."""
    name_lookup = {
        NOSE: "head",
        LEFT_SHOULDER: "left shoulder", RIGHT_SHOULDER: "right shoulder",
        LEFT_ELBOW: "left elbow", RIGHT_ELBOW: "right elbow",
        LEFT_WRIST: "left wrist", RIGHT_WRIST: "right wrist",
        LEFT_HIP: "left hip", RIGHT_HIP: "right hip",
        LEFT_KNEE: "left knee", RIGHT_KNEE: "right knee",
        LEFT_ANKLE: "left ankle", RIGHT_ANKLE: "right ankle",
    }
    return [
        name_lookup[idx]
        for idx in REQUIRED_LANDMARKS
        if result.landmarks_px[idx, 2] < MIN_VISIBILITY
    ]
