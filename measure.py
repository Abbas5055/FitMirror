"""Body measurement extraction from a fitted SMPL mesh.

Strategy:
  - Slice mesh at canonical body heights (chest, waist, hip, shoulder)
  - Extract 2D cross-section contour at each slice
  - Circumference = convex hull perimeter of the contour (conservative)
  - Height = ankle-to-top-of-head vertex span × scale
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from fitmirror.body.smpl_fit import SMPLFitResult

# Canonical SMPL vertex indices for anatomical height landmarks
# (approximate, gender-neutral SMPL topology)
SMPL_VERTEX = {
    "head_top": 412,
    "left_ankle": 3438,
    "right_ankle": 6838,
    "left_shoulder": 3011,
    "right_shoulder": 606,
    "chest_mid": 3500,   # mid-sternum approx
    "waist_mid": 3500,   # overridden by fraction
    "hip_mid": 1799,
}

# Fraction of body height (0=feet, 1=head) for each measurement plane
SLICE_FRACTIONS = {
    "chest": 0.74,
    "waist": 0.62,
    "hip": 0.52,
}


@dataclass
class Measurements:
    """Body measurements in centimetres."""
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_width_cm: float
    height_cm: float

    def to_dict(self) -> dict[str, float]:
        return {
            "chest_cm": round(self.chest_cm, 1),
            "waist_cm": round(self.waist_cm, 1),
            "hip_cm": round(self.hip_cm, 1),
            "shoulder_width_cm": round(self.shoulder_width_cm, 1),
            "height_cm": round(self.height_cm, 1),
        }


def _convex_hull_perimeter(points_2d: np.ndarray) -> float:
    """Perimeter of convex hull of a 2D point cloud."""
    if len(points_2d) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_2d)
        verts = points_2d[hull.vertices]
        perimeter = np.linalg.norm(np.diff(verts, axis=0, append=verts[:1]), axis=1).sum()
        return float(perimeter)
    except Exception:
        # Degenerate: fallback to bounding-box perimeter
        span = points_2d.max(0) - points_2d.min(0)
        return float(2 * (span[0] + span[1]))


def _slice_girth_m(vertices: np.ndarray, y_level: float, thickness: float = 0.02) -> float:
    """Extract XZ cross-section at y=y_level ± thickness, return convex-hull perimeter (metres)."""
    mask = np.abs(vertices[:, 1] - y_level) < thickness
    if mask.sum() < 3:
        return 0.0
    xz = vertices[mask][:, [0, 2]]   # XZ plane
    return _convex_hull_perimeter(xz)


def compute_measurements(result: SMPLFitResult) -> Measurements:
    """Derive body measurements from fitted SMPL mesh.

    SMPL Y-axis points upward. Vertices are in metres.
    """
    verts = result.vertices  # (6890, 3)

    # Height
    y_max = verts[:, 1].max()
    y_min = verts[:, 1].min()
    height_m = float(y_max - y_min)
    height_cm = height_m * 100.0

    # Measurement planes as absolute Y positions
    chest_y = y_min + SLICE_FRACTIONS["chest"] * height_m
    waist_y = y_min + SLICE_FRACTIONS["waist"] * height_m
    hip_y = y_min + SLICE_FRACTIONS["hip"] * height_m

    chest_m = _slice_girth_m(verts, chest_y)
    waist_m = _slice_girth_m(verts, waist_y)
    hip_m = _slice_girth_m(verts, hip_y)

    # Shoulder width: Euclidean distance between shoulder vertices
    ls = verts[SMPL_VERTEX["left_shoulder"]]
    rs = verts[SMPL_VERTEX["right_shoulder"]]
    shoulder_m = float(np.linalg.norm(ls - rs))

    return Measurements(
        chest_cm=chest_m * 100.0,
        waist_cm=waist_m * 100.0,
        hip_cm=hip_m * 100.0,
        shoulder_width_cm=shoulder_m * 100.0,
        height_cm=height_cm,
    )
