"""
fitmirror.sizing
================

Indian-wear size charts and recommendation engine.

Garment types supported in v1:
  - "mens_kurta"          (chest-driven)
  - "womens_kurta"        (chest + waist + hip)
  - "womens_anarkali"     (chest + waist; hip ignored — anarkali is loose at hip)
  - "saree_blouse"        (bust-driven, numeric sizes 32..44)

Logic:
  Each chart entry is a (label, {dim: (lo, hi)}) tuple, ranges in cm, inclusive.
  For a given measurement set we pick, per dimension, the size whose range the
  measurement falls into (or the closest one if it falls between buckets).
  We then aggregate into a single recommendation:
    - if all dimensions agree -> that size (high confidence)
    - if they disagree        -> the size most dimensions vote for, biased UP
                                  (better to be loose than tight) and we surface
                                  the disagreement in the reasoning text.

Honest about limits — see Measurements docs in measure.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GarmentType = Literal["mens_kurta", "womens_kurta", "womens_anarkali", "saree_blouse"]


# --- Charts ---------------------------------------------------------------

# Men's kurta: chest cm. Standard Indian size chart, M ≈ 38" chest = 96.5 cm.
MENS_KURTA = [
    ("XS",  {"chest": (84.0,  89.0)}),
    ("S",   {"chest": (89.1,  94.0)}),
    ("M",   {"chest": (94.1,  99.0)}),
    ("L",   {"chest": (99.1, 104.0)}),
    ("XL",  {"chest": (104.1, 109.0)}),
    ("XXL", {"chest": (109.1, 114.0)}),
    ("XXXL",{"chest": (114.1, 120.0)}),
]

# Women's kurta: chest + waist + hip, all in cm. S corresponds to 34" = 86.4 cm bust.
WOMENS_KURTA = [
    ("XS",  {"chest": (78.0,  83.0), "waist": (62.0, 67.0), "hip": (84.0,  91.0)}),
    ("S",   {"chest": (83.1,  87.0), "waist": (67.1, 71.0), "hip": (91.1,  95.0)}),
    ("M",   {"chest": (87.1,  91.0), "waist": (71.1, 75.0), "hip": (95.1,  99.0)}),
    ("L",   {"chest": (91.1,  96.0), "waist": (75.1, 80.0), "hip": (99.1, 104.0)}),
    ("XL",  {"chest": (96.1, 101.0), "waist": (80.1, 85.0), "hip": (104.1, 109.0)}),
    ("XXL", {"chest": (101.1, 106.0), "waist": (85.1, 90.0), "hip": (109.1, 114.0)}),
]

# Women's anarkali: same chest/waist as kurta; hip excluded (loose flare at hip).
WOMENS_ANARKALI = [
    (label, {k: v for k, v in dims.items() if k != "hip"})
    for label, dims in WOMENS_KURTA
]

# Saree blouse: numeric sizes (band sizes), bust-driven.
SAREE_BLOUSE = [
    ("32", {"chest": (81.0, 83.5)}),
    ("34", {"chest": (83.6, 86.5)}),
    ("36", {"chest": (86.6, 89.5)}),
    ("38", {"chest": (89.6, 93.5)}),
    ("40", {"chest": (93.6, 97.5)}),
    ("42", {"chest": (97.6, 101.5)}),
    ("44", {"chest": (101.6, 106.0)}),
]


CHARTS: dict[GarmentType, list] = {
    "mens_kurta":      MENS_KURTA,
    "womens_kurta":    WOMENS_KURTA,
    "womens_anarkali": WOMENS_ANARKALI,
    "saree_blouse":    SAREE_BLOUSE,
}


# --- Recommendation -------------------------------------------------------

@dataclass
class Recommendation:
    garment: GarmentType
    size: str                         # e.g. "M" or "34"
    confidence: str                   # "high" | "medium" | "low"
    per_dimension: dict               # {dim: {"size": "M", "value_cm": 96.0}}
    reasoning: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        garment_label = {
            "mens_kurta":      "Men's Kurta",
            "womens_kurta":    "Women's Kurta",
            "womens_anarkali": "Women's Anarkali",
            "saree_blouse":    "Saree Blouse",
        }[self.garment]

        lines = [
            f"### Recommended size: **{self.size}** ({garment_label})",
            f"_Confidence: {self.confidence}_",
            "",
            "**Per-dimension fit:**",
        ]
        for dim, info in self.per_dimension.items():
            lines.append(
                f"- {dim.capitalize()}: {info['value_cm']} cm → size **{info['size']}**"
            )
        if self.reasoning:
            lines.append("")
            lines.append("**Notes:**")
            for r in self.reasoning:
                lines.append(f"- {r}")
        return "\n".join(lines)


def _size_for_dim(chart: list, dim: str, value_cm: float) -> str:
    """Return the size label whose bucket contains value_cm, or the nearest one."""
    candidates = [(label, dims[dim]) for label, dims in chart if dim in dims]
    for label, (lo, hi) in candidates:
        if lo <= value_cm <= hi:
            return label

    # Fall back to nearest by midpoint distance.
    def dist(label_range):
        lo, hi = label_range[1]
        mid = (lo + hi) / 2.0
        return abs(value_cm - mid)

    candidates.sort(key=dist)
    return candidates[0][0]


def _aggregate(per_dim_sizes: list[str], chart: list) -> tuple[str, str]:
    """Pick a single size from a list of per-dim votes. Returns (size, confidence)."""
    order = [label for label, _ in chart]
    indices = [order.index(s) for s in per_dim_sizes]

    # All agree -> high confidence.
    if len(set(indices)) == 1:
        return order[indices[0]], "high"

    # Disagree by 1 -> bias UP (looser fit). Medium confidence.
    spread = max(indices) - min(indices)
    if spread == 1:
        return order[max(indices)], "medium"

    # Wider spread -> still bias up, but low confidence.
    return order[max(indices)], "low"


def recommend(
    garment: GarmentType,
    chest_cm: float,
    waist_cm: float | None = None,
    hip_cm: float | None = None,
) -> Recommendation:
    if garment not in CHARTS:
        raise ValueError(f"Unknown garment type: {garment}")

    chart = CHARTS[garment]
    measurements = {"chest": chest_cm}
    if waist_cm is not None:
        measurements["waist"] = waist_cm
    if hip_cm is not None:
        measurements["hip"] = hip_cm

    needed_dims = sorted({d for _, dims in chart for d in dims.keys()})
    per_dim_sizes = []
    per_dimension = {}
    reasoning = []

    for dim in needed_dims:
        if dim not in measurements:
            reasoning.append(
                f"{dim.capitalize()} measurement not available; using chest only."
            )
            continue
        size = _size_for_dim(chart, dim, measurements[dim])
        per_dim_sizes.append(size)
        per_dimension[dim] = {"size": size, "value_cm": round(measurements[dim], 1)}

    if not per_dim_sizes:
        raise ValueError("No usable measurements for this garment.")

    final_size, confidence = _aggregate(per_dim_sizes, chart)

    if confidence == "medium":
        reasoning.append(
            "Dimensions span two adjacent sizes; rounded **up** for a looser, "
            "more comfortable fit."
        )
    elif confidence == "low":
        reasoning.append(
            "Dimensions span more than two sizes — body shape may be between "
            "standard charts. Rounded up; consider a tailored option."
        )

    return Recommendation(
        garment=garment,
        size=final_size,
        confidence=confidence,
        per_dimension=per_dimension,
        reasoning=reasoning,
    )
