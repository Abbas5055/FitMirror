"""
fitmirror.sizing
================

Indian-wear size charts and recommendation engine.

Charts express the body-measurement THRESHOLD for each size, in cm.
Read as: "if body measurement >= threshold AND < next threshold, you are
this size". Below the smallest threshold the user is still labelled as the
smallest size.

Borderline rule:
  If a measurement falls within BORDERLINE_CM of the boundary to an
  adjacent size, the per-dim result is reported as a pair (e.g. "S or M")
  so the user can pick based on personal fit preference. The same logic
  is applied at aggregation time when the per-dim votes span two sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GarmentType = Literal["mens_kurta", "womens_kurta", "womens_anarkali", "saree_blouse"]


# Within this many cm of an adjacent boundary, surface BOTH sizes for the user.
BORDERLINE_CM = 5.0


def _inch(x: float) -> float:
    """Inches to cm."""
    return x * 2.54


# --- Charts ---------------------------------------------------------------
# Each entry: (size_label, body_measurement_threshold_cm).
# Sorted ascending by threshold. The bucket for size i is
# [thresholds[i].threshold, thresholds[i+1].threshold).

# Men's kurta — provided by user, in inches; converted here.
MENS_KURTA = {
    "chest": [
        ("XS",   _inch(34)),
        ("S",    _inch(36)),
        ("M",    _inch(38)),
        ("L",    _inch(40)),
        ("XL",   _inch(42)),
        ("XXL",  _inch(44)),
        ("XXXL", _inch(46)),
    ],
    "waist": [
        ("XS",   _inch(28)),
        ("S",    _inch(30)),
        ("M",    _inch(32)),
        ("L",    _inch(34)),
        ("XL",   _inch(36)),
        ("XXL",  _inch(38)),
        ("XXXL", _inch(40)),
    ],
    "hip": [
        ("XS",   _inch(36)),
        ("S",    _inch(38)),
        ("M",    _inch(40)),
        ("L",    _inch(42)),
        ("XL",   _inch(44)),
        ("XXL",  _inch(46)),
        ("XXXL", _inch(48)),
    ],
}

# Women's kurta — standard Indian sizes (best-guess; replace with brand chart
# when available). Sizes 32 / 34 / 36 / 38 etc. correspond to bust in inches.
WOMENS_KURTA = {
    "chest": [
        ("XS",  _inch(32)),
        ("S",   _inch(34)),
        ("M",   _inch(36)),
        ("L",   _inch(38)),
        ("XL",  _inch(40)),
        ("XXL", _inch(42)),
    ],
    "waist": [
        ("XS",  _inch(26)),
        ("S",   _inch(28)),
        ("M",   _inch(30)),
        ("L",   _inch(32)),
        ("XL",  _inch(34)),
        ("XXL", _inch(36)),
    ],
    "hip": [
        ("XS",  _inch(34)),
        ("S",   _inch(36)),
        ("M",   _inch(38)),
        ("L",   _inch(40)),
        ("XL",  _inch(42)),
        ("XXL", _inch(44)),
    ],
}

# Anarkali — chest + waist only; hip is loose/flared.
WOMENS_ANARKALI = {
    "chest": WOMENS_KURTA["chest"],
    "waist": WOMENS_KURTA["waist"],
}

# Saree blouse — bust-driven, numeric size labels.
SAREE_BLOUSE = {
    "chest": [
        ("32", _inch(32)),
        ("34", _inch(34)),
        ("36", _inch(36)),
        ("38", _inch(38)),
        ("40", _inch(40)),
        ("42", _inch(42)),
        ("44", _inch(44)),
    ],
}


CHARTS: dict[GarmentType, dict[str, list]] = {
    "mens_kurta":      MENS_KURTA,
    "womens_kurta":    WOMENS_KURTA,
    "womens_anarkali": WOMENS_ANARKALI,
    "saree_blouse":    SAREE_BLOUSE,
}


# --- Recommendation -------------------------------------------------------

@dataclass
class Recommendation:
    garment: GarmentType
    size_label: str                       # e.g. "S" or "S or M"
    confidence: str                       # "high" | "medium" | "low"
    per_dimension: dict                   # dim -> {"size": str, "value_cm": float}
    reasoning: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        garment_label = {
            "mens_kurta":      "Men's Kurta",
            "womens_kurta":    "Women's Kurta",
            "womens_anarkali": "Women's Anarkali",
            "saree_blouse":    "Saree Blouse",
        }[self.garment]

        lines = [
            f"### Recommended size: **{self.size_label}** ({garment_label})",
            f"_Confidence: {self.confidence}_",
            "",
            "**Per-dimension fit:**",
        ]
        for dim, info in self.per_dimension.items():
            lines.append(
                f"- {dim.capitalize()}: {info['value_cm']} cm -> size **{info['size']}**"
            )
        if self.reasoning:
            lines.append("")
            lines.append("**Notes:**")
            for r in self.reasoning:
                lines.append(f"- {r}")
        return "\n".join(lines)


def _size_for_dim(thresholds: list[tuple[str, float]], value_cm: float) -> str:
    """Return size label for value_cm, possibly paired ('S or M') if borderline.

    Logic:
      - Below smallest threshold: return the smallest size.
      - Otherwise find the bucket where threshold[i] <= value < threshold[i+1].
      - If value is within BORDERLINE_CM of the boundary to the next size up,
        return 'primary or next'.
      - If value is within BORDERLINE_CM of the boundary to the previous size
        (i.e. just barely inside this bucket), return 'previous or primary'.
    """
    if value_cm < thresholds[0][1]:
        return thresholds[0][0]

    primary_idx = 0
    for i in range(len(thresholds)):
        if value_cm >= thresholds[i][1]:
            primary_idx = i
        else:
            break

    primary_label, primary_threshold = thresholds[primary_idx]

    # Above primary: within 5 cm of next size up?
    if primary_idx + 1 < len(thresholds):
        next_label, next_threshold = thresholds[primary_idx + 1]
        if next_threshold - value_cm < BORDERLINE_CM:
            return f"{primary_label} or {next_label}"

    # Just barely above the lower boundary (and not the smallest size)?
    if primary_idx > 0:
        if value_cm - primary_threshold < BORDERLINE_CM:
            prev_label = thresholds[primary_idx - 1][0]
            return f"{prev_label} or {primary_label}"

    return primary_label


def _all_sizes_in_order(chart: dict[str, list]) -> list[str]:
    """Canonical size order from the first dim of the chart."""
    first_dim = next(iter(chart.values()))
    return [label for label, _ in first_dim]


def _aggregate(per_dim_sizes: list[str], chart: dict[str, list]) -> tuple[str, str, list[str]]:
    """Combine per-dim size strings into (final_label, confidence, reasoning)."""
    order = _all_sizes_in_order(chart)

    # Collect all unique size labels mentioned across per-dim results.
    mentioned = set()
    for s in per_dim_sizes:
        for part in s.split(" or "):
            mentioned.add(part)

    ordered = sorted(mentioned, key=lambda x: order.index(x) if x in order else len(order))

    if len(ordered) == 1:
        return ordered[0], "high", []

    if len(ordered) == 2 and order.index(ordered[1]) - order.index(ordered[0]) == 1:
        return (
            f"{ordered[0]} or {ordered[1]}",
            "medium",
            ["Body measurements sit between two adjacent sizes; "
             "either should fit acceptably. Pick the larger if you prefer "
             "a looser drape."],
        )

    # Wider spread — round up to the largest mentioned size.
    return (
        ordered[-1],
        "low",
        ["Body measurements span more than two sizes. "
         "Rounded up; consider a tailored option."],
    )


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

    per_dim_sizes: list[str] = []
    per_dimension: dict = {}

    for dim, thresholds in chart.items():
        if dim not in measurements:
            continue
        size = _size_for_dim(thresholds, measurements[dim])
        per_dimension[dim] = {"size": size, "value_cm": round(measurements[dim], 1)}
        per_dim_sizes.append(size)

    if not per_dim_sizes:
        raise ValueError("No usable measurements for this garment.")

    final, confidence, reasoning = _aggregate(per_dim_sizes, chart)

    return Recommendation(
        garment=garment,
        size_label=final,
        confidence=confidence,
        per_dimension=per_dimension,
        reasoning=reasoning,
    )
