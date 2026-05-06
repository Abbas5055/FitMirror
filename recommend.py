"""Size recommendation engine with cm-level explainability.

Core business value: not just the size, but WHY — "chest 96 cm fits L (94–98)".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitmirror.body.measure import Measurements

_CHART_PATH = Path(__file__).parent / "size_charts.csv"


@dataclass
class SizeRecommendation:
    """Output of size recommendation."""
    garment_type: str
    recommended_size: str
    confidence: float            # 0–1 based on how centred in range
    reasoning: list[str]         # human-readable per-measurement explanation
    all_scores: dict[str, float] # size → fit score for UI display

    def to_dict(self) -> dict:
        return {
            "garment_type": self.garment_type,
            "recommended_size": self.recommended_size,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "all_scores": {k: round(v, 3) for k, v in self.all_scores.items()},
        }


class SizeRecommender:
    """Rule-based size recommender backed by garment-specific size charts."""

    def __init__(self, chart_path: str | Path = _CHART_PATH) -> None:
        self._df = pd.read_csv(chart_path)

    def recommend(
        self,
        measurements: Measurements,
        garment_type: str = "kurta_men",
    ) -> SizeRecommendation:
        """Return best-fit size + full reasoning.

        Parameters
        ----------
        measurements : Measurements (in cm)
        garment_type : one of kurta_men / kurta_women / saree_blouse
        """
        chart = self._df[self._df["garment_type"] == garment_type].copy()
        if chart.empty:
            raise ValueError(f"No size chart for garment_type='{garment_type}'")

        scores: dict[str, float] = {}
        reasoning_map: dict[str, list[str]] = {}

        m = measurements

        for _, row in chart.iterrows():
            size = row["size"]
            fit_scores = []
            reasons = []

            # Chest
            c_min, c_max = row.get("chest_min"), row.get("chest_max")
            if pd.notna(c_min) and pd.notna(c_max):
                score, reason = self._fit(m.chest_cm, c_min, c_max, "chest")
                fit_scores.append(score)
                reasons.append(reason)

            # Waist
            w_min, w_max = row.get("waist_min"), row.get("waist_max")
            if pd.notna(w_min) and pd.notna(w_max):
                score, reason = self._fit(m.waist_cm, w_min, w_max, "waist")
                fit_scores.append(score)
                reasons.append(reason)

            # Hip
            h_min, h_max = row.get("hip_min"), row.get("hip_max")
            if pd.notna(h_min) and pd.notna(h_max):
                score, reason = self._fit(m.hip_cm, h_min, h_max, "hip")
                fit_scores.append(score)
                reasons.append(reason)

            scores[size] = float(sum(fit_scores) / max(len(fit_scores), 1))
            reasoning_map[size] = reasons

        best_size = max(scores, key=scores.__getitem__)
        best_score = scores[best_size]

        return SizeRecommendation(
            garment_type=garment_type,
            recommended_size=best_size,
            confidence=best_score,
            reasoning=reasoning_map[best_size],
            all_scores=scores,
        )

    @staticmethod
    def _fit(
        value: float, lo: float, hi: float, label: str
    ) -> tuple[float, str]:
        """Score how well `value` fits [lo, hi] range.

        Returns (score in [0,1], human-readable explanation).
        """
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        dist = abs(value - mid)

        if lo <= value <= hi:
            # Inside range: score by centrality
            score = 1.0 - (dist / (half_range + 1e-6)) * 0.3
            reason = (
                f"{label} {value:.1f} cm fits this range ({lo:.0f}–{hi:.0f} cm) ✓"
            )
        else:
            # Outside range: penalise by overflow
            overflow = dist - half_range
            score = max(0.0, 1.0 - overflow / (half_range + 1e-6))
            side = "too small" if value < lo else "too large"
            reason = (
                f"{label} {value:.1f} cm is {overflow:.1f} cm {side} "
                f"for this range ({lo:.0f}–{hi:.0f} cm)"
            )
        return score, reason
