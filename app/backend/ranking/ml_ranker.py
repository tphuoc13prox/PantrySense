from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.backend.config import get_ranker_model_path, get_ranking_mode
from app.backend.ranking.features import extract_features
from app.backend.recipes.schemas import RecipeSummary

logger = logging.getLogger(__name__)


class MLRecipeRanker:
    """Machine Learning Recipe Re-Ranker using LightGBM / Gradient Boosting."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_ranker_model_path()
        self._model: Any | None = None
        self._loaded = False

    def load_model(self) -> bool:
        """Attempt to load the trained ML model from disk."""
        if self._loaded and self._model is not None:
            return True

        if not self.model_path.exists():
            return False

        try:
            import joblib
            self._model = joblib.load(self.model_path)
            self._loaded = True
            logger.info("Loaded ML Ranker model from %s", self.model_path)
            return True
        except Exception as e:
            logger.warning("Failed to load ML Ranker from %s: %s", self.model_path, e)
            self._model = None
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self.load_model()

    def rank(
        self,
        recipes: list[RecipeSummary],
        details_map: dict[int, dict[str, Any]] | None = None,
        minimum_coverage: float = 0.34,
        force_heuristic: bool = False,
    ) -> list[RecipeSummary]:
        """Rank candidate recipes using ML Ranker (or Heuristic fallback)."""
        if not recipes:
            return []

        # Filter by minimum coverage threshold
        filtered = [r for r in recipes if r.coverage >= minimum_coverage]
        if not filtered:
            return []

        ranking_mode = get_ranking_mode()
        use_ml = (not force_heuristic) and (ranking_mode == "ml") and self.load_model()

        if use_ml and self._model is not None:
            try:
                feature_matrix = [
                    extract_features(r, details_map.get(r.id) if details_map else None)
                    for r in filtered
                ]
                predictions = self._model.predict(feature_matrix)

                scored_recipes: list[RecipeSummary] = []
                for summary, score in zip(filtered, predictions):
                    updated = summary.model_copy(update={"ml_score": round(float(score), 4)})
                    scored_recipes.append(updated)

                # Sort descending by ML predicted score, then coverage, then matched_count
                scored_recipes.sort(
                    key=lambda r: (
                        -(r.ml_score if r.ml_score is not None else -999.0),
                        -r.coverage,
                        -r.matched_count,
                        len(r.missing_ingredients),
                        r.title.lower(),
                    )
                )
                return scored_recipes
            except Exception as e:
                logger.warning("ML ranking prediction failed, falling back to heuristic: %s", e)

        # Heuristic fallback (v0.5 sorting)
        filtered.sort(
            key=lambda r: (
                -r.coverage,
                -r.matched_count,
                len(r.missing_ingredients),
                r.title.lower(),
            )
        )
        return filtered
