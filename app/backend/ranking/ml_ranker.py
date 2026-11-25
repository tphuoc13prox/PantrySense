from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.backend.config import get_ranker_model_path, get_ranking_mode
from app.backend.ranking.features import LTR_FEATURE_NAMES, build_feature_matrix, extract_features
from app.backend.recipes.schemas import RecipeSummary

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "pantrysense-ltr"


def heuristic_sort_key(r: RecipeSummary) -> tuple:
    return (-r.coverage, -r.matched_count, len(r.missing_ingredients), r.title.lower())


class MLRecipeRanker:
    """Learning-to-rank re-ranker with a coverage heuristic fallback.

    The model file is either a bundle dict written by scripts/train_ranker.py
    ({"format": "pantrysense-ltr", "model": ..., "feature_names": [...]}) or,
    for older files, a bare estimator over the legacy 9 features.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_ranker_model_path()
        self._model: Any | None = None
        self._feature_idx: list[int] | None = None
        self._predict_kwargs: dict[str, Any] = {}
        self._loaded = False

    def load_model(self) -> bool:
        if self._loaded and self._model is not None:
            return True

        if not self.model_path.exists():
            return False

        try:
            import joblib
            obj = joblib.load(self.model_path)
            if isinstance(obj, dict) and obj.get("format") == BUNDLE_FORMAT:
                names = list(obj["feature_names"])
                unknown = [n for n in names if n not in LTR_FEATURE_NAMES]
                if unknown:
                    raise ValueError(f"model uses unknown features: {unknown}")
                self._model = obj["model"]
                self._feature_idx = [LTR_FEATURE_NAMES.index(n) for n in names]
            else:
                self._model = obj
                self._feature_idx = None
            # LightGBM otherwise starts an OpenMP pool on every core for a
            # ~20-row predict, and the spinning threads slow down the next
            # query's FAISS/BM25/embedding work.
            is_lgbm = type(self._model).__module__.startswith("lightgbm")
            self._predict_kwargs = {"num_threads": 1} if is_lgbm else {}
            self._loaded = True
            logger.info("Loaded ML Ranker model from %s", self.model_path)
            return True
        except Exception as e:
            logger.warning("Failed to load ML Ranker from %s: %s", self.model_path, e)
            self._model = None
            self._feature_idx = None
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self.load_model()

    def score(
        self,
        recipes: list[RecipeSummary],
        details_map: dict[int, dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Raw model scores for an already-filtered candidate list."""
        if not self.load_model() or self._model is None:
            raise RuntimeError("ranker model not loaded")
        if self._feature_idx is None:
            matrix = np.array(
                [extract_features(r, details_map.get(r.id) if details_map else None) for r in recipes],
                dtype=np.float32,
            )
        else:
            matrix = build_feature_matrix(recipes, details_map)[:, self._feature_idx]
        return np.asarray(self._model.predict(matrix, **self._predict_kwargs), dtype=np.float64)

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

        filtered = [r for r in recipes if r.coverage >= minimum_coverage]
        if not filtered:
            return []

        ranking_mode = get_ranking_mode()
        use_ml = (not force_heuristic) and (ranking_mode == "ml") and self.load_model()

        if use_ml and self._model is not None:
            try:
                predictions = self.score(filtered, details_map)
                scored = [
                    summary.model_copy(update={"ml_score": round(float(score), 4)})
                    for summary, score in zip(filtered, predictions, strict=True)
                ]
                scored.sort(key=lambda r: (-(r.ml_score if r.ml_score is not None else -999.0), *heuristic_sort_key(r)))
                return scored
            except Exception as e:
                logger.warning("ML ranking prediction failed, falling back to heuristic: %s", e)

        filtered.sort(key=heuristic_sort_key)
        return filtered
