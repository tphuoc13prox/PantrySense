from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.backend.config import get_embedding_model_name, get_ranker_model_path, get_ranking_mode
from app.backend.ranking.features import LTR_FEATURE_NAMES, build_feature_matrix
from app.backend.recipes.schemas import RecipeSummary

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "pantrysense-ltr"


def heuristic_sort_key(r: RecipeSummary) -> tuple:
    return (-r.coverage, -r.matched_count, len(r.missing_ingredients), r.title.lower())


def retrieval_sort_key(r: RecipeSummary) -> tuple:
    """Retriever order (RRF rank in hybrid mode); ties and unranked rows by the heuristic."""
    rank = r.retrieval_rank if r.retrieval_rank is not None else 10**9
    return (rank, *heuristic_sort_key(r))


class MLRecipeRanker:
    """Learning-to-rank re-ranker.

    The model file is a bundle dict written by scripts/train_ranker.py
    ({"format": "pantrysense-ltr", "model": ..., "feature_names": [...]}).
    Without a usable bundle, candidates keep the retriever's order: on the
    official eval that beats both the coverage heuristic and the old bare
    LightGBM files, which are therefore ignored.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_ranker_model_path()
        self._model: Any | None = None
        self._feature_idx: list[int] | None = None
        self._predict_kwargs: dict[str, Any] = {}
        self._loaded = False
        self._warned = False

    def load_model(self) -> bool:
        if self._loaded and self._model is not None:
            return True

        if not self.model_path.exists():
            if not self._warned:
                logger.info("No ranker bundle at %s; keeping the retriever's order.", self.model_path)
                self._warned = True
            return False

        try:
            import joblib
            obj = joblib.load(self.model_path)
            if not (isinstance(obj, dict) and obj.get("format") == BUNDLE_FORMAT):
                raise ValueError("not a pantrysense-ltr bundle (an old bare model?); retrain with scripts/train_ranker.py")
            names = list(obj["feature_names"])
            unknown = [n for n in names if n not in LTR_FEATURE_NAMES]
            if unknown:
                raise ValueError(f"model uses unknown features: {unknown}")
            self._model = obj["model"]
            self._feature_idx = [LTR_FEATURE_NAMES.index(n) for n in names]
            self._warn_on_embedding_mismatch(obj.get("embedding_model"))
            # LightGBM otherwise starts an OpenMP pool on every core for a
            # ~20-row predict, and the spinning threads slow down the next
            # query's FAISS/BM25/embedding work.
            is_lgbm = type(self._model).__module__.startswith("lightgbm")
            self._predict_kwargs = {"num_threads": 1} if is_lgbm else {}
            self._loaded = True
            logger.info("Loaded LTR ranker bundle from %s", self.model_path)
            return True
        except Exception as e:
            if not self._warned:
                logger.warning("Ignoring ranker file %s (%s); keeping the retriever's order.", self.model_path, e)
                self._warned = True
            self._model = None
            self._feature_idx = None
            self._loaded = False
            return False

    @staticmethod
    def _warn_on_embedding_mismatch(trained_on: str | None) -> None:
        if not trained_on:
            return
        from app.backend.retrieval.embedder import same_embedding_model

        current = get_embedding_model_name()
        if not same_embedding_model(trained_on, current):
            logger.warning(
                "Ranker was trained on candidates from %s but queries use %s; scores may be off. "
                "Rebuild the LTR dataset and retrain (docs/ltr.md).",
                trained_on,
                current,
            )

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
        assert self._feature_idx is not None
        matrix = build_feature_matrix(recipes, details_map)[:, self._feature_idx]
        return np.asarray(self._model.predict(matrix, **self._predict_kwargs), dtype=np.float64)

    def rank(
        self,
        recipes: list[RecipeSummary],
        details_map: dict[int, dict[str, Any]] | None = None,
        minimum_coverage: float = 0.34,
        force_heuristic: bool = False,
    ) -> list[RecipeSummary]:
        """Rank candidates with the LTR model.

        force_heuristic or PANTRYSENSE_RANKING_MODE != "ml" sorts by coverage.
        A missing or failing model keeps the retriever's order.
        """
        if not recipes:
            return []

        filtered = [r for r in recipes if r.coverage >= minimum_coverage]
        if not filtered:
            return []

        if force_heuristic or get_ranking_mode() != "ml":
            filtered.sort(key=heuristic_sort_key)
            return filtered

        if self.load_model() and self._model is not None:
            try:
                predictions = self.score(filtered, details_map)
                scored = [
                    summary.model_copy(update={"ml_score": round(float(score), 4)})
                    for summary, score in zip(filtered, predictions, strict=True)
                ]
                scored.sort(key=lambda r: (-(r.ml_score if r.ml_score is not None else -999.0), *heuristic_sort_key(r)))
                return scored
            except Exception as e:
                logger.warning("LTR prediction failed, keeping the retriever's order: %s", e)

        filtered.sort(key=retrieval_sort_key)
        return filtered
