from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Counter

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize a text string into lowercased word tokens."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Store:
    """Sparse lexical index implementing BM25Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.recipe_ids: list[int] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.total_docs: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total_docs == 0

    def add_corpus(self, recipe_ids: list[int], documents: list[str]) -> None:
        """Build the BM25 index from a list of recipe IDs and their raw text representations."""
        if len(recipe_ids) != len(documents):
            raise ValueError("Number of recipe IDs must match number of documents.")

        self.recipe_ids = list(recipe_ids)
        self.total_docs = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_length = 0
        for doc in documents:
            tokens = tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = dict(Counter(tokens))
            self.term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_length / self.total_docs if self.total_docs > 0 else 0.0

    def _idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones IDF."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str | list[str], top_k: int = 20) -> list[tuple[int, float, int]]:
        """Search the corpus using BM25Okapi scoring.

        Returns:
            List of (recipe_id, bm25_score, rank_1_indexed) tuples.
        """
        if self.is_empty:
            return []

        if isinstance(query, str):
            tokens = tokenize(query)
        else:
            tokens = [t.lower() for t in query if t]

        if not tokens:
            return []

        scores: list[float] = [0.0] * self.total_docs

        for term in tokens:
            if term not in self.doc_freqs:
                continue

            idf = self._idf(term)
            for idx in range(self.total_docs):
                tf = self.term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf * (numerator / denominator)

        # Filter out zero scores and sort descending
        scored_candidates = [
            (self.recipe_ids[idx], score)
            for idx, score in enumerate(scores)
            if score > 0.0
        ]
        scored_candidates.sort(key=lambda x: -x[1])

        top_candidates = scored_candidates[:top_k]
        return [
            (recipe_id, round(score, 4), rank)
            for rank, (recipe_id, score) in enumerate(top_candidates, start=1)
        ]

    def save(self, path: Path) -> None:
        """Serialize BM25 index state to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "recipe_ids": self.recipe_ids,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
            "total_docs": self.total_docs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved BM25 index to %s", path)

    def load(self, path: Path) -> bool:
        """Load BM25 index state from JSON."""
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.k1 = float(data["k1"])
            self.b = float(data["b"])
            self.recipe_ids = [int(i) for i in data["recipe_ids"]]
            self.doc_lengths = [int(i) for i in data["doc_lengths"]]
            self.avg_doc_len = float(data["avg_doc_len"])
            self.doc_freqs = {str(k): int(v) for k, v in data["doc_freqs"].items()}
            self.term_freqs = [{str(k): int(v) for k, v in tf.items()} for tf in data["term_freqs"]]
            self.total_docs = int(data["total_docs"])
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 index from %s: %s", path, e)
            return False
