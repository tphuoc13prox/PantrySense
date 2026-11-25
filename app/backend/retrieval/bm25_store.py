from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.backend.retrieval.filtering import AllowedIds

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")
NPZ_FORMAT_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Tokenize a text string into lowercased word tokens."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def preferred_index_path(path: Path) -> Path:
    """Return the ``.npz`` sibling of a legacy ``.json`` path when it exists.

    Lets existing configuration keep pointing at ``recipe_bm25.json`` while
    the faster binary file, written next to it, is what actually gets loaded.
    """
    if path.suffix.lower() == ".json":
        npz = path.with_suffix(".npz")
        if npz.exists() and (not path.exists() or npz.stat().st_mtime >= path.stat().st_mtime):
            return npz
    return path


def bm25_index_exists(path: Path) -> bool:
    return preferred_index_path(path).exists()


class BM25Store:
    """BM25Okapi index stored as term-major postings (CSR) in numpy arrays.

    Postings for term ``t`` live in ``doc_idx[indptr[t]:indptr[t+1]]`` with the
    matching term frequencies in ``tf``. The per-posting BM25 weight is
    precomputed at load time so a query is a few vectorized scatter-adds.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len: float = 0.0
        self.total_docs: int = 0
        self._ids = np.empty(0, dtype=np.int64)
        self._doc_len = np.empty(0, dtype=np.int32)
        self._vocab: dict[str, int] = {}
        self._df = np.empty(0, dtype=np.int32)
        self._indptr = np.zeros(1, dtype=np.int64)
        self._doc_idx = np.empty(0, dtype=np.int32)
        self._tf = np.empty(0, dtype=np.int32)
        self._idf = np.empty(0, dtype=np.float64)
        self._weights = np.empty(0, dtype=np.float64)
        self._recipe_ids_list: list[int] | None = None

    # ------------------------------------------------------------------
    # Read-only views kept for callers of the previous dict-based store
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self.total_docs == 0

    @property
    def recipe_ids(self) -> list[int]:
        if self._recipe_ids_list is None:
            self._recipe_ids_list = self._ids.tolist()
        return self._recipe_ids_list

    @property
    def recipe_id_array(self) -> np.ndarray:
        return self._ids

    @property
    def doc_lengths(self) -> list[int]:
        return self._doc_len.tolist()

    @property
    def doc_freqs(self) -> dict[str, int]:
        return {term: int(self._df[tid]) for term, tid in self._vocab.items()}

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_corpus(self, recipe_ids: list[int], documents: list[str]) -> None:
        """Build the index from recipe IDs and their raw text representations."""
        if len(recipe_ids) != len(documents):
            raise ValueError("Number of recipe IDs must match number of documents.")

        vocab: dict[str, int] = {}
        term_ids: list[int] = []
        doc_ids: list[int] = []
        freqs: list[int] = []
        doc_len = np.zeros(len(documents), dtype=np.int32)

        for d, doc in enumerate(documents):
            tokens = tokenize(doc)
            doc_len[d] = len(tokens)
            for term, count in Counter(tokens).items():
                tid = vocab.get(term)
                if tid is None:
                    tid = vocab[term] = len(vocab)
                term_ids.append(tid)
                doc_ids.append(d)
                freqs.append(count)

        self._set_postings(
            recipe_ids=np.asarray(recipe_ids, dtype=np.int64),
            doc_len=doc_len,
            vocab=vocab,
            term_ids=np.asarray(term_ids, dtype=np.int64),
            doc_ids=np.asarray(doc_ids, dtype=np.int32),
            freqs=np.asarray(freqs, dtype=np.int32),
        )

    def _set_postings(
        self,
        recipe_ids: np.ndarray,
        doc_len: np.ndarray,
        vocab: dict[str, int],
        term_ids: np.ndarray,
        doc_ids: np.ndarray,
        freqs: np.ndarray,
    ) -> None:
        order = np.lexsort((doc_ids, term_ids))
        counts = np.bincount(term_ids, minlength=len(vocab)) if term_ids.size else np.zeros(len(vocab), np.int64)
        indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        self._ids = recipe_ids
        self._doc_len = doc_len.astype(np.int32, copy=False)
        self._vocab = vocab
        self._indptr = indptr
        self._doc_idx = doc_ids[order].astype(np.int32, copy=False)
        self._tf = freqs[order].astype(np.int32, copy=False)
        self._df = counts.astype(np.int32)
        self.total_docs = int(recipe_ids.shape[0])
        self.avg_doc_len = float(doc_len.sum()) / self.total_docs if self.total_docs else 0.0
        self._finalize()

    def _finalize(self) -> None:
        """Precompute IDF per term and the BM25 weight of every posting."""
        self._recipe_ids_list = None
        n = self.total_docs
        df = self._df.astype(np.float64)
        self._idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0) if n else np.empty(0)
        if self._doc_idx.size == 0 or self.avg_doc_len == 0:
            self._weights = np.zeros(self._doc_idx.shape[0], dtype=np.float64)
            return
        tf = self._tf.astype(np.float64)
        dl = self._doc_len[self._doc_idx].astype(np.float64)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
        self._weights = tf * (self.k1 + 1.0) / denom

    def _idf_scalar(self, term: str) -> float:
        tid = self._vocab.get(term)
        df = int(self._df[tid]) if tid is not None else 0
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def score_all(self, query: str | list[str]) -> np.ndarray | None:
        """BM25 score of every document for ``query`` (None when no term matches)."""
        if self.is_empty:
            return None
        tokens = tokenize(query) if isinstance(query, str) else [t.lower() for t in query if t]
        if not tokens:
            return None
        scores: np.ndarray | None = None
        # Repeated query tokens count once per occurrence, as in the original scorer.
        for term in tokens:
            tid = self._vocab.get(term)
            if tid is None:
                continue
            start, end = self._indptr[tid], self._indptr[tid + 1]
            if start == end:
                continue
            if scores is None:
                scores = np.zeros(self.total_docs, dtype=np.float64)
            scores[self._doc_idx[start:end]] += self._idf[tid] * self._weights[start:end]
        return scores

    def search(
        self,
        query: str | list[str],
        top_k: int = 20,
        allowed: AllowedIds | None = None,
    ) -> list[tuple[int, float, int]]:
        """Top-k BM25 matches, optionally restricted to ``allowed`` recipe ids.

        Returns:
            List of (recipe_id, bm25_score, rank_1_indexed) tuples.
        """
        if top_k <= 0:
            return []
        scores = self.score_all(query)
        if scores is None:
            return []
        if allowed is not None:
            scores[~allowed.mask_for(self._ids)] = 0.0

        hits = np.flatnonzero(scores > 0.0)
        if hits.size == 0:
            return []
        hit_scores = scores[hits]
        if hits.size > top_k:
            keep = np.argpartition(-hit_scores, top_k - 1)[:top_k]
            hits, hit_scores = hits[keep], hit_scores[keep]
        order = np.lexsort((hits, -hit_scores))
        return [
            (int(self._ids[hits[i]]), round(float(hit_scores[i]), 4), rank)
            for rank, i in enumerate(order, start=1)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the index; ``.npz`` paths use the binary format, others legacy JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npz":
            self.save_npz(path)
        else:
            self.save_json(path)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = [""] * len(self._vocab)
        for term, tid in self._vocab.items():
            terms[tid] = term
        vocab_blob = np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            format_version=np.int32(NPZ_FORMAT_VERSION),
            params=np.array([self.k1, self.b, self.avg_doc_len], dtype=np.float64),
            recipe_ids=self._ids,
            doc_lengths=self._doc_len,
            vocab=vocab_blob,
            vocab_size=np.int64(len(terms)),
            df=self._df,
            indptr=self._indptr,
            doc_idx=self._doc_idx,
            tf=self._tf.astype(np.uint16) if self._tf.size and int(self._tf.max()) < 65536 else self._tf,
        )
        tmp.replace(path)
        logger.info("Saved BM25 index (npz) to %s", path)

    def save_json(self, path: Path) -> None:
        """Write the legacy JSON layout read by older versions."""
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = [""] * len(self._vocab)
        for term, tid in self._vocab.items():
            terms[tid] = term
        term_freqs: list[dict[str, int]] = [{} for _ in range(self.total_docs)]
        for tid, term in enumerate(terms):
            start, end = self._indptr[tid], self._indptr[tid + 1]
            for d, f in zip(self._doc_idx[start:end].tolist(), self._tf[start:end].tolist()):
                term_freqs[d][term] = int(f)
        data = {
            "k1": self.k1,
            "b": self.b,
            "recipe_ids": self.recipe_ids,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "doc_freqs": self.doc_freqs,
            "term_freqs": term_freqs,
            "total_docs": self.total_docs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved BM25 index (legacy JSON) to %s", path)

    def load(self, path: Path) -> bool:
        """Load an index in either format; the format is detected from the file."""
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                magic = f.read(2)
            if magic == b"PK":
                self._load_npz(path)
            else:
                self._load_json(path)
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 index from %s: %s", path, e)
            return False

    def _load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != NPZ_FORMAT_VERSION:
                raise ValueError(f"Unsupported BM25 npz format version {version}")
            k1, b, avg = (float(x) for x in data["params"])
            vocab_size = int(data["vocab_size"])
            blob = data["vocab"].tobytes().decode("utf-8")
            terms = blob.split("\n") if vocab_size else []
            if len(terms) != vocab_size:
                raise ValueError("Corrupt BM25 vocabulary")
            self.k1, self.b, self.avg_doc_len = k1, b, avg
            self._ids = data["recipe_ids"].astype(np.int64, copy=False)
            self._doc_len = data["doc_lengths"].astype(np.int32, copy=False)
            self._df = data["df"].astype(np.int32, copy=False)
            self._indptr = data["indptr"].astype(np.int64, copy=False)
            self._doc_idx = data["doc_idx"].astype(np.int32, copy=False)
            self._tf = data["tf"].astype(np.int32)
        self._vocab = {term: i for i, term in enumerate(terms)}
        self.total_docs = int(self._ids.shape[0])
        self._finalize()

    def _load_json(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = float(data["k1"])
        self.b = float(data["b"])
        vocab: dict[str, int] = {}
        term_ids: list[int] = []
        doc_ids: list[int] = []
        freqs: list[int] = []
        for d, tf in enumerate(data["term_freqs"]):
            for term, count in tf.items():
                tid = vocab.get(term)
                if tid is None:
                    tid = vocab[term] = len(vocab)
                term_ids.append(tid)
                doc_ids.append(d)
                freqs.append(int(count))
        self._set_postings(
            recipe_ids=np.asarray([int(i) for i in data["recipe_ids"]], dtype=np.int64),
            doc_len=np.asarray(data["doc_lengths"], dtype=np.int32),
            vocab=vocab,
            term_ids=np.asarray(term_ids, dtype=np.int64),
            doc_ids=np.asarray(doc_ids, dtype=np.int32),
            freqs=np.asarray(freqs, dtype=np.int32),
        )
        # Keep the stored average so scores match the file exactly.
        self.avg_doc_len = float(data["avg_doc_len"])
        self._finalize()
        logger.info(
            "Loaded legacy JSON BM25 index from %s; convert it with "
            "'python scripts/build_vector_index.py --convert-bm25' for faster startup.",
            path,
        )


def convert_bm25_index(source: Path, target: Path) -> BM25Store:
    """Re-save an index (typically legacy JSON) in another format."""
    store = BM25Store()
    if not store.load(source):
        raise FileNotFoundError(f"Could not load BM25 index from {source}")
    store.save(target)
    return store
