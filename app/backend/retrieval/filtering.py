from __future__ import annotations

import threading

import numpy as np


class PositionCache:
    """Cached positions of a store's id order inside a sorted id universe."""

    __slots__ = ("entries", "lock")

    def __init__(self) -> None:
        self.entries: dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
        self.lock = threading.Lock()


class AllowedIds:
    """Set of recipe ids that pass a metadata filter.

    Retrievers keep their own id order (FAISS row order, BM25 doc order), so
    the set hands out a boolean mask aligned to whichever order a store asks
    for. Position lookups are cached per id array, and when a store's order
    already matches the sorted universe (the usual case, both indices are
    built with ORDER BY id) the mask is returned as-is.

    ``position_cache`` lets a long-lived owner (the metadata index) share the
    id-to-position lookups between the short-lived sets it creates per query.
    """

    __slots__ = ("_sorted_ids", "_mask", "_count", "_cache", "_lock")

    def __init__(
        self,
        sorted_ids: np.ndarray,
        mask: np.ndarray,
        position_cache: "PositionCache | None" = None,
    ) -> None:
        if sorted_ids.shape != mask.shape:
            raise ValueError("sorted_ids and mask must have the same shape")
        self._sorted_ids = sorted_ids
        self._mask = mask.astype(bool, copy=False)
        self._count = int(self._mask.sum())
        cache = position_cache if position_cache is not None else PositionCache()
        self._cache = cache.entries
        self._lock = cache.lock

    @classmethod
    def from_ids(cls, recipe_ids: list[int] | np.ndarray) -> "AllowedIds":
        ids = np.unique(np.asarray(recipe_ids, dtype=np.int64))
        return cls(ids, np.ones(ids.shape[0], dtype=bool))

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    def __len__(self) -> int:
        return self._count

    def __contains__(self, recipe_id: object) -> bool:
        try:
            rid = int(recipe_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        pos = int(np.searchsorted(self._sorted_ids, rid))
        return pos < self._sorted_ids.shape[0] and int(self._sorted_ids[pos]) == rid and bool(self._mask[pos])

    def ids(self) -> np.ndarray:
        return self._sorted_ids[self._mask]

    def mask_for(self, order_ids: np.ndarray) -> np.ndarray:
        """Boolean mask over ``order_ids`` marking the allowed entries."""
        key = id(order_ids)
        cached = self._cache.get(key)
        if cached is None or cached[0] is not order_ids:
            cached = self._positions(order_ids)
            with self._lock:
                self._cache[key] = cached
        _, positions, valid = cached
        if positions is None:
            return self._mask
        out = self._mask[positions]
        if valid is not None:
            out &= valid
        return out

    def _positions(self, order_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        universe = self._sorted_ids
        if order_ids.shape == universe.shape and np.array_equal(order_ids, universe):
            return order_ids, None, None
        positions = np.searchsorted(universe, order_ids)
        np.clip(positions, 0, max(universe.shape[0] - 1, 0), out=positions)
        if universe.shape[0] == 0:
            return order_ids, positions, np.zeros(order_ids.shape[0], dtype=bool)
        valid = universe[positions] == order_ids
        return order_ids, positions, None if valid.all() else valid
