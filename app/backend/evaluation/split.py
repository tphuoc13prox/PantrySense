"""Fixed train/eval partition of the recipe corpus.

A recipe's bucket depends only on its id, so the split is stable across corpus
rebuilds and identical for every script that imports this module. Buckets 0-1
(~20%) are reserved for evaluation queries; buckets 2-9 are for training
(e.g. the LTR ranker). Never sample training queries from the eval pool.
"""
from __future__ import annotations

import hashlib

N_BUCKETS = 10
EVAL_BUCKETS = frozenset({0, 1})


def split_bucket(recipe_id: int) -> int:
    return int(hashlib.md5(str(recipe_id).encode()).hexdigest(), 16) % N_BUCKETS


def is_eval_recipe(recipe_id: int) -> bool:
    return split_bucket(recipe_id) in EVAL_BUCKETS


def is_train_recipe(recipe_id: int) -> bool:
    return not is_eval_recipe(recipe_id)
