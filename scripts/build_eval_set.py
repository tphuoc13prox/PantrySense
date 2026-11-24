"""Build the deterministic evaluation query set from the real recipe corpus.

Usage:
    python scripts/build_eval_set.py                       # 1000 queries -> eval/queries_v1.jsonl
    python scripts/build_eval_set.py --n 5000 --seed 7 \
        --exclude eval/queries_v1.jsonl --out eval/train_queries.jsonl   # disjoint set, e.g. for LTR

The database is opened read-only (SQLite URI ``mode=ro``); nothing is written
outside the output path. Point ``PANTRYSENSE_DB_PATH`` (or ``--db``) at the corpus.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import get_database_path
from app.backend.evaluation.query_set import (
    QuerySetConfig,
    build_query_set,
    load_query_set,
    to_corpus_recipe,
    write_query_set,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_eval_set")

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "eval" / "queries_v1.jsonl"


def load_corpus(db_path: Path):
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()
    return [to_corpus_recipe(rid, title, json.loads(ings or "[]")) for rid, title, ings in rows]


def main(argv: list[str] | None = None) -> int:
    defaults = QuerySetConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=None, help="recipes.db (default: PANTRYSENSE_DB_PATH)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=defaults.n_queries, help="number of queries")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--staple-df", type=float, default=defaults.staple_df_threshold,
                   help="ingredients in >= this fraction of recipes are never used in queries")
    p.add_argument("--min-pool", type=int, default=defaults.min_pool_size,
                   help="minimum non-staple ingredients a recipe needs to be sampled")
    p.add_argument("--max-dup-group", type=int, default=defaults.max_duplicate_group,
                   help="skip recipes with more near-duplicates than this")
    p.add_argument("--pool", choices=("eval", "train", "all"), default=defaults.pool,
                   help="which split buckets source recipes come from (eval = buckets 0-1)")
    p.add_argument("--exclude", type=Path, action="append", default=[],
                   help="existing query JSONL whose recipes (and duplicates) must not be reused")
    args = p.parse_args(argv)

    db_path = args.db or get_database_path()
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    exclude: set[int] = set()
    for path in args.exclude:
        for q in load_query_set(path):
            exclude.update(q.relevant_ids)

    t0 = time.perf_counter()
    corpus = load_corpus(db_path)
    logger.info("Loaded %d recipes from %s in %.1fs", len(corpus), db_path, time.perf_counter() - t0)

    config = QuerySetConfig(
        n_queries=args.n,
        seed=args.seed,
        staple_df_threshold=args.staple_df,
        min_pool_size=args.min_pool,
        max_duplicate_group=args.max_dup_group,
        pool=args.pool,
        exclude_recipe_ids=frozenset(exclude),
    )
    queries, metadata = build_query_set(corpus, config)
    metadata["excluded_sets"] = [p.as_posix() for p in args.exclude]
    meta_path = write_query_set(args.out, queries, metadata)

    logger.info("Wrote %d queries to %s (metadata: %s)", len(queries), args.out, meta_path)
    logger.info("Staples excluded from queries: %s", ", ".join(s["ingredient"] for s in metadata["staples"]))
    logger.info("Stats: %s", json.dumps(metadata["stats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
