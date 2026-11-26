"""Download the recipe dataset and build recipes.db with its precomputed metadata.

    python scripts/build_database.py               # 380,000-row limit, as the app's setup
    python scripts/build_database.py --limit 3000  # small corpus for trying things out

These are steps 1-3 of the in-app setup (schema, Hugging Face parquet, ingest,
dietary and nutrition metadata) plus the ingredient vocabulary used by
autocomplete, without building any index.

Recipe ids are assigned at insert time and the eval and training splits hash
them, so a corpus that matches the published numbers needs an empty database
and the default limit. An existing recipes.db is refused unless --replace is
given, which deletes it first. Afterwards the corpus is compared with the one
eval/queries_v1.jsonl was built from (the fingerprint in its .meta.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import get_database_path, get_dataset_download_limit, get_ingredient_vocab_path  # noqa: E402
from app.backend.evaluation.query_set import corpus_fingerprint  # noqa: E402
from app.backend.recipes.suggester import IngredientSuggester  # noqa: E402
from app.backend.system.setup_pipeline import ingest_dataset  # noqa: E402
from scripts.build_eval_set import load_corpus  # noqa: E402

EVAL_META = Path(__file__).resolve().parents[1] / "eval" / "queries_v1.meta.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_database")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=None, help="default: PANTRYSENSE_DB_PATH or <data dir>/recipes.db")
    parser.add_argument("--limit", type=int, default=get_dataset_download_limit(),
                        help="rows read from the dataset (default %(default)s)")
    parser.add_argument("--replace", action="store_true", help="delete an existing database first")
    args = parser.parse_args(argv)

    db_path = args.db or get_database_path()
    if db_path.exists():
        if not args.replace:
            print(f"{db_path} already exists. Pass --replace to rebuild it from scratch.", file=sys.stderr)
            return 1
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    # The vocabulary is a cache of the database's ingredients, so it goes too.
    get_ingredient_vocab_path().unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    total = ingest_dataset(db_path, args.limit)
    logger.info("Stored %d recipes in %s in %.0fs", total, db_path, time.perf_counter() - start)
    IngredientSuggester(database_path=db_path).ensure_loaded()  # writes the vocabulary cache
    check_corpus(db_path)
    return 0


def check_corpus(db_path: Path) -> bool:
    """Log whether ``db_path`` holds the corpus the official eval set was built from."""
    expected = json.loads(EVAL_META.read_text(encoding="utf-8"))["corpus"]
    corpus = load_corpus(db_path)
    if corpus_fingerprint(corpus) == expected["fingerprint_sha256"]:
        logger.info("Corpus matches the one eval/queries_v1.jsonl was built from (%d recipes).", len(corpus))
        return True
    logger.warning("Corpus differs from the one eval/queries_v1.jsonl was built from (%d vs %d recipes); "
                   "eval and training splits will not line up with the published numbers.",
                   len(corpus), expected["n_recipes"])
    return False


if __name__ == "__main__":
    sys.exit(main())
