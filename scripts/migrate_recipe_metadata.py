"""Compute dietary tags, allergens and per-serving nutrition for every recipe.

Writes the ``recipe_metadata`` side table that search reads instead of
re-running the estimators per query. Re-run it after changing
``app/backend/recipes/dietary.py`` or ``nutrition.py``.

    python scripts/migrate_recipe_metadata.py --db data/recipes.db
    python scripts/migrate_recipe_metadata.py --db data/recipes.db --copy-from /path/to/original.db

``--copy-from`` copies a source database to ``--db`` first, so the migration
never touches the original file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.backend.config import get_database_path  # noqa: E402
from app.backend.database.recipe_metadata import populate_recipe_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=None, help="Database to migrate (default: PANTRYSENSE_DB_PATH)")
    parser.add_argument("--copy-from", type=Path, default=None, help="Copy this database to --db before migrating")
    parser.add_argument("--only-missing", action="store_true", help="Keep existing rows, fill the gaps")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    target = (args.db or get_database_path()).resolve()
    if args.copy_from is not None:
        source = args.copy_from.resolve()
        if source == target:
            parser.error("--copy-from and --db must be different files")
        free = shutil.disk_usage(target.parent).free
        if free < source.stat().st_size * 2:
            parser.error(f"Not enough free space in {target.parent} to copy {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {source} -> {target}")
        shutil.copy2(source, target)
    if not target.exists():
        parser.error(f"Database not found: {target}")

    start = time.perf_counter()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 2.0 or done == total:
            last[0] = now
            print(f"  {done:,}/{total:,} recipes ({done / max(total, 1):.0%})", flush=True)

    written = populate_recipe_metadata(
        target,
        only_missing=args.only_missing,
        workers=args.workers,
        batch_size=args.batch_size,
        progress=progress,
    )
    print(f"Wrote metadata for {written:,} recipes in {time.perf_counter() - start:.1f}s ({target})")


if __name__ == "__main__":
    main()
