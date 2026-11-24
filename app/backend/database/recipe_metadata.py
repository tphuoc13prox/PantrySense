"""Precomputed per-recipe metadata: dietary tags, allergens and nutrition.

Values are produced by the public functions in ``recipes.dietary`` and
``recipes.nutrition`` and stored in the ``recipe_metadata`` side table, so the
search path reads them instead of re-running the keyword regexes for every
candidate. Re-run ``scripts/migrate_recipe_metadata.py`` whenever those
modules change; ``metadata_fingerprint()`` is stored alongside the rows so a
stale table can be detected.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from app.backend.database.connection import get_connection
from app.backend.recipes import dietary, nutrition
from app.backend.retrieval.filtering import AllowedIds, PositionCache

logger = logging.getLogger(__name__)

DIETARY_FLAGS = ("vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free", "keto_low_carb")
NUTRIENTS = ("calories", "protein", "carbs", "fat", "fiber")
FINGERPRINT_KEY = "metadata_fingerprint"

# Databases with at most this many recipes lacking metadata get it filled in
# by initialize_database(); larger gaps need the migration script.
INLINE_BACKFILL_LIMIT = 5000

_FLAG_COLUMNS = ",\n    ".join(f"{flag} INTEGER NOT NULL DEFAULT 0" for flag in DIETARY_FLAGS)
_NUTRIENT_COLUMNS = ",\n    ".join(f"{name} REAL NOT NULL DEFAULT 0" for name in NUTRIENTS)

CREATE_METADATA_SQL = f"""
CREATE TABLE IF NOT EXISTS recipe_metadata (
    recipe_id INTEGER PRIMARY KEY,
    {_FLAG_COLUMNS},
    dietary_tags TEXT NOT NULL DEFAULT '{{}}',
    allergens TEXT NOT NULL DEFAULT '[]',
    servings INTEGER NOT NULL DEFAULT 4,
    {_NUTRIENT_COLUMNS},
    nutrition_total TEXT NOT NULL DEFAULT '{{}}'
)
"""

CREATE_INFO_SQL = """
CREATE TABLE IF NOT EXISTS recipe_metadata_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_INSERT_COLUMNS = (
    ("recipe_id",) + DIETARY_FLAGS + ("dietary_tags", "allergens", "servings") + NUTRIENTS + ("nutrition_total",)
)
INSERT_METADATA_SQL = (
    f"INSERT OR REPLACE INTO recipe_metadata ({', '.join(_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _INSERT_COLUMNS)})"
)

# Columns selected (with table alias ``m``) wherever search needs stored metadata.
METADATA_SELECT_COLUMNS = ", ".join(
    ["m.recipe_id AS meta_recipe_id", "m.dietary_tags AS meta_dietary_tags", "m.allergens AS meta_allergens",
     "m.servings AS meta_servings", "m.nutrition_total AS meta_nutrition_total"]
    + [f"m.{name} AS meta_{name}" for name in NUTRIENTS]
)


@dataclass
class RecipeMetadata:
    dietary_tags: dict[str, bool]
    allergens: list[str]
    servings: int
    per_serving: dict[str, float]
    total: dict[str, float]
    breakdown: list[dict[str, Any]] = field(default_factory=list)

    def nutrition_dict(self, include_breakdown: bool = False) -> dict[str, Any]:
        return {
            "servings": self.servings,
            "per_serving": dict(self.per_serving),
            "total": dict(self.total),
            "breakdown": list(self.breakdown) if include_breakdown else [],
        }


def compute_recipe_metadata(ingredients: list[str], servings: int | None) -> RecipeMetadata:
    """Run the dietary, allergen and nutrition estimators for one recipe."""
    tags = dietary.classify_recipe_dietary(ingredients)
    allergens = dietary.detect_recipe_allergens(ingredients)
    nut = nutrition.calculate_recipe_nutrition(ingredients, servings=servings)
    return RecipeMetadata(
        dietary_tags={str(k): bool(v) for k, v in tags.items()},
        allergens=[str(a) for a in allergens],
        servings=int(nut.get("servings") or 4),
        per_serving={k: float(nut.get("per_serving", {}).get(k, 0.0)) for k in NUTRIENTS},
        total={k: float(nut.get("total", {}).get(k, 0.0)) for k in NUTRIENTS},
        breakdown=list(nut.get("breakdown", [])),
    )


def metadata_from_row(row: sqlite3.Row) -> RecipeMetadata | None:
    """Build metadata from a row selected with METADATA_SELECT_COLUMNS, or None if absent."""
    if row["meta_recipe_id"] is None:
        return None
    return RecipeMetadata(
        dietary_tags=json.loads(row["meta_dietary_tags"] or "{}"),
        allergens=json.loads(row["meta_allergens"] or "[]"),
        servings=int(row["meta_servings"] or 4),
        per_serving={k: float(row[f"meta_{k}"] or 0.0) for k in NUTRIENTS},
        total=json.loads(row["meta_nutrition_total"] or "{}"),
    )


def metadata_fingerprint() -> str:
    """Hash of the estimator sources; changes whenever their logic changes."""
    digest = hashlib.sha256()
    for module in (dietary, nutrition):
        try:
            digest.update(inspect.getsource(module).encode("utf-8"))
        except (OSError, TypeError):
            digest.update(module.__name__.encode("utf-8"))
    return digest.hexdigest()[:16]


def ensure_metadata_tables(connection: sqlite3.Connection) -> None:
    connection.execute(CREATE_METADATA_SQL)
    connection.execute(CREATE_INFO_SQL)


def metadata_record(recipe_id: int, meta: RecipeMetadata) -> tuple[Any, ...]:
    return (
        (recipe_id,)
        + tuple(int(bool(meta.dietary_tags.get(flag, False))) for flag in DIETARY_FLAGS)
        + (json.dumps(meta.dietary_tags, sort_keys=True), json.dumps(meta.allergens), meta.servings)
        + tuple(meta.per_serving.get(k, 0.0) for k in NUTRIENTS)
        + (json.dumps(meta.total, sort_keys=True),)
    )


def _compute_batch(rows: list[tuple[int, str, int | None]]) -> list[tuple[Any, ...]]:
    out = []
    for recipe_id, ingredients_json, servings in rows:
        ingredients = json.loads(ingredients_json or "[]")
        out.append(metadata_record(recipe_id, compute_recipe_metadata(ingredients, servings)))
    return out


def _iter_batches(
    connection: sqlite3.Connection, only_missing: bool, batch_size: int
) -> Iterator[list[tuple[int, str, int | None]]]:
    sql = "SELECT r.id, r.ingredients, r.servings FROM recipes r"
    if only_missing:
        sql += " LEFT JOIN recipe_metadata m ON m.recipe_id = r.id WHERE m.recipe_id IS NULL"
    sql += " ORDER BY r.id"
    cursor = connection.execute(sql)
    while True:
        chunk = cursor.fetchmany(batch_size)
        if not chunk:
            return
        yield [(int(r[0]), r[1], r[2]) for r in chunk]


def populate_recipe_metadata(
    database_path: Path | None = None,
    only_missing: bool = False,
    workers: int = 1,
    batch_size: int = 2000,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Compute and store metadata for every recipe (or only those missing it).

    Returns the number of rows written.
    """
    with get_connection(database_path) as conn:
        ensure_metadata_tables(conn)
        if only_missing:
            total = conn.execute(
                "SELECT COUNT(*) FROM recipes r LEFT JOIN recipe_metadata m ON m.recipe_id = r.id "
                "WHERE m.recipe_id IS NULL"
            ).fetchone()[0]
        else:
            total = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
            conn.execute("DELETE FROM recipe_metadata")
        conn.commit()

    if total == 0:
        _store_fingerprint(database_path)
        return 0

    written = 0
    read_conn = get_connection(database_path)
    write_conn = get_connection(database_path)
    try:
        batches: Iterable[list[tuple[int, str, int | None]]] = _iter_batches(read_conn, only_missing, batch_size)
        if only_missing:
            # Rows are inserted while the cursor is open; materialize ids first.
            batches = list(batches)

        def _write(records: list[tuple[Any, ...]]) -> None:
            nonlocal written
            write_conn.executemany(INSERT_METADATA_SQL, records)
            write_conn.commit()
            written += len(records)
            if progress:
                progress(written, total)

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for records in pool.map(_compute_batch, batches):
                    _write(records)
        else:
            for batch in batches:
                _write(_compute_batch(batch))
    finally:
        read_conn.close()
        write_conn.close()

    _store_fingerprint(database_path)
    return written


def _store_fingerprint(database_path: Path | None) -> None:
    with get_connection(database_path) as conn:
        ensure_metadata_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO recipe_metadata_info (key, value) VALUES (?, ?)",
            (FINGERPRINT_KEY, metadata_fingerprint()),
        )
        conn.commit()


def stored_fingerprint(connection: sqlite3.Connection) -> str | None:
    try:
        row = connection.execute(
            "SELECT value FROM recipe_metadata_info WHERE key = ?", (FINGERPRINT_KEY,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else None


def backfill_small_gaps(database_path: Path | None = None) -> int:
    """Fill missing metadata when only a few rows lack it (seed data, tests)."""
    with get_connection(database_path) as conn:
        ensure_metadata_tables(conn)
        missing = conn.execute(
            "SELECT COUNT(*) FROM recipes r LEFT JOIN recipe_metadata m ON m.recipe_id = r.id "
            "WHERE m.recipe_id IS NULL"
        ).fetchone()[0]
        stale = stored_fingerprint(conn) not in (None, metadata_fingerprint())
        total = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
    if stale and total <= INLINE_BACKFILL_LIMIT:
        return populate_recipe_metadata(database_path, only_missing=False)
    if missing == 0:
        if stale:
            logger.warning(
                "recipe_metadata was computed by an older version of the dietary/nutrition "
                "estimators. Run 'python scripts/migrate_recipe_metadata.py' to refresh it."
            )
        return 0
    if missing <= INLINE_BACKFILL_LIMIT:
        return populate_recipe_metadata(database_path, only_missing=True)
    logger.warning(
        "%d recipes have no precomputed metadata. Run 'python scripts/migrate_recipe_metadata.py'; "
        "until then filtered searches compute it in memory on first use.",
        missing,
    )
    return 0


# ---------------------------------------------------------------------------
# In-memory filter index
# ---------------------------------------------------------------------------

class MetadataIndex:
    """Column arrays of filterable metadata for every recipe, sorted by id.

    Loaded once per database on the first filtered search (roughly a second
    for 315k recipes); each query then turns its filters into a boolean mask
    with a handful of vectorized comparisons.
    """

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path
        self._lock = threading.Lock()
        self._loaded = False
        self.ids = np.empty(0, dtype=np.int64)
        self.flags: dict[str, np.ndarray] = {}
        self.cooking_time = np.empty(0, dtype=np.int32)
        self.category_codes = np.empty(0, dtype=np.int32)
        self.categories: list[str] = []
        self.allergen_bits = np.empty(0, dtype=np.int64)
        self.allergen_names: list[str] = []
        self.missing_rows = 0
        self._positions = PositionCache()

    @property
    def size(self) -> int:
        self.ensure_loaded()
        return int(self.ids.shape[0])

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True

    def _load(self) -> None:
        flag_cols = ", ".join(f"m.{flag}" for flag in DIETARY_FLAGS)
        with get_connection(self.database_path) as conn:
            ensure_metadata_tables(conn)
            fp = stored_fingerprint(conn)
            if fp is not None and fp != metadata_fingerprint():
                logger.warning(
                    "recipe_metadata is stale (estimators changed). "
                    "Run 'python scripts/migrate_recipe_metadata.py' to refresh it."
                )
            rows = conn.execute(
                f"""
                SELECT r.id, r.cooking_time, r.category, r.ingredients, r.servings,
                       m.recipe_id, {flag_cols}, m.allergens
                FROM recipes r LEFT JOIN recipe_metadata m ON m.recipe_id = r.id
                ORDER BY r.id
                """
            ).fetchall()

        n = len(rows)
        ids = np.empty(n, dtype=np.int64)
        cooking = np.full(n, -1, dtype=np.int32)
        cat_codes = np.full(n, -1, dtype=np.int32)
        flags = {flag: np.zeros(n, dtype=bool) for flag in DIETARY_FLAGS}
        bits = np.zeros(n, dtype=np.int64)
        cat_lookup: dict[str, int] = {}
        allergen_lookup: dict[str, int] = {}
        allergen_cache: dict[str, int] = {}
        missing = 0

        def allergen_mask(values: list[str]) -> int:
            mask = 0
            for name in values:
                key = str(name).lower().strip()
                if key not in allergen_lookup:
                    if len(allergen_lookup) >= 63:
                        raise ValueError("Too many distinct allergen labels for a bitmask")
                    allergen_lookup[key] = len(allergen_lookup)
                mask |= 1 << allergen_lookup[key]
            return mask

        for i, row in enumerate(rows):
            ids[i] = row[0]
            if row[1] is not None:
                cooking[i] = int(row[1])
            category = row[2]
            if category:
                code = cat_lookup.get(category)
                if code is None:
                    code = cat_lookup[category] = len(cat_lookup)
                cat_codes[i] = code

            if row[5] is None:
                missing += 1
                meta = compute_recipe_metadata(json.loads(row[3] or "[]"), row[4])
                for flag in DIETARY_FLAGS:
                    flags[flag][i] = bool(meta.dietary_tags.get(flag, False))
                bits[i] = allergen_mask(meta.allergens)
                continue

            for j, flag in enumerate(DIETARY_FLAGS):
                flags[flag][i] = bool(row[6 + j])
            raw_allergens = row[6 + len(DIETARY_FLAGS)] or "[]"
            cached = allergen_cache.get(raw_allergens)
            if cached is None:
                cached = allergen_cache[raw_allergens] = allergen_mask(json.loads(raw_allergens))
            bits[i] = cached

        if missing:
            logger.warning(
                "Computed metadata in memory for %d recipes without stored rows; "
                "run 'python scripts/migrate_recipe_metadata.py' to persist it.",
                missing,
            )

        self.ids = ids
        self.cooking_time = cooking
        self.category_codes = cat_codes
        self.categories = [name for name, _ in sorted(cat_lookup.items(), key=lambda kv: kv[1])]
        self.flags = flags
        self.allergen_bits = bits
        self.allergen_names = [name for name, _ in sorted(allergen_lookup.items(), key=lambda kv: kv[1])]
        self.missing_rows = missing
        self._positions = PositionCache()

    def mask(
        self,
        filters: Any | None = None,
        max_cooking_time: int | None = None,
        category: str | None = None,
    ) -> np.ndarray | None:
        """Boolean mask over ``self.ids``, or None when nothing is filtered.

        Mirrors the semantics of the post-filter it replaces: a missing cooking
        time passes a time limit, the category test is a case-insensitive
        substring match, and filter-level values override the arguments.
        """
        eff_time = max_cooking_time
        eff_cat = category
        wanted_flags: list[str] = []
        excluded: list[str] = []
        if filters is not None:
            wanted_flags = [flag for flag in DIETARY_FLAGS if getattr(filters, flag, False)]
            excluded = [a.lower().strip() for a in (getattr(filters, "excluded_allergens", None) or []) if a]
            if getattr(filters, "max_cooking_time", None) is not None:
                eff_time = filters.max_cooking_time
            if getattr(filters, "category", None) is not None:
                eff_cat = filters.category

        req_cat = eff_cat.strip().lower() if eff_cat and eff_cat.strip() else None
        if not wanted_flags and not excluded and eff_time is None and req_cat is None:
            return None

        self.ensure_loaded()
        mask = np.ones(self.ids.shape[0], dtype=bool)
        for flag in wanted_flags:
            mask &= self.flags[flag]
        if excluded:
            lookup = {name: bit for bit, name in enumerate(self.allergen_names)}
            excl_bits = 0
            for name in excluded:
                if name in lookup:
                    excl_bits |= 1 << lookup[name]
            if excl_bits:
                mask &= (self.allergen_bits & excl_bits) == 0
        if eff_time is not None:
            mask &= (self.cooking_time < 0) | (self.cooking_time <= int(eff_time))
        if req_cat is not None:
            codes = [code for code, name in enumerate(self.categories) if req_cat in name.lower()]
            mask &= np.isin(self.category_codes, np.asarray(codes, dtype=np.int32))
        return mask

    def allowed(
        self,
        filters: Any | None = None,
        max_cooking_time: int | None = None,
        category: str | None = None,
    ) -> AllowedIds | None:
        mask = self.mask(filters, max_cooking_time, category)
        if mask is None:
            return None
        return AllowedIds(self.ids, mask, position_cache=self._positions)
