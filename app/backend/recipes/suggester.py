from __future__ import annotations

import collections
import difflib
import json
import logging
import threading
from pathlib import Path
from typing import Any

from app.backend.config import get_database_path
from app.backend.database.connection import get_connection
from app.backend.recipes.matching import normalize_ingredient

logger = logging.getLogger(__name__)

# Fallback core culinary ingredient vocabulary with baseline frequencies
FALLBACK_INGREDIENTS: dict[str, int] = {
    "salt": 112000,
    "butter": 75000,
    "sugar": 61000,
    "onion": 52000,
    "olive oil": 42000,
    "flour": 35000,
    "milk": 34000,
    "garlic": 32000,
    "garlic cloves": 30000,
    "egg": 28000,
    "eggs": 27000,
    "pepper": 28000,
    "black pepper": 25000,
    "chicken": 22000,
    "chicken breast": 15000,
    "tomato": 20000,
    "tomatoes": 19000,
    "pork": 18000,
    "pork belly": 9000,
    "beef": 17000,
    "ground beef": 12000,
    "noodles": 14000,
    "pasta": 16000,
    "spaghetti": 11000,
    "parmesan cheese": 17000,
    "cheddar cheese": 15000,
    "mozzarella cheese": 14000,
    "cheese": 26000,
    "soy sauce": 18000,
    "green onion": 16000,
    "scallion": 12000,
    "rice": 16000,
    "shrimp": 11000,
    "potato": 14000,
    "potatoes": 13000,
    "carrot": 13000,
    "carrots": 12000,
    "bell pepper": 11000,
    "lemon juice": 18000,
    "vanilla": 18000,
    "baking powder": 22000,
    "baking soda": 18000,
    "honey": 10000,
    "cream": 9000,
    "heavy cream": 8000,
    "mushrooms": 9000,
    "spinach": 7000,
    "ginger": 8000,
    "cilantro": 6000,
    "sesame oil": 7000,
    "bread": 6000,
    "bacon": 9000,
    "tofu": 4000,
}


class IngredientSuggester:
    """In-memory controlled vocabulary suggester with 3-tier prefix, substring, and fuzzy matching."""

    def __init__(
        self,
        database_path: Path | None = None,
        vocab_path: Path | None = None,
        vocab_freq: dict[str, int] | None = None,
    ) -> None:
        from app.backend.config import get_ingredient_vocab_path
        self.database_path = database_path or get_database_path()
        self.vocab_path = vocab_path if vocab_path is not None else get_ingredient_vocab_path()
        self._lock = threading.Lock()
        # Prime immediately with fallback staples so queries at millisecond 0 never fail
        self._frequencies: dict[str, int] = dict(FALLBACK_INGREDIENTS)
        self._all_ingredients: list[str] = sorted(
            self._frequencies.keys(),
            key=lambda x: (-self._frequencies[x], len(x), x),
        )
        self._is_loaded = False

        if vocab_freq is not None:
            self._frequencies = dict(vocab_freq)
            self._all_ingredients = sorted(
                self._frequencies.keys(),
                key=lambda x: (-self._frequencies[x], len(x), x),
            )
            self._is_loaded = True
        elif self.vocab_path and self.vocab_path.exists():
            # Fast-path: Load pre-cached JSON vocabulary in < 2ms
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data:
                    self._frequencies = cached_data
                    self._all_ingredients = sorted(
                        self._frequencies.keys(),
                        key=lambda x: (-self._frequencies[x], len(x), x),
                    )
                    self._is_loaded = True
                    logger.debug("Fast-loaded %d ingredients from cached %s", len(self._frequencies), self.vocab_path)
            except Exception as e:
                logger.warning("Could not fast-load vocab cache from %s: %s", self.vocab_path, e)

    def ensure_loaded(self) -> None:
        if self._is_loaded:
            return
        with self._lock:
            if self._is_loaded:
                return
            self._load_vocabulary()
            self._is_loaded = True

    def _load_vocabulary(self) -> None:
        # Check if pre-cached JSON file exists
        if self.vocab_path and self.vocab_path.exists():
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data:
                    self._frequencies = cached_data
                    self._all_ingredients = sorted(
                        self._frequencies.keys(),
                        key=lambda x: (-self._frequencies[x], len(x), x),
                    )
                    logger.info("Loaded %d unique ingredients from cached %s", len(self._frequencies), self.vocab_path)
                    return
            except Exception as e:
                logger.warning("Could not read vocab from %s: %s", self.vocab_path, e)

        freq_counter: collections.Counter[str] = collections.Counter()
        db_file = self.database_path

        if db_file.exists():
            try:
                with get_connection(db_file) as conn:
                    rows = conn.execute("SELECT ingredients FROM recipes").fetchall()
                    for r in rows:
                        raw_list = json.loads(r["ingredients"] or "[]")
                        for item in raw_list:
                            clean = normalize_ingredient(str(item))
                            if clean and len(clean) > 1:
                                freq_counter[clean] += 1
                logger.info("Loaded %d unique ingredients from %s", len(freq_counter), db_file)
            except Exception as e:
                logger.warning("Could not read ingredients from DB (%s): %s", db_file, e)

        if not freq_counter:
            logger.info("Using fallback ingredients vocabulary (%d items).", len(FALLBACK_INGREDIENTS))
            for k, v in FALLBACK_INGREDIENTS.items():
                freq_counter[k] = v
        else:
            # Filter rare single-instance noise/misspellings in the raw dataset
            filtered_freq = {
                k: v for k, v in freq_counter.items()
                if v >= 2 or k in FALLBACK_INGREDIENTS
            }
            freq_counter = collections.Counter(filtered_freq)

        self._frequencies = dict(freq_counter)
        # Sort ingredients by frequency descending for fast candidate ranking
        self._all_ingredients = sorted(
            self._frequencies.keys(),
            key=lambda x: (-self._frequencies[x], len(x), x),
        )

        # Save to cached JSON file for instantaneous sub-millisecond startups next time
        if self.vocab_path and len(self._frequencies) > len(FALLBACK_INGREDIENTS):
            try:
                self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.vocab_path, "w", encoding="utf-8") as f:
                    json.dump(self._frequencies, f)
                logger.info("Cached ingredient vocabulary (%d items) to %s", len(self._frequencies), self.vocab_path)
            except Exception as e:
                logger.warning("Could not save vocab cache to %s: %s", self.vocab_path, e)

    def suggest(self, query: str, limit: int = 10) -> list[IngredientSuggestion]:
        """Finds autocomplete suggestions and typo corrections for a given query."""
        from app.backend.recipes.schemas import IngredientSuggestion

        self.ensure_loaded()
        cleaned_query = query.strip().lower()

        # If empty query, return top most frequent staple ingredients
        if not cleaned_query:
            top_staples = self._all_ingredients[:limit]
            return [
                IngredientSuggestion(
                    name=name,
                    frequency=self._frequencies.get(name, 0),
                    match_type="popular",
                    is_correction=False,
                    confidence=1.0,
                )
                for name in top_staples
            ]

        normalized_q = normalize_ingredient(cleaned_query)
        prefix_matches: list[str] = []
        substring_matches: list[str] = []
        seen_names: set[str] = set()

        # Tier 1: Exact prefix matches (e.g. 'gar' -> 'garlic', 'garlic powder')
        for item in self._all_ingredients:
            if item.startswith(cleaned_query) or (normalized_q and item.startswith(normalized_q)):
                if item not in seen_names:
                    seen_names.add(item)
                    prefix_matches.append(item)
                    if len(prefix_matches) >= limit:
                        break

        # Tier 2: Word-boundary prefix or substring matches (e.g. 'cheese' -> 'parmesan cheese', 'cheddar cheese')
        if len(prefix_matches) < limit:
            for item in self._all_ingredients:
                if item in seen_names:
                    continue
                words = item.split()
                if any(w.startswith(cleaned_query) for w in words) or cleaned_query in item:
                    seen_names.add(item)
                    substring_matches.append(item)
                    if len(prefix_matches) + len(substring_matches) >= limit:
                        break

        combined_candidates = prefix_matches + substring_matches
        results: list[IngredientSuggestion] = []

        for name in combined_candidates:
            is_pfx = name.startswith(cleaned_query) or (normalized_q and name.startswith(normalized_q))
            results.append(
                IngredientSuggestion(
                    name=name,
                    frequency=self._frequencies.get(name, 0),
                    match_type="prefix" if is_pfx else "contains",
                    is_correction=False,
                    confidence=1.0 if is_pfx else 0.85,
                )
            )

        # Tier 3: Fuzzy typo matching / spellcheck if results are scarce or exact match absent
        if len(results) < limit:
            cutoff = 0.65 if len(cleaned_query) >= 4 else 0.75
            fuzzy_scored: list[tuple[float, str]] = []

            for item in self._all_ingredients:
                if item in seen_names:
                    continue
                # Compare against whole item
                sim = difflib.SequenceMatcher(None, cleaned_query, item).ratio()
                # Compare against individual words in multi-word items (e.g. 'chicken breast')
                words = item.split()
                if len(words) > 1:
                    for w in words:
                        w_sim = difflib.SequenceMatcher(None, cleaned_query, w).ratio()
                        if w_sim > sim:
                            sim = w_sim

                if sim >= cutoff:
                    fuzzy_scored.append((sim, item))

            # Sort fuzzy candidates by similarity ratio descending, then frequency descending
            fuzzy_scored.sort(key=lambda x: (-x[0], -self._frequencies.get(x[1], 0)))

            for sim, match_name in fuzzy_scored:
                if match_name not in seen_names:
                    seen_names.add(match_name)
                    results.append(
                        IngredientSuggestion(
                            name=match_name,
                            frequency=self._frequencies.get(match_name, 0),
                            match_type="fuzzy",
                            is_correction=True,
                            confidence=round(sim, 2),
                        )
                    )
                    if len(results) >= limit:
                        break

        return results[:limit]


_SUGGESTER_INSTANCE: IngredientSuggester | None = None


def get_ingredient_suggester(
    db_path: Path | None = None,
    force_reload: bool = False,
) -> IngredientSuggester:
    global _SUGGESTER_INSTANCE
    if _SUGGESTER_INSTANCE is None or force_reload:
        _SUGGESTER_INSTANCE = IngredientSuggester(database_path=db_path)
    return _SUGGESTER_INSTANCE
