"""Deterministic synthetic query set built from the recipe corpus itself.

Each query is a random subset of one source recipe's ingredients ("what is in
my pantry"), and the source recipe (plus its near-duplicates) is the only
labelled relevant answer. Generation is fully determined by the corpus and the
parameters below, so the committed JSONL can be regenerated bit-for-bit.

The same building blocks are meant to be reused for other offline work, e.g.
generating a disjoint *training* query set for the LTR ranker by passing the
eval set's recipe ids as ``exclude_recipe_ids``.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.backend.recipes.matching import normalize_ingredient, normalize_ingredients

TIERS = ("standard", "sparse", "noisy")

# Fraction of the recipe's non-staple ingredients used as the query, per tier.
TIER_FRACTIONS: dict[str, tuple[float, float]] = {
    "standard": (0.55, 0.70),
    "sparse": (0.40, 0.55),
    "noisy": (0.40, 0.70),
}

_TITLE_STRIP_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_TITLE_SUFFIX_RE = re.compile(r"(?:\s+(?:recipe|\d+|i{1,3}|iv|v))+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusRecipe:
    recipe_id: int
    title: str
    ingredients: tuple[str, ...]  # normalised, sorted, unique


@dataclass
class EvalQuery:
    qid: str
    recipe_id: int
    title: str
    relevant_ids: list[int]
    ingredients: list[str]
    tier: str
    distractor: str | None
    n_recipe_ingredients: int
    n_candidate_pool: int
    fraction: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass
class QuerySetConfig:
    n_queries: int = 1000
    seed: int = 20261124
    staple_df_threshold: float = 0.09
    min_pool_size: int = 3
    min_query_size: int = 2
    max_duplicate_group: int = 5
    distractor_min_df: int = 100
    exclude_recipe_ids: frozenset[int] = field(default_factory=frozenset)


def normalize_title(title: str) -> str:
    """Key used to detect near-duplicate titles (case, punctuation, '2', 'II', 'recipe')."""
    t = _TITLE_STRIP_RE.sub(" ", title.lower().replace("'s ", " ").replace("'", ""))
    t = _SPACE_RE.sub(" ", t).strip()
    t = _TITLE_SUFFIX_RE.sub("", t).strip()
    return t


def _tokens(ingredient: str) -> set[str]:
    return set(_TOKEN_RE.findall(ingredient))


def to_corpus_recipe(recipe_id: int, title: str, raw_ingredients: Sequence[str]) -> CorpusRecipe:
    return CorpusRecipe(int(recipe_id), title, tuple(normalize_ingredients(list(raw_ingredients))))


def document_frequency(recipes: Iterable[CorpusRecipe]) -> Counter[str]:
    df: Counter[str] = Counter()
    for r in recipes:
        df.update(set(r.ingredients))
    return df


def corpus_fingerprint(recipes: Sequence[CorpusRecipe]) -> str:
    """Stable hash of (id, title, ingredients) so a run can detect a different corpus."""
    h = hashlib.sha256()
    for r in sorted(recipes, key=lambda x: x.recipe_id):
        h.update(f"{r.recipe_id}\t{r.title}\t{'|'.join(r.ingredients)}\n".encode("utf-8"))
    return h.hexdigest()


def duplicate_groups(recipes: Sequence[CorpusRecipe]) -> dict[int, set[int]]:
    """recipe_id -> ids treated as equivalent (same normalised title or identical ingredient set)."""
    by_title: dict[str, set[int]] = defaultdict(set)
    by_ings: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for r in recipes:
        by_title[normalize_title(r.title)].add(r.recipe_id)
        if r.ingredients:
            by_ings[r.ingredients].add(r.recipe_id)
    groups: dict[int, set[int]] = {}
    for r in recipes:
        group = set(by_title[normalize_title(r.title)])
        if r.ingredients:
            group |= by_ings[r.ingredients]
        groups[r.recipe_id] = group
    return groups


def build_query_set(
    recipes: Sequence[CorpusRecipe],
    config: QuerySetConfig,
) -> tuple[list[EvalQuery], dict]:
    """Sample ``config.n_queries`` queries. Returns (queries, metadata)."""
    n_docs = len(recipes)
    if n_docs == 0:
        raise ValueError("empty corpus")
    df = document_frequency(recipes)
    staples = sorted(
        (ing for ing, c in df.items() if c / n_docs >= config.staple_df_threshold),
        key=lambda ing: (-df[ing], ing),
    )
    staple_set = set(staples)
    groups = duplicate_groups(recipes)

    def pool_for(r: CorpusRecipe) -> list[str]:
        # Keep only ingredients that survive re-normalisation unchanged, so the
        # query string the pipeline sees is exactly what we stored.
        return [i for i in r.ingredients if i not in staple_set and normalize_ingredient(i) == i]

    eligible = [
        r
        for r in sorted(recipes, key=lambda x: x.recipe_id)
        if r.recipe_id not in config.exclude_recipe_ids
        and len(pool_for(r)) >= config.min_pool_size
        and len(groups[r.recipe_id]) <= config.max_duplicate_group
    ]

    distractor_vocab = sorted(
        ing
        for ing, c in df.items()
        if c >= config.distractor_min_df and ing not in staple_set and normalize_ingredient(ing) == ing
    )

    rng = random.Random(config.seed)
    order = list(range(len(eligible)))
    rng.shuffle(order)

    queries: list[EvalQuery] = []
    used: set[int] = set()
    for idx in order:
        if len(queries) >= config.n_queries:
            break
        r = eligible[idx]
        group = groups[r.recipe_id]
        if group & used:
            continue  # never sample two members of the same duplicate group
        used |= group
        tier = TIERS[len(queries) % len(TIERS)]
        pool = pool_for(r)
        lo, hi = TIER_FRACTIONS[tier]
        frac = rng.uniform(lo, hi)
        k = min(len(pool), max(config.min_query_size, round(frac * len(pool))))
        chosen = rng.sample(pool, k)
        distractor = None
        if tier == "noisy":
            recipe_tokens = set().union(*(_tokens(i) for i in r.ingredients))
            for _ in range(1000):
                cand = rng.choice(distractor_vocab)
                if cand not in r.ingredients and not (_tokens(cand) & recipe_tokens):
                    distractor = cand
                    break
            if distractor is not None:
                chosen.append(distractor)
        rng.shuffle(chosen)
        queries.append(
            EvalQuery(
                qid=f"q{len(queries) + 1:04d}",
                recipe_id=r.recipe_id,
                title=r.title,
                relevant_ids=sorted(group),
                ingredients=chosen,
                tier=tier,
                distractor=distractor,
                n_recipe_ingredients=len(r.ingredients),
                n_candidate_pool=len(pool),
                fraction=round(frac, 4),
            )
        )

    if len(queries) < config.n_queries:
        raise ValueError(f"only {len(queries)} eligible queries, asked for {config.n_queries}")

    tier_counts = Counter(q.tier for q in queries)
    metadata = {
        "format_version": 1,
        "config": {
            **{k: v for k, v in asdict(config).items() if k != "exclude_recipe_ids"},
            "n_excluded_recipe_ids": len(config.exclude_recipe_ids),
            "tier_fractions": TIER_FRACTIONS,
        },
        "corpus": {
            "n_recipes": n_docs,
            "fingerprint_sha256": corpus_fingerprint(recipes),
            "n_eligible": len(eligible),
            "n_with_duplicates": sum(1 for g in groups.values() if len(g) > 1),
        },
        "staples": [{"ingredient": s, "df": df[s], "df_ratio": round(df[s] / n_docs, 4)} for s in staples],
        "stats": {
            "tiers": dict(sorted(tier_counts.items())),
            "mean_query_size": round(sum(len(q.ingredients) for q in queries) / len(queries), 3),
            "queries_with_duplicates": sum(1 for q in queries if len(q.relevant_ids) > 1),
            "noisy_without_distractor": sum(1 for q in queries if q.tier == "noisy" and q.distractor is None),
        },
    }
    return queries, metadata


def write_query_set(path: Path, queries: Sequence[EvalQuery], metadata: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            f.write(q.to_json() + "\n")
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return meta_path


def load_query_set(path: Path) -> list[EvalQuery]:
    with open(path, encoding="utf-8") as f:
        return [EvalQuery(**json.loads(line)) for line in f if line.strip()]
