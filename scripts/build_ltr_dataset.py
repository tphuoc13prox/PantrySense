"""Build a learning-to-rank dataset from real retriever candidates.

Each query is a random subset of one source recipe's non-staple ingredients
(optionally plus one unrelated "distractor" ingredient). The query is run
through RecipeSearchService.search in hybrid mode, and a recording ranker
captures exactly what MLRecipeRanker.rank would receive at serving time:
the candidate RecipeSummary list, details_map and the coverage threshold.

Labels: 2 for the source recipe, 1 for its near-duplicates (same normalised
title or identical ingredient set), 0 otherwise. Labels never look at the
features the model uses.

Only recipes from the training pool (md5 bucket 2-9) are used as sources.
Bucket 9 is held out for scripts/evaluate_ranking.py, buckets 2-8 are for
training and cross-validation. Buckets 0-1 are the official eval pool and are
never sampled; sources whose duplicate group touches them are skipped too.

Usage:
    python scripts/build_ltr_dataset.py --n-train 4000 --n-heldout 1000 --top-k 100
    python scripts/build_ltr_dataset.py --n-train 4000 --n-heldout 1000 --top-k 20 --splits heldout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import BASE_DIR, get_database_path
from app.backend.ranking.features import STAPLE_INGREDIENTS
from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes.matching import normalize_ingredient, normalize_ingredients
from app.backend.recipes.schemas import RecipeSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_ltr_dataset")

DEFAULT_OUT_DIR = BASE_DIR / "data"
EVAL_BUCKETS = frozenset({0, 1})
HELDOUT_BUCKETS = frozenset({9})
TRAIN_BUCKETS = frozenset(range(2, 9))

TIERS = ("standard", "sparse", "noisy")
TIER_FRACTIONS: dict[str, tuple[float, float]] = {
    "standard": (0.55, 0.70),
    "sparse": (0.40, 0.55),
    "noisy": (0.40, 0.70),
    "pantry": (0.40, 0.60),
}
# The "pantry" tier is not a subset of the source recipe: it adds 2-4 extra
# ingredients taken from other recipes that share an ingredient with the
# query, so the extras tend to show up among the candidates.
PANTRY_EXTRAS = (2, 4)
STAPLE_DF_THRESHOLD = 0.09
MIN_POOL_SIZE = 3
MIN_QUERY_SIZE = 2
MAX_DUPLICATE_GROUP = 5
DISTRACTOR_MIN_DF = 100

_TITLE_STRIP_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_TITLE_SUFFIX_RE = re.compile(r"(?:\s+(?:recipe|\d+|i{1,3}|iv|v))+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bucket_of(recipe_id: int) -> int:
    return int(hashlib.md5(str(recipe_id).encode()).hexdigest(), 16) % 10


def normalize_title(title: str) -> str:
    t = _TITLE_STRIP_RE.sub(" ", title.lower().replace("'s ", " ").replace("'", ""))
    t = _SPACE_RE.sub(" ", t).strip()
    return _TITLE_SUFFIX_RE.sub("", t).strip()


@dataclass
class LTRQuery:
    qid: str
    split: str
    tier: str
    recipe_id: int
    title: str
    relevant_ids: list[int]
    ingredients: list[str]
    distractor: str | None
    fraction: float
    extras: list[str] = field(default_factory=list)


def load_corpus(db_path: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT id, title, ingredients FROM recipes").fetchall()
    return [(int(i), t, tuple(normalize_ingredients(json.loads(ing or "[]")))) for i, t, ing in rows]


def generate_queries(
    corpus: list[tuple[int, str, tuple[str, ...]]],
    n_train: int,
    n_heldout: int,
    seed: int,
    tiers: Sequence[str] = TIERS,
    exclude_ids: frozenset[int] = frozenset(),
    qid_prefix: str = "",
    extra_min_df: int = DISTRACTOR_MIN_DF,
) -> tuple[list[LTRQuery], dict[str, Any]]:
    n_docs = len(corpus)
    df: Counter[str] = Counter()
    for _, _, ings in corpus:
        df.update(set(ings))
    staples = {ing for ing, c in df.items() if c / n_docs >= STAPLE_DF_THRESHOLD}
    if staples != set(STAPLE_INGREDIENTS):
        logger.warning(
            "Corpus staples differ from features.STAPLE_INGREDIENTS: +%s -%s",
            sorted(staples - STAPLE_INGREDIENTS),
            sorted(STAPLE_INGREDIENTS - staples),
        )

    by_title: dict[str, set[int]] = defaultdict(set)
    by_ings: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for rid, title, ings in corpus:
        by_title[normalize_title(title)].add(rid)
        if ings:
            by_ings[ings].add(rid)

    def group_of(rid: int, title: str, ings: tuple[str, ...]) -> set[int]:
        g = set(by_title[normalize_title(title)])
        if ings:
            g |= by_ings[ings]
        return g

    def pool_for(ings: tuple[str, ...]) -> list[str]:
        return [i for i in ings if i not in staples and normalize_ingredient(i) == i]

    distractor_vocab = sorted(
        ing for ing, c in df.items()
        if c >= DISTRACTOR_MIN_DF and ing not in staples and normalize_ingredient(ing) == ing
    )

    eligible: dict[str, list[tuple[int, str, tuple[str, ...], set[int]]]] = {"train": [], "heldout": []}
    skipped_eval_overlap = 0
    for rid, title, ings in sorted(corpus):
        b = bucket_of(rid)
        if b in EVAL_BUCKETS:
            continue
        group = group_of(rid, title, ings)
        if len(group) > MAX_DUPLICATE_GROUP or len(pool_for(ings)) < MIN_POOL_SIZE:
            continue
        if any(bucket_of(g) in EVAL_BUCKETS for g in group):
            skipped_eval_overlap += 1
            continue
        split = "heldout" if b in HELDOUT_BUCKETS else "train"
        eligible[split].append((rid, title, ings, group))

    inverted: dict[str, list[int]] = defaultdict(list)
    if "pantry" in tiers:
        for pos, (_, _, ings) in enumerate(corpus):
            for ing in ings:
                inverted[ing].append(pos)

    def token_set(values: Sequence[str]) -> set[str]:
        return set().union(*(set(_TOKEN_RE.findall(v)) for v in values)) if values else set()

    def pick_extras(chosen: list[str], ings: tuple[str, ...], n: int) -> list[str]:
        src_tokens = token_set(ings)
        extras: list[str] = []
        for _ in range(300):
            if len(extras) >= n:
                break
            other = corpus[rng.choice(inverted[rng.choice(chosen)])][2]
            if not other:
                continue
            cand = rng.choice(other)
            if (
                cand not in ings and cand not in extras and cand not in staples
                and df[cand] >= extra_min_df and normalize_ingredient(cand) == cand
                and not (set(_TOKEN_RE.findall(cand)) & src_tokens)
            ):
                extras.append(cand)
        return extras

    rng = random.Random(seed)
    used: set[int] = set(exclude_ids)
    queries: list[LTRQuery] = []
    for split, target in (("train", n_train), ("heldout", n_heldout)):
        pool_items = eligible[split]
        order = list(range(len(pool_items)))
        rng.shuffle(order)
        made = 0
        for idx in order:
            if made >= target:
                break
            rid, title, ings, group = pool_items[idx]
            if group & used:
                continue
            used |= group
            tier = tiers[made % len(tiers)]
            pool = pool_for(ings)
            lo, hi = TIER_FRACTIONS[tier]
            frac = rng.uniform(lo, hi)
            k = min(len(pool), max(MIN_QUERY_SIZE, round(frac * len(pool))))
            chosen = rng.sample(pool, k)
            distractor = None
            if tier == "noisy" and distractor_vocab:
                recipe_tokens = set().union(*(set(_TOKEN_RE.findall(i)) for i in ings))
                for _ in range(1000):
                    cand = rng.choice(distractor_vocab)
                    if cand not in ings and not (set(_TOKEN_RE.findall(cand)) & recipe_tokens):
                        distractor = cand
                        break
                if distractor is not None:
                    chosen.append(distractor)
            extras: list[str] = []
            if tier == "pantry":
                extras = pick_extras(chosen, ings, rng.randint(*PANTRY_EXTRAS))
                chosen.extend(extras)
            rng.shuffle(chosen)
            made += 1
            queries.append(
                LTRQuery(
                    qid=f"{split[0]}{qid_prefix}{made:05d}",
                    split=split,
                    tier=tier,
                    recipe_id=rid,
                    title=title,
                    relevant_ids=sorted(group),
                    ingredients=chosen,
                    distractor=distractor,
                    fraction=round(frac, 4),
                    extras=extras,
                )
            )
        if made < target:
            raise ValueError(f"only {made} eligible {split} queries, asked for {target}")

    meta = {
        "n_recipes": n_docs,
        "staples": sorted(staples, key=lambda s: -df[s]),
        "n_eligible": {k: len(v) for k, v in eligible.items()},
        "skipped_dup_group_touching_eval_pool": skipped_eval_overlap,
    }
    return queries, meta


class RecordingRanker(MLRecipeRanker):
    """Stands in for MLRecipeRanker inside RecipeSearchService and keeps its inputs."""

    def __init__(self) -> None:
        super().__init__(model_path=Path("__unused__"))
        self.last: tuple[list[RecipeSummary], dict[int, dict[str, Any]], float] | None = None

    def rank(self, recipes, details_map=None, minimum_coverage=0.34, force_heuristic=False):  # type: ignore[override]
        self.last = (list(recipes), dict(details_map or {}), float(minimum_coverage))
        return []


def run_retrieval(queries: list[LTRQuery], top_k: int) -> tuple[dict[str, list], dict[str, Any]]:
    os.environ["PANTRYSENSE_SEMANTIC_TOP_K"] = str(top_k)
    from app.backend.recipes.service import RecipeSearchService
    from app.backend.retrieval.hybrid import HybridRetriever

    recorder = RecordingRanker()
    service = RecipeSearchService(
        retrieval_mode="hybrid",
        ranking_mode="ml",
        ml_ranker=recorder,
        hybrid_retriever=HybridRetriever(),
    )

    cols: dict[str, list] = defaultdict(list)
    stats = Counter()
    tier_stats: dict[str, Counter] = defaultdict(Counter)
    orig_ranks: list[int] = []
    t0 = time.time()
    for n, q in enumerate(queries, start=1):
        recorder.last = None
        service.search(q.ingredients)
        recipes, details, threshold = recorder.last if recorder.last else ([], {}, 0.0)
        relevant = set(q.relevant_ids)
        labels = [2 if r.id == q.recipe_id else (1 if r.id in relevant else 0) for r in recipes]
        passes = [r.coverage >= threshold for r in recipes]

        ts = tier_stats[q.tier]
        for c in (stats, ts):
            c["queries"] += 1
            c["candidates"] += len(recipes)
            c["candidates_passing"] += sum(passes)
            c["orig_in_candidates"] += int(2 in labels)
            c["any_rel_in_candidates"] += int(any(labels))
            c["orig_passing"] += int(any(lb == 2 and p for lb, p in zip(labels, passes)))
            c["any_rel_passing"] += int(any(lb > 0 and p for lb, p in zip(labels, passes)))
        for r, lb in zip(recipes, labels):
            if lb == 2 and r.retrieval_rank is not None:
                orig_ranks.append(int(r.retrieval_rank))

        for r, lb, p in zip(recipes, labels, passes):
            d = details.get(r.id, {})
            cols["qid"].append(q.qid)
            cols["cand_id"].append(r.id)
            cols["label"].append(lb)
            cols["passes_threshold"].append(p)
            cols["threshold"].append(threshold)
            cols["title"].append(r.title)
            cols["semantic_score"].append(r.semantic_score)
            cols["bm25_score"].append(r.bm25_score)
            cols["rrf_score"].append(r.rrf_score)
            cols["retrieval_rank"].append(r.retrieval_rank)
            cols["coverage"].append(r.coverage)
            cols["matched_count"].append(r.matched_count)
            cols["required_count"].append(r.required_count)
            cols["matched_ingredients"].append(list(r.matched_ingredients))
            cols["missing_ingredients"].append(list(r.missing_ingredients))
            cols["cooking_time"].append(r.cooking_time)
            cols["detail_cooking_time"].append(d.get("cooking_time"))
            cols["detail_difficulty"].append(d.get("difficulty"))

        if n % 250 == 0 or n == len(queries):
            elapsed = time.time() - t0
            logger.info(
                "%d/%d queries (%.2fs/q), orig in candidates %.3f, orig passing threshold %.3f",
                n, len(queries), elapsed / n, stats["orig_in_candidates"] / n, stats["orig_passing"] / n,
            )

    def rates(c: Counter) -> dict[str, float]:
        nq = max(c["queries"], 1)
        return {
            "queries": c["queries"],
            "mean_candidates": round(c["candidates"] / nq, 2),
            "mean_candidates_passing_threshold": round(c["candidates_passing"] / nq, 2),
            "recall_orig_in_candidates": round(c["orig_in_candidates"] / nq, 4),
            "recall_any_relevant_in_candidates": round(c["any_rel_in_candidates"] / nq, 4),
            "recall_orig_after_threshold": round(c["orig_passing"] / nq, 4),
            "recall_any_relevant_after_threshold": round(c["any_rel_passing"] / nq, 4),
        }

    orig_ranks.sort()
    summary = {
        "overall": rates(stats),
        "by_tier": {t: rates(c) for t, c in sorted(tier_stats.items())},
        "orig_fused_rank_quantiles": {
            str(p): orig_ranks[min(len(orig_ranks) - 1, int(p * len(orig_ranks)))] for p in (0.25, 0.5, 0.75, 0.9)
        } if orig_ranks else {},
        "seconds": round(time.time() - t0, 1),
    }
    return cols, summary


def dataset_path(out_dir: Path, split: str, top_k: int, tag: str = "") -> Path:
    return out_dir / f"ltr_{split}{'_' + tag if tag else ''}_k{top_k}.parquet"


@dataclass
class QueryGroup:
    qid: str
    recipes: list[RecipeSummary]
    details_map: dict[int, dict[str, Any]]
    labels: list[int]
    passes: list[bool]
    threshold: float

    def passing(self) -> "QueryGroup":
        keep = [i for i, p in enumerate(self.passes) if p]
        return QueryGroup(
            self.qid,
            [self.recipes[i] for i in keep],
            self.details_map,
            [self.labels[i] for i in keep],
            [True] * len(keep),
            self.threshold,
        )


def load_groups(path: Path) -> list[QueryGroup]:
    """Rebuild per-query ranker inputs (RecipeSummary + details_map) from a dataset file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    groups: dict[str, QueryGroup] = {}
    order: list[str] = []
    for i, qid in enumerate(table["qid"]):
        g = groups.get(qid)
        if g is None:
            g = QueryGroup(qid, [], {}, [], [], float(table["threshold"][i]))
            groups[qid] = g
            order.append(qid)
        cid = int(table["cand_id"][i])
        g.recipes.append(
            RecipeSummary(
                id=cid,
                title=table["title"][i],
                matched_ingredients=table["matched_ingredients"][i],
                missing_ingredients=table["missing_ingredients"][i],
                matched_count=table["matched_count"][i],
                required_count=table["required_count"][i],
                coverage=table["coverage"][i],
                cooking_time=table["cooking_time"][i],
                semantic_score=table["semantic_score"][i],
                bm25_score=table["bm25_score"][i],
                rrf_score=table["rrf_score"][i],
                retrieval_rank=table["retrieval_rank"][i],
            )
        )
        g.details_map[cid] = {
            "cooking_time": table["detail_cooking_time"][i],
            "difficulty": table["detail_difficulty"][i],
        }
        g.labels.append(int(table["label"][i]))
        g.passes.append(bool(table["passes_threshold"][i]))
    return [groups[q] for q in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-heldout", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--splits", nargs="+", default=["train", "heldout"], choices=["train", "heldout"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=sorted(TIER_FRACTIONS))
    parser.add_argument("--tag", default="", help="suffix for output files and qids, e.g. 'pantry'")
    parser.add_argument("--exclude-queries", type=Path, nargs="*", default=[],
                        help="query files whose source recipes (and duplicates) must not be reused")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    db_path = get_database_path()
    logger.info("Loading corpus from %s", db_path)
    corpus = load_corpus(db_path)
    exclude: set[int] = set()
    for path in args.exclude_queries:
        for q in json.loads(path.read_text(encoding="utf-8")):
            exclude.update(q["relevant_ids"])
    queries, gen_meta = generate_queries(
        corpus, args.n_train, args.n_heldout, args.seed,
        tiers=tuple(args.tiers), exclude_ids=frozenset(exclude), qid_prefix=args.tag[:1],
    )
    gen_meta["n_excluded_recipe_ids"] = len(exclude)
    del corpus

    args.out_dir.mkdir(parents=True, exist_ok=True)
    queries_path = args.out_dir / f"ltr_queries{'_' + args.tag if args.tag else ''}.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in queries], f, ensure_ascii=False)

    for split in args.splits:
        split_queries = [q for q in queries if q.split == split]
        logger.info("Running hybrid retrieval for %d %s queries (top_k=%d)", len(split_queries), split, args.top_k)
        cols, summary = run_retrieval(split_queries, args.top_k)
        out = dataset_path(args.out_dir, split, args.top_k, args.tag)
        pq.write_table(pa.table(dict(cols)), out)
        tier_of = {q.qid: q.tier for q in split_queries}
        meta = {
            "file": out.name,
            "split": split,
            "source_buckets": sorted(HELDOUT_BUCKETS if split == "heldout" else TRAIN_BUCKETS),
            "bucket_rule": "int(md5(str(recipe_id)).hexdigest(), 16) % 10; buckets 0-1 never used",
            "top_k": args.top_k,
            "seed": args.seed,
            "n_queries": len(split_queries),
            "n_rows": len(cols["qid"]),
            "tiers": dict(Counter(tier_of.values())),
            "tier_fractions": TIER_FRACTIONS,
            "staple_df_threshold": STAPLE_DF_THRESHOLD,
            "labels": {"2": "source recipe", "1": "near-duplicate of source (normalised title or same ingredient set)", "0": "other"},
            "rows": "every candidate the ranker receives; passes_threshold marks those above the coverage threshold",
            "generation": gen_meta,
            "retrieval": summary,
        }
        meta_path = out.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote %s and %s", out, meta_path.name)
        logger.info("Retrieval summary: %s", json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
