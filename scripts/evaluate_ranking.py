"""Compare rankers on the held-out LTR queries (source recipes from md5 bucket 9).

The held-out queries come from the training pool but never from the queries
the model was fit or tuned on. The official eval pool (buckets 0-1) is not
touched here.

Rankings are produced by MLRecipeRanker.rank, the same call the search
service makes, on the candidates recorded by scripts/build_ltr_dataset.py.

Usage:
    python scripts/evaluate_ranking.py
    python scripts/evaluate_ranking.py --dataset data/ltr_heldout_k20.parquet --compare-model path/to/old_ranker.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import BASE_DIR, get_ranker_model_path
from app.backend.ranking.metrics import ndcg_at_k, summarize
from app.backend.ranking.ml_ranker import MLRecipeRanker
from scripts.build_ltr_dataset import QueryGroup, dataset_path, load_groups

DEFAULT_DATASETS = [dataset_path(BASE_DIR / "data", "heldout", k) for k in (100, 20)]
DEFAULT_REPORT = BASE_DIR / "data" / "ltr_eval_report.json"
SEED = 42


def ranked_labels(g: QueryGroup, method: str, ranker: MLRecipeRanker | None) -> list[int]:
    label_of = {r.id: lb for r, lb in zip(g.recipes, g.labels)}
    if method == "rrf":
        kept = [r for r in g.recipes if r.coverage >= g.threshold]
        kept.sort(key=lambda r: (r.retrieval_rank or 10**6, r.id))
    else:
        assert ranker is not None
        kept = ranker.rank(
            g.recipes, details_map=g.details_map, minimum_coverage=g.threshold,
            force_heuristic=(method == "heuristic"),
        )
    return [label_of[r.id] for r in kept]


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000) -> tuple[float, float, float]:
    """Mean of (a - b) and a 95% percentile interval."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(SEED)
    means = diff[rng.randint(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_dataset(path: Path, rankers: dict[str, MLRecipeRanker], tiers: dict[str, str]) -> dict:
    groups = load_groups(path)
    methods = ["rrf", "heuristic", *rankers]
    per_method: dict[str, list[list[int]]] = {m: [] for m in methods}
    reachable: list[bool] = []
    for g in groups:
        reachable.append(any(lb > 0 and r.coverage >= g.threshold for r, lb in zip(g.recipes, g.labels)))
        for m in methods:
            if m in ("rrf", "heuristic"):
                per_method[m].append(ranked_labels(g, m, rankers["lambdamart"]))
            else:
                per_method[m].append(ranked_labels(g, "ml", rankers[m]))

    out: dict = {"file": path.name, "n_queries": len(groups), "reachable_rate": float(np.mean(reachable))}
    out["end_to_end"] = {m: summarize(v) for m, v in per_method.items()}
    out["conditional"] = {m: summarize([x for x, ok in zip(v, reachable) if ok]) for m, v in per_method.items()}

    by_tier: dict[str, dict] = defaultdict(dict)
    for tier in sorted(set(tiers.get(g.qid, "?") for g in groups)):
        idx = [i for i, g in enumerate(groups) if tiers.get(g.qid) == tier and reachable[i]]
        for m, v in per_method.items():
            by_tier[tier][m] = summarize([v[i] for i in idx])
    out["conditional_by_tier"] = dict(by_tier)

    cond_idx = [i for i, ok in enumerate(reachable) if ok]
    nd = {m: [ndcg_at_k(per_method[m][i], 10) for i in cond_idx] for m in methods}
    out["ndcg10_diff_vs_heuristic"] = {
        m: dict(zip(("mean", "ci95_low", "ci95_high"), paired_bootstrap(nd[m], nd["heuristic"])))
        for m in methods if m != "heuristic"
    }
    return out


def print_table(title: str, metrics: dict[str, dict]) -> None:
    print(f"\n{title}")
    print("| Ranking | nDCG@10 | MRR@10 | Hit@1 | Hit@10 | queries |")
    print("|---|---|---|---|---|---|")
    for m, v in metrics.items():
        print(f"| {m} | {v['ndcg@10']:.4f} | {v['mrr@10']:.4f} | {v['hit@1']:.4f} | {v['hit@10']:.4f} | {v['queries']} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", default=None)
    parser.add_argument("--model", type=Path, default=None, help="defaults to PANTRYSENSE_RANKER_MODEL_PATH")
    parser.add_argument("--compare-model", type=Path, nargs="*", default=[], help="extra model files to compare")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    os.environ["PANTRYSENSE_RANKING_MODE"] = "ml"
    model_path = args.model or get_ranker_model_path()
    rankers: dict[str, MLRecipeRanker] = {"lambdamart": MLRecipeRanker(model_path)}
    for p in args.compare_model:
        rankers[f"model:{p.name}"] = MLRecipeRanker(p)
    for name, r in rankers.items():
        if not r.is_ready:
            raise SystemExit(f"cannot load {name} from {r.model_path}")

    queries_path = BASE_DIR / "data" / "ltr_queries.json"
    tiers = {}
    if queries_path.exists():
        tiers = {q["qid"]: q["tier"] for q in json.loads(queries_path.read_text(encoding="utf-8"))}

    datasets = args.dataset or [p for p in DEFAULT_DATASETS if p.exists()]
    if not datasets:
        raise SystemExit("no held-out dataset found; run scripts/build_ltr_dataset.py first")

    report = {"model": str(model_path), "datasets": []}
    for path in datasets:
        res = evaluate_dataset(path, rankers, tiers)
        report["datasets"].append(res)
        print(f"\n=== {res['file']}: {res['n_queries']} queries, relevant recipe reaches the ranker in "
              f"{res['reachable_rate']:.1%} ===")
        print_table("Conditional (relevant recipe among ranker inputs)", res["conditional"])
        print_table("End to end (all queries)", res["end_to_end"])
        for m, d in res["ndcg10_diff_vs_heuristic"].items():
            print(f"nDCG@10 {m} - heuristic: {d['mean']:+.4f} [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
