"""Train the LambdaMART re-ranker on the dataset from scripts/build_ltr_dataset.py.

Only candidates above the coverage threshold are used (that is what the ranker
sees at serving time), and queries whose relevant recipe did not survive
retrieval + threshold are dropped from training because they carry no signal.

Model selection uses GroupKFold over queries with early stopping on nDCG@10.
The chosen config is refit on all training queries with the mean best
iteration count. Results are compared with the coverage heuristic and the
plain RRF order on the same out-of-fold queries.

Usage:
    python scripts/train_ranker.py
    python scripts/train_ranker.py --dataset data/ltr_train_k100.parquet --output data/ranker_model.joblib
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import BASE_DIR, get_ranker_model_path
from app.backend.ranking.features import FEATURE_GROUPS, LTR_FEATURE_NAMES, build_feature_matrix
from app.backend.ranking.metrics import summarize
from app.backend.ranking.ml_ranker import BUNDLE_FORMAT, heuristic_sort_key
from scripts.build_ltr_dataset import QueryGroup, dataset_path, load_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ranker")

SEED = 42
DEFAULT_DATASET = dataset_path(BASE_DIR / "data", "train", 100)
DEFAULT_REPORT = BASE_DIR / "data" / "ltr_train_report.json"

BASE_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "n_estimators": 2000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [20, 100],
}
# Best cell of GRID on ltr_train_k100 (5-fold CV nDCG@10 0.852); used with --no-grid.
BEST_KNOWN: dict[str, list[Any]] = {"num_leaves": [15], "learning_rate": [0.03], "min_child_samples": [20]}
EARLY_STOPPING_ROUNDS = 50


@dataclass
class Prepared:
    groups: list[QueryGroup]
    matrices: list[np.ndarray]
    n_total_queries: int

    def stack(self, idx: Sequence[int] | np.ndarray, cols: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = np.vstack([self.matrices[i][:, cols] for i in idx])
        y = np.concatenate([np.asarray(self.groups[i].labels, dtype=np.float32) for i in idx])
        return X, y, [len(self.groups[i].labels) for i in idx]


def prepare(groups: list[QueryGroup]) -> Prepared:
    kept: list[QueryGroup] = []
    for g in groups:
        p = g.passing()
        if len(p.labels) >= 2 and any(p.labels):
            kept.append(p)
    matrices = [build_feature_matrix(g.recipes, g.details_map) for g in kept]
    return Prepared(kept, matrices, len(groups))


def order_by_scores(g: QueryGroup, scores: Sequence[float] | np.ndarray) -> list[int]:
    """Labels in the order MLRecipeRanker.rank would return (score, then heuristic)."""
    idx = sorted(range(len(g.recipes)), key=lambda i: (-float(scores[i]), *heuristic_sort_key(g.recipes[i])))
    return [g.labels[i] for i in idx]


def order_heuristic(g: QueryGroup) -> list[int]:
    idx = sorted(range(len(g.recipes)), key=lambda i: heuristic_sort_key(g.recipes[i]))
    return [g.labels[i] for i in idx]


def order_rrf(g: QueryGroup) -> list[int]:
    idx = sorted(range(len(g.recipes)), key=lambda i: (g.recipes[i].retrieval_rank or 10**6, g.recipes[i].id))
    return [g.labels[i] for i in idx]


def group_kfold(n: int, n_folds: int, seed: int = SEED) -> list[np.ndarray]:
    """Each query is its own group, so GroupKFold reduces to a shuffled split of query indices."""
    perm = np.random.RandomState(seed).permutation(n)
    return [np.sort(perm[f::n_folds]) for f in range(n_folds)]


def _new_model(params: dict[str, Any]):
    from lightgbm import LGBMRanker

    return LGBMRanker(**params)


def cross_validate(
    data: Prepared,
    params: dict[str, Any],
    cols: list[int],
    n_folds: int,
) -> tuple[dict[str, float], list[int], list[list[int]]]:
    import lightgbm as lgb

    folds = group_kfold(len(data.groups), n_folds)
    oof_scores: dict[int, np.ndarray] = {}
    best_iters: list[int] = []
    for f, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        X_tr, y_tr, s_tr = data.stack(train_idx, cols)
        X_va, y_va, s_va = data.stack(val_idx, cols)
        model = _new_model(params)
        model.fit(
            X_tr, y_tr, group=s_tr,
            eval_set=[(X_va, y_va)], eval_group=[s_va], eval_at=[10],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best = int(model.best_iteration_ or params["n_estimators"])
        best_iters.append(best)
        pred = model.predict(X_va, num_iteration=best)
        offset = 0
        for qi, size in zip(val_idx, s_va, strict=True):
            oof_scores[int(qi)] = pred[offset: offset + size]
            offset += size
    ranked = [order_by_scores(g, oof_scores[i]) for i, g in enumerate(data.groups)]
    return summarize(ranked), best_iters, ranked


def fmt_row(name: str, m: dict[str, float]) -> str:
    return f"| {name} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} | {m['hit@1']:.4f} | {m['hit@10']:.4f} |"


def train_from_groups(
    groups: list[QueryGroup],
    output_path: Path,
    n_folds: int = 5,
    grid: dict[str, list[Any]] | None = None,
    ablation: bool = True,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    data = prepare(groups)
    if len(data.groups) < n_folds:
        raise ValueError(f"need at least {n_folds} usable queries, got {len(data.groups)}")
    logger.info(
        "%d/%d queries usable for training (%d rows, %d features)",
        len(data.groups), data.n_total_queries, sum(len(g.labels) for g in data.groups), len(LTR_FEATURE_NAMES),
    )
    all_cols = list(range(len(LTR_FEATURE_NAMES)))
    grid = GRID if grid is None else grid

    search: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], list[int], dict[str, float]] | None = None
    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, values, strict=True))
        params = {**BASE_PARAMS, **cfg}
        metrics, iters, _ = cross_validate(data, params, all_cols, n_folds)
        search.append({"params": cfg, "cv": metrics, "best_iterations": iters})
        logger.info("cv %s -> ndcg@10 %.4f (iters %s)", cfg, metrics["ndcg@10"], iters)
        if best is None or metrics["ndcg@10"] > best[0]:
            best = (metrics["ndcg@10"], cfg, iters, metrics)
    assert best is not None
    _, best_cfg, best_iters, cv_metrics = best
    best_params = {**BASE_PARAMS, **best_cfg}

    # Baselines on the same out-of-fold queries.
    heur = summarize([order_heuristic(g) for g in data.groups])
    rrf = summarize([order_rrf(g) for g in data.groups])

    ablations: dict[str, dict[str, float]] = {}
    if ablation:
        for name, feats in FEATURE_GROUPS.items():
            cols = [i for i, f in enumerate(LTR_FEATURE_NAMES) if f not in feats]
            m, _, _ = cross_validate(data, best_params, cols, n_folds)
            ablations[f"-{name}"] = m
            logger.info("ablation -%s -> ndcg@10 %.4f", name, m["ndcg@10"])
        legacy = ["semantic_score", "bm25_score", "rrf_score", "coverage", "matched_count",
                  "missing_count", "cooking_time", "difficulty_num", "required_count"]
        m, _, _ = cross_validate(data, best_params, [LTR_FEATURE_NAMES.index(f) for f in legacy], n_folds)
        ablations["legacy_9_features_only"] = m

    n_estimators = max(1, int(round(float(np.mean(best_iters)))))
    final_params = {**best_params, "n_estimators": n_estimators}
    X, y, sizes = data.stack(range(len(data.groups)), all_cols)
    model = _new_model(final_params)
    model.fit(X, y, group=sizes)
    gain = model.booster_.feature_importance(importance_type="gain")
    importance: list[dict[str, Any]] = sorted(
        ({"feature": f, "gain": float(g), "gain_share": float(g / gain.sum()) if gain.sum() else 0.0}
         for f, g in zip(LTR_FEATURE_NAMES, gain, strict=True)),
        key=lambda d: -d["gain"],
    )

    report = {
        "n_queries_total": data.n_total_queries,
        "n_queries_usable": len(data.groups),
        "n_rows": int(len(y)),
        "n_folds": n_folds,
        "seed": SEED,
        "best_params": best_cfg,
        "n_estimators": n_estimators,
        "cv_conditional": {"lambdamart": cv_metrics, "heuristic": heur, "rrf": rrf},
        "ablations": ablations,
        "feature_importance": importance,
        "grid_search": search,
        "seconds": round(time.time() - t0, 1),
        **(extra_meta or {}),
    }

    bundle = {
        "format": BUNDLE_FORMAT,
        "version": 3,
        "model": model,
        "feature_names": list(LTR_FEATURE_NAMES),
        "params": final_params,
        "cv": {"lambdamart": cv_metrics, "heuristic": heur, "rrf": rrf},
        **({"dataset": extra_meta.get("dataset")} if extra_meta else {}),
        "embedding_model": _embedding_model_of((extra_meta or {}).get("dataset")),
    }
    import joblib

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    logger.info("Saved ranker bundle to %s", output_path)

    print("\nOut-of-fold validation (queries whose relevant recipe reaches the ranker)")
    print("| Ranking | nDCG@10 | MRR@10 | Hit@1 | Hit@10 |")
    print("|---|---|---|---|---|")
    print(fmt_row("RRF order", rrf))
    print(fmt_row("Heuristic (coverage, matched)", heur))
    print(fmt_row("LambdaMART", cv_metrics))
    for name, m in ablations.items():
        print(fmt_row(f"LambdaMART {name}", m))
    print("\nTop features by gain:")
    for d in importance[:10]:
        print(f"  {d['feature']:<26} {d['gain_share']:.3f}")
    return report


def _embedding_model_of(datasets: list[dict[str, Any]] | None) -> str | None:
    """The embedding model behind the training candidates, if all datasets agree."""
    models = {(d.get("retrieval_config") or {}).get("embedding_model") for d in datasets or []}
    return models.pop() if len(models) == 1 else None


def train_ranker(
    output_path: Path | None = None,
    dataset: Path | Sequence[Path] | None = None,
    report_path: Path | None = None,
    n_folds: int = 5,
    grid: dict[str, list[Any]] | None = None,
    ablation: bool = True,
) -> Path:
    target = output_path or get_ranker_model_path()
    paths = [dataset] if isinstance(dataset, Path) else list(dataset or [DEFAULT_DATASET])
    groups: list[QueryGroup] = []
    ds_info = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/build_ltr_dataset.py first")
        groups.extend(load_groups(path))
        meta_path = path.with_suffix(".meta.json")
        ds_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ds_info.append({"file": path.name, "tiers": ds_meta.get("tiers"), "top_k": ds_meta.get("top_k"),
                        "retrieval": ds_meta.get("retrieval"),
                        "retrieval_config": ds_meta.get("retrieval_config")})
    report = train_from_groups(
        groups, target, n_folds=n_folds, grid=grid, ablation=ablation, extra_meta={"dataset": ds_info},
    )
    report_path = report_path or DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote training report to %s", report_path)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", default=[DEFAULT_DATASET])
    parser.add_argument("--output", type=Path, default=None, help="defaults to PANTRYSENSE_RANKER_MODEL_PATH")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-grid", action="store_true", help="skip the grid, use BEST_KNOWN params")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()
    grid = BEST_KNOWN if args.no_grid else None
    train_ranker(args.output, args.dataset, args.report, args.folds, grid, not args.no_ablation)


if __name__ == "__main__":
    main()
