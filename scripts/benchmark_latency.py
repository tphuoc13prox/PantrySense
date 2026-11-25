"""Latency and load-cost benchmark for RecipeSearchService.

Usage:
    python scripts/benchmark_latency.py search --queries 200 --out bench.json
    python scripts/benchmark_latency.py load
    python scripts/benchmark_latency.py ann --queries 500

Index and database paths come from the usual PANTRYSENSE_* environment
variables; the CLI flags override them. Queries are sampled from recipes
outside the evaluation buckets (0-1) so nothing here is tuned on the eval pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

EVAL_BUCKETS = {0, 1}


def _bucket(recipe_id: int) -> int:
    return int(hashlib.md5(str(recipe_id).encode()).hexdigest(), 16) % 10


def _apply_path_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "db": "PANTRYSENSE_DB_PATH",
        "vector_index": "PANTRYSENSE_VECTOR_INDEX_PATH",
        "vector_ids": "PANTRYSENSE_VECTOR_IDS_PATH",
        "bm25": "PANTRYSENSE_BM25_INDEX_PATH",
        "ranker": "PANTRYSENSE_RANKER_MODEL_PATH",
        "engine": "PANTRYSENSE_EMBEDDER_ENGINE",
    }
    for attr, env in mapping.items():
        value = getattr(args, attr, None)
        if value:
            os.environ[env] = str(value)
    os.environ.setdefault("PANTRYSENSE_AUTO_BUILD_INDEX", "false")
    os.environ.setdefault("PANTRYSENSE_AUTO_OPEN_BROWSER", "false")
    os.environ.setdefault("PANTRYSENSE_AUTO_SHUTDOWN", "false")


def _rss_mb() -> float:
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        import faiss

        info["faiss"] = faiss.__version__
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    from app.backend.config import get_embedder_engine

    info["embedder_engine"] = get_embedder_engine()
    return info


def sample_queries(n: int, seed: int, min_size: int = 2, max_size: int = 5) -> list[list[str]]:
    """Sample ingredient subsets from random non-eval recipes."""
    from app.backend.database.connection import get_connection

    rng = random.Random(seed)
    with get_connection() as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM recipes")]
        pool = [i for i in ids if _bucket(i) not in EVAL_BUCKETS]
        chosen = rng.sample(pool, min(len(pool), n * 2))
        placeholders = ",".join("?" for _ in chosen)
        rows = conn.execute(
            f"SELECT id, ingredients FROM recipes WHERE id IN ({placeholders})", chosen
        ).fetchall()

    by_id = {row["id"]: json.loads(row["ingredients"] or "[]") for row in rows}
    queries: list[list[str]] = []
    for recipe_id in chosen:
        ings = [i for i in by_id.get(recipe_id, []) if i]
        if len(ings) < min_size:
            continue
        size = rng.randint(min_size, min(max_size, len(ings)))
        queries.append(rng.sample(ings, size))
        if len(queries) >= n:
            break
    return queries


# ---------------------------------------------------------------------------
# Load phase: each component in its own process so RSS deltas are clean.
# ---------------------------------------------------------------------------

def _load_component(name: str, queue: Any) -> None:
    try:
        from app.backend.config import (
            display_path,
            get_bm25_index_path,
            get_ranker_model_path,
            get_vector_ids_path,
            get_vector_index_path,
        )

        base = _rss_mb()
        start = time.perf_counter()
        extra: dict[str, Any] = {}
        if name == "bm25":
            from app.backend.retrieval.bm25_store import BM25Store

            store = BM25Store()
            ok = store.load(get_bm25_index_path())
            extra["docs"] = store.total_docs
            extra["path"] = display_path(get_bm25_index_path())
        elif name == "faiss":
            from app.backend.retrieval.vector_store import FaissVectorStore

            store = FaissVectorStore()
            ok = store.load(get_vector_index_path(), get_vector_ids_path())
            extra["vectors"] = store.total_vectors
            index = getattr(store, "_index", None)
            extra["index_type"] = type(index).__name__ if index is not None else None
        elif name == "embedder":
            from app.backend.retrieval.embedder import IngredientEmbedder

            embedder = IngredientEmbedder()
            embedder.embed_text("chicken garlic onion")
            ok = True
            extra["engine"] = embedder.engine
        elif name == "ranker":
            from app.backend.ranking.ml_ranker import MLRecipeRanker

            ok = MLRecipeRanker(model_path=get_ranker_model_path()).load_model()
        else:
            raise ValueError(name)
        elapsed = time.perf_counter() - start
        queue.put({"component": name, "ok": bool(ok), "seconds": round(elapsed, 3),
                   "rss_delta_mb": round(_rss_mb() - base, 1), **extra})
    except Exception as exc:  # pragma: no cover - diagnostic path
        queue.put({"component": name, "ok": False, "error": repr(exc)})


def run_load(args: argparse.Namespace) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    results = []
    for name in args.components.split(","):
        queue = ctx.Queue()
        proc = ctx.Process(target=_load_component, args=(name.strip(), queue))
        proc.start()
        results.append(queue.get())
        proc.join()
    return results


# ---------------------------------------------------------------------------
# Search phase
# ---------------------------------------------------------------------------

def _make_service(mode: str, ranker_threads: int | None = None) -> Any:
    from app.backend.recipes import service as service_module

    factory = getattr(service_module, "create_search_service", None)
    if factory is not None:
        service = factory(retrieval_mode=mode)
    else:
        service = service_module.RecipeSearchService(retrieval_mode=mode)
    if ranker_threads:
        # Experiment only: cap LightGBM's OpenMP threads for the tiny
        # per-query prediction. The ranker module itself is unchanged.
        ml = service.ml_ranker
        if ml.load_model() and hasattr(ml._model, "set_params"):
            ml._model.set_params(n_jobs=ranker_threads)
    return service


def _time_mode(service: Any, queries: list[list[str]], filters: Any, warmup: int) -> dict[str, Any]:
    for q in queries[:warmup]:
        service.search(q, filters=filters)
    latencies: list[float] = []
    result_counts: list[int] = []
    for q in queries:
        start = time.perf_counter()
        results = service.search(q, filters=filters)
        latencies.append((time.perf_counter() - start) * 1000.0)
        result_counts.append(len(results))
    return {
        "n": len(latencies),
        "p50_ms": round(_percentile(latencies, 50), 1),
        "p95_ms": round(_percentile(latencies, 95), 1),
        "mean_ms": round(statistics.fmean(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "mean_results": round(statistics.fmean(result_counts), 2),
        "zero_result_rate": round(sum(1 for c in result_counts if c == 0) / len(result_counts), 3),
    }


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    from app.backend.recipes.schemas import DietaryFilters

    queries = sample_queries(args.queries, args.seed)
    filter_sets: dict[str, Any] = {"none": None}
    if args.filters:
        for name in args.filters.split(","):
            name = name.strip()
            if name:
                filter_sets[name] = DietaryFilters(**{name: True})

    report: dict[str, Any] = {"queries": len(queries), "seed": args.seed, "ranker_threads": args.ranker_threads,
                              "modes": {}}
    for mode in args.modes.split(","):
        mode = mode.strip()
        rss_before = _rss_mb()
        start = time.perf_counter()
        try:
            service = _make_service(mode, args.ranker_threads)
            # The first query pays for lazy index loading.
            service.search(queries[0])
        except Exception as exc:
            report["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[{mode:8s}] failed: {exc!r}", flush=True)
            continue
        cold = time.perf_counter() - start
        mode_report: dict[str, Any] = {
            "cold_first_query_s": round(cold, 2),
            "rss_after_load_mb": round(_rss_mb(), 1),
            "rss_delta_mb": round(_rss_mb() - rss_before, 1),
        }
        for fname, fobj in filter_sets.items():
            mode_report[fname] = _time_mode(service, queries, fobj, args.warmup)
            print(f"[{mode:8s}] filter={fname:12s} {mode_report[fname]}", flush=True)
        report["modes"][mode] = mode_report
    report["rss_final_mb"] = round(_rss_mb(), 1)
    return report


# ---------------------------------------------------------------------------
# ANN vs exact FAISS
# ---------------------------------------------------------------------------

def run_ann(args: argparse.Namespace) -> dict[str, Any]:
    import faiss
    import numpy as np

    from app.backend.config import get_vector_index_path
    from app.backend.retrieval.embedder import IngredientEmbedder
    from app.backend.retrieval.semantic import build_query_representation

    flat = faiss.read_index(str(get_vector_index_path()))
    n, dim = flat.ntotal, flat.d
    vectors = flat.reconstruct_n(0, n).astype(np.float32)

    queries = sample_queries(args.queries, args.seed)
    embedder = IngredientEmbedder()
    qvecs = embedder.embed_queries([build_query_representation(q) for q in queries]).astype(np.float32)
    k = args.k

    def timed_search(index: Any, label: str, params: dict[str, Any]) -> dict[str, Any]:
        lat = []
        found = []
        for i in range(len(qvecs)):
            start = time.perf_counter()
            _, idx = index.search(qvecs[i : i + 1], k)
            lat.append((time.perf_counter() - start) * 1000.0)
            found.append(idx[0])
        recall = statistics.fmean(
            len(set(found[i].tolist()) & set(exact[i].tolist())) / k for i in range(len(qvecs))
        )
        row = {"index": label, **params, f"recall@{k}": round(recall, 4),
               "p50_ms": round(_percentile(lat, 50), 3), "p95_ms": round(_percentile(lat, 95), 3)}
        print(row, flush=True)
        return row

    faiss.omp_set_num_threads(args.threads)
    exact_lat = []
    exact = []
    for i in range(len(qvecs)):
        start = time.perf_counter()
        _, idx = flat.search(qvecs[i : i + 1], k)
        exact_lat.append((time.perf_counter() - start) * 1000.0)
        exact.append(idx[0])
    rows: list[dict[str, Any]] = [{
        "index": "Flat (exact)", f"recall@{k}": 1.0,
        "p50_ms": round(_percentile(exact_lat, 50), 3), "p95_ms": round(_percentile(exact_lat, 95), 3),
        "build_s": 0.0,
    }]

    # HNSW
    for m in (32,):
        start = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(vectors)
        build = time.perf_counter() - start
        for ef in (32, 64, 128, 256):
            hnsw.hnsw.efSearch = ef
            row = timed_search(hnsw, f"HNSW{m}", {"efSearch": ef})
            row["build_s"] = round(build, 1)
            rows.append(row)

    # IVF
    nlist = args.nlist
    start = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(args.seed)
    train = vectors[rng.choice(n, size=min(n, nlist * 40), replace=False)]
    ivf.train(train)
    ivf.add(vectors)
    build = time.perf_counter() - start
    for nprobe in (8, 16, 32, 64):
        ivf.nprobe = nprobe
        row = timed_search(ivf, f"IVF{nlist},Flat", {"nprobe": nprobe})
        row["build_s"] = round(build, 1)
        rows.append(row)

    # Filtered search: exact vs HNSW with the same IDSelector, vegan only.
    filtered_rows: list[dict[str, Any]] = []
    try:
        from app.backend.config import get_vector_ids_path
        from app.backend.database.recipe_metadata import MetadataIndex
        from app.backend.recipes.schemas import DietaryFilters

        ids_doc = json.loads(Path(get_vector_ids_path()).read_text(encoding="utf-8"))
        order = np.asarray(ids_doc["recipe_ids"], dtype=np.int64)
        allowed = MetadataIndex().allowed(DietaryFilters(vegan=True))
        mask = allowed.mask_for(order)
        sel = faiss.IDSelectorBitmap(np.packbits(mask, bitorder="little"))
        exact_f = [flat.search(qvecs[i : i + 1], k, params=faiss.SearchParameters(sel=sel))[1][0]
                   for i in range(len(qvecs))]
        hnsw.hnsw.efSearch = 128
        for ef in (128, 256, 512):
            lat = []
            rec = []
            for i in range(len(qvecs)):
                params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
                start = time.perf_counter()
                _, idx = hnsw.search(qvecs[i : i + 1], k, params=params)
                lat.append((time.perf_counter() - start) * 1000.0)
                rec.append(len(set(idx[0].tolist()) & set(exact_f[i].tolist())) / k)
            row = {"index": "HNSW32 + vegan selector", "efSearch": ef, "allowed_fraction": round(float(mask.mean()), 3),
                   f"recall@{k}": round(statistics.fmean(rec), 4),
                   "p50_ms": round(_percentile(lat, 50), 3), "p95_ms": round(_percentile(lat, 95), 3)}
            print(row, flush=True)
            filtered_rows.append(row)
    except Exception as exc:  # metadata table missing, etc.
        print(f"filtered ANN comparison skipped: {exc!r}")

    return {"vectors": n, "dim": dim, "queries": len(qvecs), "k": k, "threads": args.threads,
            "rows": rows, "filtered_rows": filtered_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["search", "load", "ann"])
    parser.add_argument("--db")
    parser.add_argument("--vector-index")
    parser.add_argument("--vector-ids")
    parser.add_argument("--bm25")
    parser.add_argument("--ranker")
    parser.add_argument("--engine", help="Embedder engine: cuda, onnx or cpu")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", default="lexical,semantic,hybrid")
    parser.add_argument("--filters", default="vegan,gluten_free",
                        help="Comma-separated DietaryFilters flags to time in addition to no filter")
    parser.add_argument("--components", default="bm25,faiss,embedder,ranker")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--ranker-threads", type=int, default=None,
                        help="Experiment: set LightGBM n_jobs on the loaded ranker before timing")
    parser.add_argument("--out", help="Write the JSON report here")
    args = parser.parse_args()

    _apply_path_overrides(args)
    report: dict[str, Any] = {"machine": machine_info(), "command": args.command}
    if args.command == "search":
        report["search"] = run_search(args)
    elif args.command == "load":
        report["load"] = run_load(args)
        for row in report["load"]:
            print(row)
    else:
        report["ann"] = run_ann(args)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
