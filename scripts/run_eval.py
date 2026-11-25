"""Offline evaluation of the PantrySense retrieval + ranking pipeline on the real corpus.

Measures two layers for every query in a query set built by build_eval_set.py:

  (a) retriever candidates, before coverage filtering or re-ranking
      bm25 / dense / hybrid_rrf at depth --candidate-k (default 100);
      hybrid is reproduced exactly as HybridRetriever.retrieve(top_k=K)
      (dense 2K + BM25 2K fused by RRF to K).
  (b) final output of RecipeSearchService.search() with the production
      configuration (PANTRYSENSE_SEMANTIC_TOP_K, coverage threshold, ...).

It only reads the data files; nothing is rebuilt or written next to them.

Example:
    python scripts/run_eval.py --data-dir data
    python scripts/run_eval.py --data-dir ... --limit 50 --name smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_FILES = {
    "PANTRYSENSE_DB_PATH": "recipes.db",
    "PANTRYSENSE_VECTOR_INDEX_PATH": "recipe_vectors.index",
    "PANTRYSENSE_VECTOR_IDS_PATH": "recipe_vector_ids.json",
    "PANTRYSENSE_BM25_INDEX_PATH": "recipe_bm25.json",
    "PANTRYSENSE_RANKER_MODEL_PATH": "ranker_model.joblib",
    "PANTRYSENSE_INGREDIENT_VOCAB_PATH": "ingredient_vocab.json",
}

KS = (10, 50, 100)
METRIC_NAMES = ("recall@10", "recall@50", "recall@100", "mrr@10", "ndcg@10", "hit@1")

# (layer, config, description)
CONFIGS = [
    ("a", "bm25", "BM25 top-K candidates"),
    ("a", "dense", "FAISS dense top-K candidates (semantic threshold applied)"),
    ("a", "hybrid_rrf", "BM25 + dense fused with RRF, top-K"),
    ("b", "bm25", "lexical mode, coverage filter, BM25 order"),
    ("b", "bm25_heuristic", "lexical mode, coverage filter, heuristic ranking"),
    ("b", "dense", "semantic mode*, coverage filter, dense order"),
    ("b", "dense_heuristic", "semantic mode*, coverage filter, heuristic ranking"),
    ("b", "hybrid_rrf", "hybrid mode, coverage filter, RRF order"),
    ("b", "hybrid_heuristic", "hybrid mode, coverage filter, heuristic ranking"),
    ("b", "hybrid_ml", "hybrid mode, coverage filter, LightGBM ranker (app default)"),
]

PAIRED_COMPARISONS = [
    ("a", "hybrid_rrf", "bm25"),
    ("a", "hybrid_rrf", "dense"),
    ("b", "hybrid_rrf", "bm25"),
    ("b", "hybrid_heuristic", "hybrid_rrf"),
    ("b", "hybrid_ml", "hybrid_rrf"),
    ("b", "hybrid_ml", "hybrid_heuristic"),
]

logger = logging.getLogger("run_eval")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries_v1.jsonl")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="folder with recipes.db, indices and ranker; sets any PANTRYSENSE_*_PATH not already set")
    p.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--name", default=None, help="result file stem (default: <queries stem>_topk<N>)")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--candidate-k", type=int, default=100, help="depth of layer (a) candidate lists")
    p.add_argument("--pipeline-top-k", type=int, default=None,
                   help="override PANTRYSENSE_SEMANTIC_TOP_K for layer (b) (default: app setting, 20)")
    p.add_argument("--warmup", type=int, default=5, help="untimed warm-up queries")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--rule-based-n", type=int, default=0,
                   help="also run rule_based mode (full DB scan) on the first N queries; 0 = skip")
    p.add_argument("--allow-ml-fallback", action="store_true",
                   help="do not abort if the LightGBM model cannot be loaded")
    p.add_argument("--note", action="append", default=[], help="free-text note stored in the results")
    p.add_argument("--render-only", type=Path, default=None,
                   help="re-render the markdown for an existing results JSON (adding any --note) and exit")
    p.add_argument("--dump-candidates", type=Path, default=None,
                   help="write layer (a) hybrid candidate lists (JSONL) here, e.g. for LTR training")
    return p.parse_args(argv)


def configure_env(args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        for var, fname in DATA_FILES.items():
            os.environ.setdefault(var, str((args.data_dir / fname).resolve()))
    # The embedding model is expected in the local HF cache; skip network checks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for var in ("PANTRYSENSE_AUTO_BUILD_INDEX", "PANTRYSENSE_AUTO_OPEN_BROWSER", "PANTRYSENSE_AUTO_SHUTDOWN"):
        os.environ[var] = "false"
    # MLRecipeRanker gates on this env var internally; the heuristic configs
    # pass ranking_mode explicitly, so forcing it here only prevents a silent
    # ML -> heuristic fallback in the hybrid_ml config.
    os.environ["PANTRYSENSE_RANKING_MODE"] = "ml"
    if not os.environ.get("PANTRYSENSE_EMBEDDER_ENGINE"):
        import importlib.util

        if importlib.util.find_spec("fastembed") is None:
            # The default 'onnx' engine retries `import fastembed` (and logs a
            # warning) on every query before falling back to sentence-transformers
            # on CPU; go straight to that fallback so latency is not inflated.
            os.environ["PANTRYSENSE_EMBEDDER_ENGINE"] = "cpu"
    if args.pipeline_top_k is not None:
        os.environ["PANTRYSENSE_SEMANTIC_TOP_K"] = str(args.pipeline_top_k)
    missing = [v for v in DATA_FILES if not Path(os.environ.get(v, "")).is_file()]
    if missing:
        raise SystemExit(f"Missing data files (set --data-dir or env vars): {', '.join(missing)}")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collapse(ranked: list[int], equivalents: dict[int, int]) -> list[int]:
    """Map near-duplicates of the target onto the target id (one relevant item per query)."""
    from app.backend.evaluation.metrics import dedupe

    return dedupe(equivalents.get(r, r) for r in ranked)


def query_metrics(ranked: list[int], target: int) -> dict[str, float]:
    from app.backend.evaluation import metrics as m

    rel = {target}
    out = {f"recall@{k}": m.recall_at_k(ranked, rel, k) for k in KS}
    out["mrr@10"] = m.mrr_at_k(ranked, rel, 10)
    out["ndcg@10"] = m.ndcg_at_k(ranked, rel, 10)
    out["hit@1"] = m.hit_at_k(ranked, rel, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.render_only is not None:
        res = json.loads(args.render_only.read_text(encoding="utf-8"))
        res["notes"] = res.get("notes", []) + args.note
        with open(args.render_only, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        md_path = args.render_only.with_suffix(".md")
        md_path.write_text(render_markdown(res), encoding="utf-8", newline="\n")
        logger.info("Re-rendered %s", md_path)
        return 0
    configure_env(args)

    # App modules read configuration at import time, so import after configure_env.
    from app.backend import config as cfg
    from app.backend.evaluation.metrics import bootstrap_ci, paired_bootstrap_diff, percentile
    from app.backend.evaluation.query_set import load_query_set
    from app.backend.ranking.ml_ranker import MLRecipeRanker
    from app.backend.recipes.service import RecipeSearchService
    from app.backend.retrieval.bm25_store import BM25Store, preferred_index_path
    from app.backend.retrieval.embedder import read_index_metadata
    from app.backend.retrieval.fusion import reciprocal_rank_fusion
    from app.backend.retrieval.hybrid import HybridRetriever
    from app.backend.retrieval.semantic import SemanticRetriever, build_query_representation

    queries = load_query_set(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    pipeline_top_k = cfg.get_semantic_top_k()
    name = args.name or f"{args.queries.stem}_topk{pipeline_top_k}"
    K = args.candidate_k

    # ---- load shared indices once -------------------------------------------------
    load_times = {}
    t0 = time.perf_counter()
    bm25 = BM25Store()
    bm25_path = preferred_index_path(cfg.get_bm25_index_path())
    if not bm25.load(bm25_path):
        raise SystemExit("BM25 index failed to load")
    load_times["bm25_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sem = SemanticRetriever()
    if not sem.ensure_loaded():
        raise SystemExit("FAISS index failed to load")
    load_times["faiss_s"] = time.perf_counter() - t0
    hybrid = HybridRetriever(semantic_retriever=sem, bm25_store=bm25)
    ml = MLRecipeRanker()
    ml_loaded = ml.load_model()
    if not ml_loaded and not args.allow_ml_fallback:
        raise SystemExit(
            f"ML ranker could not be loaded from {ml.model_path} (is lightgbm installed?). "
            "hybrid_ml would silently fall back to the heuristic; pass --allow-ml-fallback to run anyway."
        )
    logger.info("Indices loaded: %s; ML ranker loaded: %s", {k: round(v, 1) for k, v in load_times.items()}, ml_loaded)

    def service(mode: str, ranking: str) -> RecipeSearchService:
        return RecipeSearchService(
            retrieval_mode=mode, ranking_mode=ranking, ml_ranker=ml,
            retriever=sem, hybrid_retriever=hybrid, bm25_store=bm25,
        )

    svc = {
        "lexical": service("lexical", "heuristic"),
        "semantic": service("semantic", "heuristic"),
        "hybrid_h": service("hybrid", "heuristic"),
        "hybrid_ml": service("hybrid", "ml"),
    }
    semantic_threshold = cfg.get_semantic_threshold()
    rrf_k = cfg.get_rrf_k()

    def layer_a(ings: list[str]) -> tuple[dict[str, list], dict[str, float]]:
        qstr = build_query_representation(ings)
        t = time.perf_counter()
        vec = sem.embedder.embed_text(qstr)
        t_embed = time.perf_counter() - t
        t = time.perf_counter()
        dense2 = sem.vector_store.search(vec, top_k=2 * K, min_threshold=semantic_threshold)
        t_faiss = time.perf_counter() - t
        t = time.perf_counter()
        sparse2 = bm25.search(qstr, top_k=2 * K)
        t_bm25 = time.perf_counter() - t
        t = time.perf_counter()
        fused = reciprocal_rank_fusion(dense2, sparse2, k=rrf_k, top_k=K)
        t_rrf = time.perf_counter() - t
        lists = {
            "bm25": sparse2[:K],
            "dense": dense2[:K],
            "hybrid_rrf": fused,
        }
        timings = {"embed": t_embed, "faiss": t_faiss, "bm25": t_bm25, "rrf": t_rrf}
        return lists, timings

    def layer_b(ings: list[str]) -> tuple[dict[str, list], dict[str, float]]:
        results, timings = {}, {}
        for key, s in svc.items():
            t = time.perf_counter()
            results[key] = s.search(ings)
            timings[key] = time.perf_counter() - t
        return results, timings

    # Sanity check: our layer (a) hybrid list must equal HybridRetriever.retrieve(top_k=K).
    for q in queries[: min(3, len(queries))]:
        mine = [c.recipe_id for c in layer_a(q.ingredients)[0]["hybrid_rrf"]]
        ref = [c.recipe_id for c in hybrid.retrieve(q.ingredients, top_k=K)]
        if mine != ref:
            raise SystemExit(f"layer (a) hybrid list differs from HybridRetriever.retrieve for {q.qid}")

    logger.info("Warm-up (%d queries)...", args.warmup)
    for q in queries[: args.warmup]:
        layer_a(q.ingredients)
        layer_b(q.ingredients)

    per_query = []
    ml_scored_queries = 0
    dump = open(args.dump_candidates, "w", encoding="utf-8", newline="\n") if args.dump_candidates else None
    t_start = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        equivalents = {rid: q.recipe_id for rid in q.relevant_ids}
        lists_a, tim_a = layer_a(q.ingredients)
        res_b, tim_b = layer_b(q.ingredients)

        ranked: dict[tuple[str, str], list[int]] = {}
        for cname in ("bm25", "dense"):
            ranked[("a", cname)] = [rid for rid, _score, _rank in lists_a[cname]]
        ranked[("a", "hybrid_rrf")] = [c.recipe_id for c in lists_a["hybrid_rrf"]]

        def by_retrieval(rs):
            return [r.id for r in sorted(rs, key=lambda r: (r.retrieval_rank or 10**9))]

        ranked[("b", "bm25")] = by_retrieval(res_b["lexical"])
        ranked[("b", "bm25_heuristic")] = [r.id for r in res_b["lexical"]]
        ranked[("b", "dense")] = by_retrieval(res_b["semantic"])
        ranked[("b", "dense_heuristic")] = [r.id for r in res_b["semantic"]]
        ranked[("b", "hybrid_rrf")] = by_retrieval(res_b["hybrid_h"])
        ranked[("b", "hybrid_heuristic")] = [r.id for r in res_b["hybrid_h"]]
        ranked[("b", "hybrid_ml")] = [r.id for r in res_b["hybrid_ml"]]
        if any(r.ml_score is not None for r in res_b["hybrid_ml"]):
            ml_scored_queries += 1

        latency_ms = {
            ("a", "bm25"): tim_a["bm25"],
            ("a", "dense"): tim_a["embed"] + tim_a["faiss"],
            ("a", "hybrid_rrf"): sum(tim_a.values()),
            ("b", "bm25"): tim_b["lexical"],
            ("b", "bm25_heuristic"): tim_b["lexical"],
            ("b", "dense"): tim_b["semantic"],
            ("b", "dense_heuristic"): tim_b["semantic"],
            ("b", "hybrid_rrf"): tim_b["hybrid_h"],
            ("b", "hybrid_heuristic"): tim_b["hybrid_h"],
            ("b", "hybrid_ml"): tim_b["hybrid_ml"],
        }

        record = {"qid": q.qid, "tier": q.tier, "recipe_id": q.recipe_id, "configs": {},
                  "components_ms": {k: round(v * 1000, 3) for k, v in tim_a.items()}}
        for (layer, cname), ids in ranked.items():
            collapsed = collapse(ids, equivalents)
            mets = query_metrics(collapsed, q.recipe_id)
            rank = collapsed.index(q.recipe_id) + 1 if q.recipe_id in collapsed else None
            record["configs"][f"{layer}:{cname}"] = {
                "rank": rank,
                "n": len(collapsed),
                "latency_ms": round(latency_ms[(layer, cname)] * 1000, 3),
                **{k: round(v, 6) for k, v in mets.items()},
            }
        per_query.append(record)

        if dump:
            dump.write(json.dumps({
                "qid": q.qid, "recipe_id": q.recipe_id, "relevant_ids": q.relevant_ids,
                "ingredients": q.ingredients, "tier": q.tier,
                "hybrid_candidates": [
                    {"recipe_id": c.recipe_id, "rrf_rank": c.retrieval_rank, "rrf_score": c.rrf_score,
                     "semantic_score": c.semantic_score, "bm25_score": c.bm25_score,
                     "label": int(c.recipe_id in equivalents)}
                    for c in lists_a["hybrid_rrf"]
                ],
            }) + "\n")

        if i % 50 == 0 or i == len(queries):
            el = time.perf_counter() - t_start
            logger.info("%d/%d queries (%.1fs, %.2fs/query)", i, len(queries), el, el / i)
    if dump:
        dump.close()
    if not ml_loaded or ml_scored_queries == 0:
        logger.warning("hybrid_ml never produced ml_score: the ML ranker fell back to the heuristic")

    # ---- optional rule_based on a small subset ------------------------------------
    rule_based = None
    if args.rule_based_n > 0:
        rb = RecipeSearchService(retrieval_mode="rule_based")
        subset = queries[: args.rule_based_n]
        rows = []
        for q in subset:
            t = time.perf_counter()
            res = rb.search(q.ingredients)
            lat = time.perf_counter() - t
            collapsed = collapse([r.id for r in res], {rid: q.recipe_id for rid in q.relevant_ids})
            rows.append((query_metrics(collapsed, q.recipe_id), lat))
            logger.info("rule_based %s: %.1fs", q.qid, lat)
        rule_based = {"n_queries": len(subset), "metrics": {}, "latency_ms": {}}
        for mname in METRIC_NAMES:
            vals = [r[0][mname] for r in rows]
            rule_based["metrics"][mname] = dict(zip(("mean", "ci_low", "ci_high"),
                                                    bootstrap_ci(vals, n_resamples=args.bootstrap, seed=args.seed), strict=True))
        lats = [r[1] * 1000 for r in rows]
        rule_based["latency_ms"] = {"p50": percentile(lats, 50), "p95": percentile(lats, 95)}

    # ---- aggregate ------------------------------------------------------------------
    def agg(records, key):
        out = {"metrics": {}, "latency_ms": {}}
        for mname in METRIC_NAMES:
            vals = [r["configs"][key][mname] for r in records]
            out["metrics"][mname] = dict(zip(("mean", "ci_low", "ci_high"),
                                             bootstrap_ci(vals, n_resamples=args.bootstrap, seed=args.seed), strict=True))
        lats = [r["configs"][key]["latency_ms"] for r in records]
        for pname, q in (("p50", 50), ("p95", 95)):
            out["latency_ms"][pname] = dict(zip(
                ("value", "ci_low", "ci_high"),
                bootstrap_ci(lats, statistic=lambda v, q=q: percentile(v, q),
                             n_resamples=min(args.bootstrap, 500), seed=args.seed),
                strict=True,
            ))
        out["mean_results"] = sum(r["configs"][key]["n"] for r in records) / len(records)
        out["found_any_depth"] = sum(1 for r in records if r["configs"][key]["rank"] is not None) / len(records)
        return out

    summary = {}
    for layer, cname, desc in CONFIGS:
        key = f"{layer}:{cname}"
        summary[key] = {"layer": layer, "config": cname, "description": desc, **agg(per_query, key)}

    tiers = sorted({r["tier"] for r in per_query})
    by_tier = {}
    for tier in tiers:
        recs = [r for r in per_query if r["tier"] == tier]
        by_tier[tier] = {"n": len(recs)}
        for layer, cname, _ in CONFIGS:
            key = f"{layer}:{cname}"
            by_tier[tier][key] = {m: sum(r["configs"][key][m] for r in recs) / len(recs)
                                  for m in ("recall@10", "recall@100", "mrr@10", "hit@1")}

    comparisons = []
    for layer, a, b in PAIRED_COMPARISONS:
        ka, kb = f"{layer}:{a}", f"{layer}:{b}"
        row = {"layer": layer, "a": a, "b": b}
        for mname in ("recall@10", "mrr@10", "ndcg@10", "hit@1"):
            va = [r["configs"][ka][mname] for r in per_query]
            vb = [r["configs"][kb][mname] for r in per_query]
            row[mname] = dict(zip(("diff", "ci_low", "ci_high"),
                                  paired_bootstrap_diff(va, vb, n_resamples=args.bootstrap, seed=args.seed), strict=True))
        comparisons.append(row)

    components = {}
    for comp in ("embed", "faiss", "bm25", "rrf"):
        vals = [r["components_ms"][comp] for r in per_query]
        components[comp] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "mean": sum(vals) / len(vals)}

    embedder = sem.embedder
    embed_backend = f"{embedder.backend}-{embedder._device or 'cpu'}"
    index_meta = read_index_metadata(cfg.get_vector_index_path())
    result = {
        "name": name,
        "created_unix": int(time.time()),
        "git_commit": git_commit(),
        "query_set": {"path": args.queries.as_posix(), "sha256": sha256_file(args.queries),
                      "n_queries": len(per_query), "limit": args.limit},
        "settings": {
            "candidate_k": K,
            "pipeline_semantic_top_k": pipeline_top_k,
            "lexical_pipeline_top_k": 20,
            "semantic_threshold": semantic_threshold,
            "rrf_k": rrf_k,
            "coverage_threshold": cfg.get_match_threshold(),
            "coverage_threshold_le2_ingredients": min(cfg.get_match_threshold(), 0.10),
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
            "warmup_queries": args.warmup,
            "relevance": "source recipe; near-duplicates collapsed onto it (1 relevant item per query)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "embedder_engine": embedder.engine,
            "embedder_backend": embed_backend,
            "embedding_model": embedder.model_name,
            "vector_index": cfg.get_vector_index_path().name,
            "vector_index_metadata": index_meta,
            "bm25_index": bm25_path.name,
            "ml_ranker_loaded": ml_loaded,
            "ml_ranker_model": type(ml._model).__name__ if ml._model is not None else None,
            "ml_scored_queries": ml_scored_queries,
            "index_load_s": load_times,
        },
        "summary": summary,
        "by_tier": by_tier,
        "paired_comparisons": comparisons,
        "latency_components_ms": components,
        "rule_based": rule_based,
        "semantic_via_service": True,
        "notes": list(args.note),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{name}.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with open(args.out_dir / f"{name}.per_query.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in per_query:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    md_path = args.out_dir / f"{name}.md"
    md_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    logger.info("Wrote %s and %s", json_path, md_path)
    print(render_markdown(result))
    return 0


def _ci(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def _ms(m: dict) -> str:
    return f"{m['value']:.1f}"


def render_markdown(res: dict) -> str:
    s = res["settings"]
    env = res["environment"]
    lines = [
        f"# Evaluation results: `{res['name']}`",
        "",
        f"- Query set: `{res['query_set']['path'].split('/')[-1]}` ({res['query_set']['n_queries']} queries, "
        f"sha256 `{res['query_set']['sha256'][:12]}`)",
        f"- Git commit: `{(res['git_commit'] or 'unknown')[:10]}`",
        f"- Layer (a) candidate depth K={s['candidate_k']}; layer (b) pipeline SEMANTIC_TOP_K={s['pipeline_semantic_top_k']} "
        f"(lexical mode is hard-coded to {s['lexical_pipeline_top_k']}); semantic threshold {s['semantic_threshold']}; "
        f"RRF k={s['rrf_k']}; coverage threshold {s['coverage_threshold']} ({s['coverage_threshold_le2_ingredients']} for <=2 ingredients)",
        f"- Embedder: {env['embedder_backend']}, model `{env.get('embedding_model', 'unknown')}`; "
        f"ML ranker: {env['ml_ranker_model']} "
        f"(loaded={env['ml_ranker_loaded']}, produced scores on {env['ml_scored_queries']} queries)",
        f"- {env['platform']}, Python {env['python']}; single process, sequential queries, {s['warmup_queries']} warm-up queries",
        f"- Metrics: mean with 95% percentile-bootstrap CI over queries ({s['bootstrap_resamples']} resamples). "
        "One relevant recipe per query (near-duplicates collapsed), so Recall@k == Hit@k.",
        *[f"- Note: {n}" for n in res.get("notes", [])],
        "",
    ]
    for layer, title in (("a", "Layer (a): retriever candidates (before coverage filter / re-ranking)"),
                         ("b", "Layer (b): final pipeline output (`RecipeSearchService.search`)")):
        lines += [f"## {title}", "",
                  "| Config | Recall@10 | Recall@50 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms | p95 ms | avg #results |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in res["summary"].values():
            if row["layer"] != layer:
                continue
            m = row["metrics"]
            lines.append(
                f"| `{row['config']}` | {_ci(m['recall@10'])} | {_ci(m['recall@50'])} | {_ci(m['recall@100'])} | "
                f"{_ci(m['mrr@10'])} | {_ci(m['ndcg@10'])} | {_ci(m['hit@1'])} | "
                f"{_ms(row['latency_ms']['p50'])} | {_ms(row['latency_ms']['p95'])} | {row['mean_results']:.1f} |"
            )
        lines.append("")
        if layer == "b":
            if res.get("semantic_via_service", False):
                lines.append("\\* dense rows use `RecipeSearchService.search(retrieval_mode=\"semantic\")`.")
            else:
                lines.append("\\* `retrieval_mode=\"semantic\"` raised AttributeError in `RecipeSearchService.search` "
                             "when this was run; the dense rows called `SemanticRetriever.retrieve` + "
                             "`_process_candidates` directly.")
            lines.append("")
            lines.append("Configs sharing a service call (`bm25`/`bm25_heuristic`, `dense`/`dense_heuristic`, "
                         "`hybrid_rrf`/`hybrid_heuristic`) share its latency.")
            lines.append("")

    lines += ["## Paired differences (A − B, 95% paired bootstrap CI)", "",
              "| Layer | A | B | ΔRecall@10 | ΔMRR@10 | ΔnDCG@10 | ΔHit@1 |", "|---|---|---|---|---|---|---|"]
    for c in res["paired_comparisons"]:
        cells = [f"{c[m]['diff']:+.3f} [{c[m]['ci_low']:+.3f}, {c[m]['ci_high']:+.3f}]"
                 for m in ("recall@10", "mrr@10", "ndcg@10", "hit@1")]
        lines.append(f"| {c['layer']} | `{c['a']}` | `{c['b']}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## By difficulty tier (mean)", "",
              "| Tier | n | Config | Recall@10 | Recall@100 | MRR@10 | Hit@1 |", "|---|---|---|---|---|---|---|"]
    for tier, block in res["by_tier"].items():
        for key, m in block.items():
            if key == "n" or key not in ("a:bm25", "a:dense", "a:hybrid_rrf", "b:hybrid_rrf",
                                         "b:hybrid_heuristic", "b:hybrid_ml"):
                continue
            lines.append(f"| {tier} | {block['n']} | `{key}` | {m['recall@10']:.3f} | {m['recall@100']:.3f} | "
                         f"{m['mrr@10']:.3f} | {m['hit@1']:.3f} |")
    lines.append("")

    lines += ["## Latency components (layer (a), ms)", "", "| Component | p50 | p95 | mean |", "|---|---|---|---|"]
    for comp, v in res["latency_components_ms"].items():
        lines.append(f"| {comp} | {v['p50']:.2f} | {v['p95']:.2f} | {v['mean']:.2f} |")
    lines.append("")

    rb = res.get("rule_based")
    if rb:
        m = rb["metrics"]
        lines += [f"## rule_based (first {rb['n_queries']} queries only)", "",
                  "| Recall@10 | MRR@10 | Hit@1 | p50 ms | p95 ms |", "|---|---|---|---|---|",
                  f"| {_ci(m['recall@10'])} | {_ci(m['mrr@10'])} | {_ci(m['hit@1'])} | "
                  f"{rb['latency_ms']['p50']:.0f} | {rb['latency_ms']['p95']:.0f} |", ""]
    else:
        lines += ["`rule_based` was not run: it scores all ~315k recipes per query (about 10 minutes "
                  "per query on the reference machine). Use `--rule-based-n N` to evaluate it on a small subset.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
