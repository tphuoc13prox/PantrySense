# Evaluation results: `queries_v1_topk20_v1_1_minilm`

- Query set: `queries_v1.jsonl` (1000 queries, sha256 `eb051d0d17f5`)
- Git commit: `e7ab0d8d57`
- Layer (a) candidate depth K=100; layer (b) pipeline SEMANTIC_TOP_K=20 (lexical mode is hard-coded to 20); semantic threshold 0.35; RRF k=60; coverage threshold 0.28 (0.1 for <=2 ingredients)
- Embedder: sentence-transformers-cpu, model `sentence-transformers/all-MiniLM-L6-v2`; ML ranker: LGBMRanker (loaded=True, produced scores on 993 queries)
- Windows-10-10.0.26300-SP0, Python 3.11.16; single process, sequential queries, 5 warm-up queries
- Metrics: mean with 95% percentile-bootstrap CI over queries (2000 resamples). One relevant recipe per query (near-duplicates collapsed), so Recall@k == Hit@k.
- Note: v1.1 code before the LTR retrain; DB is a copy with recipe_metadata migrated; BM25 is the original recipe_bm25.json converted to npz (same content).
- Note: Machine otherwise idle during the run (no other heavy job).
- Note: Dense model: all-MiniLM-L6-v2 with the original index.

## Layer (a): retriever candidates (before coverage filter / re-ranking)

| Config | Recall@10 | Recall@50 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms | p95 ms | avg #results |
|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.645 [0.615, 0.675] | 0.801 [0.776, 0.827] | 0.855 [0.833, 0.877] | 0.456 [0.430, 0.483] | 0.501 [0.476, 0.527] | 0.369 [0.341, 0.399] | 7.2 | 14.0 | 100.0 |
| `dense` | 0.291 [0.264, 0.320] | 0.444 [0.414, 0.475] | 0.535 [0.504, 0.565] | 0.163 [0.144, 0.184] | 0.193 [0.173, 0.215] | 0.112 [0.093, 0.132] | 41.2 | 60.2 | 99.9 |
| `hybrid_rrf` | 0.478 [0.448, 0.507] | 0.746 [0.720, 0.773] | 0.821 [0.798, 0.845] | 0.284 [0.261, 0.310] | 0.330 [0.306, 0.355] | 0.209 [0.184, 0.235] | 50.1 | 69.4 | 100.0 |

## Layer (b): final pipeline output (`RecipeSearchService.search`)

| Config | Recall@10 | Recall@50 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms | p95 ms | avg #results |
|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.655 [0.626, 0.685] | 0.688 [0.660, 0.718] | 0.688 [0.660, 0.718] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 12.0 | 19.0 | 11.2 |
| `bm25_heuristic` | 0.664 [0.635, 0.693] | 0.688 [0.660, 0.718] | 0.688 [0.660, 0.718] | 0.375 [0.352, 0.399] | 0.445 [0.421, 0.469] | 0.238 [0.212, 0.264] | 12.0 | 19.0 | 11.2 |
| `dense` | 0.331 [0.304, 0.361] | 0.350 [0.322, 0.381] | 0.350 [0.322, 0.381] | 0.213 [0.191, 0.237] | 0.241 [0.219, 0.266] | 0.160 [0.138, 0.183] | 41.8 | 60.7 | 8.9 |
| `dense_heuristic` | 0.341 [0.313, 0.371] | 0.350 [0.322, 0.381] | 0.350 [0.322, 0.381] | 0.224 [0.202, 0.247] | 0.252 [0.231, 0.277] | 0.160 [0.138, 0.183] | 41.8 | 60.7 | 8.9 |
| `hybrid_rrf` | 0.604 [0.574, 0.635] | 0.655 [0.626, 0.685] | 0.655 [0.626, 0.685] | 0.364 [0.339, 0.391] | 0.421 [0.397, 0.448] | 0.265 [0.239, 0.293] | 48.9 | 70.1 | 10.7 |
| `hybrid_heuristic` | 0.638 [0.609, 0.668] | 0.655 [0.626, 0.685] | 0.655 [0.626, 0.685] | 0.359 [0.336, 0.382] | 0.426 [0.403, 0.450] | 0.227 [0.201, 0.253] | 48.9 | 70.1 | 10.7 |
| `hybrid_ml` | 0.601 [0.572, 0.632] | 0.655 [0.626, 0.685] | 0.655 [0.626, 0.685] | 0.331 [0.307, 0.355] | 0.395 [0.371, 0.419] | 0.221 [0.195, 0.247] | 58.8 | 83.9 | 10.7 |

\* dense rows use `RecipeSearchService.search(retrieval_mode="semantic")`.

Configs sharing a service call (`bm25`/`bm25_heuristic`, `dense`/`dense_heuristic`, `hybrid_rrf`/`hybrid_heuristic`) share its latency.

## Paired differences (A − B, 95% paired bootstrap CI)

| Layer | A | B | ΔRecall@10 | ΔMRR@10 | ΔnDCG@10 | ΔHit@1 |
|---|---|---|---|---|---|---|
| a | `hybrid_rrf` | `bm25` | -0.167 [-0.193, -0.139] | -0.171 [-0.194, -0.150] | -0.171 [-0.192, -0.151] | -0.160 [-0.186, -0.133] |
| a | `hybrid_rrf` | `dense` | +0.187 [+0.163, +0.212] | +0.121 [+0.107, +0.136] | +0.137 [+0.123, +0.151] | +0.097 [+0.079, +0.116] |
| b | `hybrid_rrf` | `bm25` | -0.051 [-0.068, -0.033] | -0.118 [-0.135, -0.100] | -0.103 [-0.117, -0.088] | -0.128 [-0.152, -0.103] |
| b | `hybrid_heuristic` | `hybrid_rrf` | +0.034 [+0.020, +0.048] | -0.006 [-0.027, +0.014] | +0.005 [-0.012, +0.020] | -0.038 [-0.067, -0.009] |
| b | `hybrid_ml` | `hybrid_rrf` | -0.003 [-0.016, +0.010] | -0.033 [-0.047, -0.020] | -0.026 [-0.038, -0.016] | -0.044 [-0.067, -0.023] |
| b | `hybrid_ml` | `hybrid_heuristic` | -0.037 [-0.051, -0.024] | -0.028 [-0.044, -0.012] | -0.031 [-0.044, -0.017] | -0.006 [-0.028, +0.015] |

## By difficulty tier (mean)

| Tier | n | Config | Recall@10 | Recall@100 | MRR@10 | Hit@1 |
|---|---|---|---|---|---|---|
| noisy | 333 | `a:bm25` | 0.556 | 0.778 | 0.410 | 0.345 |
| noisy | 333 | `a:dense` | 0.243 | 0.480 | 0.145 | 0.102 |
| noisy | 333 | `a:hybrid_rrf` | 0.432 | 0.751 | 0.252 | 0.183 |
| noisy | 333 | `b:hybrid_rrf` | 0.559 | 0.574 | 0.348 | 0.252 |
| noisy | 333 | `b:hybrid_heuristic` | 0.571 | 0.574 | 0.338 | 0.213 |
| noisy | 333 | `b:hybrid_ml` | 0.559 | 0.574 | 0.319 | 0.204 |
| sparse | 333 | `a:bm25` | 0.613 | 0.871 | 0.350 | 0.240 |
| sparse | 333 | `a:dense` | 0.240 | 0.498 | 0.106 | 0.066 |
| sparse | 333 | `a:hybrid_rrf` | 0.423 | 0.838 | 0.215 | 0.135 |
| sparse | 333 | `b:hybrid_rrf` | 0.532 | 0.619 | 0.282 | 0.183 |
| sparse | 333 | `b:hybrid_heuristic` | 0.583 | 0.619 | 0.268 | 0.132 |
| sparse | 333 | `b:hybrid_ml` | 0.532 | 0.619 | 0.244 | 0.135 |
| standard | 334 | `a:bm25` | 0.766 | 0.916 | 0.607 | 0.521 |
| standard | 334 | `a:dense` | 0.389 | 0.626 | 0.239 | 0.168 |
| standard | 334 | `a:hybrid_rrf` | 0.578 | 0.874 | 0.387 | 0.308 |
| standard | 334 | `b:hybrid_rrf` | 0.722 | 0.772 | 0.463 | 0.359 |
| standard | 334 | `b:hybrid_heuristic` | 0.760 | 0.772 | 0.469 | 0.335 |
| standard | 334 | `b:hybrid_ml` | 0.713 | 0.772 | 0.430 | 0.323 |

## Latency components (layer (a), ms)

| Component | p50 | p95 | mean |
|---|---|---|---|
| embed | 15.93 | 23.88 | 17.05 |
| faiss | 25.09 | 39.07 | 26.23 |
| bm25 | 7.15 | 14.01 | 7.78 |
| rrf | 0.89 | 1.25 | 0.87 |

`rule_based` was not run: it scores all ~315k recipes per query (about 10 minutes per query on the reference machine). Use `--rule-based-n N` to evaluate it on a small subset.
