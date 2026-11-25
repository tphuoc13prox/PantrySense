# Evaluation results: `queries_v1_topk20_v1_1_e5ft_ltrv2`

- Query set: `queries_v1.jsonl` (1000 queries, sha256 `eb051d0d17f5`)
- Git commit: `f606e284a6`
- Layer (a) candidate depth K=100; layer (b) pipeline SEMANTIC_TOP_K=20 (lexical mode is hard-coded to 20); semantic threshold 0.35; RRF k=60; coverage threshold 0.28 (0.1 for <=2 ingredients)
- Embedder: sentence-transformers-cpu, model `data/models/e5-small-pantry`; ML ranker: LGBMRanker (loaded=True, produced scores on 997 queries)
- Windows-10-10.0.26300-SP0, Python 3.11.16; single process, sequential queries, 5 warm-up queries
- Metrics: mean with 95% percentile-bootstrap CI over queries (2000 resamples). One relevant recipe per query (near-duplicates collapsed), so Recall@k == Hit@k.
- Note: v1.1 code with the LTR v2 bundle; DB is a copy with recipe_metadata migrated; BM25 is the original recipe_bm25.json converted to npz (same content).
- Note: Machine otherwise idle during the run (no other heavy job).
- Note: ML ranker: LTR v2 bundle (sha256 3990497d68fe), trained on hybrid candidates from MiniLM.
- Note: Dense model: fine-tuned e5-small-v2. The LTR v2 model was NOT retrained for e5: semantic_score and dense ranks now come from a different distribution than in training.

## Layer (a): retriever candidates (before coverage filter / re-ranking)

| Config | Recall@10 | Recall@50 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms | p95 ms | avg #results |
|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.645 [0.615, 0.675] | 0.801 [0.776, 0.827] | 0.855 [0.833, 0.877] | 0.456 [0.430, 0.483] | 0.501 [0.476, 0.527] | 0.369 [0.341, 0.399] | 6.5 | 12.1 | 100.0 |
| `dense` | 0.722 [0.694, 0.750] | 0.863 [0.841, 0.884] | 0.907 [0.888, 0.925] | 0.529 [0.502, 0.556] | 0.575 [0.549, 0.601] | 0.439 [0.409, 0.470] | 40.7 | 49.3 | 99.6 |
| `hybrid_rrf` | 0.729 [0.702, 0.756] | 0.856 [0.833, 0.878] | 0.903 [0.884, 0.922] | 0.535 [0.508, 0.562] | 0.582 [0.555, 0.607] | 0.449 [0.418, 0.480] | 48.2 | 59.5 | 100.0 |

## Layer (b): final pipeline output (`RecipeSearchService.search`)

| Config | Recall@10 | Recall@50 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms | p95 ms | avg #results |
|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.655 [0.626, 0.685] | 0.688 [0.660, 0.718] | 0.688 [0.660, 0.718] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 10.6 | 18.4 | 11.2 |
| `bm25_heuristic` | 0.664 [0.635, 0.693] | 0.688 [0.660, 0.718] | 0.688 [0.660, 0.718] | 0.375 [0.352, 0.399] | 0.445 [0.421, 0.469] | 0.238 [0.212, 0.264] | 10.6 | 18.4 | 11.2 |
| `dense` | 0.717 [0.689, 0.745] | 0.752 [0.725, 0.780] | 0.752 [0.725, 0.780] | 0.551 [0.523, 0.579] | 0.592 [0.565, 0.619] | 0.464 [0.432, 0.494] | 49.5 | 66.0 | 10.6 |
| `dense_heuristic` | 0.718 [0.690, 0.746] | 0.752 [0.725, 0.780] | 0.752 [0.725, 0.780] | 0.474 [0.449, 0.501] | 0.533 [0.508, 0.559] | 0.362 [0.332, 0.393] | 49.5 | 66.0 | 10.6 |
| `hybrid_rrf` | 0.729 [0.701, 0.757] | 0.761 [0.734, 0.788] | 0.761 [0.734, 0.788] | 0.555 [0.527, 0.584] | 0.597 [0.570, 0.624] | 0.475 [0.443, 0.507] | 55.4 | 72.1 | 11.3 |
| `hybrid_heuristic` | 0.722 [0.694, 0.751] | 0.761 [0.734, 0.788] | 0.761 [0.734, 0.788] | 0.418 [0.394, 0.443] | 0.491 [0.468, 0.516] | 0.276 [0.249, 0.304] | 55.4 | 72.1 | 11.3 |
| `hybrid_ml` | 0.755 [0.728, 0.782] | 0.761 [0.734, 0.788] | 0.761 [0.734, 0.788] | 0.648 [0.621, 0.677] | 0.675 [0.649, 0.702] | 0.589 [0.559, 0.620] | 58.0 | 73.6 | 11.3 |

\* dense rows use `RecipeSearchService.search(retrieval_mode="semantic")`.

Configs sharing a service call (`bm25`/`bm25_heuristic`, `dense`/`dense_heuristic`, `hybrid_rrf`/`hybrid_heuristic`) share its latency.

## Paired differences (A − B, 95% paired bootstrap CI)

| Layer | A | B | ΔRecall@10 | ΔMRR@10 | ΔnDCG@10 | ΔHit@1 |
|---|---|---|---|---|---|---|
| a | `hybrid_rrf` | `bm25` | +0.084 [+0.066, +0.103] | +0.080 [+0.065, +0.095] | +0.080 [+0.067, +0.094] | +0.080 [+0.058, +0.102] |
| a | `hybrid_rrf` | `dense` | +0.007 [-0.011, +0.024] | +0.006 [-0.007, +0.020] | +0.006 [-0.006, +0.018] | +0.010 [-0.009, +0.029] |
| b | `hybrid_rrf` | `bm25` | +0.074 [+0.058, +0.093] | +0.073 [+0.059, +0.087] | +0.073 [+0.061, +0.085] | +0.082 [+0.061, +0.104] |
| b | `hybrid_heuristic` | `hybrid_rrf` | -0.007 [-0.020, +0.007] | -0.137 [-0.158, -0.115] | -0.105 [-0.122, -0.088] | -0.199 [-0.231, -0.166] |
| b | `hybrid_ml` | `hybrid_rrf` | +0.026 [+0.016, +0.037] | +0.094 [+0.079, +0.110] | +0.078 [+0.066, +0.091] | +0.114 [+0.093, +0.136] |
| b | `hybrid_ml` | `hybrid_heuristic` | +0.033 [+0.022, +0.045] | +0.230 [+0.209, +0.251] | +0.183 [+0.167, +0.200] | +0.313 [+0.283, +0.343] |

## By difficulty tier (mean)

| Tier | n | Config | Recall@10 | Recall@100 | MRR@10 | Hit@1 |
|---|---|---|---|---|---|---|
| noisy | 333 | `a:bm25` | 0.556 | 0.778 | 0.410 | 0.345 |
| noisy | 333 | `a:dense` | 0.670 | 0.844 | 0.492 | 0.414 |
| noisy | 333 | `a:hybrid_rrf` | 0.676 | 0.841 | 0.511 | 0.438 |
| noisy | 333 | `b:hybrid_rrf` | 0.688 | 0.694 | 0.557 | 0.489 |
| noisy | 333 | `b:hybrid_heuristic` | 0.682 | 0.694 | 0.441 | 0.321 |
| noisy | 333 | `b:hybrid_ml` | 0.694 | 0.694 | 0.632 | 0.598 |
| sparse | 333 | `a:bm25` | 0.613 | 0.871 | 0.350 | 0.240 |
| sparse | 333 | `a:dense` | 0.661 | 0.904 | 0.415 | 0.300 |
| sparse | 333 | `a:hybrid_rrf` | 0.697 | 0.904 | 0.419 | 0.297 |
| sparse | 333 | `b:hybrid_rrf` | 0.685 | 0.730 | 0.425 | 0.315 |
| sparse | 333 | `b:hybrid_heuristic` | 0.646 | 0.730 | 0.302 | 0.156 |
| sparse | 333 | `b:hybrid_ml` | 0.715 | 0.730 | 0.557 | 0.471 |
| standard | 334 | `a:bm25` | 0.766 | 0.916 | 0.607 | 0.521 |
| standard | 334 | `a:dense` | 0.835 | 0.973 | 0.680 | 0.602 |
| standard | 334 | `a:hybrid_rrf` | 0.814 | 0.964 | 0.676 | 0.611 |
| standard | 334 | `b:hybrid_rrf` | 0.814 | 0.859 | 0.682 | 0.620 |
| standard | 334 | `b:hybrid_heuristic` | 0.838 | 0.859 | 0.512 | 0.350 |
| standard | 334 | `b:hybrid_ml` | 0.856 | 0.859 | 0.756 | 0.698 |

## Latency components (layer (a), ms)

| Component | p50 | p95 | mean |
|---|---|---|---|
| embed | 19.09 | 24.48 | 19.42 |
| faiss | 21.57 | 27.58 | 21.95 |
| bm25 | 6.52 | 12.07 | 6.83 |
| rrf | 0.75 | 1.23 | 0.79 |

`rule_based` was not run: it scores all ~315k recipes per query (about 10 minutes per query on the reference machine). Use `--rule-based-n N` to evaluate it on a small subset.
