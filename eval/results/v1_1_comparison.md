# v1.1: official eval compared with the v1.0 baseline

Query set `eval/queries_v1.jsonl`: 1000 queries from buckets 0-1 (the official
eval pool, never used for training or tuning). Relevance: the source recipe, with
near-duplicates collapsed onto it. Layer (b) uses the app settings
(`SEMANTIC_TOP_K=20`, coverage filter). Mean with 95% bootstrap CI.

Runs:

- `baseline`: `queries_v1_topk20` (the v1.0 retrieval and ranking code, unchanged).
- `minilm` / `e5ft`: `queries_v1_topk20_v1_1_{minilm,e5ft}`. The v1.1
  code before the LTR retrain, with the old v1.0 LightGBM model as the ML ranker.
- `minilm_ltrv2` / `e5ft_ltrv2`: `queries_v1_topk20_v1_1_{minilm,e5ft}_ltrv2`.
  The same code with the LTR v2 bundle (sha256 `3990497d68fe`) as the
  ML ranker. Only the `hybrid_ml` rows differ from the runs above.
- `e5ft*` use the fine-tuned e5-small-v2 and a full 315k index built from it.
  **LTR v2 was trained on hybrid candidates from MiniLM and has not been retrained
  for e5**, so with e5 its `semantic_score` and dense-rank features come from a
  different distribution than in training.
- `e5ft_ltrv3`: `queries_v1_topk20_v1_1_e5ft_ltrv3`. LTR v3 bundle (sha256
  `e2ea4bd076f4`), retrained on hybrid e5-ft candidates with the same protocol
  (buckets 2-8, four query tiers, `top_k=100`; see docs/ltr.md). Run once, after
  v3 had been chosen on the bucket-9 held-out set; it was not used to pick
  anything. Same DB copy, BM25 and e5 index as `e5ft_ltrv2`. Its query file hash
  is `0b9aabd48bb2`, the same as the baseline, now that `.gitattributes` keeps
  `eval/*.jsonl` LF.

All new runs used the same DB copy (with `recipe_metadata`) and the same BM25
content (main JSON converted to `.npz`), and ran one at a time on an otherwise
idle machine. The baseline ran next to other CPU-heavy jobs, so its latency is not
a fair comparison: its p95 is inflated, and its p50 includes the old pure-Python BM25.

The query file hash in the new result files (`eb051d0d17f5`) differs from the
baseline (`0b9aabd48bb2`) only because of CRLF line endings in this checkout
(`core.autocrlf=true`). With line endings normalised it is `0b9aabd48bb2`.

## Layer (b): final pipeline output, 1000 queries (buckets 0-1)

| Config | Run | recall@10 | mrr@10 | ndcg@10 | hit@1 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| BM25 (lexical mode) | baseline | 0.655 [0.626, 0.685] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 309.9 | 1105.7 |
| BM25 (lexical mode) | minilm | 0.655 [0.626, 0.685] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 12.0 | 19.0 |
| Dense MiniLM | baseline | 0.331 [0.304, 0.361] | 0.213 [0.191, 0.237] | 0.241 [0.219, 0.266] | 0.160 [0.138, 0.183] | 77.1 | 1401.4 |
| Dense MiniLM | minilm | 0.331 [0.304, 0.361] | 0.213 [0.191, 0.237] | 0.241 [0.219, 0.266] | 0.160 [0.138, 0.183] | 41.8 | 60.7 |
| Dense e5-small-ft | e5ft | 0.717 [0.689, 0.745] | 0.551 [0.523, 0.579] | 0.592 [0.565, 0.619] | 0.464 [0.432, 0.494] | 52.2 | 65.1 |
| Hybrid RRF, MiniLM | baseline | 0.603 [0.573, 0.634] | 0.364 [0.339, 0.391] | 0.421 [0.396, 0.448] | 0.265 [0.239, 0.293] | 419.0 | 1907.9 |
| Hybrid RRF, MiniLM | minilm | 0.604 [0.574, 0.635] | 0.364 [0.339, 0.391] | 0.421 [0.397, 0.448] | 0.265 [0.239, 0.293] | 48.9 | 70.1 |
| Hybrid RRF, e5-small-ft | e5ft | 0.729 [0.701, 0.757] | 0.555 [0.527, 0.584] | 0.597 [0.570, 0.624] | 0.475 [0.443, 0.507] | 59.2 | 74.8 |
| Hybrid MiniLM + heuristic | minilm | 0.638 [0.609, 0.668] | 0.359 [0.336, 0.382] | 0.426 [0.403, 0.450] | 0.227 [0.201, 0.253] | 48.9 | 70.1 |
| Hybrid e5-small-ft + heuristic | e5ft | 0.722 [0.694, 0.751] | 0.418 [0.394, 0.443] | 0.491 [0.468, 0.516] | 0.276 [0.249, 0.304] | 59.2 | 74.8 |
| Hybrid MiniLM + old LightGBM (app default on main) | baseline | 0.599 [0.570, 0.630] | 0.331 [0.307, 0.356] | 0.395 [0.370, 0.419] | 0.221 [0.195, 0.247] | 412.7 | 2227.9 |
| Hybrid MiniLM + old LightGBM | minilm | 0.601 [0.572, 0.632] | 0.331 [0.307, 0.355] | 0.395 [0.371, 0.419] | 0.221 [0.195, 0.247] | 58.8 | 83.9 |
| Hybrid e5-small-ft + old LightGBM | e5ft | 0.724 [0.696, 0.753] | 0.523 [0.497, 0.551] | 0.571 [0.546, 0.598] | 0.430 [0.400, 0.462] | 68.8 | 83.6 |
| Hybrid MiniLM + LTR v2 | minilm_ltrv2 | 0.651 [0.622, 0.681] | 0.590 [0.562, 0.621] | 0.606 [0.577, 0.635] | 0.553 [0.521, 0.585] | 45.4 | 70.5 |
| Hybrid e5-small-ft + LTR v2 (not retrained for e5) | e5ft_ltrv2 | 0.755 [0.728, 0.782] | 0.648 [0.621, 0.677] | 0.675 [0.649, 0.702] | 0.589 [0.559, 0.620] | 58.0 | 73.6 |
| Hybrid e5-small-ft + LTR v3 (retrained on e5, new app default) | e5ft_ltrv3 | 0.756 [0.730, 0.783] | 0.644 [0.616, 0.672] | 0.671 [0.645, 0.698] | 0.585 [0.554, 0.617] | 62.3 | 77.2 |

## Layer (a): retriever candidates

| Config | Run | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| BM25 top-100 | minilm | 0.645 [0.615, 0.675] | 0.855 [0.833, 0.877] | 0.456 [0.430, 0.483] | 0.501 [0.476, 0.527] | 7.2 | 14.0 |
| Dense MiniLM top-100 | minilm | 0.291 [0.264, 0.320] | 0.535 [0.504, 0.565] | 0.163 [0.144, 0.184] | 0.193 [0.173, 0.215] | 41.2 | 60.2 |
| Dense e5-small-ft top-100 | e5ft | 0.722 [0.694, 0.750] | 0.907 [0.888, 0.925] | 0.529 [0.502, 0.556] | 0.575 [0.549, 0.601] | 50.4 | 62.7 |
| Hybrid RRF MiniLM top-100 | minilm | 0.478 [0.448, 0.507] | 0.821 [0.798, 0.845] | 0.284 [0.261, 0.310] | 0.330 [0.306, 0.355] | 50.1 | 69.4 |
| Hybrid RRF e5-small-ft top-100 | e5ft | 0.729 [0.702, 0.756] | 0.903 [0.884, 0.922] | 0.535 [0.508, 0.562] | 0.582 [0.555, 0.607] | 58.9 | 72.5 |

## Paired differences (A - B, 95% paired bootstrap CI)

| A vs B | ΔRecall@10 | ΔMRR@10 | ΔnDCG@10 | ΔHit@1 |
|---|---|---|---|---|
| (a) dense e5-ft vs dense MiniLM | +0.431 [+0.398, +0.462] | +0.366 [+0.338, +0.393] | +0.382 [+0.355, +0.408] | +0.327 [+0.296, +0.359] |
| (b) dense e5-ft vs dense MiniLM | +0.386 [+0.354, +0.417] | +0.338 [+0.310, +0.365] | +0.350 [+0.323, +0.376] | +0.304 [+0.273, +0.336] |
| (a) hybrid e5-ft vs hybrid MiniLM | +0.251 [+0.222, +0.280] | +0.251 [+0.227, +0.276] | +0.252 [+0.229, +0.275] | +0.240 [+0.211, +0.270] |
| (b) hybrid e5-ft vs hybrid MiniLM | +0.125 [+0.103, +0.148] | +0.191 [+0.171, +0.211] | +0.175 [+0.158, +0.192] | +0.210 [+0.183, +0.239] |
| (b) hybrid e5-ft vs BM25 | +0.074 [+0.058, +0.093] | +0.073 [+0.059, +0.087] | +0.073 [+0.061, +0.085] | +0.082 [+0.061, +0.104] |
| (b) e5-ft: heuristic vs RRF order | -0.007 [-0.020, +0.007] | -0.137 [-0.158, -0.115] | -0.105 [-0.122, -0.088] | -0.199 [-0.231, -0.166] |
| (b) e5-ft: old LightGBM vs RRF order | -0.005 [-0.017, +0.008] | -0.032 [-0.048, -0.015] | -0.025 [-0.039, -0.012] | -0.045 [-0.070, -0.020] |
| (b) e5-ft + old LGBM vs app default on main | +0.125 [+0.101, +0.149] | +0.192 [+0.171, +0.215] | +0.177 [+0.158, +0.196] | +0.209 [+0.181, +0.238] |
| (b) MiniLM: LTR v2 vs RRF order | +0.047 [+0.034, +0.061] | +0.226 [+0.206, +0.248] | +0.184 [+0.168, +0.203] | +0.288 [+0.259, +0.317] |
| (b) MiniLM: LTR v2 vs old LightGBM | +0.050 [+0.036, +0.064] | +0.260 [+0.238, +0.282] | +0.211 [+0.192, +0.229] | +0.332 [+0.303, +0.363] |
| (b) e5-ft: LTR v2 vs RRF order | +0.026 [+0.016, +0.037] | +0.094 [+0.079, +0.110] | +0.078 [+0.066, +0.091] | +0.114 [+0.093, +0.136] |
| (b) e5-ft: LTR v2 vs old LightGBM | +0.031 [+0.020, +0.042] | +0.125 [+0.108, +0.144] | +0.103 [+0.089, +0.118] | +0.159 [+0.133, +0.185] |
| (b) e5-ft + LTR v2 vs app default on main | +0.156 [+0.133, +0.180] | +0.318 [+0.294, +0.342] | +0.280 [+0.259, +0.301] | +0.368 [+0.339, +0.399] |
| (b) LTR v2: e5-ft vs MiniLM retrieval | +0.104 [+0.084, +0.126] | +0.058 [+0.043, +0.074] | +0.069 [+0.054, +0.086] | +0.036 [+0.019, +0.052] |
| (b) e5-ft + LTR v2 vs BM25 | +0.100 [+0.081, +0.120] | +0.167 [+0.148, +0.186] | +0.151 [+0.134, +0.168] | +0.196 [+0.171, +0.222] |
| (b) e5-ft: LTR v3 vs RRF order | +0.027 [+0.016, +0.039] | +0.089 [+0.074, +0.105] | +0.074 [+0.062, +0.088] | +0.110 [+0.089, +0.132] |
| (b) e5-ft: LTR v3 vs LTR v2 | +0.001 [-0.002, +0.004] | -0.005 [-0.014, +0.004] | -0.004 [-0.011, +0.003] | -0.004 [-0.018, +0.010] |
| (b) e5-ft + LTR v3 vs app default on main | +0.157 [+0.133, +0.181] | +0.313 [+0.289, +0.337] | +0.276 [+0.255, +0.297] | +0.364 [+0.333, +0.395] |

## Reading

- With MiniLM and the old model, the v1.1 code reproduces the baseline
  quality. Every layer (b) metric matches to three decimals. End-to-end latency
  drops 8-9x at p50 (hybrid 419 -> 49 ms, lexical 310 -> 12 ms).
- Fine-tuned e5 is the biggest retrieval gain. Dense goes from 0.213 to 0.551
  MRR@10, hybrid RRF from 0.364 to 0.555, and hybrid now beats BM25.
- LTR v2 is the first re-ranker that helps on the official pool:
  - With MiniLM: +0.226 MRR@10 over RRF order (0.364 -> 0.590).
  - With e5: +0.094 MRR@10 over RRF order (0.555 -> 0.648), even without retraining.
  - The best configuration is hybrid e5-ft + LTR v2: MRR@10 0.648, Hit@1 0.589,
    Recall@10 0.755, p50 58 ms. Against the `main` app default that is +0.318 MRR@10.
- The heuristic ranker and the old LightGBM model still lose to plain RRF order.
- Caveats:
  - Both the e5 fine-tune and the LTR training queries were generated the same
    way as this eval set (subsets of a recipe's ingredients, from buckets 2-9),
    so the query style is in-distribution. The gain on real pantry lists is likely
    smaller.
  - The LTR card notes that `query_coverage` carries most of its gain, a feature
    the synthetic protocol favours.
  - LTR v2 with e5 is a mismatched pairing. Retraining it on e5 candidates is the
    obvious next step and may add more.
- Retraining on e5 (LTR v3) did not add anything measurable:
  - On the official pool v3 and v2 tie: -0.005 MRR@10 [-0.014, +0.004].
  - On the bucket-9 held-out sets the differences were also within noise (-0.001
    to +0.013 nDCG@10, every CI containing 0).
  - v2 gets most of its gain from features that do not depend on the dense model
    (`query_coverage` 0.54, `bm25_rel` 0.17, `matched_count` 0.13). So the shift
    in the dense features cost it little. v3 leans more on the dense gap
    (`dense_gap` 0.24) but ends up at the same quality.
  - v3 stays the default because it was chosen before this run on the held-out
    sets, where it was level or slightly ahead. It also records the embedding
    model it was trained with, so a mismatch gets logged.
  - Picking v2 now, from this run, would mean tuning on the eval pool.
  - The p50 latency difference (62 vs 58 ms) was not investigated. v3 has fewer
    trees than v2 (124 vs 210), so the model itself is an unlikely cause.
