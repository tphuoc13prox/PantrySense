# Learning-to-rank re-ranker

LightGBM LambdaMART model that re-orders the candidates produced by the hybrid
retriever (BM25 + FAISS, fused with RRF) after the coverage threshold. It
replaces the previous model, which was fit on 16 hand-written examples with
made-up feature values.

Code: `app/backend/ranking/` (features, model loading, metrics),
`scripts/build_ltr_dataset.py`, `scripts/train_ranker.py`,
`scripts/evaluate_ranking.py`.

## Default configuration

The app picks the best measured setup when its files are present and falls
back to something that runs on a fresh clone otherwise. Startup logs which one
is in use.

| Part | Used when present | Fallback |
|---|---|---|
| Dense model | `data/models/e5-small-pantry` (fine-tuned e5-small-v2) with `data/recipe_vectors_e5ft.index`, its `.meta.json` and `data/recipe_vector_ids_e5ft.json` | `all-MiniLM-L6-v2` with `data/recipe_vectors.index` |
| Ranker | LTR bundle at `data/ranker_model.joblib` | hybrid RRF order |

- `PANTRYSENSE_EMBEDDING_MODEL`, `PANTRYSENSE_VECTOR_INDEX_PATH`,
  `PANTRYSENSE_VECTOR_IDS_PATH` and `PANTRYSENSE_RANKER_MODEL_PATH` override the
  defaults. If only the model is set, the index paths follow it: the e5 paths
  for the fine-tuned folder, the MiniLM paths for anything else.
- Without a bundle the ranker keeps the retriever's order rather than the
  coverage heuristic, because the heuristic loses to RRF on the official eval
  (MRR@10 0.418 vs 0.555 with e5). Old bare LightGBM files (not a
  `pantrysense-ltr` bundle) are ignored for the same reason (0.523 vs 0.555).
  `PANTRYSENSE_RANKING_MODE=heuristic` still selects the heuristic.
- A bundle records the embedding model its training candidates came from.
  Loading it with a different model logs a warning, because the dense-score
  features then come from another distribution.
- Startup and the setup pipeline no longer train the ranker. Its dataset takes
  about 10k hybrid searches, so the ranker is built offline (below).

The model files are gitignored. `python tasks.py fetch-models` installs the
fine-tuned model and the v3 bundle from the GitHub release, and the next setup
or app start builds the e5 index (see README.md). To build everything yourself
instead, `python tasks.py reproduce` runs the whole chain (docs/reproduce.md).
By hand:

1. Fine-tuned model: train it with `scripts/finetune_embedder.py train --base
   intfloat/e5-small-v2 ...` (docs/embedding.md, about 9 min on an RTX 4050).
2. Index: `python scripts/build_vector_index.py` builds the e5 index once the
   model folder is in `data/` (35 min on CPU for 315k recipes). A copied index
   also works: the metadata check accepts a model folder with the same name in
   another location.
3. Ranker: build the dataset with the e5 setup active, then train:

       python scripts/build_ltr_dataset.py --out-dir data/e5 --top-k 100
       python scripts/build_ltr_dataset.py --out-dir data/e5 --top-k 100 --tiers pantry --tag pantry --n-train 1333 --n-heldout 600 --seed 11 --exclude-queries data/e5/ltr_queries.json
       python scripts/train_ranker.py --dataset data/e5/ltr_train_k100.parquet data/e5/ltr_train_pantry_k100.parquet --output data/ranker_model.joblib

   The two dataset builds take about 15 min on CPU, and training takes 3 min.
   Add `--top-k 20 --splits heldout` runs for the held-out sets used by
   `scripts/evaluate_ranking.py`.

## Data

- Source recipes come only from the training pool:
  `bucket = int(md5(str(recipe_id)).hexdigest(), 16) % 10`, buckets 2-8 for
  training, bucket 9 held out. Buckets 0-1 (official eval pool) are never
  sampled, and a source is skipped if its duplicate group touches them.
  Candidates come from the full corpus, as in production.
- Query: a random 40-70% subset of the recipe's ingredients after removing
  staples (ingredients in at least 9% of recipes: salt, butter, egg, onion,
  sugar, water, garlic clove, olive oil, milk, flour, pepper). There are three
  tiers in equal parts: standard (55-70%), sparse (40-55%), and noisy (40-70%
  plus one unrelated ingredient). This is the same protocol as the eval
  harness. Seeded (seed 7).
- A fourth tier, "pantry", is a query that is *not* a subset of the source:
  40-60% of the source's non-staple ingredients plus 2-4 ingredients from
  other recipes that share a query ingredient. The extras are chosen with
  df >= 100, are not staples, and share no token with the source's
  ingredients. Because they co-occur with the query, they usually match some
  candidates and enter the query-size estimate. On average a query has 3.1
  source ingredients and 3.0 extras. Sources are disjoint from the first three
  tiers (seed 11).
- Each query runs through `RecipeSearchService.search` in hybrid mode. A
  recording ranker in place of `MLRecipeRanker` stores exactly what `rank()`
  receives: the `RecipeSummary` list, `details_map` and the coverage threshold.
  Features are then computed by the same `build_feature_matrix` used at serving
  time, so there is no separate training-time feature code.
- Training uses 4000 standard/sparse/noisy queries plus 1333 pantry queries
  (25% of the total), all at `top_k=100`. The held-out set is 1000
  standard/sparse/noisy queries plus 600 pantry queries, each recorded at both
  `top_k=100` and `top_k=20` (the serving default).

## Labels

2 for the source recipe, 1 for a near-duplicate (same normalised title, or an
identical normalised ingredient set), 0 for everything else. Labels never use
model features such as coverage. Titles are unique in the DB, so near-duplicates
are rare and the task is close to "find the one source recipe".

## Retriever recall ceiling

The ranker can only reorder what it receives. The figures below are the share
of queries whose source recipe reaches it.

| | train k100 | held-out k100 | held-out k20 |
|---|---|---|---|
| standard/sparse/noisy: source among candidates | 83.3% | 82.1% | 68.1% |
| ... and above the coverage threshold | 81.3% | 79.3% | 66.5% |
| pantry: source among candidates | 50.3% | 51.5% | 33.2% |
| ... and above the coverage threshold | 48.5% | 49.2% | 32.5% |

At `top_k=100` the source recipe's median RRF rank is 7 for the subset tiers
and 19 for pantry. Extra ingredients pull retrieval toward other recipes, so
for a realistic pantry list the retriever, not the ranker, is the main
bottleneck.

## Features

23 features in six groups (`FEATURE_GROUPS` in `features.py`). All of them are
computed over the candidate list of one query.

- retrieval scores: semantic, BM25, RRF, and flags for presence in the dense and
  BM25 lists
- list-relative: dense and BM25 rank among candidates (1000 if absent), fused
  rank, BM25 divided by the list max, gap to the best dense score, RRF divided
  by the list max
- ingredient match: coverage, matched, missing, required, and missing excluding
  staples
- query level: estimated query size, and the share of the query a recipe uses
  (`query_coverage`)
- title: token overlap between the title and the query, as a ratio and a count
- metadata: cooking time (defaults to the corpus median of 20 when missing, with
  a flag), difficulty

`rank()` does not receive the raw query, so the query is estimated as the union
of matched ingredients across candidates. Train and serve use the same estimate.

## Training

The model is trained only on candidates above the coverage threshold. Queries
with no relevant candidate are dropped. Validation is 5-fold CV split by query,
with early stopping on nDCG@10 (patience 50). Seeds are fixed, and LightGBM runs
with `deterministic=True`.

1. **v1** was trained on the 4000 subset-tier queries (3254 usable, 131k rows).
   The grid covers `num_leaves` {15, 31, 63}, `learning_rate` {0.03, 0.1} and
   `min_child_samples` {20, 100}. The best setting was 15 leaves, lr 0.03 and
   min_child 20, refit on all queries for 202 trees (the mean of the best
   iterations).
2. **v2** adds the 1333 pantry training queries and keeps the v1
   parameters without a new grid (`--no-grid`). It has 3903 usable queries,
   150k rows and 210 trees. Only 649 pantry queries are usable because retrieval
   misses half of them, so pantry makes up 17% of the effective training set.
3. **v3 (shipped)** uses the same queries and protocol as v2, with candidates
   from hybrid retrieval on the fine-tuned e5 index. Training ran the full grid
   again. All 12 settings fall within 0.785-0.793 CV nDCG@10; the best is 15
   leaves, lr 0.1 and min_child 100, at 124 trees. It has 4254 usable queries
   and 179k rows (about 710 of the queries are pantry). See "v3: retrained on fine-tuned
   e5 candidates" below.

Commands:

    python scripts/train_ranker.py                     # v1, with grid
    python scripts/train_ranker.py --no-grid --dataset data/ltr_train_k100.parquet data/ltr_train_pantry_k100.parquet

`predict` runs with `num_threads=1`. A 20-row predict otherwise wakes an OpenMP
pool on every core, and those threads slowed the next query's retrieval (hybrid
p50 242 ms vs 70 ms, measured with the search latency benchmark).

The model file is a dict bundle (`format: pantrysense-ltr`) holding the
estimator, its feature names and (from v3) the embedding model of the training
candidates. When the file is missing or unusable, including old bare 9-feature
estimators, `MLRecipeRanker` keeps the retriever's order (see "Default
configuration").

## Results

Metrics: nDCG@10 with gain 2^label - 1, MRR@10, and Hit@1/Hit@10 (counting any
label above 0). "Conditional" means the queries whose relevant recipe reached
the ranker. "End to end" means all queries, where a miss scores 0.

Baselines:
- heuristic: the previous fallback, ordering by coverage, then matched count,
  then missing count
- RRF: the retriever order
- old model: the previous `ranker_model.joblib`

The tables in this section are for v1 on the subset tiers. See the next section
for the pantry tier and v2.

**Cross-validation, out of fold** (3254 training queries). The same folds were
used for early stopping, so this is slightly optimistic.

| Ranking | nDCG@10 | MRR@10 | Hit@1 | Hit@10 |
|---|---|---|---|---|
| RRF order | 0.507 | 0.438 | 0.315 | 0.734 |
| Heuristic | 0.427 | 0.338 | 0.194 | 0.718 |
| LambdaMART | **0.852** | **0.818** | **0.743** | **0.963** |

**Held-out, bucket 9** (never used for fitting or tuning)

| top_k=100 | cond. nDCG@10 | cond. MRR@10 | cond. Hit@1 | e2e nDCG@10 | e2e Hit@1 |
|---|---|---|---|---|---|
| RRF order | 0.532 | 0.464 | 0.351 | 0.422 | 0.278 |
| Heuristic | 0.458 | 0.365 | 0.227 | 0.363 | 0.180 |
| Old model | 0.458 | 0.385 | 0.264 | 0.364 | 0.209 |
| LambdaMART | **0.866** | **0.834** | **0.765** | **0.687** | **0.607** |

| top_k=20 (serving default) | cond. nDCG@10 | cond. MRR@10 | cond. Hit@1 | e2e nDCG@10 | e2e Hit@1 |
|---|---|---|---|---|---|
| RRF order | 0.656 | 0.573 | 0.433 | 0.437 | 0.289 |
| Heuristic | 0.669 | 0.572 | 0.379 | 0.446 | 0.253 |
| Old model | 0.620 | 0.535 | 0.381 | 0.414 | 0.254 |
| LambdaMART | **0.937** | **0.916** | **0.862** | **0.625** | **0.575** |

These are paired bootstrap 95% intervals for the nDCG@10 difference from the
heuristic, on conditional queries:

| | top_k=100 | top_k=20 |
|---|---|---|
| LambdaMART | +0.408 [+0.383, +0.434] | +0.268 [+0.245, +0.289] |
| Old model | +0.001 [-0.021, +0.022] | -0.049 [-0.071, -0.029] |

The model was trained on 100-candidate lists and still transfers to 20-candidate
lists. Retrieval caps the end-to-end numbers at `top_k=20`, and raising `top_k`
would lift them, at a latency cost not measured here.

**Ablation** (5-fold CV nDCG@10, dropping one group with the same parameters)

| Variant | nDCG@10 |
|---|---|
| all features | 0.852 |
| without retrieval scores | 0.853 |
| without list-relative | 0.858 |
| without ingredient match | 0.805 |
| without query level | 0.850 |
| without title | 0.852 |
| without metadata | 0.851 |
| legacy 9 features only | 0.816 |

**Feature importance** (gain share in v1): query_coverage 0.65, bm25_rel 0.14,
missing_nonstaple_count 0.05, matched_count 0.05, bm25_rank 0.05, and everything
else below 0.02. The cooking-time-missing flag and difficulty get no splits.

## Pantry tier: how much of the gain is the subset artifact

In the subset tiers the source recipe has `query_coverage` = 1 in 81% (k100) to
88% (k20) of the queries where it reaches the ranker. In the pantry tier this
drops to 1% at k100 and 11% at k20 (smaller lists match fewer of the extras),
with a mean of 0.59. The pantry tier is therefore the honest test of whether the
model ranks well when "uses everything I typed" no longer identifies the answer.

**Held-out pantry, conditional** (295 queries at k100, 195 at k20)

| Ranking | k100 nDCG@10 | k100 MRR@10 | k100 Hit@1 | k20 nDCG@10 | k20 MRR@10 | k20 Hit@1 |
|---|---|---|---|---|---|---|
| RRF order | 0.400 | 0.318 | 0.193 | 0.631 | 0.527 | 0.354 |
| Heuristic | 0.390 | 0.277 | 0.112 | 0.655 | 0.545 | 0.313 |
| v1 (subset tiers only) | 0.703 | 0.641 | 0.522 | 0.884 | 0.847 | 0.759 |
| v2 (+ pantry) | **0.743** | **0.684** | **0.569** | **0.891** | **0.855** | 0.754 |

End-to-end nDCG@10 on all 600 pantry queries is 0.196 for RRF, 0.192 for the
heuristic, 0.346 for v1 and 0.365 for v2 at k100. At k20 the figures are 0.205,
0.213, 0.287 and 0.290.

**Effect of adding pantry training data** (paired bootstrap 95% CI, v2 - v1,
conditional nDCG@10)

| Held-out set | v2 - v1 |
|---|---|
| subset tiers k100 | -0.004 [-0.009, +0.002] |
| subset tiers k20 | -0.002 [-0.007, +0.003] |
| pantry k100 | **+0.039 [+0.019, +0.060]** |
| pantry k20 | +0.007 [-0.011, +0.025] |

v2 on the subset tiers: k100 conditional nDCG@10 0.862, MRR 0.830, Hit@1 0.760;
k20 0.935, 0.914, 0.858. v2 minus heuristic is +0.405 [+0.381, +0.430] at k100
and +0.266 [+0.244, +0.287] at k20.

Findings:

- With the artifact removed, the learned ranker still beats both baselines by a
  wide margin. On pantry k100 v1 already scores +0.31 nDCG@10 over the
  heuristic, and v2 scores +0.35 [+0.31, +0.39]. The subset artifact inflated
  the subset-tier numbers (0.866 to 0.703 for v1), but it does not explain the
  gain.
- Pantry training data helps on pantry queries at k100 and costs nothing
  measurable on the subset tiers, so v2 is the shipped model. At k20 the
  difference is within noise: lists are short, and only 195 queries reach the
  ranker.
- In v2 the importance of `query_coverage` falls from 0.65 to 0.54 of gain,
  while `matched_count` (0.13) and `bm25_rel` (0.17) take a larger share.
- 5-fold CV for v2 on the mixed training set: nDCG@10 0.833, against 0.416 for
  the heuristic and 0.488 for RRF. Ablations follow the v1 pattern: removing
  ingredient match drops to 0.787, the legacy 9 features alone reach 0.799, and
  removing list-relative gives 0.839.

## v3: retrained on fine-tuned e5 candidates

With e5-ft as the default dense model, the datasets were rebuilt from scratch
in `data/e5/`. They use the same queries (same seeds and tiers), but the
candidates come from hybrid e5-ft retrieval. Recall is much higher than with
MiniLM:

| Source recipe reaches the ranker | MiniLM | e5-ft |
|---|---|---|
| subset tiers, held-out k100 | 79.3% | 88.1% |
| subset tiers, held-out k20 | 66.5% | 76.1% |
| pantry, held-out k100 | 49.2% | 54.3% |
| pantry, held-out k20 | 32.5% | 38.7% |

With e5 the source recipe's median RRF rank is 1 in the subset tiers (7 with
MiniLM) and 8 in pantry.

**Held-out (bucket 9) on e5-ft candidates**, conditional nDCG@10. v2 is the
MiniLM-trained model used unchanged.

| Held-out set | queries | RRF | Heuristic | v2 | v3 |
|---|---|---|---|---|---|
| subset tiers k100 | 882 | 0.704 | 0.448 | 0.806 | 0.805 |
| subset tiers k20 | 761 | 0.807 | 0.664 | 0.892 | 0.897 |
| pantry k100 | 326 | 0.550 | 0.400 | 0.708 | 0.720 |
| pantry k20 | 232 | 0.736 | 0.674 | 0.858 | 0.872 |

Paired bootstrap 95% CI:

| Held-out set | nDCG@10 v3 - v2 | MRR@10 v3 - v2 | nDCG@10 v3 - RRF |
|---|---|---|---|
| subset tiers k100 | -0.001 [-0.011, +0.008] | -0.001 [-0.012, +0.010] | +0.101 [+0.087, +0.116] |
| subset tiers k20 | +0.005 [-0.002, +0.013] | +0.005 [-0.004, +0.015] | +0.090 [+0.075, +0.104] |
| pantry k100 | +0.012 [-0.006, +0.030] | +0.016 [-0.004, +0.037] | +0.169 [+0.135, +0.201] |
| pantry k20 | +0.013 [-0.005, +0.033] | +0.019 [-0.005, +0.045] | +0.136 [+0.102, +0.170] |

**Official eval** (1000 queries, buckets 0-1, `scripts/run_eval.py`, run once
for the final model, `eval/results/queries_v1_topk20_v1_1_e5ft_ltrv3`), final
output MRR@10:

| Config | MRR@10 | Hit@1 | Recall@10 |
|---|---|---|---|
| hybrid e5-ft, RRF order (fallback without a bundle) | 0.555 | 0.475 | 0.729 |
| hybrid e5-ft + v2 | 0.648 | 0.589 | 0.755 |
| hybrid e5-ft + v3 (default) | 0.644 | 0.585 | 0.756 |

v3 minus v2 is -0.005 MRR@10 [-0.014, +0.004]. v3 minus RRF is +0.089
[+0.074, +0.105].

Findings:

- Retraining on e5 candidates did not measurably change quality. Every v3 - v2
  interval contains 0, on the held-out sets and on the official pool. Point
  estimates are slightly positive on three of four held-out sets and slightly
  negative on the official pool.
- v2 transfers well because most of its gain comes from features that do not
  depend on the dense model. v3 uses the dense signal more (`dense_gap` 0.24 of
  gain, `query_coverage` 0.39) and reaches the same quality.
- v3 is the default. It was chosen on the held-out sets before the official
  run, where it was level or ahead. Its bundle also records the embedding model,
  so pairing it with another one gets logged. Switching to v2 because of the
  official run would be tuning on the eval pool.
- The learned ranker is still clearly worth having over RRF order: +0.09 to
  +0.17 nDCG@10 held-out, and +0.089 MRR@10 on the official pool.
- CV (out of fold, 4254 queries): nDCG@10 0.793 against 0.672 for RRF and 0.422
  for the heuristic. Ablation: removing ingredient match costs the most (0.751),
  and the legacy 9 features alone reach 0.757. The other groups are within
  0.007.

## Limitations

- **The labels are synthetic.** In the subset tiers the source recipe uses the
  whole query, so `query_coverage` nearly identifies it, and the subset-tier
  numbers overstate real quality. Even the noisy tier's distractor rarely
  matches a candidate, so it does not change this. The pantry tier removes the
  artifact: quality drops (0.866 to 0.743 conditional nDCG@10 at k100), but it
  stays far above the baselines. Pantry queries are still synthetic. The extras
  are ingredients that co-occur with the query, not a real user's pantry.
- **The ranker does not see the real query.** `rank()` receives only the
  candidate list, so query features use the union of matched ingredients across
  candidates. A query ingredient that matches no candidate is invisible, and
  query size is underestimated. Passing the raw query in needs a one-line change
  in `RecipeSearchService._process_candidates` (`service.py`), followed by a
  retrain because the features would change. Not done yet.
- **Retrieval is the bottleneck for realistic lists.** Even with e5-ft only 54%
  (k100) and 39% (k20) of pantry sources reach the ranker (49% and 33% with
  MiniLM). Improving recall, through a larger
  `top_k`, a better embedding or prefiltering, matters more than further ranker
  work.
- Most groups add almost nothing on top of the ingredient-match features. The
  differences between ablations are about 0.005, which is near noise. Removing
  the list-relative features scored slightly higher in CV, but I kept them
  because the held-out results were already fixed and switching on that basis
  would be tuning on held-out.
- There is one relevant recipe per query, so "relevant" means "the recipe the
  query was built from". Other good answers count as negatives, and this pushes
  every absolute number down.
- With e5-ft, 12-24% of subset-tier queries and 46-61% of pantry queries never
  reach the ranker (retrieval recall plus the coverage threshold). The ranker
  cannot fix those.
- No online or user signal exists. The official pool (buckets 0-1) was run
  once per model through `scripts/run_eval.py` and never used for choices.
