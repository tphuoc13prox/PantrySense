# Learning-to-rank re-ranker

LightGBM LambdaMART model that re-orders the candidates produced by the hybrid
retriever (BM25 + FAISS, fused with RRF) after the coverage threshold. It
replaces the previous model, which was fit on 16 hand-written examples with
made-up feature values.

Code: `app/backend/ranking/` (features, model loading, metrics),
`scripts/build_ltr_dataset.py`, `scripts/train_ranker.py`,
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
- Each query runs through `RecipeSearchService.search` in hybrid mode. A
  recording ranker in place of `MLRecipeRanker` stores exactly what `rank()`
  receives: the `RecipeSummary` list, `details_map` and the coverage threshold.
  Features are then computed by the same `build_feature_matrix` used at serving
  time, so there is no separate training-time feature code.
- 4000 training queries (400k candidate rows at `top_k=100`). The held-out set
  is 1000 queries, recorded twice: once at `top_k=100` and once at `top_k=20`,
  which is the serving default.

## Labels

2 for the source recipe, 1 for a near-duplicate (same normalised title, or an
identical normalised ingredient set), 0 for everything else. Labels never use
model features such as coverage. Titles are unique in the DB, so near-duplicates
are rare and the task is close to "find the one source recipe".

## Retriever recall ceiling

The ranker can only reorder what it receives. The figures below are the share
of queries whose source recipe reaches it.

| | top_k=100 train | top_k=100 held-out | top_k=20 held-out |
|---|---|---|---|
| source recipe among candidates | 83.3% | 82.1% | 68.1% |
| ... and above the coverage threshold | 81.3% | 79.3% | 66.5% |

At `top_k=100` the source recipe's median RRF rank is 7 and its 90th
percentile is 47. Sparse queries suffer most at `top_k=20`, where only 55% of
sources reach the ranker.

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
with no relevant candidate are dropped, leaving 3254 of 4000 queries and 131k
rows. Validation is 5-fold CV split by query, with early stopping on nDCG@10
(patience 50). The grid covers `num_leaves` {15, 31, 63}, `learning_rate`
{0.03, 0.1} and `min_child_samples` {20, 100}. The best setting was 15 leaves,
lr 0.03 and min_child 20, refit on all training queries for 202 trees (the mean
of the best iterations). Seeds are fixed, and LightGBM runs with
`deterministic=True`.

The model file is a dict bundle (`format: pantrysense-ltr`) holding the
estimator and its feature names. `MLRecipeRanker` still accepts a bare legacy
9-feature estimator, and falls back to the coverage heuristic when the file is
missing or unusable.

## Results

Metrics: nDCG@10 with gain 2^label - 1, MRR@10, and Hit@1/Hit@10 (counting any
label above 0). "Conditional" means the queries whose relevant recipe reached
the ranker. "End to end" means all queries, where a miss scores 0.

Baselines:
- heuristic: the previous fallback, ordering by coverage, then matched count,
  then missing count
- RRF: the retriever order
- old model: the previous `ranker_model.joblib`

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

**Feature importance** (gain share in the final model): query_coverage 0.65,
bm25_rel 0.14, missing_nonstaple_count 0.05, matched_count 0.05, bm25_rank 0.05,
and everything else below 0.02. The cooking-time-missing flag and difficulty
get no splits.

## Limitations

- **The labels are synthetic, and they favour the main feature.** Every query is
  a subset of one recipe, so that recipe uses all of the query's ingredients.
  Within the candidate list, `query_coverage` and `matched_count` therefore pick
  it out almost directly. The model mostly learns "prefer the recipe that uses
  everything I listed, then the one with the strongest BM25 match". That is a
  reasonable preference for a pantry app. But the size of the gain over the
  heuristic, which sorts by recipe coverage and so favours short recipes,
  probably overstates the gain for real pantry lists, which are not subsets of
  one recipe. Even the distractor in the noisy tier barely changes this: it
  rarely matches any candidate, so it is invisible to the query-size estimate.
- Most groups add almost nothing on top of the ingredient-match features. The
  differences between ablations are about 0.005, which is near noise. Removing
  the list-relative features scored slightly higher in CV, but I kept them
  because the held-out results were already fixed and switching on that basis
  would be tuning on held-out.
- There is one relevant recipe per query, so "relevant" means "the recipe the
  query was built from". Other good answers count as negatives, and this pushes
  every absolute number down.
- 21-34% of queries never reach the ranker (retrieval recall plus the coverage
  threshold). The ranker cannot fix those.
- No online or user signal exists. The final numbers on the official eval pool
  (buckets 0-1) are left to the eval harness.
