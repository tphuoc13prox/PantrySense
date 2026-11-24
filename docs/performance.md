# Search performance

Measurements of `RecipeSearchService.search()` before and after moving
filtering ahead of retrieval, precomputing recipe metadata and switching BM25
to a numpy/npz index. All numbers were produced by `scripts/benchmark_latency.py`;
the raw JSON reports are in `docs/benchmarks/`.

## Setup

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H (16 cores / 22 threads) |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop (6 GB), embedder engine `cuda` |
| Disk | Samsung PM9A1 NVMe |
| OS / Python | Windows 11, Python 3.13.12, faiss-cpu 1.14.3, numpy 2.4.6, lightgbm 4.7.0 |
| Corpus | 315,447 recipes, 384-dim all-MiniLM-L6-v2 vectors, `IndexFlatIP` |

Queries: 200 ingredient lists of 2-5 items sampled (seed 42) from recipes
outside eval buckets 0-1, so nothing here was tuned on the eval pool. Each
mode runs 5 warm-up queries, then the 200 timed ones, with no filter, with
`vegan` and with `gluten_free`. Ranking is the default `ml` mode. Each
configuration was run once, so expect a few milliseconds of run-to-run noise.

"Before" is commit `9d5a78a` (main) exported to a clean folder; "after" is this branch.

```
python scripts/benchmark_latency.py load
python scripts/benchmark_latency.py search --queries 200
python scripts/benchmark_latency.py ann --queries 500 --k 40
```

## Search latency and result counts

`results` is the mean number of recipes returned per query; `0 results` is
the share of queries that returned nothing.

| Mode | Filter | p50 before | p50 after | p95 before | p95 after | results before | results after | 0 results before | 0 results after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | none | 303.5 ms | 14.3 ms | 584.7 ms | 21.5 ms | 14.96 | 14.98 | 0.5% | 0.5% |
| lexical | vegan | 349.4 ms | 16.1 ms | 710.7 ms | 22.9 ms | 3.34 | 11.00 | 42.5% | 7.5% |
| lexical | gluten_free | 407.8 ms | 18.8 ms | 755.2 ms | 25.8 ms | 10.81 | 14.49 | 7.5% | 1.0% |
| semantic | none | crash | 103.6 ms | crash | 143.9 ms | - | 13.77 | - | 0.5% |
| semantic | vegan | crash | 63.0 ms | crash | 118.6 ms | - | 10.01 | - | 8.5% |
| semantic | gluten_free | crash | 160.8 ms | crash | 413.1 ms | - | 13.51 | - | 1.0% |
| hybrid | none | 331.9 ms | 242.3 ms | 684.1 ms | 488.4 ms | 14.82 | 14.82 | 0.5% | 0.5% |
| hybrid | vegan | 384.5 ms | 202.1 ms | 718.9 ms | 385.8 ms | 3.75 | 10.96 | 35.5% | 5.0% |
| hybrid | gluten_free | 361.7 ms | 445.6 ms | 623.6 ms | 1197.9 ms | 11.02 | 14.44 | 6.0% | 1.0% |

Notes:

- Semantic mode crashed on main (`AttributeError: 'CandidateMatch' object has
  no attribute 'bm25_score'`). The semantic retriever had its own
  `CandidateMatch` without BM25/RRF fields; there is now a single class in
  `retrieval/candidates.py`.
- With filters, the old code took the global top 20 and then dropped what
  failed the filter, so a vegan query returned nothing 42.5% of the time.
  Filtering now happens inside retrieval, so the top-k is taken from the
  allowed set. The queries that still return nothing under a filter have
  no allowed recipe above the coverage threshold.
- Unfiltered result counts are unchanged; BM25 scores at every rank are
  identical to the old implementation (checked on 200 queries, top 40).
  Only the choice between recipes with exactly equal scores can differ.

### What is left in the hybrid path: LightGBM threads

Timing hybrid search by stage showed the exact FAISS search taking 80-275 ms
inside the service, against about 40 ms when run alone. LightGBM's
`predict()` on the roughly 20 candidates starts its OpenMP pool on every
thread. Those threads keep spinning after the call returns and slow down the
embedding, FAISS and BM25 work of the next query. With
`PANTRYSENSE_RANKING_MODE=heuristic` the hybrid search takes about 52 ms.

The ranker module belongs to the LTR work and is unchanged here. The
benchmark can reproduce the effect of one fix: `--ranker-threads 1` sets
`n_jobs=1` on the loaded model before timing.

| Mode | Filter | p50 (ranker default) | p50 (ranker `n_jobs=1`) | p95 (default) | p95 (`n_jobs=1`) |
|---|---|---:|---:|---:|---:|
| lexical | none | 14.3 ms | 15.5 ms | 21.5 ms | 23.1 ms |
| semantic | vegan | 63.0 ms | 42.9 ms | 118.6 ms | 79.4 ms |
| semantic | gluten_free | 160.8 ms | 65.1 ms | 413.1 ms | 83.7 ms |
| hybrid | none | 242.3 ms | 70.4 ms | 488.4 ms | 108.3 ms |
| hybrid | vegan | 202.1 ms | 55.0 ms | 385.8 ms | 76.1 ms |
| hybrid | gluten_free | 445.6 ms | 85.1 ms | 1197.9 ms | 102.8 ms |

Recommendation for `ranking/ml_ranker.py`: call
`self._model.predict(feature_matrix, num_threads=1)`, or set `n_jobs=1` when
the model is loaded. Prediction on 20 rows gains nothing from threads.
Semantic/none in the `n_jobs=1` run read 171 ms p50, slower than every other
semantic run, so it is treated as noise from the first queries after the mode
started and is left out of the table.

## Index load time and memory

Each component was loaded in a fresh process; RAM is the RSS increase.

| Component | Load before | Load after | RAM before | RAM after |
|---|---:|---:|---:|---:|
| BM25 (JSON 65 MB -> npz 33 MB) | 2.76 s | 0.26 s * | 529 MB | 93 MB |
| FAISS `IndexFlatIP` (unchanged) | 0.48 s | 2.88 s ** | 499 MB | 500 MB |
| Embedder (unchanged) | 24.3 s | 45.6 s ** | 1070 MB | 1069 MB |
| LightGBM ranker (unchanged) | 2.0 s | 3.2 s ** | 98 MB | 97 MB |

\* 1.90 s in the subprocess run, which includes importing numpy and the
retrieval package; 0.26 s is the `load()` call itself in a warm process.
\** Code unchanged. The difference is OS file cache and model start-up
noise, as the "after" load run was the first run to touch those files.

Process RSS after loading all three modes: 3,160 MB before, 1,770 MB after.
Lexical and hybrid modes used to load separate copies of the BM25 index;
they now share one store through `get_search_components()`.

The filter index (`MetadataIndex`) loads on the first filtered search: 3.6 s
and 9.5 MB of numpy arrays for 315k recipes, mostly spent reading the joined
rows from SQLite. After that a filter mask takes 1-2.5 ms to build.

## Precomputed metadata

`scripts/migrate_recipe_metadata.py` fills the `recipe_metadata` table using
`classify_recipe_dietary`, `detect_recipe_allergens` and
`calculate_recipe_nutrition`. On the full corpus it took 51 s with 20 worker
processes and added 88 MB to the database. A fingerprint of the estimator
sources is stored with the rows; a stale table is logged at load time.

## FAISS: exact vs approximate

All 315,447 vectors, 500 queries, k = 40, 22 threads, each query searched
alone. Recall is measured against the exact `IndexFlatIP` results.

| Index | Params | recall@40 | p50 | p95 | Build |
|---|---|---:|---:|---:|---:|
| Flat (exact) | - | 1.000 | 30.0 ms | 47.9 ms | - |
| HNSW32 | efSearch=32 | 0.843 | 0.41 ms | 1.35 ms | 28.5 s |
| HNSW32 | efSearch=64 | 0.921 | 0.61 ms | 0.89 ms | 28.5 s |
| HNSW32 | efSearch=128 | 0.965 | 0.95 ms | 1.33 ms | 28.5 s |
| HNSW32 | efSearch=256 | 0.986 | 1.76 ms | 5.19 ms | 28.5 s |
| IVF1024,Flat | nprobe=8 | 0.707 | 0.61 ms | 0.93 ms | 4.6 s |
| IVF1024,Flat | nprobe=16 | 0.806 | 1.02 ms | 2.72 ms | 4.6 s |
| IVF1024,Flat | nprobe=32 | 0.880 | 1.95 ms | 4.25 ms | 4.6 s |
| IVF1024,Flat | nprobe=64 | 0.935 | 3.99 ms | 17.4 ms | 4.6 s |

With the vegan filter (20.9% of recipes allowed) applied through the same
`IDSelectorBitmap`, recall is measured against exact filtered search:

| Index | Params | recall@40 | p50 | p95 |
|---|---|---:|---:|---:|
| HNSW32 + selector | efSearch=128 | 0.737 | 3.9 ms | 5.7 ms |
| HNSW32 + selector | efSearch=256 | 0.822 | 7.0 ms | 12.7 ms |
| HNSW32 + selector | efSearch=512 | 0.891 | 13.7 ms | 23.3 ms |

Decision: keep `IndexFlatIP` as the default. HNSW with efSearch=128 would
save about 30 ms per query, but filtered recall falls to 0.74, which
undoes the point of filtering before retrieval. Once the ranker threading is
fixed the whole hybrid query is about 70 ms, so the flat search is
acceptable at this corpus size. To try HNSW anyway:

```
python scripts/build_vector_index.py --convert-faiss-hnsw data/recipe_vectors.index data/recipe_vectors_hnsw.index --ef-search 128
```

and point `PANTRYSENSE_VECTOR_INDEX_PATH` at the new file (the ids file is
unchanged). `FaissVectorStore` loads any FAISS index type and passes the
selector through the matching `SearchParameters` class.
