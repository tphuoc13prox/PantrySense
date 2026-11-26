# PantrySense

PantrySense suggests recipes from the ingredients you have. It searches about
315,000 Food.com recipes and runs locally: a FastAPI backend, SQLite, and a
static web frontend, with no external services once the data is downloaded.

Search is a two-stage pipeline. BM25 and a dense FAISS index (e5-small-v2
fine-tuned on this corpus) each retrieve candidates. Reciprocal rank fusion
merges the two lists, and a LightGBM LambdaMART model with 23 features re-ranks
the result. Around search, the app has dietary and allergen filters, nutrition
estimates, ingredient substitutions, a pantry with expiry dates, a weekly meal
planner with a shopping list, and a step-by-step cooking view with timers.

Version 1.2.0.

## Quickstart

Python 3.11 or 3.12. Plan for about 1.5 GB in `data/` (dataset, database,
indices, models).

```
git clone https://github.com/tphuoc13prox/pantrysense.git
cd pantrysense
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python tasks.py install         # requirements-dev.txt, CPU-only torch
python tasks.py fetch-models    # fine-tuned e5 model + LTR ranker, about 112 MB
python tasks.py run             # http://localhost:8000
```

`python tasks.py install` adds the test and lint tools. For the app alone:
`pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`.

`fetch-models` downloads `pantrysense-models-v1.2.0.zip` from the
[v1.2.0 release](https://github.com/tphuoc13prox/pantrysense/releases/tag/v1.2.0).
It checks the zip, and every file in it, against the sha256 values in
`scripts/model_assets.json`, then unpacks it into `data/`. `--from-dir DIR`
installs from a zip you already downloaded.

On the first start the browser opens a setup screen. Setup downloads the
dataset from Hugging Face (about 230 MB), builds `data/recipes.db` with dietary
and nutrition metadata, and then builds the BM25 index and the e5 index. On the
laptop CPU used for the measurements below, the e5 index for the full corpus
took about 35 minutes. Later starts reuse the built files.

Without `fetch-models` everything still works, on the fallback configuration:
all-MiniLM-L6-v2 embeddings and results in RRF order, which scores clearly lower
(see Evaluation). The setup screen says so. If you add the models later and
restart, the app builds the e5 index from the existing database.

To check an install end to end without the full dataset, run setup on a small
corpus in a scratch folder and send one search to the running server:

```
python scripts/smoke_test.py --data-dir /tmp/pantry-smoke --limit 3000
```

Docker: `python tasks.py docker-build` and `python tasks.py docker-run`. The
image holds no data. It expects `data/` mounted at `/app/data`, prepared on the
host with `fetch-models` and one run of setup. It does not build indices on
startup.

## Evaluation

The official query set is `eval/queries_v1.jsonl`: 1000 queries from recipes in
hash buckets 0-1, a pool that is never used for training or tuning. Each query
is a random 40-70% of one recipe's ingredients, with very common ingredients
removed and, in one tier, an unrelated ingredient added. The source recipe and
its near-duplicates count as relevant. The numbers below are for the final
output (20 candidates, coverage filter, ranking), with 95% bootstrap intervals,
from `eval/results/v1_1_comparison.md`.

| Configuration | Recall@10 | MRR@10 | nDCG@10 | Hit@1 | p50 ms |
|---|---|---|---|---|---|
| BM25 only | 0.655 [0.626, 0.685] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 12 |
| Dense e5-small-ft only | 0.717 [0.689, 0.745] | 0.551 [0.523, 0.579] | 0.592 [0.565, 0.619] | 0.464 [0.432, 0.494] | 52 |
| Hybrid MiniLM, RRF order (fallback without fetch-models) | 0.604 [0.574, 0.635] | 0.364 [0.339, 0.391] | 0.421 [0.397, 0.448] | 0.265 [0.239, 0.293] | 49 |
| Hybrid e5-small-ft, RRF order | 0.729 [0.701, 0.757] | 0.555 [0.527, 0.584] | 0.597 [0.570, 0.624] | 0.475 [0.443, 0.507] | 59 |
| Hybrid e5-small-ft + LTR v3 (default) | 0.756 [0.730, 0.783] | 0.644 [0.616, 0.672] | 0.671 [0.645, 0.698] | 0.585 [0.554, 0.617] | 62 |

Latency is end to end in one process on a laptop CPU (Intel Core Ultra 7 155H),
with the query embedded by sentence-transformers.

What to keep in mind:

- The queries are synthetic, and the e5 fine-tune and the LTR training data
  were generated the same way (from other buckets). The query style is
  in-distribution, so the gain on real pantry lists is probably smaller.
- Each query has one relevant recipe plus duplicates, so a good alternative
  recipe counts as a miss and every metric is a lower bound.
- The ranker gets most of its gain from `query_coverage`, a feature that this
  query protocol favours (docs/ltr.md).

`python scripts/run_eval.py` reruns the evaluation (`--limit 50` for a quick run).
It picks the embedding setup the same way the app does.

## Reproducing from scratch

`fetch-models` gives you the published models. To rebuild them from the raw
dataset instead:

```
python tasks.py reproduce             # data -> finetune -> index -> ltr-data -> ltr-train -> eval
python tasks.py reproduce --quick     # tiny settings, checks the chain runs
```

The defaults are the settings behind the published models. The full chain
takes about two hours on the laptop above, with the fine-tune on its RTX 4050.
The steps, timings and what should come out are in
[docs/reproduce.md](docs/reproduce.md).

## How it works

- **Data** ([docs/data_card.md](docs/data_card.md)): the Hugging Face dataset
  `untitledwebsite123/food-recipes`, a copy of Food.com recipes. At most 12
  ingredients per recipe are stored. Dietary tags, allergens and nutrition are
  computed once at ingest ([docs/nutrition_dietary.md](docs/nutrition_dietary.md)).
- **Retrieval**: BM25 over title and ingredients (numpy/npz index) and an exact
  FAISS inner-product index over e5 embeddings, with `query: ` and `passage: `
  prefixes. Filters run before retrieval. Each index's `.meta.json` records the
  model, prefixes and backend, and the app refuses to query an index built with
  a different setup ([docs/embedding.md](docs/embedding.md)).
- **Fusion and ranking**: RRF (k=60) over both candidate lists, a coverage
  threshold, then LambdaMART on 23 features: retrieval scores and ranks,
  ingredient coverage, title overlap, cooking time and difficulty. Without a ranker
  bundle the RRF order is kept ([docs/ltr.md](docs/ltr.md)).
- **Speed**: [docs/performance.md](docs/performance.md). Open problems:
  [docs/known_issues.md](docs/known_issues.md).

Code layout: `app/backend/retrieval/` (BM25, FAISS, embedder, fusion),
`app/backend/ranking/` (features, LTR), `app/backend/recipes/` (search service,
filters, nutrition), `app/backend/system/` (setup pipeline, dataset download),
`scripts/` (training, evaluation, release), `frontend/` (no build step).

## Configuration

Environment variables, all optional:

| Variable | Default | Effect |
|---|---|---|
| `PANTRYSENSE_DATA_DIR` | `data/` | Folder for the database, indices and models |
| `PANTRYSENSE_EMBEDDING_MODEL` | fine-tuned e5 if present, else MiniLM | Embedding model, with the matching index |
| `PANTRYSENSE_RANKING_MODE` | `ml` | `heuristic` sorts by ingredient coverage instead |
| `PANTRYSENSE_RETRIEVAL_MODE` | `hybrid` | `lexical` or `semantic` for one retriever |
| `PANTRYSENSE_EMBEDDER_ENGINE` | `cuda` if available, else `onnx` | Backend for embedding |
| `PANTRYSENSE_AUTO_OPEN_BROWSER`, `PANTRYSENSE_AUTO_SHUTDOWN` | `true` | Open a tab on start; stop when the last tab closes |

Per-file paths (`PANTRYSENSE_DB_PATH`, `PANTRYSENSE_VECTOR_INDEX_PATH`, ...)
override `PANTRYSENSE_DATA_DIR`. They are listed in `app/backend/config.py`.

## Development

```
python tasks.py test        # pytest -m "not slow", as in CI
python tasks.py lint        # ruff
python tasks.py typecheck   # mypy
```

CI runs ruff, mypy and the tests on Python 3.11 and 3.12. It also runs an
end-to-end job (`e2e-smoke`) that sets up a corpus of about 2,000 recipes on an empty data
folder and searches it, once on the fallback and once with the models from the
release. Tests that need the real dataset or built indices skip themselves when
`data/` is empty.

## Data and model licenses

The recipe data is not redistributed here. Its Hugging Face page states no
license, so setup downloads it from Hugging Face on your machine
(docs/data_card.md). The release contains only model weights and the ranker,
no recipe text. The e5 model is a fine-tune of `intfloat/e5-small-v2` (MIT
license, included in the model folder). The fallback model is
`sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0), downloaded on first use.
