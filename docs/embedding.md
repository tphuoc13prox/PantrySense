# Dense retrieval: embedding models

This page covers how the dense side of retrieval embeds queries and recipes, how the
fine-tuned models were trained, and how they compare. Everything here can be
reproduced with `scripts/finetune_embedder.py`.

## Engine consistency

Recipes are embedded as `title + "\n" + normalized ingredients`
(`build_recipe_representation`). Queries are the normalized ingredient list
(`build_query_representation`). Query and index vectors have to come from the same
model with the same prefixes, otherwise cosine scores are meaningless.

Two problems existed before this change:

- The FastEmbed (ONNX) path always loaded `all-MiniLM-L6-v2` and ignored
  `PANTRYSENSE_EMBEDDING_MODEL`. An index built on CUDA with any other model would
  be queried with MiniLM vectors on a CPU-only machine.
- The model cache was a single global per engine, keyed by nothing, so a second
  model name in the same process got the first model back.

Measured drift between engines for `all-MiniLM-L6-v2`
(`finetune_embedder.py drift`, 368 queries and 400 recipe texts):

| Pair | mean cosine | min cosine |
|---|---|---|
| sentence-transformers vs FastEmbed, queries | 1.000000 | 0.99999988 |
| sentence-transformers vs FastEmbed, recipes | 1.000000 | 0.99999994 |
| sentence-transformers CUDA vs CPU, recipes (against the stored index, 300 recipes) | 0.997893 for both | 0.869625 for both |

So for MiniLM the two engines are interchangeable. The actual bug was the
hardcoded model. FastEmbed's `bge-small-en-v1.5` is a quantized export (`-onnx-q`),
so it would not match an index built with sentence-transformers. It was not
measured here.

What `embedder.py` does now:

- FastEmbed is used only for models listed in `FASTEMBED_EQUIVALENT_MODELS`
  (currently MiniLM, verified above). Any other model, including local fine-tuned
  directories, runs through sentence-transformers on CUDA or CPU.
- Query and passage prefixes come from the model directory's
  `config_sentence_transformers.json` (`prompts`) or from a small table (e5 uses
  `query: ` / `passage: `). `embed_text` embeds a query and `embed_texts` embeds
  passages.
- When an index is saved, `write_index_metadata` stores
  `<index>.meta.json` with model, backend, dimension and prefixes.
  `SemanticRetriever.ensure_loaded` calls `verify_index`. On a mismatch it logs
  the reason and reports the index as not ready, instead of returning wrong results.
  Indexes without metadata (everything built before this change) are assumed
  to be MiniLM, and only a warning is logged.

Side finding: re-embedding 300 random recipes and comparing them with the vectors in
the current `data/recipe_vectors.index` gives mean cosine 0.998 but minimum 0.87.
Some recipe texts changed after the index was built (the database is newer than
the index), so the shipped index is slightly stale.

## Training data

- Split: `bucket = md5(recipe_id) % 10`. Buckets 0-1 are the official eval pool and
  are excluded from everything below. Inside buckets 2-9, a second hash
  (`md5("heldout:<id>") % 100 < 2`) moves 5,107 recipes into a held-out split.
  This leaves 247,045 training recipes.
- Query: 40-70% of a recipe's normalized ingredients, sampled at random (at least 2).
  Ingredients that appear in more than 15% of training recipes are never used in
  queries (5 ingredients at this threshold). Recipes with fewer than 3 usable
  ingredients are skipped.
- Positive: the recipe representation exactly as it is indexed.
- Loss: symmetric MultipleNegativesRankingLoss (in-batch negatives, scale 20). Duplicate
  recipe texts inside a batch are dropped so they are not treated as negatives.
- 100,000 pairs, 1 epoch, batch 128, AdamW lr 5e-5, 5% warmup then linear decay,
  fp16 autocast, seed 13.

```
python scripts/finetune_embedder.py --db <recipes.db> train --max-pairs 100000 --output data/models/minilm-pantry
python scripts/finetune_embedder.py --db <recipes.db> train --base intfloat/e5-small-v2 --max-pairs 100000 --output data/models/e5-small-pantry
```

## Results

Setup: 500 queries generated from held-out recipes (same generator, different seed).
The corpus is a controlled 50,000-recipe subset: the 500 targets plus 49,500 random
recipes from buckets 2-9. The subset is used only to keep the runs short.
On the full 315k index, absolute recall will be lower.
Retrieval uses exact FAISS inner product. A hit means the target recipe, or a recipe
with exactly the same representation text, appears in the top K.

Queries with 40-70% of ingredients (the training distribution):

| Model | Recall@10 | Recall@50 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (current) | 0.542 | 0.708 | 0.776 | 0.363 |
| bge-small-en-v1.5 | 0.490 | 0.672 | 0.766 | 0.333 |
| bge-small-en-v1.5 + query instruction | 0.480 | 0.670 | 0.766 | 0.327 |
| e5-small-v2 | 0.612 | 0.780 | 0.864 | 0.425 |
| MiniLM fine-tuned | 0.890 | 0.970 | 0.986 | 0.720 |
| e5-small-v2 fine-tuned | **0.934** | **0.984** | **0.994** | **0.776** |

Shorter queries with 20-40% of ingredients, a harder setting not seen in training:

| Model | Recall@10 | Recall@50 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.274 | 0.456 | 0.526 | 0.159 |
| e5-small-v2 | 0.340 | 0.500 | 0.612 | 0.206 |
| MiniLM fine-tuned | 0.548 | 0.786 | 0.874 | 0.352 |
| e5-small-v2 fine-tuned | 0.610 | 0.802 | 0.886 | 0.410 |

The fine-tuned models still lead when the queries are shorter than anything in
training, so the gain does not come only from matching the 40-70% sampling ratio.

```
python scripts/finetune_embedder.py --db <recipes.db> bench --queries 500 --corpus-size 50000 \
  --models "minilm=sentence-transformers/all-MiniLM-L6-v2" "e5-small=intfloat/e5-small-v2" \
           "minilm-ft=data/models/minilm-pantry" "e5-small-ft=data/models/e5-small-pantry"
```

## Cost

Measured on a laptop with an RTX 4050 (6 GB) and a 16-thread CPU.

| Model | Params | Disk | Train time | Embed 50k recipes (GPU) | Query, CPU median / p95 | Query, CUDA median |
|---|---|---|---|---|---|---|
| MiniLM via FastEmbed | 22.7M | 88 MB | - | - | 2.8 / 5.0 ms | - |
| MiniLM via sentence-transformers | 22.7M | 88 MB | - | 43 s | 27 / 41 ms | - |
| MiniLM fine-tuned | 22.7M | 88 MB | 296 s | 29 s | 24 / 37 ms | 15 ms |
| e5-small-v2 | 33.4M | 129 MB | - | 45 s | 40 / 56 ms | - |
| e5-small-v2 fine-tuned | 33.4M | 129 MB | 549 s | 57 s | 40 / 57 ms | 24 ms |

Query latency is for one `IngredientEmbedder.embed_text` call (300 queries after
warm-up). All models produce 384-dimension vectors, so the full flat FAISS index
is the same size for each, about 484 MB for 315k recipes. Building the
full index takes roughly 6x the 50k time.

## Recommendation

Use the fine-tuned e5-small-v2 (`data/models/e5-small-pantry`) as the dense
model. It is the most accurate in both settings. A query costs about 40 ms on CPU,
which is small next to the rest of the request. If CPU latency matters more,
fine-tuned MiniLM keeps most of the gain at about half the train and index cost.

The app uses it by default once the model folder, its index, the index metadata and
the id file are in `data/`, and falls back to MiniLM otherwise (see "Default
configuration" in docs/ltr.md). To build the index with the same model so the
metadata matches:

```
python scripts/finetune_embedder.py --db data/recipes.db build-index --model data/models/e5-small-pantry --index data/recipe_vectors_e5ft.index --ids data/recipe_vector_ids_e5ft.json
```

## Limitations

- Queries are synthetic: random subsets of a recipe's own ingredients. Real users
  list what is in their pantry. That includes items the recipe does not use and
  misses some it needs. These numbers measure how well a model finds the source
  recipe from part of its ingredients. They do not measure user satisfaction. The
  final number has to come from the official eval set (buckets 0-1) once
  `eval-harness` is merged, and it has not been run here.
- Every held-out query has exactly one relevant recipe (plus exact-text
  duplicates). Near-duplicate recipes count as misses, which understates every model.
- The 50k subset makes absolute recall optimistic compared with the full index.
  The ranking between models is what should carry over.
- The fine-tuned models have no verified ONNX export, so on CPU they run through
  sentence-transformers at roughly 10x FastEmbed's latency. An ONNX export of the
  fine-tuned model, checked with `drift`, would bring that back down.
- One seed, one epoch, no hyperparameter search. Variance between runs was not measured.
- The fine-tuned models are not in git (`data/models/` is ignored) and have to be
  retrained with the commands above or shipped separately.
