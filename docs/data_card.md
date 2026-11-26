# Data card: PantrySense recipe corpus

This document describes the recipe data PantrySense searches over: where it
comes from, how it is transformed, what ends up in `data/recipes.db`, and what
to keep in mind when interpreting results or evaluation numbers. Counts below
were read from a local `recipes.db` (opened read-only) in November 2026.

## Source

| | |
|---|---|
| Dataset | Hugging Face `untitledwebsite123/food-recipes` |
| URL | https://huggingface.co/datasets/untitledwebsite123/food-recipes |
| Uploaded | 2025-08-31 by user `untitledwebsite123` |
| Files in the repo | `recipes.csv` (about 704 MB); the app uses the Hub's auto-converted Parquet branch (`refs/convert/parquet/default/train/0000.parquet`, about 229 MB locally) |
| License | Not stated. The dataset page has no dataset card and no license metadata. Treat as unclear. |
| Original provider | Not stated on the dataset page. The CSV header (`RecipeId`, `RecipeIngredientParts`, `AggregatedRating`, ...), the `c("a", "b")` list encoding from R, image URLs on `img.sndimg.com` and descriptions of the form "Make and share this ... recipe from Food.com." indicate the data is a copy of Food.com recipes, probably the widely circulated Kaggle export. This is an inference from the file contents, not something the page documents. |

Because the license and the upstream terms are unclear, the dataset and the
databases and indices derived from it are not redistributed with this
repository (`data/` is git-ignored and excluded from the Docker image). Anyone
rebuilding the corpus downloads it from Hugging Face themselves.

Two trained models are published as a release asset (from v1.2.0): the
fine-tuned e5-small-v2 embedding model and the LTR ranker bundle. They contain
weights, tree ensembles, feature names and training statistics, but no recipe
text, ids or ingredient lists. Before publishing them, these terms were checked
(November 2026):

- The Hugging Face dataset page states no license and no usage terms.
- The most likely upstream, the Kaggle dataset "Food.com - Recipes and Reviews"
  (`irkaal/foodcom-recipes-and-reviews`), is marked CC0: Public Domain.
- Food.com's footer links the Discovery Visitor Agreement (last updated January
  2023). It forbids distributing or reproducing copies of the site's materials,
  and has no clause on derived data, data mining or model training.
- The base model `intfloat/e5-small-v2` is MIT-licensed; its license is
  included in the model folder.

None of these forbids publishing models trained on the data. Whether the
Kaggle CC0 label was the uploader's to give is unknown, which is one more
reason the recipe data itself stays out of the repository and the release.

The raw row count of the Parquet file was not measured here (the Parquet
reader is not installed in the environment used to write this card). The Hub
tags put it in the 100K to 1M range.

## What is in `recipes.db`

SQLite database, table `recipes`, **315,447 rows**, all with distinct titles.
`id` is an autoincrement integer (range 286 to 315,732, not contiguous). The database also holds the
`pantry_items`, `favorite_recipes` and `meal_plans` tables, which are user state
and not part of the corpus.

| Column | Type | Meaning |
|---|---|---|
| `id` | integer | Local primary key. Not the upstream `RecipeId`. |
| `title` | text, unique | Recipe name (upstream `Name`). |
| `ingredients` | JSON list of strings | Normalised ingredient names, 1 to 12 per recipe. |
| `ingredient_details` | JSON list of objects | `{name, quantity, unit}` aligned with `ingredients`. |
| `instructions` | JSON list of strings | Steps, 1 to 10 per recipe. |
| `cooking_time` | integer minutes | 5 to 180. Mean 39. |
| `difficulty` | text | `easy` (182,144) or `medium` (133,303). Derived, see below. |
| `servings` | integer | 1 to 12. |
| `category` | text | Coarse category, see below. |

Category counts: dinner 213,163; dessert 56,174; breakfast 22,702; lunch
22,556; soup 845; and 7 rows from the built-in catalogue with other labels
(`rice dish`, `main dish`, `pasta`).

Besides the Hugging Face rows, the ingest also inserts a small hand-written
catalogue (`CORE_RECIPE_CATALOG` in `dataset_loader.py`, about two dozen
recipes with real quantities and units) so the app is usable on a tiny corpus.

Derived artifacts built from the table, all under `data/`: `recipe_vectors.index`
(FAISS flat inner-product index of 384-dim embeddings), `recipe_vector_ids.json`,
`recipe_bm25.json` (BM25 index), `ingredient_vocab.json` (autocomplete
vocabulary) and `ranker_model.joblib` (LightGBM or gradient boosting ranker).

## Fields read from the source

Only 7 of the 28 upstream columns are used: `Name`, `RecipeIngredientParts`,
`RecipeIngredientQuantities`, `RecipeInstructions`, `CookTime`,
`RecipeCategory`, `RecipeServings`. Everything else is dropped at ingest:
ratings and review counts, nutrition columns (`Calories`, `FatContent`, ...),
keywords, description, images, author, dates, `PrepTime`, `TotalTime` and
`RecipeYield`. Ratings are therefore not available as a relevance signal.

## Preprocessing

Implemented in `app/backend/system/dataset_loader.py`
(`load_recipes_from_parquet`, `_parse_*` helpers, `SetupManager`).

1. **List decoding.** Upstream list fields are R-style `c("a", "b")` strings;
   they are parsed with a regex (falling back to JSON, then newline split).
2. **Ingredient names.** Lower-cased, punctuation removed, truncated to the
   first 3 words, de-duplicated within a recipe, and capped at the first 12.
   Recipes with no usable ingredient are dropped.
3. **Quantities.** Parsed from `RecipeIngredientQuantities` (integers, decimals
   and simple fractions such as `1/4`), defaulting to 1.0. The unit is always
   set to `portion`, so upstream units are lost for corpus recipes.
4. **Instructions.** Capped at the first 10 steps. When a recipe has none, the
   placeholder `Prepare ingredients and cook <title> to desired taste.` is
   inserted (8,986 rows carry it).
5. **Cooking time.** Taken from `CookTime` only (not prep or total time).
   ISO-8601 durations such as `PT25M` are converted to minutes; anything
   unparseable or missing becomes 20. The value is clamped to 5 to 180.
6. **Difficulty.** Not an upstream field: `medium` when cooking time is over 25
   minutes, otherwise `easy`.
7. **Servings.** Parsed as an integer, default 2, clamped to 1 to 12.
8. **Category.** Keyword match on `RecipeCategory`: breakfast, lunch, soup,
   dessert (also pie, cookie), beverage or drink (mapped to `breakfast`);
   everything else becomes `dinner`.
9. **Title de-duplication.** Case-insensitive on title; the first occurrence
   wins, and the built-in catalogue is inserted first. Titles shorter than 2
   characters are dropped.
10. **Row limit.** `PANTRYSENSE_DATASET_LIMIT` (default 380,000) caps how many
    parsed recipes are read from the Parquet file. The final count is lower
    than that because of the filters and de-duplication above; the count
    before de-duplication was not recorded.

## Train and evaluation split

The split is fixed and depends only on the recipe id, so it survives corpus
rebuilds as long as ids are stable:

```
bucket = int(md5(str(recipe_id)).hexdigest(), 16) % 10
```

- Buckets 0 and 1 are the **evaluation pool**: 63,295 recipes (20.1%).
  Evaluation queries are generated only from these recipes. Nothing trained or
  tuned may sample queries from them.
- Buckets 2 to 9 are the **training pool**: 252,152 recipes. The learning-to-rank
  work further holds out bucket 9 for comparing rankers and uses buckets 2 to 8
  for training and cross-validation.

Bucket sizes in the current database: 0: 31,782; 1: 31,513; 2: 31,479;
3: 31,529; 4: 31,761; 5: 31,533; 6: 31,321; 7: 31,548; 8: 31,371; 9: 31,610.

The helper that implements this rule lives in
`app/backend/evaluation/split.py` on the evaluation work's branch and is not
part of this branch yet.

Note that the split is by recipe, not by near-duplicate group. The corpus
contains recipes that differ only slightly in title or ingredients, so a
query built from an eval recipe can still have a close sibling in the training
pool. Treat absolute metrics as optimistic.

## Known biases and limitations

- **Source skew.** Food.com is user-submitted and English-language, with a
  North American bias in dishes, ingredient names and units. Other cuisines
  are under-represented and often described through Americanised ingredient
  lists.
- **Category imbalance.** About 68% of rows are `dinner`, which is also the
  fallback for anything unmatched, so it mixes true dinners with side dishes,
  sauces, snacks and drinks other than the beverage keywords. Soups are 0.3%.
  Beverages are labelled `breakfast`.
- **Lossy ingredient normalisation.** Truncating to 3 words loses qualifiers
  (`extra virgin olive oil` becomes `extra virgin olive`), and variants stay
  separate (`garlic` vs `garlic cloves`). The corpus has 6,455
  distinct ingredient strings, 1,227 of which occur once. This inflates the
  vocabulary and can hurt exact-match coverage.
- **Caps are hit often.** 45,050 recipes (14.3%) have exactly 12 ingredients
  and 69,903 (22.2%) have exactly 10 steps, so many are truncated and their
  coverage scores are computed on partial lists.
- **Default values look like data.** 79,480 recipes (25.2%) have a cooking time
  of exactly 20 minutes, which is the fallback for missing or unparseable
  values, and 125,429 have 2 servings (the fallback). Difficulty is a function
  of that time, so it is only a rough proxy.
- **Quantities and units** are unreliable for corpus recipes (unit always
  `portion`). Anything that depends on amounts, such as nutrition estimates or
  grocery quantities, is approximate for these rows.
- **No ground truth.** The corpus has no user relevance judgments. Evaluation
  queries are synthetic, built from the recipes themselves, so they favour
  lexical overlap and say little about how real users phrase a pantry.
- **Safety and dietary information is heuristic.** Allergen and dietary
  filters work from ingredient strings, not from verified labels, and must not
  be relied on for medical decisions.
- **Licensing.** See Source: the upstream terms are unknown, so the data should
  be used for local research and demonstration only.

## Rebuilding

The in-app setup flow (`SetupManager`) downloads the Parquet file to
`data/food_recipes.parquet`, parses it, fills `recipes.db`, then builds the
FAISS and BM25 indices. It does not train the ranker. `scripts/build_database.py`
runs only the database part, and `scripts/build_vector_index.py` rebuilds only
the indices from an existing database.

Row ids are assigned at insert time, so rebuilding from a different row limit
changes ids and therefore bucket membership; keep the same limit when comparing
evaluation runs. A fresh database built with the default limit numbers its
recipes from 286, like the published corpus, and gives exactly that corpus:
`build_database.py` checks it against the fingerprint in
`eval/queries_v1.meta.json`.
