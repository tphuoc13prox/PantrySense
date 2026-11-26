# Known issues

Open problems and limitations as of v1.2.0, roughly in order of impact. Each
entry says what is wrong, what it affects and what a fix would involve.

## Data

- **Ingredient quantities and names are paired by position.** The source has
  separate quantity and name lists, and in about 76% of recipes they have
  different lengths, because Food.com drops unlinked ingredients from the names
  only. The loader pairs them by index, so the quantity-aware nutrition
  estimate often uses another ingredient's amount: the median is a plausible
  392 kcal per serving, but p99 is 3,355. Fixing it needs the loader to keep
  only aligned quantities or store an alignment flag (see
  `docs/nutrition_dietary.md`).
- **Dietary tags are a word list over incomplete data.** Food.com often leaves
  the main protein out of the ingredient list, so vegan and vegetarian tags
  also read the title, Food.com's keywords and category, and the ingredients
  past the stored twelve (`docs/nutrition_dietary.md`). A meat dish that none
  of these name is still tagged vegetarian. The rules lean the other way on
  purpose, so some vegetarian recipes lose the tag: sauces "for chicken", egg
  dishes Food.com tagged "Poultry", titles with a quoted stand-in. The extra
  fields come from `food_recipes.parquet`; a database migrated without it (or
  without `pyarrow`) relies on titles and ingredients alone.
- **The MiniLM index has drifted a little from the database.** It covers
  exactly the same 315,447 ids, but a few recipe texts changed after it was
  built. Rebuilding it with `scripts/build_vector_index.py` should fix this. The
  fine-tuned e5 index was built from the current database.

## Ranking

- **The ranker does not see the query.** `MLRecipeRanker.rank()` only gets
  the candidate list, so query-level features are estimated from the union of
  ingredients the candidates matched. A query ingredient that no candidate
  matches is invisible, and query size is underestimated. Passing the parsed
  query through `RecipeSearchService._process_candidates` is a small change,
  but the features change with it, so the ranker has to be retrained.
- **Retrieval limits quality on realistic pantry lists.** On the pantry query
  tier only 54% (top-100) and 39% (top-20) of source recipes reach the ranker
  with e5, and 49% / 33% with MiniLM. Better recall (a larger candidate depth,
  a better embedding, or using the prefilter more aggressively) would help more
  than further ranker work.
- **LTR v3 is slightly slower than v2 for no clear reason.** On the official
  eval p50 is 62 ms with v3 against 58 ms with v2, although v3 has fewer trees
  (124 against 210). Not investigated yet.
- **`train_ranker.py` uses a deprecated LightGBM argument.** `LGBMRanker.fit`
  warns that `eval_set` is deprecated; the cross-validation should switch to
  the new evaluation arguments before a LightGBM upgrade removes it.

## Evaluation

- **All queries are synthetic.** The eval set, the e5 fine-tune pairs and the
  LTR training queries are all built from subsets of a recipe's own
  ingredients (buckets 0-1 for eval, 2-9 for training). The recipes are
  disjoint, but the query style is in-distribution, so gains on real pantry
  lists are likely smaller than the reported numbers. `query_coverage`, the
  strongest LTR feature, benefits most from this. The pantry tier reduces the
  effect but its extra ingredients are co-occurring ones, not a real user's
  pantry.
- **Old result files have a different query-file hash.** Runs made from a
  checkout with `core.autocrlf=true` before `.gitattributes` pinned
  `eval/**/*.jsonl` to LF record sha256 `eb051d0d17f5` instead of
  `0b9aabd48bb2`. The content is the same; only line endings differ.

## Deployment and environment

- **The fine-tuned model, its index and the ranker are not in the
  repository.** The app uses e5 only when `data/models/e5-small-pantry/`,
  `data/recipe_vectors_e5ft.index` with its `.meta.json` and
  `data/recipe_vector_ids_e5ft.json` are all present, and the learned ranker
  only when `data/ranker_model.joblib` holds an LTR bundle. Otherwise it falls
  back to MiniLM and to plain RRF order, and logs which files are missing.
- **Index metadata stores the model path as written at build time.** The
  compatibility check accepts a model folder that was copied elsewhere as long
  as its name matches, but the `.meta.json` itself still holds the original
  location.
- **Building the e5 index on CPU is slow.** With CPU-only torch, embedding the
  full corpus takes about 35 minutes. The GPU path has not been measured for
  the full index.
