# Nutrition, dietary tags and allergens

> **Disclaimer.** These numbers and labels are estimates computed from ingredient names.
> They are not medical, nutritional or allergy advice. Products differ by brand and
> region; anyone with an allergy or a medical diet must read the actual product labels.

## Code

| File | Role |
|---|---|
| `app/backend/recipes/ingredient_parser.py` | Rule-based parser: line or `ingredient_details` record -> quantity (range kept as `quantity`/`quantity_max`), canonical unit, package size, normalised name, preparation notes. |
| `app/backend/recipes/nutrition.py` | Matches each ingredient to a food, converts to grams, scales per-100 g values, reports coverage. |
| `app/backend/recipes/dietary.py` | Phrase rules for allergens (contains / may contain) and dietary tags. |
| `scripts/evaluate_nutrition_dietary.py` | Reproduces every number below. The baseline is the code at commit `9d5a78a`, loaded from git. |

## Data

`recipes.db` comes from the Food.com dump (`data/food_recipes.parquet`). Two properties
shape everything here:

1. **No units.** The source has quantities (`RecipeIngredientQuantities`) and names
   (`RecipeIngredientParts`) but no units, so `ingredient_details.unit` is `"portion"`
   on 99.97% of rows.
2. **Misaligned quantities.** In 76% of source recipes the two lists have different
   lengths (Food.com drops unlinked ingredients from `Parts` only). The loader pairs them
   by index, so for those recipes the stored quantity often belongs to another ingredient.

The parquet also carries Food.com's own per-serving nutrition, computed from the full
ingredient text with units. That is used as the reference for calories.

## Method

**Parsing.** Regex and word lists handle fractions, mixed numbers (`1 1/2`, `1-1/2`),
unicode fractions, ranges (`1-2`, `1 to 2`), abbreviations and plurals (`T`/`t`, `Tbsp.`,
`lbs`), number words, parenthetical sizes (`1 (14.5 oz) can`), units after the name
(`garlic cloves`) and Food.com forms such as `lemons, rind of`. Participles like
`chopped` move to `preparation` unless they are part of a product name
(`toasted sesame oil`, `crushed red pepper`).

**Nutrition.**
- The food table holds 250 rows with about 1,800 aliases. Values are per 100 g,
  transcribed by hand from USDA FoodData Central SR Legacy and rounded. Some rows stand
  for groups of foods, for example hard cheeses or dried spices (spices use an average
  profile).
- Matching picks the longest alias that appears as a contiguous run of words. Ties go to
  the rightmost match, since the head noun usually comes last ("beef broth" -> broth).
- Grams:
  - Mass units convert directly.
  - Volume units use the food's density.
  - Count units use a typical piece weight (egg 50 g, medium onion 110 g, garlic clove 3 g).
  - Containers use the size in parentheses, otherwise a typical package weight.
- Bare Food.com numbers use a per-food default unit, e.g. `butter 0.5` means cup and
  `butter 2` means tablespoons. On top of that:
  - 20 or more of a food measured by volume is read as grams.
  - 4 to 20 of a canned food is read as the can size in ounces.
  - 5 to 20 "cups" is read as tablespoons.
  - One ingredient is capped at 2.5 kg.
- Each breakdown row carries `status`: `estimated`, `assumed_unit`,
  `assumed_quantity` or `unknown`. Unknown ingredients add nothing, and there is no default
  profile any more.

**Allergens and tags.**
- Rules run from specific phrases to generic words. A matched phrase is blanked out, so
  "coconut milk", "peanut butter", "cream of tartar" and "eggplant" never reach the
  generic dairy, egg or tree-nut words.
- Hidden allergens are explicit: worcestershire -> fish, pesto -> tree nuts + dairy,
  oyster sauce -> shellfish, nougat, marzipan, ghee, tahini, hoisin, and others.
- Qualifiers remove what they rule out: `gluten-free`, `dairy-free`, `vegan`,
  `plant-based`, `vegetarian`. `lactose-free` does not remove dairy, because the milk
  protein is still there. `non-dairy` only downgrades dairy to "may contain", because
  such products often contain casein.
- Sesame was added as an allergen.
- **Trade-off: missing an allergen is worse than a false warning.** Oats are flagged
  "may contain gluten". Chocolate chips and margarine "may contain dairy/soy". Generic
  "broth" counts as non-vegetarian. `detect_recipe_allergens` returns contains ∪ may
  contain, and `detect_recipe_allergens_detailed` keeps the two levels apart for the UI.
  The price is lower precision, mostly for soy and fish (table below).

## Evaluation sets (`tests/fixtures/`)

- `ingredient_lines_{dev,test}.jsonl`: 113 + 112 ingredient lines written by hand in
  common recipe formats and labelled by hand (quantity, unit, name, preparation). The DB
  has no raw lines with units, so the lines are not taken from it. The rules were
  developed on dev only. Both splits come from the same author and conventions, so the
  test score is optimistic about unseen formats.
- `allergen_recipes.json`: 100 DB recipes. 40 were picked for tricky terms and 60 are
  uniform random (seed 20261124). All 397 unique ingredient names were labelled by hand
  *before* the new rules were written, and the labelling policy is in the file's
  `_meta`. One label (oyster sauce) was missing from the first pass and was corrected
  later. All numbers below use the corrected labels, including the baseline.
- `nutrition_reference.json`: 300 recipes whose quantity and name lists line up, plus
  150 misaligned ones, sampled from the parquet, each with Food.com per-serving
  reference values.

## Results

Regenerate with `python scripts/evaluate_nutrition_dietary.py report`.

### Parser

The baseline is what the code did before: leading number, a short unit list, and the
name cut to three words as in `dataset_loader`.

| Split | Method | quantity | unit | name | preparation | whole row |
|---|---|---|---|---|---|---|
| dev (113) | baseline | 0.920 | 0.451 | 0.230 | 0.690 | 0.159 |
| dev (113) | rule-based | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| test (112) | baseline | 0.911 | 0.527 | 0.241 | 0.670 | 0.188 |
| test (112) | rule-based, first run | 1.000 | 1.000 | 0.991 | 1.000 | 0.991 |
| test (112) | rule-based, final | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The one first-run miss was `lemons, rind of`. A learned tagger (CRF or NER) was not
tried: `sklearn-crfsuite` is not installed, and with 225 author-written lines the
comparison would not mean much.

### Allergens (100 recipes; positives = contains ∪ may contain)

| Allergen | baseline P / R | new P / R |
|---|---|---|
| dairy | 0.905 / 0.905 | 0.984 / 1.000 |
| eggs | 1.000 / 0.947 | 0.974 / 1.000 |
| fish | 1.000 / 0.769 | 0.867 / 1.000 |
| gluten | 0.950 / 0.809 | 0.959 / 1.000 |
| peanuts | 1.000 / 1.000 | 1.000 / 1.000 |
| sesame | – / 0.000 | 1.000 / 1.000 |
| shellfish | 1.000 / 1.000 | 1.000 / 1.000 |
| soy | 1.000 / 0.370 | 0.839 / 0.963 |
| tree nuts | 1.000 / 0.333 | 1.000 / 1.000 |
| **micro** | **0.955 / 0.760** | **0.952 / 0.995** |

- On the 60 random recipes only, micro P/R went from 0.990 / 0.811 to 0.969 / 0.992.
- Dietary tag accuracy, from baseline to new:
  - vegetarian: 0.96 -> 1.00
  - vegan: 0.93 -> 1.00
  - gluten-free: 0.89 -> 0.98
  - dairy-free: 0.88 -> 0.99
- The first run of the new rules, before looking at their errors and with the uncorrected
  label, scored micro P/R 0.944 / 0.995. The final rules were then adjusted on this set,
  so the final row is partly fitted to it. The labeller and the rule author are the
  same person, which also flatters both numbers.

### Calories per serving vs Food.com

| Set | Method | median abs % error | within ±25% | MAE kcal |
|---|---|---|---|---|
| aligned (300) | baseline (names only) | 72.5% | 6.0% | 258 |
| aligned (300) | new, names only | 43.2% | 28.0% | 197 |
| aligned (300) | new, with quantities | 33.5% | 42.7% | 206 |
| misaligned (150) | baseline | 76.9% | 4.7% | 279 |
| misaligned (150) | new, with quantities | 48.5% | 26.0% | 233 |
| fresh aligned sample (400, never inspected) | baseline | 71.4% | 6.8% | 222 |
| fresh aligned sample (400, never inspected) | new, with quantities | 36.0% | 41.2% | 228 |

The bare-number rules were tuned on the 300 aligned recipes, so the fresh sample (seed 7,
not stored in the repo) is the fairer figure. Median error halves. Mean absolute error
does not improve, because a few large overestimates remain: a bare number read with the
wrong unit, such as `brown sugar 2` taken as cups in a sauce. Ingredient-level coverage
on 2,010 random DB recipes is 98.4%, and 88.9% of recipes are fully covered.

## Interface changes

- `calculate_recipe_nutrition(ingredients, servings=None, ingredient_details=None)`: the
  new argument is optional. When it is given, the stored quantities are used.
- The nutrition result keeps `servings`, `per_serving`, `total` and `breakdown`. It also
  gains optional fields, which `_build_nutrition_model` ignores until the schema is
  extended:
  - top-level `coverage`: `{ratio, matched, total, unknown, assumed_quantity_or_unit}`.
  - on each breakdown item: `matched_food`, `grams` and `status`.
- `match_ingredient_nutrition(name)` now returns per-100 g values, or `None` when nothing
  matches, instead of a fixed per-entry profile. Nothing else in the repo calls it.
- `detect_recipe_allergens` can now return `"sesame"`.
- New helpers: `dietary.detect_recipe_allergens_detailed`, `dietary.analyze_ingredient`
  and `nutrition.match_food`.

## Limitations

- Units for Food.com data are guessed per food. This is the main source of calorie
  error, and misaligned quantities (76% of recipes) make it worse. Fixing it needs the
  loader to keep only aligned quantities or store an alignment flag.
- The food table is a hand-transcribed subset, not an FDC import, and grouped rows hide
  differences within a group (all hard cheeses share one row). Importing the SR Legacy
  CSVs would be better, but that is a download the user has to approve.
- DB names are cut to three words, e.g. "fat free sour" or "gluten free baking", which
  loses information. Some rules exist only to catch these cut-off forms.
- The allergen rules work on names. They cannot see brand recipes, cross-contact in
  factories, or regional formulations. Coconut is not treated as a tree nut, and molluscs
  are counted as shellfish.
- The evaluation sets are small and labelled by one person.
