# PantrySense

PantrySense is an offline-first, local AI/ML-oriented recipe discovery, pantry management, and culinary assistant application. In **v1.0.1**, PantrySense delivers a complete smart kitchen ecosystem featuring **Dietary & Allergen Filtering**, **Smart Ingredient Substitution**, **Macro & Calorie Nutrition Estimation**, **Virtual Pantry & Expiry Tracker**, **Weekly 7-Day Meal Planner**, **Pantry-Aware Grocery Shopping List Generator**, **Interactive Step-by-Step Cooking Assistant with Audio Timers & Step Navigator**, **Controlled Ingredient Autocomplete & Fuzzy Spellchecker** (4,588-ingredient culinary vocabulary), and **Two-Stage Hybrid AI Retrieval + Learning-to-Rank (LTR)**.

Current version: **v1.1.0**

## Architecture

```text
                                Web Browser (localhost:8000)
                                            │
           ┌────────────────┬───────────────┼───────────────┬────────────────┐
           ▼                ▼               ▼               ▼                ▼
     🔍 Search & Filter  🥑 My Pantry   📅 Meal Planner  ⭐ Favorites   👨‍🍳 Cooking Assistant
   (Dietary/Allergens) (Expiry Alerts) (7-Day Calendar) (Bookmarks)    (Step Timers & Audio)
           │                │               │               │                │
           ▼                ▼               ▼               ▼                ▼
  POST /api/recipes/search  /api/pantry   /api/meal-plan  /api/favorites /api/assistant/parse-steps
  (Hybrid + ML Ranker)   (SQLite DB)   (Shopping List)  (SQLite DB)   (Countdown & Chimes)
```

## Implemented Features (v1.0.0)

### 1. 🔍 Dietary & Allergen Intelligence
- **Dietary Filter Chips**: 1-click filtering for Vegetarian (`🌱`), Vegan (`🌿`), Gluten-Free (`🌾`), Dairy-Free (`🥛`), Nut-Free (`🥜`), and Keto / Low-Carb (`🥩`).
- **Allergen Detection & Exclusion**: Multi-allergen exclusion engine detecting and filtering Peanuts, Tree Nuts, Dairy, Eggs, Gluten, Shellfish, Fish, and Soy.
- **Cooking Time & Category Constraints**: Filter recipes by max cooking time (<15m, <30m, <45m, <60m) and meal category.

### 2. 💡 Smart Culinary Ingredient Substitutions
- **50+ Knowledge Base Substitutions**: Curated culinary substitutions with exact conversion ratios and context notes (e.g. baking, sauces, high-heat cooking).
- **Pantry-Aware Substitute Recommendations**: Analyzes missing ingredients in recipes and prioritizes substitutes that already exist in your pantry.

### 3. 🔥 Nutrition & Macronutrient Estimation
- **Per-Serving Calculations**: Real-time automated estimation of Calories, Protein (g), Carbohydrates (g), Fats (g), and Dietary Fiber (g).
- **Per-Ingredient Breakdown**: Complete nutritional breakdown per ingredient based on standard culinary portion weights.

### 4. 🥑 Virtual Pantry & Expiry Tracker (My Pantry)
- **Inventory Management**: Track ingredient names, quantities, units, categories, and expiration dates.
- **Dynamic Expiry Status Badges**:
  - `Fresh` (Green): > 3 days remaining.
  - `Expiring Soon` (Amber): 1 to 3 days remaining.
  - `Expired` (Red): ≤ 0 days.
- **1-Click Recipe Search**: "Find Recipes with My Pantry" instantly searches dishes using all in-stock pantry items.

### 5. 📅 Weekly 7-Day Meal Planner & Shopping List
- **7-Day Meal Grid**: Schedule recipes across Monday–Sunday for Breakfast, Lunch, Dinner, and Snack slots.
- **Pantry-Deducted Grocery List**: Aggregates all missing ingredients needed for the entire week and automatically subtracts items already in your virtual pantry.
- **1-Click Copy**: Copy formatted shopping checklist to clipboard.

### 6. 👨‍🍳 Interactive Step-by-Step Cooking Assistant
- **Instruction Step Viewer**: Step-by-step cooking modal with visual progress bar and large, readable instructions.
- **Auto-Extracted Countdown Timers**: Automatically parses cooking durations (e.g. "simmer for 15 minutes", "bake 1 hour") into interactive timers with start/pause/reset controls.
- **Web Audio API Chimes**: Plays audible completion chimes when timers reach zero.

### 7. ⭐ Favorite Recipe Bookmarks
- Bookmark favorite recipes with one click from search results or recipe detail views.
- Persistent offline SQLite storage with instant access from the navigation bar.

### 8. 🔤 Controlled Ingredient Autocomplete & Fuzzy Spellchecker
- **4,588-Ingredient Vocabulary**: Instant in-memory frequency dictionary (< 2ms load time).
- **3-Tier Matching Algorithm**: Exact prefix matching, substring/token boundary matching, and fuzzy typo correction (`✨ Did you mean?`).
- **Keyboard Navigation**: Full `ArrowUp`, `ArrowDown`, `Enter`, `Tab`, and `Escape` support.

### 9. 🚀 High-Speed AI Retrieval & ML Re-Ranker
- **Dual-Engine Acceleration**: PyTorch CUDA (GPU) & FastEmbed ONNX INT8 (CPU) with instant header dropdown switching.
- **Hybrid Retrieval**: Inverted-index BM25 lexical retrieval + FAISS dense semantic embeddings fused via Reciprocal Rank Fusion (RRF).
- **Learning-to-Rank (LTR)**: LightGBM GBDT ranking model evaluating 9-dimensional relevance features.

---

## Requirements

- Python 3.11 or newer
- Dependencies in `requirements.txt` (FastAPI, Uvicorn, pytest, httpx, fastembed, onnxruntime, sentence-transformers, faiss-cpu, numpy, rank-bm25, lightgbm, scikit-learn, joblib)

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the backend server:

```powershell
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

> **Behavior**:
> 1. Server starts and verifies local indices and ML ranker model.
> 2. Automatically opens your default external browser at `http://localhost:8000`.
> 3. If first run, the **Onboarding Setup Screen** automatically prepares vector indices and database.
> 4. Server stays active as long as your browser tab is open and closes cleanly when all tabs are closed.

## Run Automated Tests

```powershell
python -m pytest
```

All 46 unit and integration tests covering API routes, dietary tagging, substitutions, nutrition, pantry CRUD, meal planning, and hybrid ranking will execute and pass.

## Evaluation

Offline evaluation of retrieval and ranking on the full corpus (315,447 recipes). Code lives in `app/backend/evaluation/` and `scripts/`; committed artifacts in `eval/`.

**Query set** (`eval/queries_v1.jsonl`, built by `scripts/build_eval_set.py`, seed 20261124). Each query is a random 40-70% of one recipe's ingredients; that recipe is the answer. Ingredients appearing in at least 9% of recipes (salt, butter, egg, onion, sugar, water, garlic clove, olive oil, milk, flour, pepper) are never used in queries. Recipes with the same normalised title or an identical ingredient set are treated as one answer. Three tiers of ~333 queries each: `standard` (55-70%), `sparse` (40-55%), `noisy` (40-70% plus one unrelated ingredient).

**Train/eval split.** `split_bucket(id) = int(md5(str(id)).hexdigest(), 16) % 10` in `app/backend/evaluation/split.py`. Buckets 0-1 are the eval pool and the only source of eval queries (a recipe qualifies only if all its duplicates are also in buckets 0-1). Buckets 2-9 are reserved for training, e.g. the LTR ranker (`build_eval_set.py --pool train`).

**Limitations.** Only one relevant recipe per query, while many other recipes can be equally good answers, so every metric is a lower bound on real quality. Queries are literal subsets of the target's ingredient list, which favours lexical matching (BM25) over the dense retriever. Ingredient lists in the corpus are often truncated (at most 12 per recipe).

**Running it.**

```powershell
python scripts/build_eval_set.py --db "D:/path/to/data/recipes.db"
python scripts/run_eval.py --data-dir "D:/path/to/data"            # ~50 min for 1000 queries
python scripts/run_eval.py --data-dir "D:/path/to/data" --limit 50 --name smoke
```

`run_eval.py` only reads the data files. It needs `lightgbm` installed, otherwise the ML ranker cannot load; the script aborts instead of silently measuring the heuristic. Layer (a) scores raw retriever candidates at depth 100. Layer (b) scores `RecipeSearchService.search` with the production settings: 20 candidates, coverage filter, then ranking. Output goes to `eval/results/` (JSON, markdown, per-query JSONL).

**Results** (`eval/results/queries_v1_topk20.md`, 1000 queries, 95% bootstrap CI). Latency is a single process on CPU (sentence-transformers, no fastembed). Other jobs were running on the machine, so treat p95 as an upper bound.

Layer (a), retriever candidates:

| Config | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Hit@1 | p50 ms |
|---|---|---|---|---|---|---|
| BM25 | 0.645 [0.615, 0.675] | 0.855 [0.833, 0.877] | 0.456 [0.430, 0.483] | 0.501 [0.476, 0.527] | 0.369 [0.341, 0.399] | 310 |
| Dense (MiniLM) | 0.291 [0.264, 0.320] | 0.535 [0.504, 0.565] | 0.163 [0.144, 0.184] | 0.193 [0.173, 0.215] | 0.112 [0.093, 0.132] | 68 |
| Hybrid RRF | 0.478 [0.448, 0.507] | 0.821 [0.798, 0.845] | 0.284 [0.261, 0.310] | 0.330 [0.306, 0.355] | 0.209 [0.184, 0.235] | 406 |

Layer (b), final output:

| Config | Recall@10 | MRR@10 | nDCG@10 | Hit@1 | p50 ms |
|---|---|---|---|---|---|
| BM25, retrieval order | 0.655 [0.626, 0.685] | 0.482 [0.456, 0.509] | 0.524 [0.498, 0.551] | 0.393 [0.364, 0.423] | 310 |
| Dense, retrieval order | 0.331 [0.304, 0.361] | 0.213 [0.191, 0.237] | 0.241 [0.219, 0.266] | 0.160 [0.138, 0.183] | 77 |
| Hybrid, RRF order | 0.603 [0.573, 0.634] | 0.364 [0.339, 0.391] | 0.421 [0.396, 0.448] | 0.265 [0.239, 0.293] | 419 |
| Hybrid + heuristic ranking | 0.638 [0.609, 0.668] | 0.359 [0.336, 0.383] | 0.426 [0.403, 0.450] | 0.227 [0.201, 0.253] | 419 |
| Hybrid + LightGBM ranker (default) | 0.599 [0.570, 0.630] | 0.331 [0.307, 0.356] | 0.395 [0.370, 0.419] | 0.221 [0.195, 0.247] | 413 |

What the numbers say:

- Hybrid RRF is worse than BM25 alone at both layers: MRR@10 is 0.171 lower at layer (a) (paired CI [-0.194, -0.150]). On this query set the dense retriever adds noise rather than recall. Part of this is the lexical bias of the set.
- The LightGBM ranker is worse than the heuristic it replaced: Recall@10 -0.039 [-0.053, -0.026], MRR@10 -0.028 [-0.044, -0.013]. It is also worse than plain RRF order. Both rankers lower Hit@1 compared with not re-ranking.
- The app default (hybrid + ML) has MRR@10 0.331, against 0.482 for BM25 order with the coverage filter.
- Recall is capped by the candidate count, not the ranker. Hybrid reaches 0.821 recall at 100 candidates, but the pipeline keeps only 20, so final recall tops out at 0.655. About 0.7% of queries return nothing after the coverage filter.
- Latency is dominated by BM25 scoring in pure Python (p50 310 ms). Query embedding (p50 32 ms), FAISS flat search (34 ms) and RRF (1 ms) are much cheaper.
- `rule_based` was not measured. It scores every recipe per query and took about 10 minutes for a single query.
- `retrieval_mode="semantic"` currently raises `AttributeError` in `RecipeSearchService.search`. The dense layer (b) rows reproduce the intended path directly.

The legacy scripts `scripts/evaluate_retrieval.py` and `scripts/evaluate_ranking.py` only check hand-written seed examples. They are kept as smoke checks and are not measurements.
