# PantrySense

PantrySense is an offline-first, local AI/ML-oriented recipe discovery, pantry management, and culinary assistant application. In **v1.0.1**, PantrySense delivers a complete smart kitchen ecosystem featuring **Dietary & Allergen Filtering**, **Smart Ingredient Substitution**, **Macro & Calorie Nutrition Estimation**, **Virtual Pantry & Expiry Tracker**, **Weekly 7-Day Meal Planner**, **Pantry-Aware Grocery Shopping List Generator**, **Interactive Step-by-Step Cooking Assistant with Audio Timers & Step Navigator**, **Controlled Ingredient Autocomplete & Fuzzy Spellchecker** (4,588-ingredient culinary vocabulary), and **Two-Stage Hybrid AI Retrieval + Learning-to-Rank (LTR)**.

Current version: **v1.0.1**

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
