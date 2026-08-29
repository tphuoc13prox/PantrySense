# PantrySense

PantrySense is an offline-first recipe discovery application. In v0.3.1, the application runs as a local web application accessible through your browser. Users enter ingredients they have at home, receive ranked recipe matches from a local SQLite database, see ingredient coverage and missing ingredients, and select a recipe to view details.

Current version: v0.3.1

## Architecture

```text
Web Browser (HTML5 / CSS / Vanilla JS)
        |
        | localhost HTTP (http://localhost:8000)
        v
FastAPI Python Backend
        |
        v
SQLite Recipe Database
        |
        v
Smart Ingredient Matching
```

The browser UI sends ingredient names to the FastAPI backend. The backend normalizes strings, searches SQLite, performs deterministic set-based matching, calculates ingredient coverage and missing ingredients, ranks results with simple rules, and returns recipe IDs, titles, and match information.

When a user selects a recipe, the web UI requests details from the backend (`GET /api/recipes/{recipe_id}`) and displays the local database content.

## Implemented Features

- Local web application UI (HTML5, Vanilla CSS, Vanilla JavaScript)
- Ingredient input, add (with Enter key support), individual remove ('x'), and clear actions
- Recipe search button and user-friendly status/loading messages
- Backend error state handling in the web UI
- `GET /` (serves the PantrySense web application)
- `GET /health`
- `POST /api/recipes/search`
- `GET /api/recipes/{recipe_id}`
- SQLite recipe database initialization
- Small seed/demo recipe dataset
- Recipe detail fields: ingredients, quantities where available, instructions, cooking time, difficulty, servings, and category
- Single-page view switching between search results and recipe details (with back navigation)
- Smart Ingredient Matching with matched ingredients, missing ingredients, matched count, required count, and coverage
- Small deterministic synonym map for terms such as `tomatoes`, `eggs`, `bell pepper`, and `capsicum`
- Rule-based ranking by coverage, matched count, missing count, and title
- Configurable minimum coverage through `PANTRYSENSE_MINIMUM_COVERAGE`
- Backend tests for root HTML serving, static assets, health, search, search regression, recipe details, missing recipes, empty input, unknown ingredients, and malformed requests

## Not Implemented in v0.3.1

- BM25
- FAISS
- Embeddings
- ML ranking
- Personalization
- Computer vision
- LLM or cloud AI APIs
- User accounts or authentication

## Requirements

- Python 3.11 or newer
- Dependencies listed in `requirements.txt` (FastAPI, Uvicorn, pytest, httpx)

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Initialize the Database

The backend initializes the SQLite database automatically on startup. You can also initialize it manually:

```powershell
python scripts/initialize_db.py
```

The default database path is:

```text
data/recipes.db
```

To use a different database path:

```powershell
$env:PANTRYSENSE_DB_PATH = "C:\path\to\recipes.db"
python scripts/initialize_db.py
```

## Run the Application

Start the backend server:

```powershell
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Then open your web browser and navigate to:

```text
http://localhost:8000
```

or:

```text
http://127.0.0.1:8000
```

### Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Recipe Search Example

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/recipes/search `
  -ContentType "application/json" `
  -Body '{"ingredients":["chicken","tomato","egg"]}'
```

Search response shape:

```json
{
  "recipes": [
    {
      "id": 3,
      "title": "Chicken Omelette",
      "matched_ingredients": ["chicken", "egg"],
      "missing_ingredients": ["butter", "onion"],
      "matched_count": 2,
      "required_count": 4,
      "coverage": 0.5
    },
    {
      "id": 1,
      "title": "Chicken Tomato Stir Fry",
      "matched_ingredients": ["chicken", "tomato"],
      "missing_ingredients": ["cooking oil", "garlic", "onion"],
      "matched_count": 2,
      "required_count": 5,
      "coverage": 0.4
    }
  ]
}
```

### Recipe Detail Example

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/recipes/3
```

## Run Tests

```powershell
python -m pytest
```

## Recipe Database

The v0.3.1 database continues to use one simple table:

```text
recipes
- id INTEGER PRIMARY KEY
- title TEXT
- ingredients TEXT
- ingredient_details TEXT
- instructions TEXT
- cooking_time INTEGER
- difficulty TEXT
- servings INTEGER
- category TEXT
```

Ingredients, ingredient details, and instructions are stored as JSON text for the seed/demo dataset.

## Smart Ingredient Matching

The backend compares normalized user ingredients with each recipe's ingredient set. A recipe is returned when ingredient coverage reaches the configured threshold.

Normalization currently handles:

- lowercase
- leading/trailing whitespace
- repeated spaces
- simple safe plural forms
- a small synonym map

Default threshold:

```text
PANTRYSENSE_MINIMUM_COVERAGE=0.34
```

`PANTRYSENSE_MATCH_THRESHOLD` is still accepted as a backward-compatible fallback.

Coverage is calculated as:

```text
matched_required_ingredients / total_required_ingredients
```

Results are ranked by:

```text
1. coverage descending
2. matched ingredient count descending
3. missing ingredient count ascending
4. title ascending
```
