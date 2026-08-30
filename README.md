# PantrySense

PantrySense is an offline-first, local AI/ML-oriented recipe discovery and recommendation application. In v0.5.0, PantrySense introduces **Hybrid Retrieval**, combining **Sparse Lexical Matching (BM25)** with **Dense Semantic Vector Search (FAISS)** using **Reciprocal Rank Fusion (RRF)** for optimal candidate recipe retrieval, backed by deterministic ingredient feasibility analysis.

Current version: v0.5.0

## Architecture

```text
                                Web Browser (localhost:8000)
                                            │
                                            ▼ (Heartbeat ping every 2.5s)
                                     FastAPI Backend
                                            │
                                            ▼
                                  RecipeSearchService
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
    Rule-Based Mode                  Semantic Mode                     Hybrid Mode
        (v0.3)                           (v0.4)                       (v0.5 MỚI)
           │                                │                                │
           │                                │                ┌───────────────┴───────────────┐
           │                                │                ▼                               ▼
           │                                │           Sparse BM25                     Dense FAISS
           │                                │         (Lexical Match)                (Semantic Vector)
           │                                │                │                               │
           │                                │                └───────────────┬───────────────┘
           │                                │                                ▼
           │                                │                     Reciprocal Rank Fusion
           │                                │                             (RRF)
           │                                │                                │
           │                                └────────────────┬───────────────┘
           │                                                 ▼
           └────────────────────────────────────────┬────────┘
                                                    ▼
                                         Top-K Candidate Recipes
                                                    │
                                                    ▼
                                        Ingredient Feasibility (v0.3.x)
                                          ├── Matched Ingredients
                                          ├── Missing Ingredients
                                          └── Coverage Calculation
                                                    │
                                                    ▼
                                          RecipeRanker (v0.3.x)
                                                    │
                                                    ▼
                                              Final Results
```

### Why Hybrid Retrieval?
- **Sparse BM25**: Excels at exact keyword matching and rare/unique ingredient terms.
- **Dense FAISS**: Excels at semantic context, synonyms, and morphological variations.
- **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists without needing fragile score calibration:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k=60)$$

## Implemented Features

- Local web application UI (HTML5, Vanilla CSS, Vanilla JavaScript).
- **Auto-Open Browser**: Automatically opens a new tab in your default browser once the server is ready.
- **Auto-Shutdown on Inactivity**: Automatically shuts down server when no active tabs remain for 10 seconds.
- **Hybrid Retrieval Layer (v0.5.0)**:
  - BM25Okapi sparse lexical indexing and scoring.
  - Sentence Transformers (`all-MiniLM-L6-v2`) dense vector embedding.
  - FAISS (`IndexFlatIP`) cosine similarity search.
  - Reciprocal Rank Fusion (RRF) candidate fusion.
- **Multi-Mode Retrieval Toggle (`PANTRYSENSE_RETRIEVAL_MODE`)**:
  - `hybrid` (default): BM25 + FAISS + RRF.
  - `semantic`: FAISS dense retrieval only.
  - `lexical`: BM25 sparse lexical search only.
  - `rule_based`: Deterministic v0.3.x set-matching baseline.
- Deterministic Ingredient Matching: matched ingredients, missing ingredients, matched count, required count, and coverage.
- Single-page view switching between search results and recipe details (with back navigation).
- Offline Vector & BM25 Index Build Script (`scripts/build_vector_index.py`).
- 4-Way Retrieval Benchmark Script (`scripts/evaluate_retrieval.py`).
- Full automated test suite covering unit tests, vector store, BM25, RRF fusion, retriever, heartbeat monitor, and regression.

## Not Implemented in v0.5.0

- Learning-to-Rank / Machine Learning Ranking (LambdaMART, LightGBM)
- User Personalization & Cooking history
- Controlled Ingredient Autocomplete / Pantry Management (v0.8.0)
- Computer Vision / Image ingredient recognition
- Cloud / LLM external APIs

## Requirements

- Python 3.11 or newer
- Dependencies in `requirements.txt` (FastAPI, Uvicorn, pytest, httpx, sentence-transformers, faiss-cpu, numpy, rank-bm25)

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Indexing & First-Time Setup

PantrySense features **Smart Auto-Indexing**: when starting the server, if index files are missing, it will automatically compute and cache them on the first run.

You can also manually build or refresh the indices anytime:

```powershell
python scripts/build_vector_index.py
```

This generates:
- `data/recipes.db` (SQLite recipe database)
- `data/recipe_vectors.index` (FAISS vector index)
- `data/recipe_vector_ids.json` (ID mappings and index metadata)
- `data/recipe_bm25.json` (BM25 lexical index)


## Run the Application

Start the backend server:

```powershell
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

> **Behavior**:
> 1. Server starts and loads SQLite, FAISS, and BM25 indices.
> 2. Automatically opens your default external browser at `http://localhost:8000`.
> 3. Automatically shuts down after **10 seconds** if all PantrySense tabs are closed.

## Multi-Mode Benchmark Evaluation

Compare retrieval accuracy across all 4 modes (Rule-Based vs. BM25 vs. Semantic vs. Hybrid):

```powershell
python scripts/evaluate_retrieval.py
```

## Configuration

You can configure PantrySense using environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PANTRYSENSE_RETRIEVAL_MODE` | `hybrid` | Mode: `hybrid`, `semantic`, `lexical`, `rule_based`. |
| `PANTRYSENSE_RRF_K` | `60` | Constant $k$ for Reciprocal Rank Fusion. |
| `PANTRYSENSE_AUTO_OPEN_BROWSER` | `true` | Auto-open browser on startup (`true`/`false`). |
| `PANTRYSENSE_AUTO_SHUTDOWN` | `true` | Auto-shutdown on 10s inactivity (`true`/`false`). |
| `PANTRYSENSE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier. |
| `PANTRYSENSE_SEMANTIC_TOP_K` | `20` | Maximum candidate pool size. |
| `PANTRYSENSE_SEMANTIC_THRESHOLD` | `0.35` | Minimum cosine similarity threshold for dense candidates. |
| `PANTRYSENSE_MINIMUM_COVERAGE` | `0.34` | Minimum ingredient coverage threshold for display. |
| `PANTRYSENSE_DB_PATH` | `data/recipes.db` | SQLite database path. |
| `PANTRYSENSE_VECTOR_INDEX_PATH` | `data/recipe_vectors.index` | FAISS index path. |
| `PANTRYSENSE_BM25_INDEX_PATH` | `data/recipe_bm25.json` | BM25 index path. |

## Run Automated Tests

```powershell
python -m pytest
```
