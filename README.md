# PantrySense

PantrySense is an offline-first, local AI/ML-oriented recipe discovery application. In v0.4.0, PantrySense introduces a local **Semantic Retrieval** layer powered by Sentence Transformers and FAISS, enabling intelligent candidate recipe retrieval across linguistic variations while preserving deterministic ingredient feasibility analysis.

Current version: v0.4.0

## Architecture

```text
                                Web Browser (localhost:8000)
                                            │
                                            ▼
                                     FastAPI Backend
                                            │
                                            ▼
                                  RecipeSearchService
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
             Rule-Based Mode                               Semantic Mode
                      │                                           │
                      │                                    Normalized Query
                      │                                           │
                      │                                  Query Representation
                      │                                           │
                      │                                 IngredientEmbedder
                      │                                 (all-MiniLM-L6-v2)
                      │                                           │
                      │                                   FaissVectorStore
                      │                                   (IndexFlatIP Search)
                      │                                           │
                      │                                Top-K Candidates (IDs)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                               Candidate Database Fetch
                                            │
                                            ▼
                               IngredientMatcher (v0.3.x)
                                 ├── Matched Ingredients
                                 ├── Missing Ingredients
                                 └── Coverage Ratio
                                            │
                                            ▼
                                 RecipeRanker (v0.3.x)
                                            │
                                            ▼
                                      Recipe Results
```

### Candidate Generation vs. Feasibility Analysis
- **Semantic Retrieval**: Answers *"Which recipes might be semantically relevant to the user's ingredients?"*
- **Ingredient Matching**: Answers *"How well do the user's ingredients actually satisfy the candidate recipes?"*

## Implemented Features

- Local web application UI (HTML5, Vanilla CSS, Vanilla JavaScript)
- **Local Semantic Retrieval Layer**:
  - `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
  - `FAISS` (`IndexFlatIP` cosine similarity search) with NumPy fallback
  - Deterministic recipe and user query representations
  - Pre-built offline index (loaded once at startup)
- **Dual Retrieval Modes** (`PANTRYSENSE_RETRIEVAL_MODE`):
  - `semantic` (default): Dense candidate retrieval followed by ingredient feasibility analysis.
  - `rule_based`: Deterministic v0.3.x set-matching baseline.
- Deterministic Ingredient Matching: matched ingredients, missing ingredients, matched count, required count, and coverage.
- Single-page view switching between search results and recipe details (with back navigation).
- Offline Vector Index Build Script (`scripts/build_vector_index.py`).
- Empirical Retrieval Benchmark Script (`scripts/evaluate_retrieval.py`).
- Full automated test suite covering unit tests, vector store, retriever, and regression.

## Not Implemented in v0.4.0

- BM25 / Sparse lexical indexing
- Reciprocal Rank Fusion (RRF) / Hybrid Fusion
- Learning-to-Rank / Machine Learning Ranking (LambdaMART, LightGBM)
- User Personalization & Cooking history
- Computer Vision / Image ingredient recognition
- Cloud / LLM external APIs

## Requirements

- Python 3.11 or newer
- Dependencies in `requirements.txt` (FastAPI, Uvicorn, pytest, httpx, sentence-transformers, faiss-cpu, numpy)

## Installation

```powershell
python -m pip install -r requirements.txt
```

## First-Time Setup: Build the Vector Index

Generate the local SQLite database and pre-compute recipe vector embeddings:

```powershell
python scripts/build_vector_index.py
```

This will create:
- `data/recipes.db` (SQLite recipe database)
- `data/recipe_vectors.index` (FAISS vector index)
- `data/recipe_vector_ids.json` (ID mappings and index metadata)

## Run the Application

Start the backend server:

```powershell
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open your web browser at:

```text
http://localhost:8000
```
*(or `http://127.0.0.1:8000`)*

## Retrieval Benchmark Evaluation

Compare retrieval accuracy between Rule-Based and Semantic Retrieval across 5 benchmark categories (Synonyms, Morphology, Specific/General, Multi-ingredient, Negative queries):

```powershell
python scripts/evaluate_retrieval.py
```

## Configuration

You can configure PantrySense using environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PANTRYSENSE_RETRIEVAL_MODE` | `semantic` | Retrieval mode: `semantic` or `rule_based`. |
| `PANTRYSENSE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model identifier for embeddings. |
| `PANTRYSENSE_SEMANTIC_TOP_K` | `20` | Maximum candidate recipes retrieved via vector search. |
| `PANTRYSENSE_SEMANTIC_THRESHOLD` | `0.35` | Minimum cosine similarity threshold for candidates. |
| `PANTRYSENSE_MINIMUM_COVERAGE` | `0.34` | Minimum ingredient coverage threshold for recipe display. |
| `PANTRYSENSE_DB_PATH` | `data/recipes.db` | Path to SQLite database file. |
| `PANTRYSENSE_VECTOR_INDEX_PATH` | `data/recipe_vectors.index` | Path to FAISS index file. |

## Run Automated Tests

```powershell
python -m pytest
```

## Known Limitations

`all-MiniLM-L6-v2` is a general-purpose sentence transformer, not a dedicated culinary ontology. It improves retrieval for many natural variations (e.g. `scallions`, `chicken breast`), but semantic similarity is not equivalent to perfect culinary knowledge. This serves as the dense retrieval baseline for future hybrid and learning-to-rank improvements.
