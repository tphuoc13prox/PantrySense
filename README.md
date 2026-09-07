# PantrySense

PantrySense is an offline-first, local AI/ML-oriented recipe discovery and recommendation application. In **v0.8.2**, PantrySense features a **Controlled Ingredient Autocomplete & Fuzzy Spellchecker** (6,454-ingredient culinary vocabulary with prefix, substring, and typo correction), **Dual-Engine High-Speed AI Acceleration (PyTorch CUDA & FastEmbed ONNX INT8)**, an interactive **Header Model Engine Selector Dropdown**, **Full-Scale Complete Recipe Archive Ingestion (315,447 unique recipes from 380,000 records via Hugging Face Parquet Stream)**, **Inverted Index BM25 Lexical Retrieval**, **Git Cleanliness with Zero Binary Bloat**, and **Two-Stage Search with Learning-to-Rank (LTR)** combining **Hybrid Candidate Retrieval (BM25 + FAISS + RRF)** with a **Machine Learning Re-Ranker (LightGBM / LambdaMART)**.

Current version: **v0.8.2**

## Architecture

```text
                                Web Browser (localhost:8000)
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      🔤 Controlled Autocomplete   🔍 Two-Stage Search Engine   💓 Heartbeat Pulse (2s)
      (Prefix / Substring / Typo)  (Hybrid + ML Re-Ranker)      (Auto-Shutdown Daemon)
               │                            │                            │
               ▼                            ▼                            ▼
      GET /api/ingredients/suggest  POST /api/recipes/search    POST /api/system/heartbeat
      (6,454 Vocab In-Memory DB)   (PyTorch CUDA / FastEmbed)   (Tab Tracking State)
```

## Implemented Features

- **Controlled Ingredient Autocomplete & Fuzzy Spellchecker**:
  - 🔤 **6,454-Ingredient Vocabulary**: In-memory frequency dictionary extracted directly from all 315,447 recipes in the dataset.
  - ⚡ **3-Tier Matching Algorithm**:
    - **Exact Prefix Match**: Prioritizes ingredients starting with the search string (e.g. `gar` -> `garlic`, `garlic powder`, `garam masala`).
    - **Substring / Word Boundary**: Matches internal tokens across multi-word culinary terms (e.g. `cheese` -> `cheddar cheese`, `parmesan cheese`).
    - **Fuzzy Typo Correction**: Automatic typo detection and suggestion via `difflib.SequenceMatcher` (e.g. `chikcen` -> `chicken`, `tomto` -> `tomato`, `spagetti` -> `spaghetti`).
  - 🎯 **Interactive UI Experience**:
    - Full keyboard navigation: `ArrowUp`, `ArrowDown`, `Enter`, `Tab` (instant completion), and `Escape`.
    - Real-time matched substring highlights and recipe occurrence counts (`N recipes`).
    - Visual `✨ Did you mean?` typo badges.
    - Debounced client requests (120ms) and sub-millisecond backend lookup (< 1ms).

- **Dual-Engine High-Speed Acceleration**:
  - 🚀 **PyTorch CUDA (NVIDIA GPU)**: Ultra-fast batch inference reaching ~2,600+ recipes/second on NVIDIA GPUs.
  - ⚡ **FastEmbed ONNX INT8 (CPU)**: Highly-optimized quantized ONNX runtime inference (~126 recipes/second) requiring no dedicated GPU and minimal CPU overhead.
  - **Auto-Detection**: Onboarding screen automatically identifies whether a compatible NVIDIA GPU is present and selects the optimal engine by default.

- **Complete Full-Scale Recipe Dataset Ingestion (315,447 Recipes)**:
  - **Comprehensive Culinary Corpus**: Ingests all 380,000 dataset records (yielding 315,447 unique culinary dishes) directly from the Hugging Face / Food.com Parquet archive with zero loss or downsampling.
  - **Inverted Index BM25**: Optimized inverted posting list lookup delivers sub-millisecond lexical scoring across 315,447 recipes.
  - **Seamless Initial Setup**: Auto-detects dataset presence and visualizes the 8-step pipeline with real-time percentage progress.

- **Header Model Engine Dropdown Selector**:
  - Interactive dropdown in the header allowing instantaneous switching between:
    - 🚀 **PyTorch CUDA (all-MiniLM-L6-v2)**: Optimized for PCs with dedicated NVIDIA GPUs (~2,600+ items/s).
    - ⚡ **FastEmbed ONNX INT8 (all-MiniLM-L6-v2)**: Quantized SIMD CPU inference (~126 items/s).
  - Automatically detects GPU hardware on startup and sets the optimal model engine.

- **High-Performance Parquet Ingestion (`pyarrow`)**:
  - Direct streaming from the 522k recipe Food.com / Hugging Face Parquet archive (`0000.parquet`).
  - Sub-second parsing and high-throughput batch SQL ingestion.

- **Enhanced Multi-Ingredient Search & Ranking**:
  - Fixed recipe ranking so multi-ingredient searches (e.g., `Pork` + `Noodles`) strictly prioritize dishes containing all requested ingredients.
  - Expanded built-in catalogs with savory pork dishes, noodle soups, braised pork, and diverse Asian/Western combinations.

- **Learning-to-Rank (LTR) Machine Learning Model**:
  - 9-dimensional query-candidate feature extraction.
  - LightGBM LambdaMART / GBDT model training and real-time inference.
  - Graceful heuristic fallback if model is absent.

- **Hybrid Retrieval Layer**:
  - BM25Okapi sparse lexical indexing and scoring.
  - Sentence Transformers / FastEmbed dense vector embedding (`all-MiniLM-L6-v2` / `BAAI/bge-small-en-v1.5`).
  - FAISS (`IndexFlatIP`) cosine similarity search.
  - Reciprocal Rank Fusion (RRF) candidate fusion.

- **Auto-Open Browser & Tab-Lifecycle Auto-Shutdown**: Automatically opens default browser upon server launch and keeps server alive as long as any browser tab is open (using robust Web Worker heartbeat pulses). Shuts down cleanly only when all browser tabs are closed.

- **Multi-Mode Toggles**:
  - Retrieval modes (`PANTRYSENSE_RETRIEVAL_MODE`): `hybrid`, `semantic`, `lexical`, `rule_based`.
  - Ranking modes (`PANTRYSENSE_RANKING_MODE`): `ml`, `heuristic`.
  - Embedder engine (`PANTRYSENSE_EMBEDDER_ENGINE`): `auto`, `cuda`, `onnx`, `cpu`.

- Deterministic Ingredient Matching: matched ingredients, missing ingredients, matched count, required count, and coverage.
- Single-page view switching between search results and recipe details (with back navigation).
- Offline Scripts for Indexing, Model Training, and Benchmark Evaluations.
- Full automated test suite (32/32 passing tests) covering all modules and regression safety.

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
> 1. Server starts and verifies local indices and ML ranker model (auto-initializing if first run).
> 2. Automatically opens your default external browser at `http://localhost:8000`.
> 3. If first run, the **Onboarding Setup Screen** displays hardware selection (GPU vs CPU FastEmbed) and dataset size choices.
> 4. Automatically keeps the server alive via background Web Worker pulses; shuts down only after all PantrySense tabs are closed.

## Offline Training & Evaluation Scripts

### 1. Build Vector & Lexical Indices
```powershell
python scripts/build_vector_index.py
```

### 2. Train the Machine Learning Ranker
```powershell
python scripts/train_ranker.py
```

### 3. Evaluate Ranking Quality (NDCG@3, NDCG@5, MRR, Precision@1)
```powershell
python scripts/evaluate_ranking.py
```

### 4. Evaluate Retrieval Quality (4-Way Retrieval Benchmark)
```powershell
python scripts/evaluate_retrieval.py
```

## Configuration

You can configure PantrySense using environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PANTRYSENSE_EMBEDDER_ENGINE` | `auto` | Embedding engine: `auto`, `cuda`, `onnx`, `cpu`. |
| `PANTRYSENSE_DATASET_DOWNLOAD_LIMIT` | `2000` | Dataset download ingestion limit (e.g. 500 or 2000). |
| `PANTRYSENSE_RANKING_MODE` | `ml` | Ranking mode: `ml` or `heuristic`. |
| `PANTRYSENSE_RETRIEVAL_MODE` | `hybrid` | Retrieval mode: `hybrid`, `semantic`, `lexical`, `rule_based`. |
| `PANTRYSENSE_RANKER_MODEL_PATH` | `data/ranker_model.joblib` | Path to trained ML ranker artifact. |
| `PANTRYSENSE_RRF_K` | `60` | Constant $k$ for Reciprocal Rank Fusion. |
| `PANTRYSENSE_AUTO_OPEN_BROWSER` | `true` | Auto-open browser on startup (`true`/`false`). |
| `PANTRYSENSE_AUTO_SHUTDOWN` | `true` | Auto-shutdown on tab close (`true`/`false`). |
| `PANTRYSENSE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier. |
| `PANTRYSENSE_SEMANTIC_TOP_K` | `20` | Maximum candidate pool size. |
| `PANTRYSENSE_SEMANTIC_THRESHOLD` | `0.35` | Minimum cosine similarity threshold for dense candidates. |
| `PANTRYSENSE_MINIMUM_COVERAGE` | `0.28` | Minimum ingredient coverage threshold for display. |
| `PANTRYSENSE_DB_PATH` | `data/recipes.db` | SQLite database path. |
| `PANTRYSENSE_VECTOR_INDEX_PATH` | `data/recipe_vectors.index` | FAISS index path. |
| `PANTRYSENSE_BM25_INDEX_PATH` | `data/recipe_bm25.json` | BM25 index path. |

## Run Automated Tests

```powershell
python -m pytest
```

