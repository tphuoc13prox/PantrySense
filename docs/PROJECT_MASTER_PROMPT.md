# PantrySense — Project Master Prompt

> **This document is the single source of truth for the PantrySense project.**
>
> Any AI coding agent, developer, or contributor working on PantrySense MUST read and understand this document before modifying the project.

---

# 1. PROJECT IDENTITY

## Project Name
**PantrySense**

## Project Type
Local-first AI/ML-oriented recipe discovery and recommendation application.

## Current Platform
**Local Web Application**
Accessed through a web browser at `http://localhost:8000`.

---

# 2. PROJECT VISION

PantrySense helps users answer a simple everyday question:
> **"I have these ingredients. What can I make?"**

The long-term goal is to evolve PantrySense from a basic ingredient-to-recipe search tool into an intelligent food recommendation system using:
- Information Retrieval
- Natural Language Processing
- Machine Learning
- Deep Learning
- Semantic Search
- Recommendation Systems
- Computer Vision

---

# 3. CORE PRINCIPLE

> **Build a working end-to-end system first, then progressively replace simple components with intelligent ones.**

```text
Rule-based baseline (v0.3.0)
        ↓
Semantic retrieval (v0.4.0)
        ↓
Hybrid retrieval (v0.5.0)
        ↓
Learning-to-Rank (v0.6.0)
        ↓
Personalization (v0.7.0)
        ↓
Pantry Management & Ingredient Autocomplete (v0.8.0)
        ↓
Multimodal AI / Computer Vision (v0.9.0)
        ↓
Intelligent PantrySense (v1.0.0)
```

---

# 4. VERSION ROADMAP & SPECS

## v0.1.0 — Basic End-to-End Recipe Search
- Basic ingredient input, search, SQLite database, FastAPI backend, result list.

## v0.2.0 — Recipe Details
- Recipe detail view: quantities, units, cooking instructions, cooking time, difficulty, servings, category.

## v0.3.0 / v0.3.1 — Smart Matching & Web Migration
- Deterministic string normalization, safe plural handling, synonym mapping.
- Matched ingredients, missing ingredients, ingredient coverage percentage, rule-based ranking.
- Pure local web application (HTML5, Vanilla CSS, Vanilla JS).

## v0.4.0 / v0.4.1 — Semantic Retrieval & Auto-Lifecycle
- Dense semantic retrieval with Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS (`IndexFlatIP`).
- Dual retrieval modes (`PANTRYSENSE_RETRIEVAL_MODE` = `semantic` / `rule_based`).
- Auto-open default browser on startup & 10s inactivity auto-shutdown via client heartbeat.

## v0.5.0 — Hybrid Retrieval
- Combine Lexical Matching (BM25) with Dense Semantic Retrieval (FAISS).
- Reciprocal Rank Fusion (RRF) / Score Fusion.

## v0.6.0 — Learning-to-Rank
- Replace hand-crafted ranking with Machine Learning Ranker (LightGBM Ranker / LambdaMART).
- Feature extraction: coverage, semantic similarity, BM25 score, cooking time, difficulty, missing count.
- Evaluation metrics: NDCG, MRR, Precision@K, Recall@K.

## v0.7.0 — Personalization
- User profiles, saved recipes, cooking history, category preferences, disliked/allergen ingredients.

## v0.8.0 — Pantry Management & Controlled Ingredient Autocomplete
- **Real-time Ingredient Autocomplete & Suggestion**:
  - Gợi ý tên nguyên liệu chính xác theo thời gian thực khi người dùng gõ phím (`GET /api/ingredients/suggest?q=...`).
  - Hỗ trợ kho từ vựng nguyên liệu chuẩn (*Controlled Ingredient Vocabulary / Trie Index / Fuzzy search*).
  - Triệt tiêu lỗi chính tả (Zero-Typo) và ngăn người dùng nhập các query gây lỗi, giúp chuẩn hóa dữ liệu đầu vào.
- **My Pantry (Tủ nguyên liệu tại nhà)**:
  - Lưu trữ trạng thái kho nguyên liệu hiện có trong gia đình vào SQLite.
  - Thêm, sửa, xóa, cập nhật định lượng, đánh dấu hạn sử dụng (expiry dates).
  - Gợi ý món ăn trực tiếp từ kho nguyên liệu có sẵn mà không cần gõ lại mỗi lần.

## v0.9.0 — Computer Vision
- Nhận diện nguyên liệu từ hình ảnh chụp thực phẩm/tủ lạnh (Object Detection / YOLO / Classification) và tự động đưa vào Pantry.

## v1.0.0 — Intelligent PantrySense
- Hệ thống hoàn chỉnh tích hợp toàn bộ pipeline: Pantry State + Image Input + Hybrid Retrieval + ML Ranking + Personalization.

---

# 5. AI/ML DEVELOPMENT PHILOSOPHY

Do not add AI simply because the project is labeled AI. Every AI/ML component must answer:
> **What problem does this solve better than the previous baseline?**

---

# 6. LOCAL-FIRST & PRIVACY

PantrySense is designed to work completely offline once software, models, and data are installed locally. User pantry data and cooking habits remain strictly local.
