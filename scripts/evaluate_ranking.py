from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.ranking.ml_ranker import MLRecipeRanker
from app.backend.recipes.schemas import RecipeSummary

TEST_RANKING_CASES = [
    {
        "query": "quick chicken egg breakfast",
        "candidates": [
            (
                RecipeSummary(id=1, title="Tomato Egg Stir Fry", coverage=0.5, matched_count=1, missing_ingredients=["tomato"], semantic_score=0.70, bm25_score=2.0, rrf_score=0.018),
                {"cooking_time": 10, "difficulty": "easy", "ingredients": ["tomato", "egg", "green onion"]},
                1,  # Partial relevance
            ),
            (
                RecipeSummary(id=2, title="Chicken Omelette", coverage=0.8, matched_count=2, missing_ingredients=["onion"], semantic_score=0.92, bm25_score=4.5, rrf_score=0.032),
                {"cooking_time": 15, "difficulty": "easy", "ingredients": ["chicken", "egg", "onion", "butter"]},
                3,  # Target ideal top-1
            ),
            (
                RecipeSummary(id=3, title="Chicken Tomato Stir Fry", coverage=0.5, matched_count=1, missing_ingredients=["tomato", "garlic"], semantic_score=0.65, bm25_score=2.2, rrf_score=0.016),
                {"cooking_time": 20, "difficulty": "easy", "ingredients": ["chicken", "tomato", "garlic", "soy sauce"]},
                2,  # Good partial
            ),
        ],
    },
    {
        "query": "quick garlic pasta",
        "candidates": [
            (
                RecipeSummary(id=4, title="Creamy Garlic Parmesan Pasta", coverage=1.0, matched_count=2, missing_ingredients=[], semantic_score=0.95, bm25_score=5.0, rrf_score=0.033),
                {"cooking_time": 18, "difficulty": "medium", "ingredients": ["pasta", "garlic", "cream", "cheese"]},
                3,  # Target ideal top-1
            ),
            (
                RecipeSummary(id=6, title="Classic Beef Bolognese", coverage=0.5, matched_count=1, missing_ingredients=["beef", "tomato"], semantic_score=0.60, bm25_score=2.0, rrf_score=0.015),
                {"cooking_time": 35, "difficulty": "medium", "ingredients": ["beef", "pasta", "tomato", "onion", "garlic"]},
                2,  # Slower, more complex
            ),
        ],
    },
]


def dcg_at_k(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return score


def ndcg_at_k(relevances: list[float], k: int) -> float:
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_ranking() -> None:
    print("=" * 80)
    print("PantrySense Ranking Evaluation: Heuristic (v0.5) vs. ML Ranker (v0.6)")
    print("=" * 80)

    ranker = MLRecipeRanker()
    if not ranker.is_ready:
        print("[NOTICE] Training ML ranker model first...")
        from scripts.train_ranker import train_ranker
        train_ranker()
        ranker.load_model()

    heuristic_ndcg3, ml_ndcg3 = [], []
    heuristic_p1, ml_p1 = [], []

    for idx, case in enumerate(TEST_RANKING_CASES, start=1):
        query = case["query"]
        candidates = [c[0] for c in case["candidates"]]
        details_map = {c[0].id: c[1] for c in case["candidates"]}
        ground_truth = {c[0].id: c[2] for c in case["candidates"]}

        # 1. Heuristic Ranking
        h_ranked = ranker.rank(candidates, details_map=details_map, force_heuristic=True)
        h_rels = [ground_truth[r.id] for r in h_ranked]
        h_ndcg = ndcg_at_k(h_rels, k=3)
        heuristic_ndcg3.append(h_ndcg)
        heuristic_p1.append(1.0 if h_rels and h_rels[0] == max(ground_truth.values()) else 0.0)

        # 2. ML Ranking
        ml_ranked = ranker.rank(candidates, details_map=details_map, force_heuristic=False)
        ml_rels = [ground_truth[r.id] for r in ml_ranked]
        m_ndcg = ndcg_at_k(ml_rels, k=3)
        ml_ndcg3.append(m_ndcg)
        ml_p1.append(1.0 if ml_rels and ml_rels[0] == max(ground_truth.values()) else 0.0)

        print(f"\nQuery {idx}: '{query}'")
        print(f"  Heuristic Top-1: {h_ranked[0].title:<30} (NDCG@3 = {h_ndcg:.4f})")
        print(f"  ML Ranker Top-1: {ml_ranked[0].title:<30} (NDCG@3 = {m_ndcg:.4f}, ML Score = {ml_ranked[0].ml_score})")

    print("\n" + "=" * 80)
    print("Aggregate Ranking Benchmark Summary:")
    print(f"{'Ranking Method':<25} | {'Avg NDCG@3':<15} | {'Precision@1':<15}")
    print("-" * 60)
    print(f"{'Heuristic Sorting (v0.5)':<25} | {sum(heuristic_ndcg3)/len(heuristic_ndcg3):>10.4f}     | {sum(heuristic_p1)/len(heuristic_p1)*100:>10.1f}%")
    print(f"{'ML Ranker / LTR (v0.6)':<25} | {sum(ml_ndcg3)/len(ml_ndcg3):>10.4f}     | {sum(ml_p1)/len(ml_p1)*100:>10.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_ranking()
