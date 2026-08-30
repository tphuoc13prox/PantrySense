from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.recipes.service import RecipeSearchService

logging.basicConfig(level=logging.WARNING)

BENCHMARK_CASES = [
    # Category A: Synonym-like variation
    {
        "category": "A. Synonym variation",
        "query": ["scallion"],
        "expected_titles": ["Tomato Egg Stir Fry"],  # Has green onion
    },
    {
        "category": "A. Synonym variation",
        "query": ["capsicum"],
        "expected_titles": ["Chicken Tomato Stir Fry"],
    },
    # Category B: Morphological variation
    {
        "category": "B. Morphological variation",
        "query": ["tomatoes"],
        "expected_titles": ["Chicken Tomato Stir Fry", "Tomato Egg Stir Fry"],
    },
    {
        "category": "B. Morphological variation",
        "query": ["eggs"],
        "expected_titles": ["Chicken Omelette", "Tomato Egg Stir Fry"],
    },
    # Category C: Specific / General relationship
    {
        "category": "C. Specific/General",
        "query": ["chicken breast"],
        "expected_titles": ["Chicken Omelette", "Chicken Tomato Stir Fry"],
    },
    # Category D: Multi-ingredient queries
    {
        "category": "D. Multi-ingredient",
        "query": ["chicken", "egg"],
        "expected_titles": ["Chicken Omelette"],
    },
    {
        "category": "D. Multi-ingredient",
        "query": ["chicken", "tomato", "egg"],
        "expected_titles": ["Chicken Omelette", "Chicken Tomato Stir Fry", "Tomato Egg Stir Fry"],
    },
    # Category E: Negative queries (unrelated)
    {
        "category": "E. Negative queries",
        "query": ["durian"],
        "expected_titles": [],
    },
    {
        "category": "E. Negative queries",
        "query": ["chocolate"],
        "expected_titles": [],
    },
]


def run_benchmark() -> None:
    print("=" * 105)
    print("PantrySense Retrieval Benchmark: Rule-Based vs. BM25 vs. Semantic vs. Hybrid (RRF)")
    print("=" * 105)

    rule_service = RecipeSearchService(retrieval_mode="rule_based", match_threshold=0.0)
    bm25_service = RecipeSearchService(retrieval_mode="lexical", match_threshold=0.0)
    semantic_service = RecipeSearchService(retrieval_mode="semantic", match_threshold=0.0)
    hybrid_service = RecipeSearchService(retrieval_mode="hybrid", match_threshold=0.0)

    # Check readiness
    if not hybrid_service.hybrid_retriever.is_ready():
        print("\n[ERROR] Hybrid indices (FAISS/BM25) not found. Please build them first with:")
        print("    python scripts/build_vector_index.py\n")
        sys.exit(1)

    modes = [
        ("Rule-Based", rule_service),
        ("BM25 (Lexical)", bm25_service),
        ("Semantic (Dense)", semantic_service),
        ("Hybrid (RRF)", hybrid_service),
    ]

    stats = {
        name: {"hits": 0, "rr_list": []}
        for name, _ in modes
    }
    total_evaluable = 0

    header = f"{'Query':<22} | {'Rule-Based':<14} | {'BM25':<14} | {'Semantic':<18} | {'Hybrid (RRF)':<20}"
    print(header)
    print("-" * len(header))

    for case in BENCHMARK_CASES:
        query = case["query"]
        expected = case["expected_titles"]
        query_str = ", ".join(query)
        is_negative = len(expected) == 0

        row_displays = []

        if not is_negative:
            total_evaluable += 1

        for name, srv in modes:
            results = srv.search(query)
            titles = [r.title for r in results]

            if not is_negative:
                rank = next((i + 1 for i, t in enumerate(titles) if t in expected), None)
                if rank is not None:
                    stats[name]["hits"] += 1
                    stats[name]["rr_list"].append(1.0 / rank)
                    disp = f"Rank {rank}"
                else:
                    stats[name]["rr_list"].append(0.0)
                    disp = "NOT FOUND"
            else:
                disp = "PASS (0)" if len(results) == 0 else f"FOUND ({len(results)})"

            row_displays.append(disp)

        print(f"{query_str:<22} | {row_displays[0]:<14} | {row_displays[1]:<14} | {row_displays[2]:<18} | {row_displays[3]:<20}")

    print("=" * 105)
    print("Aggregate Benchmark Summary (Positive Retrieval Queries):")
    print(f"{'Mode':<20} | {'Hits / Total':<15} | {'Recall':<12} | {'MRR':<10}")
    print("-" * 65)

    for name, _ in modes:
        hits = stats[name]["hits"]
        recall = (hits / total_evaluable) * 100 if total_evaluable else 0.0
        rr_list = stats[name]["rr_list"]
        mrr = sum(rr_list) / len(rr_list) if rr_list else 0.0
        print(f"{name:<20} | {hits}/{total_evaluable:<13} | {recall:>6.1f}%     | {mrr:>6.3f}")

    print("=" * 105)


if __name__ == "__main__":
    run_benchmark()
