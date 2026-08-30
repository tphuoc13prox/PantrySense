from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.database.init_db import initialize_database
from app.backend.recipes.service import RecipeSearchService
from app.backend.retrieval.semantic import SemanticRetriever

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
        "expected_titles": ["Chicken Tomato Stir Fry"],  # or pepper
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
    print("=" * 80)
    print("PantrySense Retrieval Benchmark: v0.3.x Rule-Based vs. v0.4.0 Semantic Retrieval")
    print("=" * 80)

    # Initialize services
    rule_service = RecipeSearchService(retrieval_mode="rule_based", match_threshold=0.0)
    semantic_service = RecipeSearchService(retrieval_mode="semantic", match_threshold=0.0)

    # Check if semantic retriever is ready
    if not semantic_service.retriever.is_ready():
        print("\n[ERROR] Semantic index not found. Please build the index first with:")
        print("    python scripts/build_vector_index.py\n")
        sys.exit(1)

    rule_hits = 0
    semantic_hits = 0
    total_evaluable = 0

    rule_reciprocal_ranks: list[float] = []
    semantic_reciprocal_ranks: list[float] = []

    print(f"{'Category':<22} | {'Query':<22} | {'Rule-Based':<14} | {'Semantic (Rank / Score)':<24}")
    print("-" * 88)

    for case in BENCHMARK_CASES:
        category = case["category"]
        query = case["query"]
        expected = case["expected_titles"]
        query_str = ", ".join(query)

        # Run Rule-Based
        rule_results = rule_service.search(query)
        rule_titles = [r.title for r in rule_results]

        # Run Semantic
        semantic_results = semantic_service.search(query)
        semantic_titles = [r.title for r in semantic_results]

        is_negative = len(expected) == 0

        if not is_negative:
            total_evaluable += 1

            # Rule-based evaluation
            rule_rank = next((i + 1 for i, t in enumerate(rule_titles) if t in expected), None)
            if rule_rank is not None:
                rule_hits += 1
                rule_reciprocal_ranks.append(1.0 / rule_rank)
                rule_display = f"FOUND (Rank {rule_rank})"
            else:
                rule_reciprocal_ranks.append(0.0)
                rule_display = "NOT FOUND"

            # Semantic evaluation
            semantic_rank = next((i + 1 for i, t in enumerate(semantic_titles) if t in expected), None)
            if semantic_rank is not None:
                semantic_hits += 1
                semantic_reciprocal_ranks.append(1.0 / semantic_rank)
                matched_recipe = next(r for r in semantic_results if r.title in expected)
                score_str = f"{matched_recipe.semantic_score:.3f}" if matched_recipe.semantic_score else "N/A"
                semantic_display = f"FOUND (Rank {semantic_rank}, {score_str})"
            else:
                semantic_reciprocal_ranks.append(0.0)
                semantic_display = "NOT FOUND"
        else:
            # Negative queries evaluation
            rule_display = "PASS (0 results)" if len(rule_results) == 0 else f"FOUND ({len(rule_results)})"
            semantic_display = "PASS (0 results)" if len(semantic_results) == 0 else f"FOUND ({len(semantic_results)})"

        print(f"{category:<22} | {query_str:<22} | {rule_display:<14} | {semantic_display:<24}")

    print("=" * 88)
    print("Aggregate Benchmark Summary (Positive Retrieval Queries):")
    rule_recall = (rule_hits / total_evaluable) * 100 if total_evaluable else 0.0
    semantic_recall = (semantic_hits / total_evaluable) * 100 if total_evaluable else 0.0

    rule_mrr = sum(rule_reciprocal_ranks) / len(rule_reciprocal_ranks) if rule_reciprocal_ranks else 0.0
    semantic_mrr = sum(semantic_reciprocal_ranks) / len(semantic_reciprocal_ranks) if semantic_reciprocal_ranks else 0.0

    print(f"  Total Test Cases Evaluated : {total_evaluable}")
    print(f"  Rule-Based Hit Rate / Recall: {rule_hits}/{total_evaluable} ({rule_recall:.1f}%) | MRR: {rule_mrr:.3f}")
    print(f"  Semantic Hit Rate / Recall  : {semantic_hits}/{total_evaluable} ({semantic_recall:.1f}%) | MRR: {semantic_mrr:.3f}")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmark()
