from __future__ import annotations

import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import get_ranker_model_path
from app.backend.ranking.features import extract_features
from app.backend.recipes.schemas import RecipeSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_training_data() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate diverse training query-candidate pairs with target relevance labels.

    Relevance labels:
        3 = Perfect match (High coverage, fast cooking, easy, high semantic relevance)
        2 = Good match (Decent coverage, minor missing ingredients)
        1 = Weak match (Low coverage, missing key components)
        0 = Irrelevant match
    """
    X_rows = []
    y_labels = []
    group_sizes = []

    # Query Scenario 1: "chicken egg breakfast"
    q1_candidates = [
        # Candidate 1: Chicken Omelette (Perfect)
        (
            RecipeSummary(id=2, title="Chicken Omelette", coverage=1.0, matched_count=2, missing_ingredients=[], semantic_score=0.92, bm25_score=4.8, rrf_score=0.032),
            {"cooking_time": 15, "difficulty": "easy", "ingredients": ["chicken", "egg", "onion", "butter"]},
            3,
        ),
        # Candidate 2: Tomato Egg Stir Fry (Good partial)
        (
            RecipeSummary(id=1, title="Tomato Egg Stir Fry", coverage=0.5, matched_count=1, missing_ingredients=["tomato"], semantic_score=0.75, bm25_score=2.2, rrf_score=0.021),
            {"cooking_time": 10, "difficulty": "easy", "ingredients": ["tomato", "egg", "green onion"]},
            2,
        ),
        # Candidate 3: Pasta Garlic (Irrelevant)
        (
            RecipeSummary(id=4, title="Garlic Pasta", coverage=0.0, matched_count=0, missing_ingredients=["pasta", "garlic"], semantic_score=0.15, bm25_score=0.0, rrf_score=0.005),
            {"cooking_time": 20, "difficulty": "medium", "ingredients": ["pasta", "garlic", "cheese"]},
            0,
        ),
    ]
    for sumry, det, lbl in q1_candidates:
        X_rows.append(extract_features(sumry, det))
        y_labels.append(lbl)
    group_sizes.append(len(q1_candidates))

    # Query Scenario 2: "quick tomato dinner"
    q2_candidates = [
        # Candidate 1: Tomato Egg Stir Fry (10 mins, easy, high match -> 3)
        (
            RecipeSummary(id=1, title="Tomato Egg Stir Fry", coverage=1.0, matched_count=2, missing_ingredients=[], semantic_score=0.90, bm25_score=4.5, rrf_score=0.032),
            {"cooking_time": 10, "difficulty": "easy", "ingredients": ["tomato", "egg", "scallion"]},
            3,
        ),
        # Candidate 2: Beef Bolognese (35 mins, medium, partial -> 2)
        (
            RecipeSummary(id=6, title="Beef Bolognese", coverage=0.6, matched_count=1, missing_ingredients=["beef", "pasta"], semantic_score=0.72, bm25_score=2.5, rrf_score=0.020),
            {"cooking_time": 35, "difficulty": "medium", "ingredients": ["beef", "pasta", "tomato", "garlic"]},
            2,
        ),
        # Candidate 3: Shrimp Fried Rice (Irrelevant -> 0)
        (
            RecipeSummary(id=5, title="Shrimp Fried Rice", coverage=0.0, matched_count=0, missing_ingredients=["shrimp", "rice"], semantic_score=0.20, bm25_score=0.0, rrf_score=0.004),
            {"cooking_time": 15, "difficulty": "easy", "ingredients": ["shrimp", "rice", "garlic"]},
            0,
        ),
    ]
    for sumry, det, lbl in q2_candidates:
        X_rows.append(extract_features(sumry, det))
        y_labels.append(lbl)
    group_sizes.append(len(q2_candidates))

    # Query Scenario 3: "garlic shrimp dinner"
    q3_candidates = [
        # Candidate 1: Shrimp Fried Rice (Perfect -> 3)
        (
            RecipeSummary(id=5, title="Shrimp Fried Rice", coverage=1.0, matched_count=2, missing_ingredients=[], semantic_score=0.95, bm25_score=5.2, rrf_score=0.033),
            {"cooking_time": 15, "difficulty": "easy", "ingredients": ["shrimp", "rice", "garlic", "egg"]},
            3,
        ),
        # Candidate 2: Garlic Parmesan Pasta (Partial -> 2)
        (
            RecipeSummary(id=4, title="Garlic Parmesan Pasta", coverage=0.5, matched_count=1, missing_ingredients=["pasta", "cheese"], semantic_score=0.68, bm25_score=2.4, rrf_score=0.019),
            {"cooking_time": 18, "difficulty": "medium", "ingredients": ["pasta", "garlic", "cream", "cheese"]},
            2,
        ),
        # Candidate 3: Chicken Omelette (Irrelevant -> 0)
        (
            RecipeSummary(id=2, title="Chicken Omelette", coverage=0.0, matched_count=0, missing_ingredients=["chicken", "egg"], semantic_score=0.10, bm25_score=0.0, rrf_score=0.002),
            {"cooking_time": 15, "difficulty": "easy", "ingredients": ["chicken", "egg", "onion"]},
            0,
        ),
    ]
    for sumry, det, lbl in q3_candidates:
        X_rows.append(extract_features(sumry, det))
        y_labels.append(lbl)
    group_sizes.append(len(q3_candidates))

    # Query Scenario 4: "pork noodles dinner"
    q4_candidates = [
        # Candidate 1: Savory Pork Garlic Stir Fried Noodles (Both Pork & Noodles -> Perfect 3)
        (
            RecipeSummary(id=7, title="Savory Pork Garlic Stir Fried Noodles", coverage=0.45, matched_count=2, missing_ingredients=["garlic", "soy sauce", "green onion"], semantic_score=0.94, bm25_score=4.9, rrf_score=0.033),
            {"cooking_time": 15, "difficulty": "easy", "ingredients": ["pork", "noodles", "garlic", "soy sauce", "green onion", "oil", "black pepper"]},
            3,
        ),
        # Candidate 2: Classic Pork Belly Ramen (Both Pork & Noodles -> Perfect 3)
        (
            RecipeSummary(id=8, title="Classic Pork Belly Ramen", coverage=0.40, matched_count=2, missing_ingredients=["egg", "green onion", "soy sauce"], semantic_score=0.91, bm25_score=4.6, rrf_score=0.031),
            {"cooking_time": 20, "difficulty": "medium", "ingredients": ["pork", "noodles", "egg", "green onion", "soy sauce", "garlic", "sesame oil"]},
            3,
        ),
        # Candidate 3: Creamy Egg and Milk Noodles (Only Noodles, 0 Pork -> Weak 1)
        (
            RecipeSummary(id=3, title="Creamy Egg and Milk Noodles", coverage=0.14, matched_count=1, missing_ingredients=["milk", "egg", "butter", "garlic", "cheese", "black pepper"], semantic_score=0.55, bm25_score=2.1, rrf_score=0.016),
            {"cooking_time": 12, "difficulty": "easy", "ingredients": ["noodles", "milk", "egg", "butter", "garlic", "cheese", "black pepper"]},
            1,
        ),
        # Candidate 4: Tomato Egg Stir Fry (Irrelevant -> 0)
        (
            RecipeSummary(id=1, title="Tomato Egg Stir Fry", coverage=0.0, matched_count=0, missing_ingredients=["tomato", "egg"], semantic_score=0.12, bm25_score=0.0, rrf_score=0.003),
            {"cooking_time": 10, "difficulty": "easy", "ingredients": ["tomato", "egg", "green onion"]},
            0,
        ),
    ]
    for sumry, det, lbl in q4_candidates:
        X_rows.append(extract_features(sumry, det))
        y_labels.append(lbl)
    group_sizes.append(len(q4_candidates))

    # Query Scenario 5: "beef pasta dinner"
    q5_candidates = [
        # Candidate 1: Classic Beef Bolognese (Both Beef & Pasta -> Perfect 3)
        (
            RecipeSummary(id=6, title="Classic Beef Bolognese", coverage=0.50, matched_count=2, missing_ingredients=["tomato", "onion", "garlic"], semantic_score=0.93, bm25_score=4.7, rrf_score=0.032),
            {"cooking_time": 35, "difficulty": "medium", "ingredients": ["beef", "pasta", "tomato", "onion", "garlic", "olive oil", "cheese"]},
            3,
        ),
        # Candidate 2: Garlic Parmesan Pasta (Only Pasta, 0 Beef -> Weak 1)
        (
            RecipeSummary(id=4, title="Creamy Garlic Parmesan Pasta", coverage=0.17, matched_count=1, missing_ingredients=["garlic", "butter", "milk", "cream", "cheese"], semantic_score=0.52, bm25_score=2.0, rrf_score=0.015),
            {"cooking_time": 18, "difficulty": "medium", "ingredients": ["pasta", "garlic", "butter", "milk", "cream", "cheese", "black pepper"]},
            1,
        ),
    ]
    for sumry, det, lbl in q5_candidates:
        X_rows.append(extract_features(sumry, det))
        y_labels.append(lbl)
    group_sizes.append(len(q5_candidates))

    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.float32), group_sizes


def train_ranker(output_path: Path | None = None) -> Path:
    target_path = output_path or get_ranker_model_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, groups = generate_training_data()
    logger.info("Training ML Ranker on %d query-recipe feature samples...", len(X))

    model = None

    # Try LightGBM LGBMRanker first
    try:
        from lightgbm import LGBMRanker
        ranker = LGBMRanker(
            objective="lambdarank",
            n_estimators=30,
            learning_rate=0.08,
            num_leaves=15,
            min_child_samples=1,
            random_state=42,
            verbose=-1,
        )
        ranker.fit(X, y, group=groups)
        model = ranker
        logger.info("Trained LightGBM LambdaMART ranker successfully.")
    except Exception as e:
        logger.warning("LightGBM ranker not available, falling back to scikit-learn GradientBoosting: %s", e)

    # Fallback to scikit-learn GradientBoostingRegressor
    if model is None:
        from sklearn.ensemble import GradientBoostingRegressor
        gbr = GradientBoostingRegressor(
            n_estimators=30,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        )
        gbr.fit(X, y)
        model = gbr
        logger.info("Trained scikit-learn GradientBoostingRegressor ranker successfully.")

    import joblib
    joblib.dump(model, target_path)
    logger.info("Saved ML Ranker model to %s", target_path)
    return target_path


if __name__ == "__main__":
    train_ranker()
