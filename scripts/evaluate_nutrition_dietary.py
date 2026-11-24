"""Evaluate the ingredient parser, nutrition estimate and allergen detection.

Usage:
    python scripts/evaluate_nutrition_dietary.py report [--json out.json]
    python scripts/evaluate_nutrition_dietary.py build-nutrition-fixture --parquet data/food_recipes.parquet

Fixtures live in tests/fixtures/ (see docs/nutrition_dietary.md for how they were built).
The "baseline" numbers come from the implementations at commit 9d5a78a, loaded from
git history so the comparison does not depend on a checked-out old tree.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"
BASELINE_COMMIT = "9d5a78a"

NUTRIENTS = ("calories", "protein", "carbs", "fat", "fiber")
REFERENCE_COLUMNS = {
    "calories": "Calories",
    "protein": "ProteinContent",
    "carbs": "CarbohydrateContent",
    "fat": "FatContent",
    "fiber": "FiberContent",
}


# ---------------------------------------------------------------------------
# Fixture building
# ---------------------------------------------------------------------------

def _c_vector(value: str | None) -> list[str | None]:
    """Parse an R-style c("a", NA, "b") string, keeping NA as None."""
    if not value:
        return []
    value = value.strip()
    if not value.startswith("c("):
        return [value.strip('"')]
    return [m[1:-1] if m != "NA" else None for m in re.findall(r'"(?:[^"\\]|\\.)*"|\bNA\b', value)]


def build_nutrition_fixture(parquet_path: Path, out_path: Path, n_aligned: int, n_misaligned: int, seed: int) -> None:
    import pyarrow.parquet as pq

    columns = ["RecipeId", "Name", "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeServings",
               *REFERENCE_COLUMNS.values()]
    aligned, misaligned = [], []
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=50000, columns=columns):
        for row in batch.to_pylist():
            parts = _c_vector(row["RecipeIngredientParts"])
            quantities = _c_vector(row["RecipeIngredientQuantities"])
            try:
                servings = int(float(row["RecipeServings"]))
                reference = {k: float(row[col]) for k, col in REFERENCE_COLUMNS.items()}
            except (TypeError, ValueError):
                continue
            if not parts or servings <= 0 or not (20 <= reference["calories"] <= 3000) or len(parts) > 20:
                continue
            item = {
                "recipe_id": row["RecipeId"],
                "title": row["Name"],
                "parts": parts,
                "quantities": quantities,
                "servings": servings,
                "reference_per_serving": reference,
            }
            (aligned if len(parts) == len(quantities) else misaligned).append(item)
    rnd = random.Random(seed)
    payload = {
        "_meta": {
            "source": "data/food_recipes.parquet (Food.com dump used to build recipes.db)",
            "reference": "Food.com per-serving nutrition columns (computed by Food.com from the full "
                         "ingredient text, which included units).",
            "filters": "servings > 0, 20 <= reference kcal/serving <= 3000, <= 20 ingredients",
            "aligned": "len(RecipeIngredientQuantities) == len(RecipeIngredientParts); quantity i belongs to part i",
            "misaligned": "lengths differ; the app pairs them by index anyway, so quantities are shifted",
            "seed": seed,
        },
        "aligned": rnd.sample(aligned, n_aligned),
        "misaligned": rnd.sample(misaligned, n_misaligned),
    }
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"aligned pool={len(aligned)} misaligned pool={len(misaligned)} -> {out_path}")


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def _load_baseline(module: str) -> types.ModuleType | None:
    path = f"app/backend/recipes/{module}.py"
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{path}"],
            cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod = types.ModuleType(f"baseline_{module}")
    exec(compile(source, f"baseline:{path}", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def eval_allergens(detect, classify, recipes: list[dict], allergens: list[str]) -> dict:
    per = {a: [0, 0, 0] for a in allergens}
    strict_fn = {a: 0 for a in allergens}
    strict_pos = {a: 0 for a in allergens}
    tags = {"vegetarian": [0, 0], "vegan": [0, 0], "gluten_free": [0, 0], "dairy_free": [0, 0]}
    for r in recipes:
        predicted = set(detect(r["ingredients"]))
        gold = set(r["allergens_contains"]) | set(r["allergens_may_contain"])
        for a in allergens:
            p, g = a in predicted, a in gold
            if p and g:
                per[a][0] += 1
            elif p:
                per[a][1] += 1
            elif g:
                per[a][2] += 1
            if a in r["allergens_contains"]:
                strict_pos[a] += 1
                strict_fn[a] += a not in predicted
        dietary = classify(r["ingredients"])
        gold_tags = {
            "vegetarian": r["vegetarian"],
            "vegan": r["vegan"],
            "gluten_free": "gluten" not in gold,
            "dairy_free": "dairy" not in gold,
        }
        for tag, value in gold_tags.items():
            tags[tag][0] += dietary.get(tag) == value
            tags[tag][1] += 1
    out = {a: _prf(*per[a]) for a in allergens}
    for a in allergens:
        out[a]["recall_contains_only"] = round(1 - strict_fn[a] / strict_pos[a], 3) if strict_pos[a] else None
    tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
    out["micro"] = _prf(tp, fp, fn)
    macro = [out[a] for a in allergens if per[a][0] + per[a][2] > 0]
    out["macro"] = {
        "precision": round(statistics.mean(m["precision"] for m in macro), 3),
        "recall": round(statistics.mean(m["recall"] for m in macro), 3),
    }
    out["dietary_tag_accuracy"] = {t: round(c / n, 3) for t, (c, n) in tags.items()}
    return out


def _details(item: dict, aligned_only: bool) -> list[dict]:
    """Rebuild ingredient_details the way the app stores them (quantity per name, unit 'portion')."""
    details = []
    quantities = [q for q in item["quantities"] if q is not None] if not aligned_only else item["quantities"]
    for i, name in enumerate(item["parts"]):
        q = quantities[i] if i < len(quantities) else None
        details.append({"name": name, "quantity": q, "unit": "portion"})
    return details


def eval_nutrition(calc, items: list[dict], use_details: bool, aligned: bool) -> dict:
    errors = {k: [] for k in NUTRIENTS}
    ape = []
    within = 0
    coverage = []
    for item in items:
        names = [p.lower() for p in item["parts"]]
        if use_details:
            from app.backend.recipes.ingredient_parser import parse_quantity

            details = _details(item, aligned_only=aligned)
            for d in details:
                d["quantity"] = parse_quantity(d["quantity"]) if isinstance(d["quantity"], str) else d["quantity"]
            result = calc(names, servings=item["servings"], ingredient_details=details)
        else:
            result = calc(names, servings=item["servings"])
        ref = item["reference_per_serving"]
        for k in NUTRIENTS:
            errors[k].append(abs(result["per_serving"][k] - ref[k]))
        rel = abs(result["per_serving"]["calories"] - ref["calories"]) / ref["calories"]
        ape.append(rel)
        within += rel <= 0.25
        if "coverage" in result:
            coverage.append(result["coverage"]["ratio"])
    out = {
        "n": len(items),
        "calories_median_ape": round(statistics.median(ape), 3),
        "calories_within_25pct": round(within / len(items), 3),
        "mae_per_serving": {k: round(statistics.mean(v), 1) for k, v in errors.items()},
    }
    if coverage:
        out["mean_coverage"] = round(statistics.mean(coverage), 3)
    return out


def eval_parser(parse, rows: list[dict]) -> dict:
    fields = ("quantity", "unit", "name", "preparation")
    correct = {f: 0 for f in fields}
    exact = 0
    failures = []
    for row in rows:
        p = parse(row["text"])
        got = {
            "quantity": p.get("quantity"),
            "unit": p.get("unit"),
            "name": p.get("name"),
            "preparation": p.get("preparation"),
        }
        ok_all = True
        for f in fields:
            gold, val = row[f], got[f]
            if f == "quantity":
                ok = (gold is None and val is None) or (
                    gold is not None and val is not None and abs(float(gold) - float(val)) < 1e-2)
            elif f == "preparation":
                ok = _norm_prep(gold) == _norm_prep(val)
            else:
                ok = (gold or None) == (val or None)
            correct[f] += ok
            ok_all &= ok
        exact += ok_all
        if not ok_all and len(failures) < 15:
            failures.append({"text": row["text"], "gold": {f: row[f] for f in fields}, "got": got})
    n = len(rows)
    return {"n": n, **{f"{f}_acc": round(correct[f] / n, 3) for f in fields},
            "row_exact": round(exact / n, 3), "sample_failures": failures}


def _norm_prep(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(sorted(re.findall(r"[a-z]+", value.lower())))


# ---------------------------------------------------------------------------
# Baseline parser: what the codebase did before (dataset_loader-style).
# ---------------------------------------------------------------------------

_BASELINE_UNITS = {"cup", "cups", "tbsp", "tsp", "g", "kg", "oz", "lb", "lbs", "ml", "l", "pinch", "clove", "cloves"}


def baseline_parse(text: str) -> dict:
    """Leading numeric token as quantity, next token as unit if it is a common unit, rest as
    the name truncated to three words (dataset_loader keeps the first three words)."""
    from app.backend.system.dataset_loader import _parse_quantity

    tokens = text.split()
    quantity = None
    if tokens and re.match(r"^[\d/.]+$", tokens[0]):
        quantity = _parse_quantity(tokens.pop(0))
    unit = None
    if tokens and tokens[0].lower().rstrip(".") in _BASELINE_UNITS:
        unit = tokens.pop(0).lower().rstrip(".")
    clean = re.sub(r"[^\w\s-]", "", " ".join(tokens).lower()).strip()
    return {"quantity": quantity, "unit": unit, "name": " ".join(clean.split()[:3]), "preparation": None}


def _new_parse(text: str) -> dict:
    from app.backend.recipes.ingredient_parser import parse_ingredient

    p = parse_ingredient(text)
    return {"quantity": p.quantity, "unit": p.unit, "name": p.name, "preparation": p.preparation}


# ---------------------------------------------------------------------------

def report() -> dict:
    from app.backend.recipes import dietary, nutrition

    results: dict = {}

    parser_sets = {}
    for split in ("dev", "test"):
        path = FIXTURES / f"ingredient_lines_{split}.jsonl"
        if path.exists():
            parser_sets[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if parser_sets:
        results["parser"] = {
            split: {"baseline": eval_parser(baseline_parse, rows), "rule_based": eval_parser(_new_parse, rows)}
            for split, rows in parser_sets.items()
        }

    allergen_data = json.loads((FIXTURES / "allergen_recipes.json").read_text(encoding="utf-8"))
    recipes = allergen_data["recipes"]
    allergens = allergen_data["_meta"]["allergens"]
    base_dietary = _load_baseline("dietary")
    results["allergens"] = {}
    for label, subset in (("all_100", recipes), ("random_60", [r for r in recipes if r["sample"] == "random"])):
        entry = {"new": eval_allergens(dietary.detect_recipe_allergens, dietary.classify_recipe_dietary, subset, allergens)}
        if base_dietary:
            entry["baseline"] = eval_allergens(base_dietary.detect_recipe_allergens,
                                               base_dietary.classify_recipe_dietary, subset, allergens)
        results["allergens"][label] = entry

    nut_path = FIXTURES / "nutrition_reference.json"
    if nut_path.exists():
        nut = json.loads(nut_path.read_text(encoding="utf-8"))
        base_nutrition = _load_baseline("nutrition")
        res = {}
        if base_nutrition:
            res["baseline_names_only_aligned"] = eval_nutrition(base_nutrition.calculate_recipe_nutrition, nut["aligned"], False, True)
            res["baseline_names_only_misaligned"] = eval_nutrition(base_nutrition.calculate_recipe_nutrition, nut["misaligned"], False, False)
        res["new_names_only_aligned"] = eval_nutrition(nutrition.calculate_recipe_nutrition, nut["aligned"], False, True)
        res["new_with_quantities_aligned"] = eval_nutrition(nutrition.calculate_recipe_nutrition, nut["aligned"], True, True)
        res["new_with_quantities_misaligned"] = eval_nutrition(nutrition.calculate_recipe_nutrition, nut["misaligned"], True, False)
        results["nutrition"] = res
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--json", type=Path)
    b = sub.add_parser("build-nutrition-fixture")
    b.add_argument("--parquet", type=Path, default=ROOT / "data" / "food_recipes.parquet")
    b.add_argument("--out", type=Path, default=FIXTURES / "nutrition_reference.json")
    b.add_argument("--aligned", type=int, default=300)
    b.add_argument("--misaligned", type=int, default=150)
    b.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()
    if args.cmd == "build-nutrition-fixture":
        build_nutrition_fixture(args.parquet, args.out, args.aligned, args.misaligned, args.seed)
        return
    results = report()
    text = json.dumps(results, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
