from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backend.recipes.ingredient_parser import (
    parse_ingredient,
    parse_ingredient_detail,
    parse_quantity,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "text, expected",
    [("1 1/2", 1.5), ("1-1/2", 1.5), ("2½", 2.5), ("1 -2", 1.0), ("3/4", 0.75), ("NA", None), (None, None), (2, 2.0)],
)
def test_parse_quantity(text, expected):
    assert parse_quantity(text) == expected


def test_range_keeps_upper_bound():
    p = parse_ingredient("2 to 3 cups flour")
    assert (p.quantity, p.quantity_max, p.unit, p.name) == (2.0, 3.0, "cup", "flour")


def test_package_size():
    p = parse_ingredient("1 (14.5 ounce) can diced tomatoes, drained")
    assert (p.quantity, p.unit, p.size_quantity, p.size_unit, p.name) == (1.0, "can", 14.5, "oz", "tomato")


def test_detail_record_with_placeholder_unit():
    p = parse_ingredient_detail({"name": "garlic cloves", "quantity": 3.0, "unit": "portion"})
    assert (p.quantity, p.unit, p.name) == (3.0, "clove", "garlic")


@pytest.mark.parametrize("split", ["dev", "test"])
def test_labelled_fixture_accuracy(split):
    rows = [json.loads(line) for line in (FIXTURES / f"ingredient_lines_{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    correct = 0
    for row in rows:
        p = parse_ingredient(row["text"])
        qty_ok = (row["quantity"] is None) == (p.quantity is None) and (
            row["quantity"] is None or abs(row["quantity"] - p.quantity) < 1e-2)
        correct += qty_ok and p.unit == row["unit"] and p.name == row["name"]
    assert correct / len(rows) >= 0.95
