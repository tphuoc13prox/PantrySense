from __future__ import annotations

from pathlib import Path

from app.backend.evaluation.split import is_eval_recipe, split_bucket
from app.backend.evaluation.query_set import (
    QuerySetConfig,
    build_query_set,
    duplicate_groups,
    load_query_set,
    normalize_title,
    to_corpus_recipe,
    write_query_set,
)


def _corpus():
    base = [
        ("Chicken Curry", ["chicken", "curry powder", "coconut milk", "ginger", "salt", "rice"]),
        ("chicken curry!", ["chicken", "curry powder", "coconut milk", "cilantro", "salt"]),
        ("Beef Stew", ["beef", "carrot", "potato", "celery", "salt", "thyme"]),
        ("Lemon Bars", ["lemon juice", "powdered sugar", "salt", "graham cracker", "lemon zest"]),
        ("Tomato Soup", ["tomato", "basil", "cream", "salt", "garlic"]),
        ("Salt Water", ["salt", "water"]),
    ]
    proteins = ["tofu", "shrimp", "pork", "lamb", "duck", "salmon", "turkey", "cod"]
    base += [
        (f"{p.title()} Bowl", [p, f"{p} sauce", "scallion", "sesame seed", "noodle", "salt"]) for p in proteins
    ]
    recipes = [to_corpus_recipe(i + 1, t, ings) for i, (t, ings) in enumerate(base)]
    # Filler recipes with unique ingredients dilute document frequency so 'salt'
    # is the only staple. Their titles collapse to one 40-member duplicate group,
    # which exceeds max_duplicate_group, so they are never sampled themselves.
    recipes += [
        to_corpus_recipe(100 + i, f"Filler {i}", [f"spice{i}", f"herb{i}", f"nut{i}", f"leaf{i}"]) for i in range(40)
    ]
    return recipes


def _config(**kw) -> QuerySetConfig:
    params = dict(n_queries=4, seed=1, pool="all", staple_df_threshold=0.20, min_pool_size=3, distractor_min_df=1)
    params.update(kw)
    return QuerySetConfig(**params)


def test_normalize_title_collapses_case_punctuation_and_suffixes() -> None:
    assert normalize_title("Chicken Curry!") == normalize_title("chicken curry")
    assert normalize_title("Chicken Curry II") == "chicken curry"
    assert normalize_title("Mom's Pie Recipe") == "mom pie"
    assert normalize_title("Chicken Curry") != normalize_title("Beef Curry")


def test_duplicate_groups_link_title_variants() -> None:
    groups = duplicate_groups(_corpus())
    assert groups[1] == {1, 2}
    assert groups[3] == {3}


def test_build_query_set_is_deterministic_and_respects_constraints() -> None:
    corpus = _corpus()
    q1, meta = build_query_set(corpus, _config())
    q2, _ = build_query_set(corpus, _config())
    assert [q.to_json() for q in q1] == [q.to_json() for q in q2]

    assert [s["ingredient"] for s in meta["staples"]] == ["salt"]
    by_id = {r.recipe_id: r for r in corpus}
    for q in q1:
        assert "salt" not in q.ingredients
        assert q.recipe_id in q.relevant_ids
        real = [i for i in q.ingredients if i != q.distractor]
        assert set(real) <= set(by_id[q.recipe_id].ingredients)
        assert len(real) >= 2
        if q.distractor:
            assert q.tier == "noisy"
            assert q.distractor not in by_id[q.recipe_id].ingredients
    # Never two members of the same duplicate group.
    seen: set[int] = set()
    for q in q1:
        assert not (set(q.relevant_ids) & seen)
        seen |= set(q.relevant_ids)
    sampled = {q.recipe_id for q in q1}
    # 'Salt Water' has no usable ingredients; fillers are an oversized duplicate group.
    assert 6 not in sampled
    assert not any(rid >= 100 for rid in sampled)


def test_different_seed_changes_sample() -> None:
    corpus = _corpus()
    a, _ = build_query_set(corpus, _config(n_queries=8, seed=1))
    b, _ = build_query_set(corpus, _config(n_queries=8, seed=2))
    assert [q.to_json() for q in a] != [q.to_json() for q in b]


def test_exclude_recipe_ids_gives_disjoint_set() -> None:
    corpus = _corpus()
    first, _ = build_query_set(corpus, _config(n_queries=4))
    excluded = frozenset(i for q in first for i in q.relevant_ids)
    second, _ = build_query_set(corpus, _config(n_queries=4, seed=9, exclude_recipe_ids=excluded))
    assert not ({q.recipe_id for q in second} & excluded)


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    queries, meta = build_query_set(_corpus(), _config())
    out = tmp_path / "q.jsonl"
    meta_path = write_query_set(out, queries, meta)
    assert meta_path.exists()
    assert [q.to_json() for q in load_query_set(out)] == [q.to_json() for q in queries]


def test_split_bucket_is_stable_and_roughly_20_percent_eval() -> None:
    # Pinned values: the convention is shared with the LTR training pipeline.
    assert [split_bucket(i) for i in (1, 2, 3, 286, 315732)] == [1, 2, 3, 4, 9]
    share = sum(is_eval_recipe(i) for i in range(1, 20001)) / 20000
    assert 0.18 < share < 0.22


def test_eval_pool_only_samples_eval_buckets() -> None:
    corpus = _corpus()
    corpus += [
        to_corpus_recipe(1000 + i, f"Dish number {chr(97 + i % 26)}{i}", [f"a{i}", f"b{i}", f"c{i}", "salt"])
        for i in range(200)
    ]
    queries, meta = build_query_set(corpus, _config(n_queries=10, pool="eval"))
    for q in queries:
        assert all(is_eval_recipe(rid) for rid in q.relevant_ids)
    train, _ = build_query_set(corpus, _config(n_queries=10, pool="train"))
    assert not any(is_eval_recipe(rid) for q in train for rid in q.relevant_ids)
