from pathlib import Path

from app.backend import config


def test_display_path_inside_project_is_relative():
    path = config.BASE_DIR / "data" / "models" / "e5-small-pantry"
    assert config.display_path(path) == "data/models/e5-small-pantry"
    assert config.display_path(str(path)) == "data/models/e5-small-pantry"


def test_display_path_keeps_model_ids_and_relative_paths():
    assert config.display_path("sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.display_path(Path("eval") / "queries_v1.jsonl") == "eval/queries_v1.jsonl"


def test_display_path_under_external_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "pantry-data"
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(data_dir / "recipes.db"))
    assert config.display_path(data_dir / "recipe_bm25.npz") == "<data-dir>/recipe_bm25.npz"


def test_display_path_elsewhere_keeps_only_the_name(tmp_path, monkeypatch):
    monkeypatch.setenv("PANTRYSENSE_DB_PATH", str(tmp_path / "db" / "recipes.db"))
    assert config.display_path(tmp_path / "other" / "model.bin") == "model.bin"
