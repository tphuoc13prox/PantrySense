"""run_eval picks the embedding setup the way the app does."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from app.backend import config

ROOT = Path(__file__).resolve().parents[1]


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_under_test", ROOT / "scripts" / "run_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


run_eval = _load_run_eval()


def _data_dir(tmp_path: Path, with_e5: bool) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("recipes.db", "recipe_bm25.json", "ranker_model.joblib", "ingredient_vocab.json",
                 "recipe_vectors.index", "recipe_vector_ids.json"):
        (data / name).write_bytes(b"")
    if with_e5:
        model = data / "models" / "e5-small-pantry"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        for name in ("recipe_vectors_e5ft.index", "recipe_vectors_e5ft.index.meta.json", "recipe_vector_ids_e5ft.json"):
            (data / name).write_bytes(b"")
    return data


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PANTRYSENSE_")}
    monkeypatch.setattr(os, "environ", env)
    return env


def _configure(data_dir: Path, embedding: str) -> str:
    args = run_eval.parse_args(["--data-dir", str(data_dir), "--embedding", embedding])
    run_eval.configure_env(args)
    return args.embedding_setup


def test_auto_uses_finetuned_e5_when_complete(tmp_path: Path, clean_env: dict[str, str]) -> None:
    data = _data_dir(tmp_path, with_e5=True)
    assert _configure(data, "auto") == "e5-finetuned"
    assert Path(clean_env["PANTRYSENSE_VECTOR_INDEX_PATH"]).name == "recipe_vectors_e5ft.index"
    assert Path(clean_env["PANTRYSENSE_VECTOR_IDS_PATH"]).name == "recipe_vector_ids_e5ft.json"
    assert Path(clean_env["PANTRYSENSE_EMBEDDING_MODEL"]) == (data / "models" / "e5-small-pantry").resolve()
    # The app's own getters agree with what was configured.
    assert config.uses_finetuned_embedding(data.resolve())
    assert config.get_vector_index_path() == Path(clean_env["PANTRYSENSE_VECTOR_INDEX_PATH"])


def test_auto_falls_back_to_minilm(tmp_path: Path, clean_env: dict[str, str]) -> None:
    data = _data_dir(tmp_path, with_e5=False)
    assert _configure(data, "auto") == "minilm"
    assert clean_env["PANTRYSENSE_EMBEDDING_MODEL"] == config.DEFAULT_EMBEDDING_MODEL_NAME
    assert Path(clean_env["PANTRYSENSE_VECTOR_INDEX_PATH"]).name == "recipe_vectors.index"


def test_minilm_can_be_forced(tmp_path: Path, clean_env: dict[str, str]) -> None:
    data = _data_dir(tmp_path, with_e5=True)
    assert _configure(data, "minilm") == "minilm"
    assert Path(clean_env["PANTRYSENSE_VECTOR_IDS_PATH"]).name == "recipe_vector_ids.json"


def test_forcing_e5_without_its_files_fails(tmp_path: Path, clean_env: dict[str, str]) -> None:
    data = _data_dir(tmp_path, with_e5=False)
    with pytest.raises(SystemExit, match="e5"):
        _configure(data, "e5")


def test_resolve_embedding_setup_matches_app_default() -> None:
    setup = config.resolve_embedding_setup()
    if config.uses_finetuned_embedding():
        assert setup.name == "e5-finetuned" and setup.index_path == config.FINETUNED_VECTOR_INDEX_PATH
    else:
        assert setup.name == "minilm" and setup.index_path == config.DEFAULT_VECTOR_INDEX_PATH
