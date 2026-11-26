"""Setup builds the dense index the app will read: e5 when its model is present, MiniLM otherwise."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.backend import config
from app.backend.database.init_db import initialize_database
from app.backend.retrieval import index_build
from app.backend.retrieval.embedder import IngredientEmbedder, read_index_metadata
from app.backend.system import setup_pipeline


class FakeEmbedder(IngredientEmbedder):
    """Deterministic vectors without loading a model."""

    def __init__(self, model_name: str | None = None, engine: str | None = None) -> None:
        super().__init__(model_name=model_name, engine=engine)
        self._backend, self._device = "sentence-transformers", "cpu"

    def _encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vecs = rng.normal(size=(len(texts), self.dimension)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in [k for k in os.environ if k.startswith("PANTRYSENSE_") and k.endswith(("_PATH", "_MODEL", "_DIR"))]:
        monkeypatch.delenv(var)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("PANTRYSENSE_DATA_DIR", str(data))
    monkeypatch.setattr(index_build, "IngredientEmbedder", FakeEmbedder)
    initialize_database(data / "recipes.db")  # inserts the seed recipes
    return data


def _add_model(data: Path) -> Path:
    model = data / "models" / "e5-small-pantry"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "query: ", "passage": "passage: "}}), encoding="utf-8")
    return model


def test_data_dir_moves_every_default(data_dir: Path) -> None:
    assert config.get_database_path() == data_dir / "recipes.db"
    assert config.get_bm25_index_path() == data_dir / "recipe_bm25.json"
    assert config.get_ranker_model_path() == data_dir / "ranker_model.joblib"
    assert config.finetuned_files()[0] == data_dir / "models" / "e5-small-pantry"
    assert config.get_vector_index_path() == data_dir / "recipe_vectors.index"


def test_build_target_follows_the_model_folder(data_dir: Path) -> None:
    assert config.get_index_build_setup().name == "minilm"
    _add_model(data_dir)
    target = config.get_index_build_setup()
    assert target.name == "e5-finetuned"
    assert target.index_path == data_dir / "recipe_vectors_e5ft.index"
    # The app itself still runs on MiniLM until the e5 index exists.
    assert config.resolve_embedding_setup().name == "minilm"


def test_setup_builds_e5_index_where_the_app_reads_it(data_dir: Path) -> None:
    model = _add_model(data_dir)
    assert not setup_pipeline.indices_ready()

    label = setup_pipeline.build_indices_step(data_dir / "recipes.db", "onnx", 0)

    assert "e5" in label
    used = config.resolve_embedding_setup()
    assert used.name == "e5-finetuned"
    assert config.get_vector_index_path() == used.index_path == data_dir / "recipe_vectors_e5ft.index"
    assert config.get_vector_ids_path() == data_dir / "recipe_vector_ids_e5ft.json"
    meta = read_index_metadata(used.index_path)
    assert meta is not None
    assert Path(meta["model"]) == model.resolve()
    assert (meta["query_prefix"], meta["passage_prefix"]) == ("query: ", "passage: ")
    assert not (data_dir / "recipe_vectors.index").exists()
    assert setup_pipeline.indices_ready()


def test_setup_without_model_keeps_minilm(data_dir: Path) -> None:
    setup_pipeline.build_indices_step(data_dir / "recipes.db", "onnx", 0)

    assert config.resolve_embedding_setup().name == "minilm"
    meta = read_index_metadata(data_dir / "recipe_vectors.index")
    assert meta is not None and meta["model"] == config.DEFAULT_EMBEDDING_MODEL_NAME
    assert not (data_dir / "recipe_vectors_e5ft.index").exists()
    assert setup_pipeline.indices_ready()


def test_minilm_index_does_not_count_once_the_model_arrives(data_dir: Path) -> None:
    setup_pipeline.build_indices_step(data_dir / "recipes.db", "onnx", 0)
    assert setup_pipeline.indices_ready()

    _add_model(data_dir)
    assert not setup_pipeline.indices_ready()
    status = setup_pipeline.SetupManager().get_status()
    assert status["has_finetuned_model"] is True
    assert status["embedding_setup"] == "e5-finetuned"
    assert status["has_indices"] is False


def test_explicit_model_override_wins(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_model(data_dir)
    monkeypatch.setenv("PANTRYSENSE_EMBEDDING_MODEL", config.DEFAULT_EMBEDDING_MODEL_NAME)
    target = config.get_index_build_setup()
    assert target.model == config.DEFAULT_EMBEDDING_MODEL_NAME
    assert target.index_path == config.get_vector_index_path() == data_dir / "recipe_vectors.index"
