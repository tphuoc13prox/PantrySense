from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.backend.retrieval import embedder as embedder_module
from app.backend.retrieval.embedder import (
    BACKEND_FASTEMBED,
    BACKEND_SENTENCE_TRANSFORMERS,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndexMismatchError,
    IngredientEmbedder,
    check_index_compatibility,
    index_metadata_path,
    read_index_metadata,
    resolve_prefixes,
    write_index_metadata,
)
from app.backend.retrieval.semantic import SemanticRetriever
from app.backend.retrieval.vector_store import FaissVectorStore


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder_module, "_cuda_available", lambda: False)


def make_embedder(model: str = DEFAULT_EMBEDDING_MODEL, engine: str = "cpu") -> IngredientEmbedder:
    return IngredientEmbedder(model_name=model, engine=engine)


def test_model_name_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PANTRYSENSE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    assert IngredientEmbedder(engine="onnx").model_name == "BAAI/bge-small-en-v1.5"


def test_fastembed_only_for_verified_models() -> None:
    assert make_embedder(engine="onnx").backend == BACKEND_FASTEMBED
    assert make_embedder("BAAI/bge-small-en-v1.5", engine="onnx").backend == BACKEND_SENTENCE_TRANSFORMERS
    assert make_embedder("data/models/my-finetune", engine="fastembed").backend == BACKEND_SENTENCE_TRANSFORMERS
    # cuda requested but unavailable: falls back without changing the model
    assert make_embedder(engine="cuda").backend == BACKEND_FASTEMBED
    assert make_embedder(engine="cpu").backend == BACKEND_SENTENCE_TRANSFORMERS


def test_prefixes_for_e5_and_local_prompts(tmp_path: Path) -> None:
    assert resolve_prefixes("intfloat/e5-small-v2") == ("query: ", "passage: ")
    assert resolve_prefixes(DEFAULT_EMBEDDING_MODEL) == ("", "")

    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "q: ", "passage": "p: "}}), encoding="utf-8"
    )
    assert resolve_prefixes(str(tmp_path)) == ("q: ", "p: ")


def test_prefixes_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    emb = make_embedder("intfloat/e5-small-v2")
    monkeypatch.setattr(emb, "_encode", lambda texts: seen.append(texts) or np.ones((len(texts), 2), np.float32))
    emb.embed_text("egg tomato")
    emb.embed_texts(["Omelette\negg"])
    assert seen == [["query: egg tomato"], ["passage: Omelette\negg"]]


def test_metadata_round_trip(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.index"
    emb = make_embedder("intfloat/e5-small-v2")
    path = write_index_metadata(index_path, emb, count=3)

    assert path == index_metadata_path(index_path)
    meta = read_index_metadata(index_path)
    assert meta is not None
    assert meta["model"] == "intfloat/e5-small-v2"
    assert meta["backend"] == BACKEND_SENTENCE_TRANSFORMERS
    assert meta["query_prefix"] == "query: "
    assert meta["count"] == 3
    emb.verify_index(index_path)


def test_model_mismatch_is_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.index"
    write_index_metadata(index_path, make_embedder("BAAI/bge-small-en-v1.5"))
    with pytest.raises(EmbeddingIndexMismatchError, match="model"):
        make_embedder(DEFAULT_EMBEDDING_MODEL).verify_index(index_path)


def test_prefix_and_dimension_mismatch_are_rejected() -> None:
    emb = make_embedder("intfloat/e5-small-v2")
    meta = emb.index_metadata() | {"query_prefix": ""}
    with pytest.raises(EmbeddingIndexMismatchError, match="prefix"):
        check_index_compatibility(meta, emb)

    emb._dimension_known = True
    meta = emb.index_metadata() | {"dimension": 768}
    with pytest.raises(EmbeddingIndexMismatchError, match="dimension"):
        check_index_compatibility(meta, emb)


def test_backend_mismatch_allowed_only_for_equivalent_models() -> None:
    minilm_onnx = make_embedder(engine="onnx")
    meta = make_embedder(engine="cpu").index_metadata()
    check_index_compatibility(meta, minilm_onnx)

    finetuned = make_embedder("data/models/ft", engine="cpu")
    meta = finetuned.index_metadata() | {"backend": BACKEND_FASTEMBED}
    with pytest.raises(EmbeddingIndexMismatchError, match="backend"):
        check_index_compatibility(meta, finetuned)


def test_local_model_paths_compare_by_location(tmp_path: Path) -> None:
    model_dir = tmp_path / "ft"
    model_dir.mkdir()
    meta = make_embedder(str(model_dir)).index_metadata()
    check_index_compatibility(meta, make_embedder(str(model_dir / ".." / "ft")))


def test_legacy_index_without_metadata_is_accepted(tmp_path: Path) -> None:
    make_embedder().verify_index(tmp_path / "missing.index")


def test_semantic_retriever_refuses_mismatched_index(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.index"
    ids_path = tmp_path / "ids.json"
    store = FaissVectorStore(dimension=2)
    store.add(np.array([[1.0, 0.0]], dtype=np.float32), [1])
    store.save(index_path, ids_path)
    write_index_metadata(index_path, make_embedder("BAAI/bge-small-en-v1.5"))

    retriever = SemanticRetriever(
        embedder=make_embedder(DEFAULT_EMBEDDING_MODEL),
        vector_store=FaissVectorStore(dimension=2),
        index_path=index_path,
        ids_path=ids_path,
    )
    assert retriever.ensure_loaded() is False
