"""package_models: what goes into the release zip."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.package_models import portable_model_path, write_zip


def test_portable_model_path() -> None:
    assert portable_model_path(r"D:\work\checkout\data\models\e5-small-pantry") == "data/models/e5-small-pantry"
    assert portable_model_path("/home/u/pantrysense/data/models/e5-small-pantry/") == "data/models/e5-small-pantry"
    assert portable_model_path("sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"


def test_zip_is_reproducible(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "models" / "m").mkdir(parents=True)
    (staging / "models" / "m" / "config.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (staging / "ranker_model.joblib").write_bytes(b"x" * 100)

    first = write_zip(staging, tmp_path / "a.zip")
    second = write_zip(staging, tmp_path / "b.zip")
    assert first == second
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "a.zip") as zf:
        assert zf.namelist() == ["models/m/config.json", "ranker_model.joblib"]
