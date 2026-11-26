"""fetch_models: checksums, safe extraction and installing into the data folder."""
from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts import fetch_models

ROOT = Path(__file__).resolve().parents[1]
ASSET = "models-test.zip"
FILES = {
    "models/e5-small-pantry/config.json": b'{"model_type": "bert"}',
    "models/e5-small-pantry/1_Pooling/config.json": b"{}",
    "ranker_model.joblib": b"not really a pickle",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(tmp_path: Path, files: dict[str, bytes] = FILES, manifest_files: dict[str, bytes] | None = None) -> tuple[Path, Path]:
    """A release folder with the zip, and a manifest describing ``manifest_files`` (default: ``files``)."""
    release = tmp_path / "release"
    release.mkdir()
    with zipfile.ZipFile(release / ASSET, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    manifest = tmp_path / "model_assets.json"
    manifest.write_text(json.dumps({
        "tag": "v0.0.1",
        "repository": "example/pantrysense",
        "asset": {
            "name": ASSET,
            "sha256": fetch_models.sha256_file(release / ASSET),
            "replace": ["models/e5-small-pantry", "ranker_model.joblib"],
            "files": {k: _sha(v) for k, v in (manifest_files or files).items()},
        },
    }), encoding="utf-8")
    return release, manifest


def test_installs_from_dir_and_skips_when_current(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path)
    data = tmp_path / "data"

    assert fetch_models.fetch(data, manifest, from_dir=release) is True
    for name, content in FILES.items():
        assert (data / name).read_bytes() == content
    assert not list(data.glob(".fetch-models-*"))

    assert fetch_models.fetch(data, manifest, from_dir=release) is False
    assert fetch_models.fetch(data, manifest, from_dir=release, force=True) is True


def test_downloads_from_url(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path)
    data = tmp_path / "data"
    assert fetch_models.fetch(data, manifest, url=(release / ASSET).as_uri()) is True
    assert (data / "ranker_model.joblib").read_bytes() == FILES["ranker_model.joblib"]


def test_replaces_a_stale_model_folder(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path)
    data = tmp_path / "data"
    stale = data / "models" / "e5-small-pantry" / "pytorch_model.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old weights")
    (data / "recipes.db").write_bytes(b"keep me")

    fetch_models.fetch(data, manifest, from_dir=release)
    assert not stale.exists()
    assert (data / "recipes.db").read_bytes() == b"keep me"


def test_rejects_wrong_archive_checksum(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path)
    with zipfile.ZipFile(release / ASSET, "a") as zf:
        zf.writestr("extra.txt", b"x")
    data = tmp_path / "data"
    with pytest.raises(fetch_models.FetchError, match="sha256 mismatch"):
        fetch_models.fetch(data, manifest, from_dir=release)
    assert not (data / "ranker_model.joblib").exists()


def test_rejects_a_file_that_does_not_match_its_hash(tmp_path: Path) -> None:
    tampered = {**FILES, "ranker_model.joblib": b"something else"}
    release, manifest = _release(tmp_path, files=tampered, manifest_files=FILES)
    data = tmp_path / "data"
    with pytest.raises(fetch_models.FetchError, match="ranker_model.joblib"):
        fetch_models.fetch(data, manifest, from_dir=release)
    assert not (data / "ranker_model.joblib").exists()
    assert not (data / "models").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "C:/evil.txt", "models/../../evil.txt"])
def test_rejects_paths_outside_the_target(tmp_path: Path, name: str) -> None:
    release, manifest = _release(tmp_path, files={**FILES, name: b"x"})
    with pytest.raises(fetch_models.FetchError, match="unsafe path"):
        fetch_models.fetch(tmp_path / "data", manifest, from_dir=release)
    assert not (tmp_path / "evil.txt").exists()


def test_rejects_files_missing_from_the_manifest(tmp_path: Path) -> None:
    release, manifest = _release(tmp_path, files={**FILES, "models/e5-small-pantry/extra.bin": b"x"},
                                 manifest_files=FILES)
    # The archive hash in the manifest matches, but the extra member is refused.
    with pytest.raises(fetch_models.FetchError, match="unexpected"):
        fetch_models.fetch(tmp_path / "data", manifest, from_dir=release)


def test_main_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, manifest = _release(tmp_path)
    code = fetch_models.main(["--from-dir", str(tmp_path / "nowhere"), "--data-dir", str(tmp_path / "data"),
                              "--manifest", str(manifest)])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_tracked_manifest_matches_this_release() -> None:
    manifest = fetch_models.load_manifest(fetch_models.DEFAULT_MANIFEST)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert manifest["tag"] == f"v{version}"
    asset = manifest["asset"]
    assert asset["name"].endswith(f"{manifest['tag']}.zip")
    assert "models/e5-small-pantry/config.json" in asset["files"]
    for rel in asset["files"]:
        assert any(rel == r or rel.startswith(r + "/") for r in asset["replace"]), rel
    assert all(len(h) == 64 for h in [asset["sha256"], *asset["files"].values()])

