"""Download the pretrained models from the GitHub release into data/.

    python scripts/fetch_models.py                  # from the release in the manifest
    python scripts/fetch_models.py --from-dir DIR   # DIR holds the release zip
    python scripts/fetch_models.py --url URL        # zip at another URL (file:// works)

The release zip holds the fine-tuned e5-small model (data/models/e5-small-pantry)
and the LTR bundle (data/ranker_model.joblib). scripts/model_assets.json, which
is tracked in git, pins the sha256 of the zip and of every file in it; anything
that does not match is rejected before it reaches data/. The ranker is a joblib
pickle, so this check is what makes loading it safe.

Files already in place with the right hashes are left alone unless --force is
given. The recipe dataset and the indices are not part of the release: the
in-app setup downloads the dataset and builds the e5 index once the model is in
data/ (see README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import get_data_dir  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().with_name("model_assets.json")
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("tag", "repository", "asset"):
        if key not in manifest:
            raise FetchError(f"{path.name}: missing '{key}'")
    asset = manifest["asset"]
    for key in ("name", "sha256", "files", "replace"):
        if key not in asset:
            raise FetchError(f"{path.name}: asset is missing '{key}'")
    return manifest


def release_url(manifest: dict[str, Any]) -> str:
    return (f"https://github.com/{manifest['repository']}/releases/download/"
            f"{manifest['tag']}/{manifest['asset']['name']}")


def installed(data_dir: Path, files: dict[str, str]) -> bool:
    """True when every file of the release is in ``data_dir`` with the pinned hash."""
    return all((data_dir / rel).is_file() and sha256_file(data_dir / rel) == digest for rel, digest in files.items())


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pantrysense-fetch-models"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        for block in iter(lambda: response.read(CHUNK), b""):
            out.write(block)
            done += len(block)
            now = time.monotonic()
            if total and (now - last > 2 or done == total):
                print(f"  {done / 2**20:.0f} / {total / 2**20:.0f} MB", flush=True)
                last = now


def verify(path: Path, expected: str, what: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FetchError(f"sha256 mismatch for {what}: expected {expected}, got {actual}")


def _safe_member(name: str) -> PurePosixPath:
    """The member's relative path, or an error for anything that could leave the target folder."""
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or ":" in name or not member.parts:
        raise FetchError(f"unsafe path in archive: {name!r}")
    return member


def extract(archive: Path, staging: Path, files: dict[str, str]) -> None:
    """Extract the files listed in the manifest into ``staging`` and check each hash.

    The archive must contain exactly those files (directories aside).
    """
    with zipfile.ZipFile(archive) as zf:
        members = {}
        for info in zf.infolist():
            rel = _safe_member(info.filename)
            if info.is_dir():
                continue
            members[rel.as_posix()] = info
        unexpected = sorted(set(members) - set(files))
        missing = sorted(set(files) - set(members))
        if unexpected or missing:
            raise FetchError(f"archive does not match the manifest (unexpected: {unexpected}, missing: {missing})")
        for rel, info in members.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            verify(target, files[rel], rel)


def install(staging: Path, data_dir: Path, replace: list[str]) -> None:
    """Move each top-level entry from ``staging`` into ``data_dir``, replacing what is there."""
    for rel in replace:
        _safe_member(rel)
        source, target = staging / rel, data_dir / rel
        if not source.exists():
            raise FetchError(f"{rel} is not in the archive")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(source), str(target))


def fetch(
    data_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    url: str | None = None,
    from_dir: Path | None = None,
    force: bool = False,
) -> bool:
    """Install the release into ``data_dir``. Returns False when it was already in place."""
    manifest = load_manifest(manifest_path)
    asset = manifest["asset"]
    if not force and installed(data_dir, asset["files"]):
        print(f"Models from {manifest['tag']} are already in {data_dir}.")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)
    # Staging inside data_dir keeps the final move on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".fetch-models-", dir=data_dir) as tmp:
        tmp_dir = Path(tmp)
        if from_dir is not None:
            archive = from_dir / asset["name"]
            if not archive.is_file():
                raise FetchError(f"{asset['name']} not found in {from_dir}")
            print(f"Using {archive}")
        else:
            archive = tmp_dir / asset["name"]
            source = url or release_url(manifest)
            print(f"Downloading {source}")
            try:
                download(source, archive)
            except OSError as exc:
                raise FetchError(f"download failed: {exc}") from exc
        verify(archive, asset["sha256"], asset["name"])
        staging = tmp_dir / "staging"
        extract(archive, staging, asset["files"])
        install(staging, data_dir, asset["replace"])
    print(f"Installed {', '.join(asset['replace'])} from {manifest['tag']} into {data_dir}.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", help="download the zip from this URL instead of the release")
    source.add_argument("--from-dir", type=Path, help="use the zip from this folder, no network")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PANTRYSENSE_DATA_DIR or data/")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="reinstall even if the files already match")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or get_data_dir()
    try:
        fetch(data_dir, args.manifest, url=args.url, from_dir=args.from_dir, force=args.force)
    except (FetchError, zipfile.BadZipFile) as exc:
        print(f"fetch-models: {exc}", file=sys.stderr)
        return 1
    print("Next: start the app (python tasks.py run). Setup builds the e5 index on first run; "
          "with an existing recipes.db, python tasks.py build-index does it directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
