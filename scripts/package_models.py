"""Build the release zip of the pretrained models and update scripts/model_assets.json.

    python scripts/package_models.py --source data --out dist/models --tag v1.2.0

Takes ``<source>/models/e5-small-pantry`` and ``<source>/ranker_model.joblib``
and writes ``pantrysense-models-<tag>.zip`` and ``SHA256SUMS`` to ``--out``.
The source folder is only read. On the way:

- the model folder gets this project's model card as README.md instead of the
  base model's, plus the base model's MIT license;
- absolute paths of the model folder recorded in the ranker bundle are
  rewritten as ``data/models/<name>``, and the rewritten bundle is checked to
  give the same scores as the original;
- zip entries are sorted and carry a fixed timestamp, so the same inputs give
  the same sha256 (with the same Python and zlib).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_models import DEFAULT_MANIFEST, sha256_file  # noqa: E402

MODEL_DIR = PurePosixPath("models/e5-small-pantry")
RANKER = PurePosixPath("ranker_model.joblib")
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
# The base model's card; replaced by model_card() below.
SKIP_FILES = {"README.md"}

MIT_LICENSE = """MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def model_card(meta: dict[str, Any]) -> str:
    return f"""# e5-small-pantry

Sentence embedding model used by PantrySense for dense recipe retrieval.
Fine-tuned from [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
(MIT license, see LICENSE) on ingredient-subset queries paired with recipe texts.

- Inputs need the e5 prefixes: `query: ` for an ingredient list, `passage: ` for
  a recipe (also stored in `config_sentence_transformers.json`).
- Output: 384-dim, L2-normalized.
- Training: {meta.get('pairs')} pairs, {meta.get('epochs')} epoch, batch {meta.get('batch_size')},
  lr {meta.get('lr')}, MultipleNegativesRankingLoss with scale {meta.get('scale')}, seed {meta.get('seed')}.
  Full details and results are in docs/embedding.md of the PantrySense repository,
  and `train_meta.json` here.
- Training data: queries generated from Food.com recipes in the Hugging Face
  dataset `untitledwebsite123/food-recipes`, which states no license. No recipe
  text is included in this folder. See docs/data_card.md for what is known about
  the data and its terms.
"""


def portable_model_path(value: str) -> str:
    """``.../data/models/<name>`` (any OS, any prefix) as ``data/models/<name>``."""
    parts = value.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2] == "models":
        return f"data/models/{parts[-1]}"
    return value


def _rewrite_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _rewrite_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(v) for v in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return portable_model_path(value)
    return value


def clean_ranker(source: Path, target: Path) -> None:
    import joblib
    import numpy as np

    bundle = joblib.load(source)
    cleaned = {k: (v if k == "model" else _rewrite_paths(v)) for k, v in bundle.items()}
    joblib.dump(cleaned, target)

    reloaded = joblib.load(target)
    rng = np.random.default_rng(0)
    features = rng.random((64, len(bundle["feature_names"])))
    if not np.array_equal(bundle["model"].predict(features), reloaded["model"].predict(features)):
        raise SystemExit("rewritten ranker bundle scores differently from the original")


def stage(source: Path, staging: Path) -> None:
    model_src = source / MODEL_DIR
    if not (model_src / "config.json").is_file():
        raise SystemExit(f"{model_src} is not a sentence-transformers model folder")
    model_dst = staging / MODEL_DIR
    for path in sorted(model_src.rglob("*")):
        rel = path.relative_to(model_src)
        if path.is_file() and rel.as_posix() not in SKIP_FILES:
            (model_dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, model_dst / rel)
    meta_path = model_src / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    (model_dst / "README.md").write_text(model_card(meta), encoding="utf-8", newline="\n")
    (model_dst / "LICENSE").write_text(MIT_LICENSE, encoding="utf-8", newline="\n")
    clean_ranker(source / RANKER, staging / RANKER)


def write_zip(staging: Path, archive: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Sorted by the posix name so the order (and the hash) is the same on every OS.
        for path in sorted((p for p in staging.rglob("*") if p.is_file()), key=lambda p: p.relative_to(staging).as_posix()):
            rel = path.relative_to(staging).as_posix()
            info = zipfile.ZipInfo(rel, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
            files[rel] = sha256_file(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="data folder with models/ and ranker_model.joblib")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", default="tphuoc13prox/pantrysense")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    name = f"pantrysense-models-{args.tag}.zip"
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / name
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage(args.source, staging)
        files = write_zip(staging, archive)

    digest = sha256_file(archive)
    (args.out / "SHA256SUMS").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    manifest = {
        "tag": args.tag,
        "repository": args.repository,
        "asset": {
            "name": name,
            "size": archive.stat().st_size,
            "sha256": digest,
            "replace": [MODEL_DIR.as_posix(), RANKER.as_posix()],
            "files": files,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{archive} ({archive.stat().st_size / 2**20:.1f} MB, sha256 {digest})")
    print(f"Manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
