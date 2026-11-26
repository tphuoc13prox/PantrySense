"""Rebuild every model from the raw dataset, then run the official eval.

    python scripts/reproduce.py                   # all steps, published settings
    python scripts/reproduce.py ltr-data ltr-train  # only these steps
    python scripts/reproduce.py --from index      # this step and the ones after it
    python scripts/reproduce.py --quick           # tiny settings, to check the commands run
    python scripts/reproduce.py --dry-run         # print the commands

Steps, in order (docs/reproduce.md has timings and hardware):

  data       download the dataset, build recipes.db and its metadata
  finetune   fine-tune intfloat/e5-small-v2 into models/e5-small-pantry
  index      e5 FAISS index and BM25
  ltr-data   LTR datasets from hybrid e5 candidates (four query tiers)
  ltr-train  LambdaMART with the grid search, writes ranker_model.joblib
  eval       scripts/run_eval.py on eval/queries_v1.jsonl

Everything is written to the data folder (PANTRYSENSE_DATA_DIR, default data/),
the eval results to <data folder>/reproduce/. The data step refuses to overwrite
an existing recipes.db; delete it first, or point PANTRYSENSE_DATA_DIR at an
empty folder. --quick results mean nothing; it only exercises each command.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backend.config import get_data_dir  # noqa: E402

STEPS = ("data", "finetune", "index", "ltr-data", "ltr-train", "eval")


def commands(step: str, data: Path, quick: bool) -> list[list[str]]:
    """The commands for one step. Defaults are the settings of the published models."""
    py = sys.executable
    db = str(data / "recipes.db")
    ltr = data / "ltr"
    model = str(data / "models" / "e5-small-pantry")
    if step == "data":
        return [[py, "scripts/build_database.py", "--db", db, *(["--limit", "3000"] if quick else [])]]
    if step == "finetune":
        # docs/embedding.md: 100k pairs, 1 epoch, batch 128, lr 5e-5, seed 13.
        small = ["--max-pairs", "256", "--batch-size", "32"] if quick else ["--max-pairs", "100000"]
        return [[py, "scripts/finetune_embedder.py", "--db", db, "train", "--base", "intfloat/e5-small-v2",
                 *small, "--output", model]]
    if step == "index":
        # With the model folder present, the build targets the e5 index files.
        return [[py, "scripts/build_vector_index.py"]]
    if step == "ltr-data":
        # docs/ltr.md, v3: standard/sparse/noisy tiers (seed 7), then pantry (seed 11)
        # on different source recipes.
        main_n = ["--n-train", "40", "--n-heldout", "10"] if quick else []
        pantry_n = ["--n-train", "20", "--n-heldout", "5"] if quick else ["--n-train", "1333", "--n-heldout", "600"]
        return [
            [py, "scripts/build_ltr_dataset.py", "--out-dir", str(ltr), "--top-k", "100", *main_n],
            [py, "scripts/build_ltr_dataset.py", "--out-dir", str(ltr), "--top-k", "100", "--tiers", "pantry",
             "--tag", "pantry", *pantry_n, "--seed", "11", "--exclude-queries", str(ltr / "ltr_queries.json")],
        ]
    if step == "ltr-train":
        small = ["--no-grid", "--folds", "2", "--no-ablation"] if quick else []
        return [[py, "scripts/train_ranker.py", "--dataset", str(ltr / "ltr_train_k100.parquet"),
                 str(ltr / "ltr_train_pantry_k100.parquet"), "--output", str(data / "ranker_model.joblib"),
                 "--report", str(ltr / "ltr_train_report.json"), *small]]
    if step == "eval":
        small = ["--limit", "20", "--bootstrap", "100", "--warmup", "1"] if quick else []
        return [[py, "scripts/run_eval.py", "--data-dir", str(data), "--embedding", "e5",
                 "--out-dir", str(data / "reproduce"),
                 "--name", "queries_v1_topk20_reproduce", *small]]
    raise ValueError(step)


def selected_steps(names: list[str], start: str | None) -> list[str]:
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s): {', '.join(unknown)} (steps: {', '.join(STEPS)})")
    if names and start:
        raise SystemExit("give step names or --from, not both")
    if start:
        return list(STEPS[STEPS.index(start):])
    return [s for s in STEPS if s in names] if names else list(STEPS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", metavar="STEP", help=", ".join(STEPS))
    parser.add_argument("--from", dest="start", choices=STEPS, help="run this step and every later one")
    parser.add_argument("--quick", action="store_true", help="tiny settings, only checks that each command runs")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)

    data = get_data_dir().resolve()
    env = {**os.environ, "PANTRYSENSE_DATA_DIR": str(data), "PANTRYSENSE_AUTO_OPEN_BROWSER": "false",
           "PANTRYSENSE_AUTO_SHUTDOWN": "false"}
    print(f"Data folder: {data}" + (" (quick mode)" if args.quick else ""), flush=True)
    for step in selected_steps(args.steps, args.start):
        for cmd in commands(step, data, args.quick):
            shown = " ".join(shlex.quote(c) for c in cmd[1:])
            print(f"\n[{step}] python {shown}", flush=True)
            if args.dry_run:
                continue
            code = subprocess.call(cmd, cwd=ROOT, env=env)
            if code != 0:
                print(f"[{step}] failed with exit code {code}", file=sys.stderr)
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
