# Reproducing the models from scratch

`python tasks.py fetch-models` downloads the fine-tuned e5 model and the LTR
ranker that the published numbers come from. This page is for rebuilding both
from the raw dataset instead. It is the same chain of scripts that produced
them, with the settings from docs/embedding.md and docs/ltr.md as defaults.

```
python tasks.py reproduce                    # every step, in order
python tasks.py reproduce --from ltr-data    # resume at a step
python tasks.py reproduce index eval         # only some steps
python tasks.py reproduce --dry-run          # print the commands
```

Everything goes into the data folder (`data/`, or `PANTRYSENSE_DATA_DIR`).
The data step refuses to overwrite an existing `recipes.db`, so either delete
it or point `PANTRYSENSE_DATA_DIR` at an empty folder:

```
PANTRYSENSE_DATA_DIR=/tmp/pantry-repro python tasks.py reproduce
```

## Steps

| Step | What it runs | Output | Time (machine below) |
|---|---|---|---|
| `data` | `scripts/build_database.py`: download the Hugging Face parquet, ingest, dietary and nutrition metadata | `recipes.db`, `food_recipes.parquet` | 7-10 min (432 s and 618 s in two runs), plus the 230 MB download |
| `finetune` | `scripts/finetune_embedder.py train --base intfloat/e5-small-v2 --max-pairs 100000` | `models/e5-small-pantry/` | 9 min on the GPU; on CPU, not measured |
| `index` | `scripts/build_vector_index.py` (picks e5 because the model folder now exists) | `recipe_vectors_e5ft.index` + `.meta.json`, `recipe_vector_ids_e5ft.json`, `recipe_bm25.npz` | 35 min on an idle CPU; 61 min in a run that shared the CPU with other jobs |
| `ltr-data` | `scripts/build_ltr_dataset.py` twice: standard, sparse and noisy tiers (4000 train + 1000 held-out queries, seed 7), then the pantry tier (1333 + 600, seed 11) on other source recipes | `ltr/ltr_*_k100.parquet` and `.meta.json` | about 15 min on CPU |
| `ltr-train` | `scripts/train_ranker.py` on both training sets, with the 12-setting grid | `ranker_model.joblib`, `ltr/ltr_train_report.json` | about 3 min |
| `eval` | `scripts/run_eval.py --embedding e5` on `eval/queries_v1.jsonl` | `reproduce/queries_v1_topk20_reproduce.{json,md,per_query.jsonl}` | 9 min (529 s) |

Total: about 1.5 hours with a CUDA GPU for the fine-tune. The fine-tune is the
only step that needs a GPU in practice. The other steps were run on CPU.

Timings come from the runs behind the published models (docs/embedding.md,
docs/ltr.md) and, for `data`, the slower `index` figure and `eval`, from runs made
while writing this page.
They were measured on one laptop:

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H (16 cores / 22 threads) |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop (6 GB), used only for the fine-tune |
| OS / Python | Windows 11, Python 3.11 to 3.13 |

## What should come out

- `data` logs whether the new `recipes.db` matches the corpus the eval set was
  built from (the fingerprint in `eval/queries_v1.meta.json`). It matches when
  the database starts empty and the default 380,000-row limit is used. Recipe
  ids are assigned at insert time and every split hashes them, so a different
  limit, or ingesting into a database that already had rows, gives different
  splits and numbers that cannot be compared.
- The eval step should land near the "Hybrid e5-small-ft + LTR v3" row of
  `eval/results/v1_1_comparison.md` (MRR@10 0.644). A check made for this page
  rebuilt the corpus with `build_database.py`, installed the released models
  with `fetch-models` and built a new e5 index. It gave MRR@10 0.641 [0.613,
  0.669], Recall@10 0.750 and Hit@1 0.583, against 0.644, 0.756 and 0.585
  published, all inside the intervals. Besides the database, only the e5 index
  was rebuilt (with newer library versions than the original); the cause of the
  small gap was not investigated. That check did not rerun the fine-tune or the
  LTR training. The fine-tune fixes its seed but runs on CUDA with fp16
  autocast, so a different GPU or driver can give slightly different weights.
  Run-to-run variance was not measured (docs/embedding.md, Limitations).
- `ltr-data` also writes `ltr_heldout_k100*.parquet`. The `--top-k 20 --splits
  heldout` variants used by `scripts/evaluate_ranking.py` are not part of the
  chain. Add them by hand if you want those numbers (docs/ltr.md).

## Checking the commands quickly

`--quick` runs every step with tiny settings: 3,000 recipes, 256 fine-tune
pairs, 40 LTR queries, no grid, 20 eval queries. The numbers it produces mean
nothing. It only shows that the chain runs end to end on your machine:

```
PANTRYSENSE_DATA_DIR=/tmp/pantry-quick python tasks.py reproduce --quick
```

On the laptop above it took under 3 minutes (166 s), with the parquet and the
base e5 model already downloaded.
