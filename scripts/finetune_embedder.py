"""Fine-tune and compare embedding models for ingredient-list -> recipe retrieval.

Subcommands:
  drift        compare sentence-transformers and FastEmbed vectors for one model
  train        fine-tune a sentence-transformers model with in-batch negatives (MNRL)
  bench        Recall@K / MRR@10 of several models on a held-out split of the train pool
  build-index  embed every recipe in the database and write a FAISS index + metadata

Data split: bucket = md5(recipe_id) % 10. Buckets 0-1 are the official eval pool and
are never read here except by build-index (which indexes the whole database).
Buckets 2-9 are the train pool; a second hash moves ~2% of it into a held-out
split used by `bench`, so held-out recipes never appear as training positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backend.config import get_database_path  # noqa: E402
from app.backend.retrieval.embedder import (  # noqa: E402
    BACKEND_SENTENCE_TRANSFORMERS,
    IngredientEmbedder,
    resolve_prefixes,
    write_index_metadata,
)
from app.backend.retrieval.semantic import (  # noqa: E402
    build_query_representation,
    build_recipe_representation,
)
from app.backend.retrieval.vector_store import FaissVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_embedder")

EVAL_BUCKETS = frozenset({0, 1})
HELDOUT_PERCENT = 2
DEFAULT_MODELS_DIR = ROOT / "data" / "models"
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def bucket(recipe_id: int) -> int:
    return int(hashlib.md5(str(recipe_id).encode()).hexdigest(), 16) % 10


def is_heldout(recipe_id: int) -> bool:
    return int(hashlib.md5(f"heldout:{recipe_id}".encode()).hexdigest(), 16) % 100 < HELDOUT_PERCENT


@dataclass(frozen=True)
class Recipe:
    id: int
    title: str
    ingredients: tuple[str, ...]

    @property
    def text(self) -> str:
        return build_recipe_representation(self.title, list(self.ingredients))


def load_recipes(db_path: Path, include_eval_pool: bool = False) -> list[Recipe]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT id, title, ingredients FROM recipes ORDER BY id").fetchall()
    out = []
    for rid, title, ingredients in rows:
        if not include_eval_pool and bucket(rid) in EVAL_BUCKETS:
            continue
        out.append(Recipe(int(rid), title or "", tuple(json.loads(ingredients or "[]"))))
    return out


def split_pool(recipes: list[Recipe]) -> tuple[list[Recipe], list[Recipe]]:
    train, heldout = [], []
    for r in recipes:
        assert bucket(r.id) not in EVAL_BUCKETS
        (heldout if is_heldout(r.id) else train).append(r)
    return train, heldout


def common_ingredients(recipes: list[Recipe], max_df: float) -> set[str]:
    """Ingredients that appear in more than `max_df` of recipes (salt, water, ...)."""
    df = Counter()
    for r in recipes:
        df.update(set(_norm(r)))
    limit = max_df * len(recipes)
    return {ing for ing, n in df.items() if n > limit}


_NORM_CACHE: dict[int, list[str]] = {}


def _norm(r: Recipe) -> list[str]:
    from app.backend.recipes.matching import normalize_ingredients

    cached = _NORM_CACHE.get(r.id)
    if cached is None:
        cached = _NORM_CACHE[r.id] = normalize_ingredients(list(r.ingredients))
    return cached


def make_query(r: Recipe, rng: random.Random, common: set[str], min_frac: float, max_frac: float) -> str | None:
    """Keep 40-70% of the recipe's non-trivial ingredients as a pantry-style query."""
    usable = [i for i in _norm(r) if i not in common]
    if len(usable) < 3:
        return None
    k = max(2, round(len(usable) * rng.uniform(min_frac, max_frac)))
    picked = rng.sample(usable, min(k, len(usable)))
    return build_query_representation(picked)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelSpec:
    label: str
    name: str
    query_prefix: str
    passage_prefix: str


def model_spec(arg: str) -> ModelSpec:
    """`label=name[|query_prefix|passage_prefix]`, or just `name`."""
    label, _, rest = arg.partition("=") if "=" in arg else (arg, "", arg)
    parts = rest.split("|")
    name = parts[0]
    q, p = resolve_prefixes(name)
    if len(parts) == 3:
        q, p = parts[1], parts[2]
    return ModelSpec(label, name, q, p)


def load_st(name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def encode(model, texts: list[str], prefix: str, batch_size: int = 256) -> np.ndarray:
    vecs = model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 20000,
    )
    return np.asarray(vecs, dtype=np.float32)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- drift


def cmd_drift(args: argparse.Namespace) -> None:
    from fastembed import TextEmbedding

    recipes = load_recipes(args.db)
    rng = random.Random(args.seed)
    sample = rng.sample(recipes, args.n)
    common = common_ingredients(recipes, args.max_df)
    queries = [q for q in (make_query(r, rng, common, 0.4, 0.7) for r in sample) if q]
    passages = [r.text for r in sample]

    st = load_st(args.model, pick_device(args.device))
    fe = TextEmbedding(model_name=args.model)
    result = {"model": args.model}
    for kind, texts in (("query", queries), ("passage", passages)):
        a = encode(st, texts, "")
        b = np.array(list(fe.embed(texts)), dtype=np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cos = (a * b).sum(1)
        result[kind] = {"n": len(texts), "mean": float(cos.mean()), "min": float(cos.min())}
    print(json.dumps(result, indent=2))


# --------------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    recipes = load_recipes(args.db)
    train, heldout = split_pool(recipes)
    common = common_ingredients(train, args.max_df)
    logger.info("train pool: %d recipes, held-out: %d, common ingredients dropped: %d",
                len(train), len(heldout), len(common))

    rng = random.Random(args.seed)
    order = list(train)
    rng.shuffle(order)
    pairs: list[tuple[str, str]] = []
    for r in order:
        q = make_query(r, rng, common, args.min_frac, args.max_frac)
        if q:
            pairs.append((q, r.text))
        if len(pairs) >= args.max_pairs:
            break
    logger.info("generated %d training pairs", len(pairs))

    model = load_st(args.base, device)
    model.max_seq_length = args.max_seq_length
    q_prefix, p_prefix = resolve_prefixes(args.base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def embed(texts: list[str]) -> torch.Tensor:
        features = model.tokenize(texts)
        features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        return F.normalize(model(features)["sentence_embedding"], dim=-1)

    model.train()
    start = time.perf_counter()
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i : i + args.batch_size]
            if len(batch) < 2:
                continue
            # Recipes with identical text would be false negatives in the same batch.
            seen, dedup = set(), []
            for q, p in batch:
                if p not in seen:
                    seen.add(p)
                    dedup.append((q, p))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                qv = embed([q_prefix + q for q, _ in dedup])
                pv = embed([p_prefix + p for _, p in dedup])
                scores = qv @ pv.T * args.scale
                labels = torch.arange(len(dedup), device=device)
                # Symmetric MNRL: query->passage and passage->query.
                loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                logger.info("epoch %d step %d/%d loss %.4f", epoch + 1, step, total_steps, loss.item())
    train_seconds = time.perf_counter() - start
    model.eval()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if q_prefix or p_prefix:
        model.prompts = {"query": q_prefix, "passage": p_prefix}
    model.save(str(out))
    meta = {
        "base_model": args.base,
        "pairs": len(pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scale": args.scale,
        "seed": args.seed,
        "max_df": args.max_df,
        "query_fraction": [args.min_frac, args.max_frac],
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_seconds": round(train_seconds, 1),
        "steps": step,
        "train_buckets": "2-9 minus held-out",
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("saved %s (%.1fs of training)", out, train_seconds)


# --------------------------------------------------------------------------- bench


def build_eval_set(args: argparse.Namespace):
    recipes = load_recipes(args.db)
    train, heldout = split_pool(recipes)
    common = common_ingredients(train, args.max_df)
    rng = random.Random(args.seed + 1)

    targets = [r for r in heldout if make_query(r, random.Random(0), common, 0.4, 0.7)]
    rng.shuffle(targets)
    targets = targets[: args.queries]
    queries = [make_query(r, rng, common, args.min_frac, args.max_frac) for r in targets]

    corpus_size = args.corpus_size or len(recipes)
    target_ids = {r.id for r in targets}
    others = [r for r in recipes if r.id not in target_ids]
    rng.shuffle(others)
    corpus = targets + others[: max(0, corpus_size - len(targets))]

    # Recipes with exactly the same representation are indistinguishable; count them as hits.
    by_text: dict[str, list[int]] = {}
    for r in corpus:
        by_text.setdefault(r.text, []).append(r.id)
    relevant = [set(by_text[r.text]) for r in targets]
    return queries, relevant, corpus


def evaluate(qv: np.ndarray, pv: np.ndarray, corpus_ids: np.ndarray, relevant: list[set[int]]) -> dict:
    import faiss

    index = faiss.IndexFlatIP(pv.shape[1])
    index.add(pv)
    _, idx = index.search(qv, 100)
    hits = {10: 0, 50: 0, 100: 0}
    mrr = 0.0
    for row, rel in zip(idx, relevant):
        ranked = corpus_ids[row]
        first = next((i for i, rid in enumerate(ranked) if rid in rel), None)
        if first is None:
            continue
        for k in hits:
            if first < k:
                hits[k] += 1
        if first < 10:
            mrr += 1.0 / (first + 1)
    n = len(relevant)
    return {f"recall@{k}": hits[k] / n for k in hits} | {"mrr@10": mrr / n}


def query_latency_ms(model, prefix: str, queries: list[str], n: int = 200) -> float:
    sample = queries[:n]
    encode(model, sample[:5], prefix)  # warm-up
    times = []
    for q in sample:
        t = time.perf_counter()
        model.encode([prefix + q], normalize_embeddings=True, show_progress_bar=False)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def cmd_bench(args: argparse.Namespace) -> None:
    import torch

    queries, relevant, corpus = build_eval_set(args)
    corpus_ids = np.array([r.id for r in corpus])
    corpus_texts = [r.text for r in corpus]
    logger.info("held-out queries: %d, corpus: %d recipes", len(queries), len(corpus))

    device = pick_device(args.device)
    results = []
    for spec in map(model_spec, args.models):
        model = load_st(spec.name, device)
        t = time.perf_counter()
        pv = encode(model, corpus_texts, spec.passage_prefix)
        index_seconds = time.perf_counter() - t
        qv = encode(model, queries, spec.query_prefix)
        metrics = evaluate(qv, pv, corpus_ids, relevant)
        row = {
            "label": spec.label,
            "model": spec.name,
            "query_prefix": spec.query_prefix,
            "passage_prefix": spec.passage_prefix,
            **{k: round(v, 4) for k, v in metrics.items()},
            "embed_corpus_seconds": round(index_seconds, 1),
            "query_ms_gpu": round(query_latency_ms(model, spec.query_prefix, queries), 2) if device == "cuda" else None,
        }
        cpu_model = load_st(spec.name, "cpu")
        row["query_ms_cpu"] = round(query_latency_ms(cpu_model, spec.query_prefix, queries), 2)
        row["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        logger.info("%s: %s", spec.label, row)
        results.append(row)
        del model, cpu_model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = {
        "queries": len(queries),
        "corpus_size": len(corpus),
        "corpus_note": "all held-out targets plus random recipes from buckets 2-9",
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(queries)} held-out queries, {len(corpus):,}-recipe corpus\n")
    print("| Model | Recall@10 | Recall@50 | Recall@100 | MRR@10 | Query ms (CPU) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['recall@10']:.3f} | {r['recall@50']:.3f} | {r['recall@100']:.3f} "
              f"| {r['mrr@10']:.3f} | {r['query_ms_cpu']:.1f} |")


# --------------------------------------------------------------------------- build-index


def cmd_build_index(args: argparse.Namespace) -> None:
    recipes = load_recipes(args.db, include_eval_pool=True)
    embedder = IngredientEmbedder(model_name=args.model, engine="cuda" if pick_device(args.device) == "cuda" else "cpu")
    if embedder.backend != BACKEND_SENTENCE_TRANSFORMERS:
        embedder = IngredientEmbedder(model_name=args.model, engine="cpu")
    start = time.perf_counter()
    vecs = embedder.embed_passages([r.text for r in recipes])
    logger.info("embedded %d recipes in %.1fs", len(recipes), time.perf_counter() - start)
    store = FaissVectorStore(dimension=vecs.shape[1])
    store.add(vecs, [r.id for r in recipes])
    store.save(args.index, args.ids)
    write_index_metadata(args.index, embedder, count=len(recipes))
    logger.info("wrote %s, %s and metadata", args.index, args.ids)


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=get_database_path())
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-df", type=float, default=0.15,
                        help="drop ingredients present in more than this share of recipes from queries")
    parser.add_argument("--min-frac", type=float, default=0.4)
    parser.add_argument("--max-frac", type=float, default=0.7)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("drift")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--n", type=int, default=400)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("train")
    p.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output", type=Path, default=DEFAULT_MODELS_DIR / "minilm-pantry")
    p.add_argument("--max-pairs", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("bench")
    p.add_argument("--models", nargs="+", required=True, help="label=name[|query_prefix|passage_prefix]")
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--corpus-size", type=int, default=50_000, help="0 = whole train pool")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "embedding_bench.json")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("build-index")
    p.add_argument("--model", required=True)
    p.add_argument("--index", type=Path, default=ROOT / "data" / "recipe_vectors.index")
    p.add_argument("--ids", type=Path, default=ROOT / "data" / "recipe_vector_ids.json")
    p.set_defaults(func=cmd_build_index)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
