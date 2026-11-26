from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.config import EmbeddingSetup, get_database_path, get_index_build_setup
from app.backend.database.init_db import initialize_database
from app.backend.retrieval.bm25_store import convert_bm25_index
from app.backend.retrieval.index_build import build_search_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_indices(
    db_path: Path | None = None,
    index_path: Path | None = None,
    ids_path: Path | None = None,
    bm25_path: Path | None = None,
    legacy_json: bool = True,
) -> int:
    """Embed every recipe and build the FAISS and BM25 indices; returns the recipe count.

    The model and default index paths come from config.get_index_build_setup():
    the fine-tuned e5 model when its folder is in the data directory, MiniLM
    otherwise.
    """
    target_db = db_path or get_database_path()
    setup = get_index_build_setup()
    setup = EmbeddingSetup(setup.name, setup.model, index_path or setup.index_path, ids_path or setup.ids_path)
    initialize_database(target_db)

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "embed":
            logger.info("Embedded %d / %d recipes", done, total)
        else:
            logger.info("Building the %s index", "FAISS" if stage == "faiss" else "BM25")

    return build_search_indices(target_db, setup, bm25_path=bm25_path, legacy_json=legacy_json,
                                progress=progress).recipes


def convert_faiss_to_hnsw(
    source: Path, target: Path, m: int = 32, ef_construction: int = 80, ef_search: int = 128
) -> None:
    """Rebuild a flat FAISS index as HNSW. Row order (and so the ids file) is unchanged."""
    import faiss

    start = time.perf_counter()
    flat = faiss.read_index(str(source))
    vectors = flat.reconstruct_n(0, flat.ntotal)
    hnsw = faiss.IndexHNSWFlat(flat.d, m, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = ef_construction
    hnsw.add(vectors)
    hnsw.hnsw.efSearch = ef_search
    target.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(hnsw, str(target))
    logger.info("Wrote HNSW%d index (%d vectors, efSearch=%d) to %s in %.1fs",
                m, hnsw.ntotal, ef_search, target, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS and BM25 retrieval indices.")
    parser.add_argument(
        "--convert-bm25",
        nargs="+",
        metavar=("SOURCE", "TARGET"),
        help="Only convert a BM25 index between formats (default target: SOURCE with .npz suffix)",
    )
    parser.add_argument("--no-legacy-json", action="store_true", help="Skip writing the JSON BM25 file")
    parser.add_argument(
        "--convert-faiss-hnsw",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Only rebuild an existing FAISS index as HNSW (approximate; see docs/performance.md)",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=128)
    args = parser.parse_args()

    if args.convert_faiss_hnsw:
        convert_faiss_to_hnsw(
            Path(args.convert_faiss_hnsw[0]), Path(args.convert_faiss_hnsw[1]),
            m=args.hnsw_m, ef_construction=args.ef_construction, ef_search=args.ef_search,
        )
        return

    if args.convert_bm25:
        source = Path(args.convert_bm25[0])
        target = Path(args.convert_bm25[1]) if len(args.convert_bm25) > 1 else source.with_suffix(".npz")
        start = time.perf_counter()
        store = convert_bm25_index(source, target)
        logger.info(
            "Converted %s -> %s (%d docs, %d terms) in %.1fs",
            source, target, store.total_docs, store.vocabulary_size, time.perf_counter() - start,
        )
        return
    build_indices(legacy_json=not args.no_legacy_json)


if __name__ == "__main__":
    main()
