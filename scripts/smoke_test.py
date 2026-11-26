"""End-to-end check on an empty data folder: setup, start the server, search.

    python scripts/smoke_test.py --data-dir /tmp/pantry-data --limit 3000
    python scripts/smoke_test.py --data-dir /tmp/pantry-data --limit 3000 --expect e5

Runs the in-app setup pipeline (dataset download, recipes.db, metadata, dense
and BM25 indices) with a small row limit, starts uvicorn on the result and
posts one search to /api/recipes/search over HTTP. --expect checks which
configuration the server ended up on: "e5" means the fine-tuned model and the
LTR bundle (run fetch-models into the data folder first), "minilm" the
fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUERY = ["chicken", "garlic", "lemon", "olive oil"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _request(url: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _wait_for(url: str, server: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}")
        try:
            _request(url)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"server did not answer within {timeout:.0f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--engine", default="onnx", help="embedding engine for setup (onnx, cuda, cpu)")
    parser.add_argument("--expect", choices=("e5", "minilm"), default=None)
    parser.add_argument("--skip-setup", action="store_true", help="only start the server and search")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Before any app import, so every default path points into the data folder.
    os.environ["PANTRYSENSE_DATA_DIR"] = str(data_dir)
    for var in ("AUTO_OPEN_BROWSER", "AUTO_SHUTDOWN", "AUTO_BUILD_INDEX"):
        os.environ[f"PANTRYSENSE_{var}"] = "false"
    sys.path.insert(0, str(ROOT))

    setup_seconds = 0.0
    if not args.skip_setup:
        from app.backend.system.setup_pipeline import get_setup_manager

        start = time.perf_counter()
        status = get_setup_manager().run_blocking(engine=args.engine, limit=args.limit)
        setup_seconds = time.perf_counter() - start
        if not status["is_ready"]:
            print(f"setup did not finish: {status['message']}", file=sys.stderr)
            return 1

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.backend.main:app", "--port", str(port)],
                              cwd=ROOT, env=os.environ.copy())
    try:
        _wait_for(f"{base}/api/system/setup-status", server, timeout=180)
        status = _request(f"{base}/api/system/setup-status")
        start = time.perf_counter()
        recipes = _request(f"{base}/api/recipes/search", {"ingredients": QUERY})["recipes"]
        search_ms = (time.perf_counter() - start) * 1000
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"smoke test: {exc}", file=sys.stderr)
        return 1
    finally:
        server.terminate()
        server.wait(timeout=30)

    summary = {
        "recipes_indexed": status["recipe_count"],
        "setup_seconds": round(setup_seconds, 1),
        "ready": status["is_ready"],
        "embedding": status["embedding_setup"],
        "ranker_bundle": status["has_model"],
        "query": QUERY,
        "results": len(recipes),
        "first_search_ms": round(search_ms, 1),
        "top": [{"title": r["title"], "ml_score": r.get("ml_score")} for r in recipes[:3]],
    }
    print(json.dumps(summary, indent=2))

    problems = []
    if not status["is_ready"]:
        problems.append("server does not report ready")
    if not recipes:
        problems.append("search returned no recipes")
    if args.expect == "e5":
        if summary["embedding"] != "e5-finetuned":
            problems.append(f"expected the fine-tuned e5 setup, got {summary['embedding']}")
        if not summary["ranker_bundle"] or (recipes and recipes[0].get("ml_score") is None):
            problems.append("expected results ranked by the LTR bundle")
    elif args.expect == "minilm" and summary["embedding"] != "minilm":
        problems.append(f"expected the MiniLM fallback, got {summary['embedding']}")
    for problem in problems:
        print(f"smoke test: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
