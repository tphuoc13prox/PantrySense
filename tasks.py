"""Common development commands, runnable on Windows, macOS and Linux.

    python tasks.py <command> [extra args passed to the underlying tool]

Commands: install, test, test-all, lint, typecheck, run, build-index,
train-ranker, docker-build, docker-run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(*cmd: str, extra: list[str] | None = None) -> int:
    full = [*cmd, *(extra or [])]
    print("+", " ".join(full), flush=True)
    return subprocess.call(full, cwd=ROOT)


def install(extra: list[str]) -> int:
    """Install runtime and dev dependencies (CPU-only torch)."""
    return _run(
        PY, "-m", "pip", "install",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        "-r", "requirements-dev.txt",
        extra=extra,
    )


def test(extra: list[str]) -> int:
    """Run the suite the way CI does (skips slow tests)."""
    return _run(PY, "-m", "pytest", "-m", "not slow", extra=extra)


def test_all(extra: list[str]) -> int:
    """Run every test, including slow ones and those needing data/ artifacts."""
    return _run(PY, "-m", "pytest", extra=extra)


def lint(extra: list[str]) -> int:
    """Ruff check (read-only; pass --fix yourself if you want fixes)."""
    return _run(PY, "-m", "ruff", "check", ".", extra=extra)


def typecheck(extra: list[str]) -> int:
    """Run mypy on app/."""
    return _run(PY, "-m", "mypy", extra=extra)


def run(extra: list[str]) -> int:
    """Start the app on http://localhost:8000 with auto-reload."""
    return _run(PY, "-m", "uvicorn", "app.backend.main:app", "--reload", "--port", "8000", extra=extra)


def build_index(extra: list[str]) -> int:
    """Build the FAISS and BM25 indices from data/recipes.db."""
    return _run(PY, "-m", "scripts.build_vector_index", extra=extra)


def train_ranker(extra: list[str]) -> int:
    """Train the ranker and write data/ranker_model.joblib."""
    return _run(PY, "-m", "scripts.train_ranker", extra=extra)


def docker_build(extra: list[str]) -> int:
    """Build the image as pantrysense."""
    return _run("docker", "build", "-t", "pantrysense", ".", extra=extra)


def docker_run(extra: list[str]) -> int:
    """Run the image with ./data mounted at /app/data."""
    return _run(
        "docker", "run", "--rm", "-p", "8000:8000",
        "-v", f"{ROOT / 'data'}:/app/data", "pantrysense",
        extra=extra,
    )


COMMANDS = {
    "install": install,
    "test": test,
    "test-all": test_all,
    "lint": lint,
    "typecheck": typecheck,
    "run": run,
    "build-index": build_index,
    "train-ranker": train_ranker,
    "docker-build": docker_build,
    "docker-run": docker_run,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        for name, fn in COMMANDS.items():
            print(f"  {name:<13} {(fn.__doc__ or '').strip()}")
        return 0 if not argv or argv[0] in ("-h", "--help", "help") else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
