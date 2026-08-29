from __future__ import annotations

import sqlite3
from pathlib import Path

from app.backend.config import get_database_path


def get_connection(database_path: Path | None = None) -> sqlite3.Connection:
    path = database_path or get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection
