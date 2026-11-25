import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.backend.database.init_db import initialize_database  # noqa: E402

if __name__ == "__main__":
    initialize_database()
    print("PantrySense recipe database initialized.")
