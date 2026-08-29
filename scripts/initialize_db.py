from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.backend.database.init_db import initialize_database


if __name__ == "__main__":
    initialize_database()
    print("PantrySense recipe database initialized.")
