import sqlite3
from pathlib import Path

conn = sqlite3.connect(Path(__file__).parent.parent / "data" / "db.sqlite3")

with open(Path(__file__).parent.parent / "data" / "schema.sql", "r") as f:
    conn.executescript(f.read())

conn.close()