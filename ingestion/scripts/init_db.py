import sqlite3

conn = sqlite3.connect("./data/db.sqlite3")

schema_file = "./data/schema.sql"

with open(schema_file, "r") as f:
    schema = f.read()

conn.executescript(schema)

conn.close()