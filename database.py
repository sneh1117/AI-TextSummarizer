import sqlite3, datetime

DB = "history.db"

def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT,
                summary TEXT,
                char_count INTEGER,
                created_at TEXT
            )
        """)

def save_summary(original, summary):
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO summaries (original_text, summary, char_count, created_at) VALUES (?,?,?,?)",
            (original, summary, len(original), datetime.datetime.now().isoformat())
        )

def get_history(limit=10):
    with sqlite3.connect(DB) as conn:
        rows = conn.execute(
            "SELECT id, summary, char_count, created_at FROM summaries ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows