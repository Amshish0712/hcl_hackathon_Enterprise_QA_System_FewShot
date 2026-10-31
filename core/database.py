"""SQLite persistence. Paths are independent of the current working directory."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from core.security import hash_password


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
DEFAULT_ADMIN_EMAIL = "admin@enterprise.com"
DEFAULT_ADMIN_PASSWORD = "Admin@123"


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(DB_PATH, timeout=10)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    try:
        with database:
            yield database
    finally:
        database.close()


def initialize_database() -> None:
    with connection() as database:
        database.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Additive Phase 2 schema: existing accounts are never replaced.
        database.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                extension TEXT NOT NULL,
                file_size INTEGER NOT NULL CHECK (file_size > 0),
                uploaded_by INTEGER NOT NULL REFERENCES users(id),
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                extraction_status TEXT NOT NULL,
                extracted_characters INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Serialize the additive migration so simultaneous Streamlit sessions
        # cannot both attempt to add the same column.
        database.execute("BEGIN IMMEDIATE")
        columns = {row["name"] for row in database.execute("PRAGMA table_info(documents)")}
        for name, definition in (
            ("indexing_status", "TEXT NOT NULL DEFAULT 'not_indexed'"),
            ("indexed_at", "TEXT"),
            ("chunk_count", "INTEGER NOT NULL DEFAULT 0"),
            ("embedding_model", "TEXT"),
        ):
            if name not in columns:
                database.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
        database.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL CHECK (length(trim(chunk_text)) > 0),
                chunk_length INTEGER NOT NULL,
                page_number INTEGER,
                section_title TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, chunk_index)
            )
        """)
        database.execute("""
            CREATE TABLE IF NOT EXISTS vector_mapping (
                position INTEGER PRIMARY KEY,
                chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        database.execute("""
            CREATE TABLE IF NOT EXISTS index_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                checksum TEXT NOT NULL,
                embedding_model TEXT NOT NULL
            )
        """)
        (DB_PATH.parent / "vector_store").mkdir(parents=True, exist_ok=True)
        existing = database.execute(
            "SELECT id FROM users WHERE email = ?", (DEFAULT_ADMIN_EMAIL,)
        ).fetchone()
        if existing is None:
            database.execute(
                """INSERT INTO users (name, email, password_hash, role)
                   VALUES (?, ?, ?, ?) ON CONFLICT(email) DO NOTHING""",
                ("Enterprise Admin", DEFAULT_ADMIN_EMAIL,
                 hash_password(DEFAULT_ADMIN_PASSWORD), "admin"),
            )
