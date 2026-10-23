"""SQLite storage: connection handling and a minimal forward-only migration runner.

Migrations are numbered SQL files in `migrations/` (001_initial.sql, 002_...). Each file is
applied once, inside a transaction, and recorded in `schema_migrations`.
"""
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL") if str(path) != ":memory:" else None
    migrate(con)
    return con


def migrate(con: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in order. Returns the names applied."""
    con.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    done = {r[0] for r in con.execute("SELECT name FROM schema_migrations")}
    applied = []
    for f in sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql")):
        if f.name in done:
            continue
        log.info("applying migration %s", f.name)
        try:
            con.executescript("BEGIN;\n" + f.read_text(encoding="utf-8") +
                              f"\nINSERT INTO schema_migrations(name) VALUES ('{f.name}');\nCOMMIT;")
        except sqlite3.Error:
            con.rollback()
            raise
        applied.append(f.name)
    return applied
