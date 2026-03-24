"""
api/db.py
---------
Database connections and path resolution for the Climbing Intelligence API.

Provides:
  - ThreadedConnectionPool for PostgreSQL (reuse connections across requests)
  - get_pg() context manager (borrow/return from pool automatically)
  - SQLite helper for kilter.db reads
  - Canonical path constants: MODEL_DIR, DATA_DIR, SQLITE_PATH
"""

import contextlib
import os
import sqlite3
from pathlib import Path

import psycopg2
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv

# ── Project root resolution ───────────────────────────────────────────────────

_API_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = _API_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

MODEL_DIR   = PROJECT_ROOT / "ml"
DATA_DIR    = PROJECT_ROOT / "data"
SQLITE_PATH = Path(os.getenv("KILTER_DB_PATH", str(PROJECT_ROOT / "kilter.db"))).expanduser()

# ── PostgreSQL connection pool ────────────────────────────────────────────────

_pool: pg_pool.ThreadedConnectionPool | None = None


def _pg_dsn() -> dict:
    return dict(
        dbname=os.getenv("DB_NAME", "climbing_platform"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER", os.getenv("USER", "")),
        password=os.getenv("DB_PASSWORD", ""),
    )


def get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(minconn=2, maxconn=10, **_pg_dsn())
    return _pool


@contextlib.contextmanager
def get_pg():
    """
    Borrow a pooled PostgreSQL connection.

    Usage:
        with get_pg() as conn:
            cur = conn.cursor()
            ...
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── SQLite helper ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def get_sqlite():
    """Open a read-only SQLite connection to kilter.db."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    try:
        yield conn
    finally:
        conn.close()
