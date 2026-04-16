"""
db.py — Lightweight SQLite storage for deploy outcomes and DORA metrics.

Why SQLite and not ChromaDB here?
- We're storing structured tabular data (outcomes with labels), not text embeddings
- We need to query by repo_name, sort by timestamp, compute aggregates
- SQLite is zero-config, file-backed, and perfect for a local demo

Two tables:
  1. deploy_outcomes — one row per recorded deployment with its result (failed/success)
     This is the training/inference data for the ML risk model.
  2. (DORA metrics are computed on-the-fly from deploy_outcomes, not stored separately)

Schema is created on first import — safe to call repeatedly.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "./dora.db")


def _get_conn() -> sqlite3.Connection:
    """
    Returns a connection with row_factory set so rows behave like dicts.
    Each call creates a new connection — SQLite connections are not thread-safe
    to share across requests in FastAPI.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create tables if they don't exist.
    Called once at app startup from main.py.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deploy_outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_name   TEXT    NOT NULL,
                pr_title    TEXT    NOT NULL,
                author      TEXT    NOT NULL,
                branch      TEXT    NOT NULL,

                -- Features used by the ML model (same as DeploymentEvent fields)
                pr_size_lines_changed   INTEGER NOT NULL,
                files_changed           INTEGER NOT NULL,
                new_test_files          INTEGER NOT NULL,
                days_since_last_deploy  REAL    NOT NULL,
                deploys_last_7_days     INTEGER NOT NULL,
                hour_of_day             INTEGER NOT NULL,
                day_of_week             INTEGER NOT NULL,

                -- Ground truth label
                failed      INTEGER NOT NULL DEFAULT 0,   -- 0=success, 1=failure

                -- Metadata
                commit_sha  TEXT,
                recorded_at TEXT    NOT NULL
            )
        """)
        conn.commit()


def record_outcome(outcome: dict) -> int:
    """
    Insert one deploy outcome row. Returns the new row id.
    outcome dict must have all columns from the schema above.
    """
    with _get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO deploy_outcomes (
                repo_name, pr_title, author, branch,
                pr_size_lines_changed, files_changed, new_test_files,
                days_since_last_deploy, deploys_last_7_days,
                hour_of_day, day_of_week,
                failed, commit_sha, recorded_at
            ) VALUES (
                :repo_name, :pr_title, :author, :branch,
                :pr_size_lines_changed, :files_changed, :new_test_files,
                :days_since_last_deploy, :deploys_last_7_days,
                :hour_of_day, :day_of_week,
                :failed, :commit_sha, :recorded_at
            )
        """, {**outcome, "recorded_at": datetime.now(timezone.utc).isoformat()})
        conn.commit()
        return cursor.lastrowid


def get_all_outcomes() -> list[dict]:
    """
    Fetch all recorded deploy outcomes as a list of dicts.
    Used by the ML scorer to retrain / score on.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM deploy_outcomes ORDER BY recorded_at ASC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_outcomes_for_repo(repo_name: str) -> list[dict]:
    """Fetch all outcomes for a specific repo — used for per-repo DORA metrics."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM deploy_outcomes WHERE repo_name = ? ORDER BY recorded_at ASC",
            (repo_name,)
        ).fetchall()
        return [dict(row) for row in rows]


def count_outcomes() -> int:
    """How many deploy outcomes are recorded — shown in /health."""
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM deploy_outcomes").fetchone()[0]


def get_all_repos() -> list[str]:
    """Return unique repo names — used by /dora-metrics to list available repos."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT repo_name FROM deploy_outcomes ORDER BY repo_name"
        ).fetchall()
        return [row["repo_name"] for row in rows]
