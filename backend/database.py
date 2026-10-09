"""
database.py — SQLite helpers for the sentiment analyzer backend.

Schema
------
feedback
  id          INTEGER  PRIMARY KEY AUTOINCREMENT
  name        TEXT     nullable
  email       TEXT     NOT NULL
  category    TEXT     NOT NULL  ('Product' | 'Service' | 'General')
  rating      INTEGER  NOT NULL  (1-5)
  feedback    TEXT     NOT NULL
  sentiment   TEXT     NOT NULL  ('POSITIVE' | 'NEGATIVE')
  confidence  REAL     NOT NULL  (0.0-1.0)
  created_at  TEXT     NOT NULL  (ISO-8601 UTC)
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.environ.get("DB_PATH", "feedback.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the feedback table if it does not already exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                email       TEXT    NOT NULL,
                category    TEXT    NOT NULL,
                rating      INTEGER NOT NULL,
                feedback    TEXT    NOT NULL,
                sentiment   TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )
        conn.commit()


def insert_feedback(
    email: str,
    category: str,
    rating: int,
    feedback: str,
    sentiment: str,
    confidence: float,
    name: Optional[str] = None,
) -> int:
    """Insert a feedback row and return the new row id."""
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback
                (name, email, category, rating, feedback, sentiment, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name or None, email, category, rating, feedback, sentiment, confidence, created_at),
        )
        conn.commit()
        return cursor.lastrowid


def get_all_feedback() -> list[dict]:
    """Return all rows as a list of dicts, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_summary_stats() -> dict:
    """Return aggregate stats for the dashboard."""
    with _connect() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = conn.execute("SELECT COUNT(*) FROM feedback WHERE sentiment='POSITIVE'").fetchone()[0]
        negative = conn.execute("SELECT COUNT(*) FROM feedback WHERE sentiment='NEGATIVE'").fetchone()[0]
        avg_raw  = conn.execute("SELECT ROUND(AVG(rating),1) FROM feedback").fetchone()[0]
        avg_rating = avg_raw if avg_raw is not None else 0.0
        cat_rows = conn.execute(
            "SELECT category, COUNT(*) AS cnt FROM feedback GROUP BY category"
        ).fetchall()
        by_category = {r["category"]: r["cnt"] for r in cat_rows}

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "avg_rating": avg_rating,
        "by_category": by_category,
        "by_sentiment": {"POSITIVE": positive, "NEGATIVE": negative},
    }
