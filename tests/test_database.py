import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


def _insert(sentiment="POSITIVE", category="Product", rating=5):
    return database.insert_feedback(
        email="a@example.com",
        category=category,
        rating=rating,
        feedback="Great product, works well.",
        sentiment=sentiment,
        confidence=0.99,
    )


def test_get_feedback_by_id():
    row_id = _insert()
    row = database.get_feedback_by_id(row_id)
    assert row["id"] == row_id
    assert row["name"] is None
    assert database.get_feedback_by_id(row_id + 1) is None


def test_summary_stats():
    _insert("POSITIVE", "Product", 5)
    _insert("NEGATIVE", "Service", 2)
    stats = database.get_summary_stats()
    assert stats["total"] == 2
    assert stats["positive"] == 1 and stats["negative"] == 1
    assert stats["avg_rating"] == 3.5
    assert stats["by_category"] == {"Product": 1, "Service": 1}


def test_empty_stats():
    assert database.get_summary_stats()["avg_rating"] == 0.0
