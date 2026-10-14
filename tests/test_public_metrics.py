"""Public growth curves contain anonymous UTC counts and perform no writes."""
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from paperleet import public_metrics


@pytest.fixture
def account_history(tmp_path, monkeypatch):
    path = tmp_path / "community.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT, status TEXT,
                                is_test_user INTEGER, created_at TEXT NOT NULL);
            CREATE TABLE accounts_private (user_id TEXT, email TEXT);
            CREATE TABLE account_sessions (user_id TEXT, token_hash TEXT);
        """)
    monkeypatch.setattr(public_metrics, "_now", lambda: datetime(2026, 1, 4, 12, tzinfo=timezone.utc))
    return path


def test_created_account_curve_includes_unused_beta_and_disabled_accounts(account_history):
    with sqlite3.connect(account_history) as db:
        db.executemany("INSERT INTO users VALUES(?,?,?,?,?)", [
            ("private-a", "alice", "active", 0, "2026-01-01T12:00:00Z"),
            ("private-b", "beta31", "active", 1, "2026-01-03T12:00:00Z"),
            ("private-c", "beta32", "disabled", 1, "2026-01-03T13:00:00Z"),
        ])
    data = public_metrics.created_account_metrics(account_history)
    assert set(data) == {"metric", "label", "definition", "updated_at", "points"}
    assert data["metric"] == "created_accounts" and data["label"] == "Created accounts"
    assert data["updated_at"] == "2026-01-04T12:00:00Z"
    assert data["points"] == [
        {"date": "2026-01-01", "value": 1}, {"date": "2026-01-02", "value": 1},
        {"date": "2026-01-03", "value": 3}, {"date": "2026-01-04", "value": 3},
    ]
    assert "pre-created beta" in data["definition"] and "hard-deleted" in data["definition"]
    encoded = json.dumps(data)
    for private in ("private-a", "private-b", "private-c", "alice", "beta31", "beta32", "username", "email", "user_id"):
        assert private not in encoded


def test_created_account_days_are_utc_and_not_timestamp_text_order(account_history):
    with sqlite3.connect(account_history) as db:
        db.executemany("INSERT INTO users VALUES(?,?,?,?,?)", [
            ("a", "one", "active", 0, "2026-01-02T00:30:00+02:00"),
            ("b", "two", "active", 0, "2026-01-02T23:30:00-02:00"),
            ("c", "three", "active", 0, "2026-01-03T00:00:00"),
        ])
    assert public_metrics.created_account_metrics(account_history)["points"] == [
        {"date": "2026-01-01", "value": 1}, {"date": "2026-01-02", "value": 1},
        {"date": "2026-01-03", "value": 3}, {"date": "2026-01-04", "value": 3},
    ]


def test_empty_initialized_database_has_zero_but_missing_database_is_not_created(account_history, tmp_path):
    assert public_metrics.created_account_metrics(account_history)["points"] == [{"date": "2026-01-04", "value": 0}]
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        public_metrics.created_account_metrics(missing)
    assert not missing.exists()


@pytest.mark.parametrize("timestamp", ["not-a-date", "2026-01-05T00:00:00Z"])
def test_invalid_history_fails_instead_of_publishing_a_partial_count(account_history, timestamp):
    with sqlite3.connect(account_history) as db:
        db.execute("INSERT INTO users VALUES(?,?,?,?,?)", ("a", "one", "active", 0, timestamp))
    with pytest.raises(ValueError):
        public_metrics.created_account_metrics(account_history)


def test_metrics_read_only_creation_timestamps_and_never_change_database(account_history, monkeypatch):
    before = account_history.read_bytes()
    original = sqlite3.connect

    def guarded_connect(path, *args, **kwargs):
        assert path.endswith("?mode=ro") and kwargs["uri"] is True
        db = original(path, *args, **kwargs)

        def authorize(action, table, column, database, context):
            if action == sqlite3.SQLITE_READ and (table != "users" or column not in ("", "created_at")):
                return sqlite3.SQLITE_DENY
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                          sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_ALTER_TABLE):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(authorize)
        return db

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    assert public_metrics.created_account_metrics(account_history)["points"][-1]["value"] == 0
    assert account_history.read_bytes() == before
