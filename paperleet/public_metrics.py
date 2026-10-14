"""Anonymous repository-chart metrics, read without account or session writes."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3


def _now():
    return datetime.now(timezone.utc)


def created_account_metrics(db_path):
    """Count retained account rows by their UTC creation date.

    Pre-created beta accounts and disabled accounts are included. This is a
    creation count, not a measure of visitors or active users. Read only the
    creation timestamp; credentials, identities, sessions and code are unused.
    A missing database or malformed history must fail instead of publishing a
    misleading zero. Hard-deleted rows cannot be recovered from this aggregate.
    """
    now = _now().astimezone(timezone.utc)
    today = now.date()
    connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        rows = connection.execute("""
            SELECT date(created_at) AS day, COUNT(*) AS count
            FROM users GROUP BY date(created_at) ORDER BY day
        """).fetchall()
    finally:
        connection.close()

    daily = {}
    for day, count in rows:
        if day is None:
            raise ValueError("Account creation history contains an invalid date")
        parsed = date.fromisoformat(day)
        if parsed > today:
            raise ValueError("Account creation history contains a future date")
        daily[parsed] = count

    points, total = [], 0
    start = min(daily) if daily else today
    for offset in range((today - start).days + 1):
        day = start + timedelta(days=offset)
        total += daily.get(day, 0)
        points.append({"date": day.isoformat(), "value": total})

    return {
        "metric": "created_accounts",
        "label": "Created accounts",
        "definition": "Cumulative accounts by UTC creation date, including pre-created beta and disabled accounts. "
                      "Counts reflect account rows retained in the database; hard-deleted accounts are not included. "
                      "This is not an active-user or visitor count.",
        "updated_at": now.isoformat().replace("+00:00", "Z"),
        "points": points,
    }
