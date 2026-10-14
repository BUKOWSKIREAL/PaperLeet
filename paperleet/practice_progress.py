"""Public practice summaries derived from trusted runs, never learner source."""
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .catalog import problems
from .practice import full_submission_query


def _now():
    return datetime.now(timezone.utc)


def _calendar_options(year, timezone_name):
    if not isinstance(timezone_name, str) or not timezone_name or len(timezone_name) > 128:
        raise ValueError("timezone must be a valid IANA timezone")
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("timezone must be a valid IANA timezone") from None
    today = _now().astimezone(zone).date()
    if year is None:
        selected = today.year
    elif type(year) is int:
        selected = year
    elif isinstance(year, str) and year.isascii() and year.isdigit() and len(year) == 4:
        selected = int(year)
    else:
        raise ValueError("year must be a calendar year")
    if not 1970 <= selected <= today.year:
        raise ValueError(f"year must be between 1970 and {today.year}")
    return zone, today, selected


def _timestamp(value):
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Earlier imports occasionally stored a bare UTC date or naive timestamp.
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    except (AttributeError, TypeError, ValueError):
        return None


def _streak(active_dates):
    longest = current = 0
    previous = None
    for day in sorted(active_dates):
        current = current + 1 if previous and (day - previous).days == 1 else 1
        longest = max(longest, current)
        previous = day
    return longest


def profile_progress(db, user_id, *, year=None, timezone_name="UTC"):
    """Read one user's safe aggregate after the caller enforces visibility.

    No service constructor or schema migration runs on this read path. Accounts
    predating hosted practice work normally even without a practice_runs table.
    All classifications share the public per-problem statistics query.
    """
    zone, today, year = _calendar_options(year, timezone_name)
    catalog = problems()
    entries = {entry["id"]: entry for entry in catalog}
    rows = []
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='practice_runs'").fetchone():
        query, params = full_submission_query(catalog, user_id)
        rows = db.execute(query + "SELECT problem_id,accepted,created_at FROM full_submissions", params)

    attempted, first_accepted, latest_accepted = set(), {}, {}
    daily = defaultdict(lambda: {"submissions": 0, "attempted": set(), "solved": set(), "newly_solved": 0})
    total = accepted_count = 0
    for row in rows:
        moment = _timestamp(row["created_at"])
        if moment is None:
            continue
        local_day = moment.astimezone(zone).date()
        if local_day > today:
            continue
        problem_id = row["problem_id"]
        accepted = row["accepted"] == 1
        attempted.add(problem_id)
        total += 1
        bucket = daily[local_day]
        bucket["submissions"] += 1
        bucket["attempted"].add(problem_id)
        if accepted:
            accepted_count += 1
            bucket["solved"].add(problem_id)
            first_accepted[problem_id] = min(first_accepted.get(problem_id, moment), moment)
            latest_accepted[problem_id] = max(latest_accepted.get(problem_id, moment), moment)
    for moment in first_accepted.values():
        daily[moment.astimezone(zone).date()]["newly_solved"] += 1

    start = date(year, 1, 1)
    end = today if year == today.year else date(year, 12, 31)
    days, selected_active, selected_submissions = [], set(), 0
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        bucket = daily.get(day)
        item = {"date": day.isoformat(), "submissions": 0, "attempted": 0, "solved": 0, "newly_solved": 0}
        if bucket:
            item.update(submissions=bucket["submissions"], attempted=len(bucket["attempted"]),
                        solved=len(bucket["solved"]), newly_solved=bucket["newly_solved"])
            selected_submissions += item["submissions"]
            selected_active.add(day)
        days.append(item)

    current_streak = 0
    cursor = today if today in daily else today - timedelta(days=1)
    while cursor in daily:
        current_streak += 1
        cursor -= timedelta(days=1)

    difficulty = {label: {"difficulty": label, "total": 0, "solved": 0} for label in ("E", "M", "H")}
    for entry in catalog:
        label = entry.get("difficulty", "M")
        # Canonical metadata uses E/M/H; retain any future catalog categories.
        bucket = difficulty.setdefault(label, {"difficulty": label, "total": 0, "solved": 0})
        bucket["total"] += 1
        bucket["solved"] += int(entry["id"] in first_accepted)

    recent = []
    for problem_id, moment in sorted(latest_accepted.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)[:10]:
        entry = entries[problem_id]
        title = entry.get("title") or entry.get("title_en") or f"Problem {entry['number']}"
        recent.append({"problem_id": problem_id, "number": entry["number"], "title": title,
                       "title_en": entry.get("title_en") or title, "difficulty": entry.get("difficulty", "M"),
                       "accepted_at": moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})

    return {
        "visible": True, "timezone": timezone_name, "year": year,
        "years": sorted({today.year, year, *(day.year for day in daily)}, reverse=True),
        "total_problems": len(catalog), "attempted": len(attempted), "solved": len(first_accepted),
        "attempting": len(attempted) - len(first_accepted),
        "total_submissions": total, "accepted_submissions": accepted_count, "failed_submissions": total - accepted_count,
        "by_difficulty": list(difficulty.values()),
        "calendar": {"start": start.isoformat(), "end": end.isoformat(), "submissions": selected_submissions,
                     "active_days": len(selected_active), "current_streak": current_streak,
                     "max_streak": _streak(selected_active), "days": days},
        "recent_accepted": recent,
    }
