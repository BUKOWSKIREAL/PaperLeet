#!/usr/bin/env python3
"""Render public aggregate README charts using only the Python standard library.

Only aggregate dates/counts are written to disk. GitHub identities and API tokens
are never included in snapshots. An unavailable upstream preserves the last
verified snapshot; its original update time remains visible in the chart.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen


UTC = timezone.utc
METRICS = {
    "stars": ("current_stargazers", "GitHub stars", "Current stargazers by star date", "#c68a13", "#f3b840"),
    "users": ("created_accounts", "Created accounts", "Cumulative accounts · includes beta invitations", "#07856f", "#40cbb1"),
    "commits": ("weekly_commits", "Commits", "Non-merge commits per UTC week · current week partial", "#7560d7", "#ac9afa"),
}


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return parsed.astimezone(UTC)


def iso_now(now: datetime) -> str:
    return now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def snapshot(key: str, points: list[dict], now: datetime | None) -> dict:
    metric, label, *_ = METRICS[key]
    return {"metric": metric, "label": label, "updated_at": iso_now(now) if now else None, "points": points}


def validate_snapshot(payload: object, key: str) -> dict:
    """Whitelist public fields; reject malformed or misleading aggregate history."""
    if not isinstance(payload, dict) or payload.get("metric") != METRICS[key][0]:
        raise ValueError("Unexpected metric")
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("Missing aggregate points")
    updated = payload.get("updated_at")
    updated_at = timestamp(updated) if isinstance(updated, str) else None
    if raw_points and updated_at is None:
        raise ValueError("Populated metrics require an update time")
    points = []
    previous_date = None
    previous_value = -1
    for point in raw_points:
        if not isinstance(point, dict):
            raise ValueError("Invalid aggregate point")
        day = date.fromisoformat(point["date"])
        value = point.get("value")
        if type(value) is not int or value < 0:
            raise ValueError("Counts must be nonnegative integers")
        if previous_date is not None and day <= previous_date:
            raise ValueError("Dates must be distinct and ascending")
        if key in ("stars", "users") and value < previous_value:
            raise ValueError("Cumulative counts must not decrease")
        if updated_at and day > updated_at.date():
            raise ValueError("A point cannot follow its observation time")
        points.append({"date": day.isoformat(), "value": value})
        previous_date, previous_value = day, value
    return snapshot(key, points, updated_at)


def fetch_json(url: str, headers: dict | None = None) -> object:
    request = Request(url, headers={"User-Agent": "PaperLeet-readme-metrics", "Accept": "application/json", **(headers or {})})
    with urlopen(request, timeout=30) as response:
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ValueError("Metric response exceeds size limit")
    return json.loads(body)


def fetch_stars(repo: str, now: datetime, token: str | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Expected owner/repository")
    headers = {"Accept": "application/vnd.github.star+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    counts: Counter = Counter()
    seen = set()
    page = 1
    while True:
        entries = fetch_json(f"https://api.github.com/repos/{repo}/stargazers?per_page=100&page={page}", headers)
        if not isinstance(entries, list):
            raise ValueError("Expected GitHub stargazer list")
        for entry in entries:
            # IDs are used only to avoid a double count if pagination moves.
            identity = entry["user"]["id"]
            if identity not in seen:
                seen.add(identity)
                counts[timestamp(entry["starred_at"]).date()] += 1
        if len(entries) < 100:
            break
        page += 1
    total = 0
    points = []
    for day in sorted(counts):
        total += counts[day]
        points.append({"date": day.isoformat(), "value": total})
    # Today's count is observed directly, including a legitimate empty repo.
    if not points or points[-1]["date"] < now.date().isoformat():
        points.append({"date": now.date().isoformat(), "value": total})
    return validate_snapshot(snapshot("stars", points, now), "stars")


def weekly_commits(now: datetime, repo_dir: Path) -> dict:
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo_dir, capture_output=True, text=True, check=True)
    if shallow.stdout.strip() == "true":
        raise ValueError("Full git history is required (checkout fetch-depth: 0)")
    history = subprocess.run(
        ["git", "log", "--no-merges", "--format=%ct", "--", ".", ":(exclude)docs/metrics"],
        cwd=repo_dir, capture_output=True, text=True, check=True,
    )
    counts: Counter = Counter()
    for raw in history.stdout.splitlines():
        day = datetime.fromtimestamp(int(raw), UTC).date()
        if day > now.date():
            raise ValueError("Git history contains a future commit")
        counts[day - timedelta(days=day.weekday())] += 1
    points = []
    if counts:
        week = min(counts)
        last_week = now.date() - timedelta(days=now.date().weekday())
        while week <= last_week:
            points.append({"date": week.isoformat(), "value": counts[week]})
            week += timedelta(days=7)
    return snapshot("commits", points, now)


def fetch_users(url: str) -> dict:
    # Intentionally no GitHub or application credential on this public request.
    return validate_snapshot(fetch_json(url), "users")


def text(x: float, y: float, value: object, fill: str, size: int = 12, extra: str = "") -> str:
    return f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-size="{size}" {extra}>{escape(str(value))}</text>'


def render_chart(key: str, data: dict, *, dark: bool = False, cached: bool = False) -> str:
    """Produce accessible static SVG, including honest one-point/empty states."""
    _, title, subtitle, light_color, dark_color = METRICS[key]
    color = dark_color if dark else light_color
    bg, border, foreground, muted, grid = ("#11151f", "#2d3545", "#f1f4fa", "#a1adbf", "#252d3b") if dark else ("#ffffff", "#e4e9f0", "#172033", "#657186", "#edf0f5")
    points = data["points"]
    total = sum(point["value"] for point in points) if key == "commits" else points[-1]["value"] if points else None
    number = f"{total:,}" if total is not None else "—"
    unit = "total commits" if key == "commits" else "accounts" if key == "users" else "stars"
    updated = timestamp(data["updated_at"]).strftime("%Y-%m-%d %H:%M UTC") if data["updated_at"] else None
    update_label = f"{'Last verified' if cached else 'Updated'} {updated}" if updated else "Awaiting first verified snapshot"
    state = "Cached snapshot" if cached and updated else "UTC" if updated else "Not yet available"
    description = f"{title}: {number} {unit}. {subtitle}. {update_label}."
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="340" viewBox="0 0 920 340" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
        f'<defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{color}" stop-opacity=".18"/><stop offset="100%" stop-color="{color}" stop-opacity=".02"/></linearGradient></defs>',
        f'<rect x=".5" y=".5" width="919" height="339" rx="16" fill="{bg}" stroke="{border}"/>',
        '<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        text(30, 35, title, foreground, 17, 'font-weight="650"'),
        text(30, 58, subtitle, muted, 12),
        text(30, 104, number, foreground, 36, 'font-weight="700"'),
        text(34 + len(number) * 23, 102, unit, muted, 13),
        f'<circle cx="{870 - len(state) * 6}" cy="31" r="4" fill="{color}"/>',
        text(890, 35, state, muted, 12, 'text-anchor="end"'),
    ]
    left, right, top, bottom = 65.0, 880.0, 138.0, 265.0
    if points:
        maximum = max(point["value"] for point in points)
        # Integer ticks, with headroom; zero is a measured value only, not history.
        raw_step = max(maximum / 3, 1)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = next(multiplier * magnitude for multiplier in (1, 2, 5, 10) if multiplier * magnitude >= raw_step)
        y_max = max(step * 3, maximum)
        for index in range(4):
            value = y_max * index / 3
            y = bottom - (bottom - top) * index / 3
            parts.append(f'<path d="M{left:g},{y:g}H{right:g}" stroke="{grid}"/>')
            parts.append(text(left - 12, y + 4, f"{int(value):,}", muted, 11, 'text-anchor="end"'))
        days = [date.fromisoformat(point["date"]) for point in points]
        span = (days[-1] - days[0]).days
        coordinates = [
            (left + (right - left) * (day - days[0]).days / span if span else (left + right) / 2,
             bottom - (bottom - top) * point["value"] / y_max)
            for day, point in zip(days, points)
        ]
        if len(coordinates) > 1:
            line = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
            parts.append(f'<path d="{line} L{coordinates[-1][0]:.2f},{bottom:g} L{coordinates[0][0]:.2f},{bottom:g} Z" fill="url(#fill)"/>')
            parts.append(f'<path d="{line}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        x, y = coordinates[-1]
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="{bg}" stroke-width="2"/>')
        if len(coordinates) == 1:
            parts.append(text(right, top + 4, "First recorded week" if key == "commits" else "First recorded day", muted, 11, 'text-anchor="end"'))
        indices = sorted({0, len(days) // 2, len(days) - 1})
        # Avoid nearly overlapping labels when dates are irregularly spaced.
        selected = [indices[0]]
        for index in indices[1:]:
            if coordinates[index][0] - coordinates[selected[-1]][0] > 115:
                selected.append(index)
        for index in selected:
            parts.append(text(coordinates[index][0], 287, days[index].strftime("%b %d, %Y"), muted, 11, 'text-anchor="middle"'))
    else:
        parts.append(text(460, 197, "Waiting for a verified data snapshot" if data["updated_at"] is None else "No recorded activity", muted, 15, 'text-anchor="middle"'))
        parts.append(text(460, 223, "A curve will appear as data becomes available.", muted, 12, 'text-anchor="middle"'))
    parts.extend([f'<path d="M30,305H890" stroke="{grid}"/>', text(30, 326, "PaperLeet", muted, 11), text(890, 326, update_label, muted, 11, 'text-anchor="end"'), '</g></svg>\n'])
    return "\n".join(parts)


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def update_metric(key: str, loader, output_dir: Path) -> None:
    destination = output_dir / f"{key}.json"
    cached = False
    try:
        data = validate_snapshot(loader(), key)
    except Exception as error:
        # Do not print URLs, payloads, tokens, or exception text from upstreams.
        print(f"Warning: {key} unavailable ({type(error).__name__}); retaining last verified snapshot.", file=sys.stderr)
        cached = True
        try:
            data = validate_snapshot(json.loads(destination.read_text(encoding="utf-8")), key)
        except (OSError, ValueError, KeyError, TypeError):
            data = snapshot(key, [], None)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    write_atomic(destination, serialized)
    for dark in (False, True):
        suffix = "-dark" if dark else ""
        write_atomic(output_dir / f"{key}{suffix}.svg", render_chart(key, data, dark=dark, cached=cached))
    print(f"Rendered {key} ({len(data['points'])} aggregate points{' · cached' if cached else ''}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Hongyang-Du/PaperLeet", help="GitHub owner/repository")
    parser.add_argument("--users-url", default="https://paperleet.com/api/public/metrics")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/metrics"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    root = Path(__file__).resolve().parents[1]
    for key, loader in (
        ("stars", lambda: fetch_stars(args.repo, now, os.environ.get("GITHUB_TOKEN"))),
        ("users", lambda: fetch_users(args.users_url)),
        ("commits", lambda: weekly_commits(now, root)),
    ):
        update_metric(key, loader, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
