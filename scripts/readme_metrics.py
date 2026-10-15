#!/usr/bin/env python3
"""Render the offline repository Star History using only the standard library.

Only aggregate dates/counts are written to disk. GitHub identities and API tokens
are never included in snapshots. An unavailable upstream preserves the last
verified snapshot; its original update time remains visible in the chart.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

try:  # Module imports in checks; direct execution in the daily workflow.
    from scripts.metrics_lettering import lettering_defs, text, text_width
except ModuleNotFoundError:
    from metrics_lettering import lettering_defs, text, text_width


UTC = timezone.utc
REPOSITORY = "Hongyang-Du/PaperLeet"
METRICS = {
    "stars": ("current_stargazers", "GitHub stars", "Current stargazers by star date", "#ee4934", "#ff745f"),
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
    return {"repository": REPOSITORY, "metric": metric, "label": label, "updated_at": iso_now(now) if now else None, "stale": False, "points": points}


def validate_snapshot(payload: object, key: str) -> dict:
    """Whitelist public fields; reject malformed or misleading aggregate history."""
    if not isinstance(payload, dict) or payload.get("metric") != METRICS[key][0]:
        raise ValueError("Unexpected metric")
    if payload.get("repository") != REPOSITORY:
        raise ValueError("Snapshot belongs to a different repository")
    if type(payload.get("stale")) is not bool:
        raise ValueError("Snapshot requires explicit freshness")
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
        if value < previous_value:
            raise ValueError("Cumulative counts must not decrease")
        if updated_at and day > updated_at.date():
            raise ValueError("A point cannot follow its observation time")
        points.append({"date": day.isoformat(), "value": value})
        previous_date, previous_value = day, value
    result = snapshot(key, points, updated_at)
    result["stale"] = payload["stale"]
    return result


def fetch_json(url: str, headers: dict | None = None) -> object:
    request = Request(url, headers={"User-Agent": "PaperLeet-readme-metrics", "Accept": "application/json", **(headers or {})})
    with urlopen(request, timeout=30) as response:
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ValueError("Metric response exceeds size limit")
    return json.loads(body)


def fetch_stars(repo: str, now: datetime, token: str | None = None) -> dict:
    if repo != REPOSITORY:
        raise ValueError("Only the offline PaperLeet repository is supported")
    now = now.astimezone(UTC)
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
        new_count = 0
        for entry in entries:
            # IDs are used only to avoid a double count if pagination moves.
            identity = entry["user"]["id"]
            if type(identity) is not int or identity <= 0:
                raise ValueError("Expected a positive GitHub user ID")
            starred_at = timestamp(entry["starred_at"])
            if starred_at > now:
                raise ValueError("Star date follows its observation time")
            if identity not in seen:
                seen.add(identity)
                counts[starred_at.date()] += 1
                new_count += 1
        if len(entries) < 100:
            break
        if new_count == 0:
            raise ValueError("GitHub pagination made no progress")
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


def ink_line(points: list[tuple[float, float]], color: str, width: float = 2.5, *, rough: bool = True) -> str:
    """Draw a pen stroke; measured series coordinates use the exact polyline."""
    commands = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        if rough:
            # Deterministic subpixel wobble is reserved for decorative strokes.
            length = math.hypot(x1 - x0, y1 - y0)
            nx, ny = (-(y1 - y0) / length, (x1 - x0) / length) if length else (0, 0)
            for step in range(1, max(2, math.ceil(length / 35))):
                fraction = step / max(2, math.ceil(length / 35))
                wobble = math.sin(step * 2.1 + index) * .85
                x, y = x0 + (x1 - x0) * fraction + nx * wobble, y0 + (y1 - y0) * fraction + ny * wobble
                commands.append(f"L{x:.2f},{y:.2f}")
        commands.append(f"L{x1:.2f},{y1:.2f}")
    return f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="{width:g}" stroke-linecap="round" stroke-linejoin="round"/>'


def cat_doodle(x: float, y: float, ink: str, paper: str) -> str:
    """A small smiling cat signature, drawn in the same pen style."""
    return (f'<g transform="translate({x:g} {y:g})" aria-hidden="true">'
            f'<path d="M2 13 L1 1 Q3 -1 10 5 Q16 2 22 5 L30 0 L30 14 '
            f'Q33 29 16 30 Q-1 28 2 13 Z" fill="{ink}" stroke="{ink}" stroke-width="1.4" stroke-linejoin="round"/>'
            f'<path d="M7 15 Q10 11 13 15 M20 15 Q23 11 26 15 M13 21 Q17 25 21 21" '
            f'fill="none" stroke="{paper}" stroke-width="1.8" stroke-linecap="round"/></g>')


def render_chart(key: str, data: dict, *, dark: bool = False, cached: bool = False) -> str:
    """Hand-drawn SVG with outlined lettering, portable to GitHub image embeds."""
    _, label, subtitle, light_color, dark_color = METRICS[key]
    title = "Star History"
    axis_label = "GitHub Stars"
    color = dark_color if dark else light_color
    paper, ink, muted = ("#0d1117", "#f0eee9", "#9faaaf") if dark else ("#ffffff", "#202124", "#757575")
    points = data["points"]
    total = points[-1]["value"] if points else None
    number = f"{total:,}" if total is not None else "—"
    unit = "stars"
    cached = cached or data["stale"]
    updated = timestamp(data["updated_at"]).strftime("%Y-%m-%d %H:%M UTC") if data["updated_at"] else None
    update_label = f"{'Last verified' if cached else 'Updated'} {updated}" if updated else "Awaiting first verified snapshot"
    description = f"{REPOSITORY}. {label}: {number} {unit}. {subtitle}. {update_label}."
    if cached:
        description += " Stale snapshot; the latest request was unavailable."
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
        lettering_defs(),
        f'<rect width="920" height="520" fill="{paper}"/>',
        cat_doodle(460 - text_width(title, 27) / 2 - 43, 10, ink, paper),
        text(460, 34, title, ink, 27, 'text-anchor="middle"'),
    ]
    left, right, top, bottom = 86.0, 866.0, 70.0, 420.0
    # Open, slightly irregular axes and a sketched legend match a notebook plot.
    parts.append(ink_line([(left + 1, top), (left, bottom), (right + 8, bottom + 1)], ink, 2.7))
    parts.append(ink_line([(left - 3, top + 7), (left - 2, bottom - 1), (right + 5, bottom - 1)], ink, .8))
    legend = REPOSITORY
    legend_right = 132 + text_width(legend, 18)
    parts.append(ink_line([(101, 76), (legend_right, 75), (legend_right + 2, 109), (100, 110), (101, 76)], ink, 1.7))
    parts.append(f'<rect x="110" y="87" width="9" height="9" rx="2" fill="{color}"/>')
    parts.append(text(127, 99, legend, ink, 18))
    parts.append(text(25, (top + bottom) / 2, axis_label, ink, 20,
                      f'text-anchor="middle" transform="rotate(-90 25 {(top + bottom) / 2:g})"'))
    parts.append(text((left + right) / 2, 477, "Date (UTC)", ink, 20, 'text-anchor="middle"'))

    if points:
        maximum = max(point["value"] for point in points)
        raw_step = max(maximum / 4, 1)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = next(multiplier * magnitude for multiplier in (1, 2, 5, 10) if multiplier * magnitude >= raw_step)
        y_max = max(step, math.ceil(maximum * 1.08 / step) * step)
        for value in range(0, int(y_max) + 1, int(step)):
            y = bottom - (bottom - top) * value / y_max
            parts.append(ink_line([(left - 3, y), (left + 3, y + .3)], ink, 1.2))
            tick = f"{value / 1_000_000:g}m" if value >= 1_000_000 else f"{value / 1_000:g}k" if value >= 1_000 else str(value)
            parts.append(text(left - 14, y + 6, tick, ink, 18, 'text-anchor="end"'))
        days = [date.fromisoformat(point["date"]) for point in points]
        span = (days[-1] - days[0]).days
        coordinates = [
            (left + (right - left) * (day - days[0]).days / span if span else (left + right) / 2,
             bottom - (bottom - top) * point["value"] / y_max)
            for day, point in zip(days, points)
        ]
        if len(coordinates) > 1:
            # Keep every observed value exact: styling must not invent growth.
            parts.append('<g aria-label="Observed data series">')
            parts.append(ink_line(coordinates, color, 3, rough=False))
            parts.append('</g>')
        x, y = coordinates[-1]
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        if len(coordinates) == 1:
            parts.append(text(x, y - 16, f"{number} {unit}", color, 21, 'text-anchor="middle"'))
            parts.append(text(right - 8, bottom - 28,
                              "First recorded day", muted, 17, 'text-anchor="end"'))
        indices = sorted({0, len(days) // 4, len(days) // 2, 3 * len(days) // 4, len(days) - 1})
        selected = [indices[0]]
        for index in indices[1:]:
            if coordinates[index][0] - coordinates[selected[-1]][0] > 125:
                selected.append(index)
        for index in selected:
            # Short date ticks, with the year stated explicitly in the footer.
            day_label = days[index].strftime("%b %d") if span < 365 else days[index].strftime("%b %Y")
            parts.append(text(coordinates[index][0], 448, day_label, ink, 18, 'text-anchor="middle"'))
    else:
        parts.append(text(476, 239, "Waiting for a verified data snapshot" if data["updated_at"] is None else "No recorded activity", muted, 22, 'text-anchor="middle"'))
        parts.append(text(476, 270, "A curve will appear as data becomes available.", muted, 17, 'text-anchor="middle"'))

    if cached:
        parts.append(text(875, 33, "Stale snapshot", muted, 15, 'text-anchor="end"'))
    parts.extend([
        text(86, 509, "Current stargazers by star date", muted, 14),
        text(875, 509, update_label, muted, 14, 'text-anchor="end"'),
        '</svg>\n',
    ])
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
    data["stale"] = cached
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    write_atomic(destination, serialized)
    for dark in (False, True):
        suffix = "-dark" if dark else ""
        write_atomic(output_dir / f"{key}{suffix}.svg", render_chart(key, data, dark=dark, cached=cached))
    print(f"Rendered {key} ({len(data['points'])} aggregate points{' · cached' if cached else ''}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPOSITORY, choices=[REPOSITORY], help="Offline repository (fixed to match the chart legend)")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/metrics"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    update_metric("stars", lambda: fetch_stars(args.repo, now, os.environ.get("GITHUB_TOKEN")), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
