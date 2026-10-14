"""Public aggregate charts: source semantics, privacy and honest failure states."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from xml.etree import ElementTree

import pytest


@pytest.fixture
def metrics():
    path = Path(__file__).resolve().parents[1] / "scripts/readme_metrics.py"
    spec = importlib.util.spec_from_file_location("paperleet_readme_metrics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)


def test_stars_paginate_deduplicate_and_export_no_github_identities(metrics, monkeypatch):
    requests = []
    first = [{"user": {"id": i, "login": f"private-name-{i}"}, "starred_at": "2026-10-12T23:30:00-04:00"} for i in range(100)]
    second = [first[-1], {"user": {"id": 100, "login": "another-name"}, "starred_at": "2026-10-14T01:00:00Z"}]

    def fetch(url, headers):
        requests.append((url, headers))
        return first if url.endswith("page=1") else second

    monkeypatch.setattr(metrics, "fetch_json", fetch)
    result = metrics.fetch_stars("example/repo", NOW, "secret-token")
    assert result["points"] == [{"date": "2026-10-13", "value": 100}, {"date": "2026-10-14", "value": 101}]
    assert len(requests) == 2
    assert requests[0][1]["Accept"] == "application/vnd.github.star+json"
    assert requests[0][1]["Authorization"] == "Bearer secret-token"
    assert "private-name" not in json.dumps(result)
    assert "secret-token" not in json.dumps(result)
    assert set(result) == {"metric", "label", "updated_at", "points"}


def test_zero_stars_is_observed_today_not_fabricated_backfill(metrics, monkeypatch):
    monkeypatch.setattr(metrics, "fetch_json", lambda *_: [])
    assert metrics.fetch_stars("example/repo", NOW)["points"] == [{"date": "2026-10-14", "value": 0}]


def test_users_request_is_credential_free_and_whitelists_aggregate_fields(metrics, monkeypatch):
    def fetch(url, *args):
        assert url == "https://paperleet.com/api/public/metrics"
        assert not args
        return {
            "metric": "created_accounts", "label": "untrusted display label", "updated_at": "2026-10-14T11:00:00-04:00",
            "points": [{"date": "2026-10-13", "value": 31, "email": "must-not-export@example.org"}],
            "accounts": ["must-not-export"],
        }

    monkeypatch.setattr(metrics, "fetch_json", fetch)
    result = metrics.fetch_users("https://paperleet.com/api/public/metrics")
    assert result == {
        "metric": "created_accounts", "label": "Created accounts", "updated_at": "2026-10-14T15:00:00Z",
        "points": [{"date": "2026-10-13", "value": 31}],
    }


@pytest.mark.parametrize("points", [
    [{"date": "2026-10-13", "value": -1}],
    [{"date": "2026-10-13", "value": True}],
    [{"date": "2026-10-13", "value": 1.5}],
    [{"date": "2026-10-15", "value": 1}],
    [{"date": "2026-10-13", "value": 2}, {"date": "2026-10-14", "value": 1}],
    [{"date": "2026-10-14", "value": 1}, {"date": "2026-10-14", "value": 1}],
])
def test_invalid_user_aggregates_are_rejected(metrics, points):
    with pytest.raises(ValueError):
        metrics.validate_snapshot(metrics.snapshot("users", points, NOW), "users")


def test_unavailable_source_preserves_counts_and_original_update_time(metrics, tmp_path, capsys):
    prior = metrics.snapshot("users", [{"date": "2026-10-13", "value": 31}], NOW)
    destination = tmp_path / "users.json"
    destination.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    original = destination.read_bytes()

    def fail():
        raise ValueError("secret-token from some private URL must not be logged")

    metrics.update_metric("users", fail, tmp_path)
    assert destination.read_bytes() == original
    light = (tmp_path / "users.svg").read_text()
    assert "Last verified 2026-10-14 15:00 UTC" in light
    assert "Cached snapshot" in light
    assert "31" in light
    assert "secret-token" not in capsys.readouterr().err


def test_missing_source_produces_unknown_not_fake_zero(metrics, tmp_path):
    def fail():
        raise OSError("Unavailable")

    metrics.update_metric("users", fail, tmp_path)
    data = json.loads((tmp_path / "users.json").read_text())
    assert data["points"] == []
    assert data["updated_at"] is None
    svg = (tmp_path / "users.svg").read_text()
    assert "Waiting for a verified data snapshot" in svg
    assert "—" in svg


def test_weekly_commits_use_utc_exclude_merges_and_metric_only_updates(metrics, tmp_path):
    def git(*args, stamp=None):
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
               "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
               "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
        if stamp:
            env.update(GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, env=env)

    def commit(path, stamp):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(stamp)
        git("add", path)
        git("commit", "-m", "fixture", stamp=stamp)

    git("init", "-b", "main")
    commit("app.txt", "2026-10-04T23:30:00-04:00")  # Monday UTC.
    commit("docs/metrics/users.svg", "2026-10-12T12:00:00Z")  # No activity feedback loop.
    git("checkout", "-b", "feature")
    commit("feature.txt", "2026-10-20T12:00:00Z")
    git("checkout", "main")
    commit("second.txt", "2026-10-20T12:30:00Z")
    git("merge", "--no-ff", "feature", "-m", "merge", stamp="2026-10-20T13:00:00Z")
    result = metrics.weekly_commits(datetime(2026, 10, 30, tzinfo=timezone.utc), tmp_path)
    assert result["points"] == [
        {"date": "2026-10-05", "value": 1}, {"date": "2026-10-12", "value": 0},
        {"date": "2026-10-19", "value": 2}, {"date": "2026-10-26", "value": 0},
    ]


@pytest.mark.parametrize("key", ["stars", "users", "commits"])
@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("values", [[], [0], [1], [8, 9, 12]])
def test_charts_are_valid_svg_for_empty_single_point_and_history(metrics, key, dark, values):
    points = [{"date": f"2026-10-{11 + index}", "value": value} for index, value in enumerate(values)]
    chart = metrics.render_chart(key, metrics.snapshot(key, points, NOW), dark=dark)
    root = ElementTree.fromstring(chart)
    assert root.attrib["role"] == "img"
    assert "nan" not in chart.lower()
    assert "Updated 2026-10-14 15:00 UTC" in chart
    assert "2026-10-10" not in chart  # No made-up origin day.
