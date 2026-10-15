"""Offline Star History: real aggregate data, privacy, and honest failure states."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
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
    first = [{"user": {"id": i, "login": f"must-not-export-{i}"}, "starred_at": "2026-10-12T23:30:00-04:00"} for i in range(1, 101)]
    second = [first[-1], {"user": {"id": 101, "login": "another-name"}, "starred_at": "2026-10-14T01:00:00Z"}]

    def fetch(url, headers):
        requests.append((url, headers))
        return first if url.endswith("page=1") else second

    monkeypatch.setattr(metrics, "fetch_json", fetch)
    result = metrics.fetch_stars(metrics.REPOSITORY, NOW, "fixture-token")
    assert result["points"] == [{"date": "2026-10-13", "value": 100}, {"date": "2026-10-14", "value": 101}]
    assert len(requests) == 2
    assert requests[0][0] == "https://api.github.com/repos/Hongyang-Du/PaperLeet/stargazers?per_page=100&page=1"
    assert requests[0][1]["Accept"] == "application/vnd.github.star+json"
    assert requests[0][1]["Authorization"] == "Bearer fixture-token"
    assert "must-not-export" not in json.dumps(result)
    assert "fixture-token" not in json.dumps(result)
    assert set(result) == {"repository", "metric", "label", "updated_at", "stale", "points"}
    assert result["repository"] == metrics.REPOSITORY
    assert result["stale"] is False


def test_zero_stars_is_observed_today_not_fabricated_backfill(metrics, monkeypatch):
    monkeypatch.setattr(metrics, "fetch_json", lambda *_: [])
    assert metrics.fetch_stars(metrics.REPOSITORY, NOW)["points"] == [{"date": "2026-10-14", "value": 0}]


def test_first_star_does_not_invent_an_earlier_zero(metrics, monkeypatch):
    monkeypatch.setattr(metrics, "fetch_json", lambda *_: [{"user": {"id": 1}, "starred_at": "2026-10-14T01:00:00Z"}])
    data = metrics.fetch_stars(metrics.REPOSITORY, NOW)
    assert data["points"] == [{"date": "2026-10-14", "value": 1}]
    chart = metrics.render_chart("stars", data)
    assert "First recorded day" in chart
    assert "Observed data series" not in chart


def test_repository_is_fixed_before_any_request_or_output(metrics, monkeypatch, tmp_path):
    def unexpected(*_):
        pytest.fail("A different repository must not be queried")
    monkeypatch.setattr(metrics, "fetch_json", unexpected)
    with pytest.raises(ValueError, match="offline"):
        metrics.fetch_stars("example/other", NOW)
    with pytest.raises(SystemExit) as error:
        metrics.main(["--repo", "example/other", "--output-dir", str(tmp_path / "unexpected")])
    assert error.value.code == 2
    assert not (tmp_path / "unexpected").exists()


def test_repeated_full_page_does_not_loop_or_publish_partial_data(metrics, monkeypatch):
    page = [{"user": {"id": i}, "starred_at": "2026-10-14T01:00:00Z"} for i in range(1, 101)]
    calls = []
    def fetch(*args):
        calls.append(args)
        assert len(calls) <= 2
        return page
    monkeypatch.setattr(metrics, "fetch_json", fetch)
    with pytest.raises(ValueError, match="no progress"):
        metrics.fetch_stars(metrics.REPOSITORY, NOW)
    assert len(calls) == 2


@pytest.mark.parametrize("points", [
    [{"date": "2026-10-13", "value": -1}],
    [{"date": "2026-10-13", "value": True}],
    [{"date": "2026-10-13", "value": 1.5}],
    [{"date": "2026-10-15", "value": 1}],
    [{"date": "2026-10-13", "value": 2}, {"date": "2026-10-14", "value": 1}],
    [{"date": "2026-10-14", "value": 1}, {"date": "2026-10-14", "value": 1}],
])
def test_invalid_aggregates_are_rejected(metrics, points):
    with pytest.raises(ValueError):
        metrics.validate_snapshot(metrics.snapshot("stars", points, NOW), "stars")


def test_snapshot_whitelists_aggregate_fields(metrics):
    raw = metrics.snapshot("stars", [{"date": "2026-10-13", "value": 31, "login": "must-not-export"}], NOW)
    raw.update(label="untrusted label", users=["must-not-export"], token="must-not-export")
    result = metrics.validate_snapshot(raw, "stars")
    assert result["label"] == "GitHub stars"
    assert "must-not-export" not in json.dumps(result)


def test_unavailable_source_preserves_counts_and_original_update_time(metrics, tmp_path, capsys):
    prior = metrics.snapshot("stars", [{"date": "2026-10-13", "value": 31}], NOW)
    destination = tmp_path / "stars.json"
    destination.write_text(json.dumps(prior), encoding="utf-8")

    def fail():
        raise ValueError("fixture-token from some private URL must not be logged")

    metrics.update_metric("stars", fail, tmp_path)
    expected = {**prior, "stale": True}
    assert json.loads(destination.read_text()) == expected
    light = (tmp_path / "stars.svg").read_text()
    assert "Last verified 2026-10-14 15:00 UTC" in light
    assert "Stale snapshot" in light
    assert "31 stars" in light
    assert "fixture-token" not in capsys.readouterr().err
    # A second failure keeps the same verified observation, not a new timestamp.
    metrics.update_metric("stars", fail, tmp_path)
    assert json.loads(destination.read_text()) == expected


def test_failure_after_page_one_preserves_prior_snapshot(metrics, monkeypatch, tmp_path):
    prior = metrics.snapshot("stars", [{"date": "2026-10-13", "value": 200}], NOW)
    (tmp_path / "stars.json").write_text(json.dumps(prior))
    def fetch(url, _headers):
        if url.endswith("page=1"):
            return [{"user": {"id": i}, "starred_at": "2026-10-14T01:00:00Z"} for i in range(1, 101)]
        raise OSError("Unavailable")
    monkeypatch.setattr(metrics, "fetch_json", fetch)
    metrics.update_metric("stars", lambda: metrics.fetch_stars(metrics.REPOSITORY, NOW), tmp_path)
    assert json.loads((tmp_path / "stars.json").read_text()) == {**prior, "stale": True}


@pytest.mark.parametrize("cache", [None, "malformed", {"repository": "another/repo"}])
def test_missing_or_invalid_source_produces_unknown_not_fake_zero(metrics, tmp_path, cache):
    if cache is not None:
        (tmp_path / "stars.json").write_text(json.dumps(cache))
    def fail():
        raise OSError("Unavailable")
    metrics.update_metric("stars", fail, tmp_path)
    data = json.loads((tmp_path / "stars.json").read_text())
    assert data["points"] == []
    assert data["updated_at"] is None
    assert data["stale"] is True
    svg = (tmp_path / "stars.svg").read_text()
    assert "Waiting for a verified data snapshot" in svg
    assert "—" in svg


def test_success_replaces_stale_data_and_honors_unstars(metrics, tmp_path):
    prior = metrics.snapshot("stars", [{"date": "2026-10-13", "value": 31}], NOW)
    prior["stale"] = True
    (tmp_path / "stars.json").write_text(json.dumps(prior))
    fresh = metrics.snapshot("stars", [{"date": "2026-10-13", "value": 30}, {"date": "2026-10-14", "value": 30}], NOW)
    metrics.update_metric("stars", lambda: fresh, tmp_path)
    assert json.loads((tmp_path / "stars.json").read_text()) == fresh
    assert "Stale snapshot" not in (tmp_path / "stars.svg").read_text()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("values", [[], [0], [1], [8, 9, 12]])
def test_charts_are_valid_svg_for_empty_single_point_and_history(metrics, dark, values):
    points = [{"date": f"2026-10-{11 + index}", "value": value} for index, value in enumerate(values)]
    chart = metrics.render_chart("stars", metrics.snapshot("stars", points, NOW), dark=dark)
    root = ElementTree.fromstring(chart)
    assert root.attrib["role"] == "img"
    assert "nan" not in chart.lower()
    assert "Updated 2026-10-14 15:00 UTC" in chart
    assert "2026-10-10" not in chart  # No made-up origin day.
    assert "Hongyang-Du/PaperLeet" in chart
    assert "SIL OPEN FONT LICENSE" in chart
    assert "<use href=\"#pl-hand-" in chart


def test_main_only_generates_stars(metrics, monkeypatch, tmp_path):
    def fetch(repo, now, token):
        assert repo == metrics.REPOSITORY
        return metrics.snapshot("stars", [{"date": now.date().isoformat(), "value": 0}], now)
    monkeypatch.setattr(metrics, "fetch_stars", fetch)
    assert metrics.main(["--output-dir", str(tmp_path)]) == 0
    assert {path.name for path in tmp_path.iterdir()} == {"stars.json", "stars.svg", "stars-dark.svg"}
