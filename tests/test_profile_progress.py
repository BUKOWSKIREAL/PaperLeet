"""Profile aggregates share trusted submission rules and enforce visibility."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3

import pytest
import yaml

from paperleet.accounts import AccountService
from paperleet.catalog import problem
from paperleet.community import CommunityService
from paperleet.practice import PracticeService
from paperleet import practice as practice_module, practice_progress


@pytest.fixture
def progress(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "local")
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    for number, difficulty in enumerate(("E", "M", "H"), 1):
        folder = tmp_path / "problems" / "demo" / f"core{number}"
        folder.mkdir(parents=True)
        (folder / "problem.yaml").write_text(yaml.safe_dump({
            "id": f"demo/core{number}", "number": number, "title": f"题目{number}",
            "title_en": f"Problem {number}", "difficulty": difficulty,
            "tests": {"T0": 1, "T1": 1, "T2": 1, "T3": 0},
        }))
    institutions = tmp_path / "institutions.yaml"
    institutions.write_text("institutions:\n  - id: institute\n    name: Research University\n    domains: [research.edu]\n")

    class Provider:
        available = True

        def verify_code(self, email, code):
            return {"id": email, "email": email, "email_confirmed_at": "2026-01-01T00:00:00Z"}

    accounts = AccountService(tmp_path / "community.sqlite3", institutions, Provider())
    _, alice = accounts.verify_code("alice@research.edu", "123456")
    _, bob = accounts.verify_code("bob@research.edu", "123456")
    accounts.update_profile(alice["id"], {"username": "alice"})
    accounts.update_profile(bob["id"], {"username": "bob"})
    community = CommunityService(accounts.db_path, accounts.institutions)
    service = PracticeService(accounts.db_path)
    monkeypatch.setattr(practice_progress, "_now", lambda: datetime(2026, 1, 3, 12, tzinfo=timezone.utc))

    def submit(number=1, when="2026-01-02T10:00:00Z", *, accepted=True, user=alice, layers=None, mutation=None):
        entry = problem(str(number))
        selected = layers or ["T0", "T1", "T2"]
        tests = [{"id": layer, "layer": layer, "passed": accepted} for layer in selected if layer != "T3"]
        candidate = {"problem": entry["id"], "layers": selected, "tests": tests,
                     "summary": {layer: [int(accepted), 1] for layer in selected if layer != "T3"}}
        if mutation is not None:
            candidate["mutation"] = mutation
        monkeypatch.setattr(practice_module, "_now", lambda: when)
        return service.save_run(user["id"], entry, "private learner source", candidate)

    return community, service, accounts, alice, bob, submit


def test_profile_counts_unique_problems_and_daily_work_across_years(progress):
    community, service, _, alice, bob, submit = progress
    submit(1, "2025-12-31T10:00:00Z")
    submit(1, "2026-01-01T10:00:00Z")
    submit(1, "2026-01-01T11:00:00Z")
    submit(1, "2026-01-02T12:00:00Z", accepted=False)
    submit(2, "2026-01-02T12:00:00Z", accepted=False)
    submit(2, "2026-01-02T13:00:00Z")
    submit(3, "2026-01-03T10:00:00Z", accepted=False)
    submit(3, user=bob)
    service.save_draft(alice["id"], "3", "Draft does not count")
    data = community.practice_progress("alice")
    assert {key: data[key] for key in ("attempted", "solved", "attempting", "total_submissions", "accepted_submissions", "failed_submissions")} == {
        "attempted": 3, "solved": 2, "attempting": 1, "total_submissions": 7, "accepted_submissions": 4, "failed_submissions": 3}
    assert data["total_problems"] == 3 and data["years"] == [2026, 2025]
    assert data["by_difficulty"] == [
        {"difficulty": "E", "total": 1, "solved": 1}, {"difficulty": "M", "total": 1, "solved": 1},
        {"difficulty": "H", "total": 1, "solved": 0}]
    calendar = data["calendar"]
    assert calendar["submissions"] == 6 and calendar["active_days"] == 3
    assert calendar["current_streak"] == 4 and calendar["max_streak"] == 3
    assert calendar["days"] == [
        {"date": "2026-01-01", "submissions": 2, "attempted": 1, "solved": 1, "newly_solved": 0},
        {"date": "2026-01-02", "submissions": 3, "attempted": 2, "solved": 1, "newly_solved": 1},
        {"date": "2026-01-03", "submissions": 1, "attempted": 1, "solved": 0, "newly_solved": 0}]
    recent = data["recent_accepted"]
    assert [item["number"] for item in recent] == [2, 1]
    assert recent[1]["accepted_at"] == "2026-01-01T11:00:00Z"
    assert recent[0]["title"] == "题目2" and recent[0]["title_en"] == "Problem 2"
    old = CommunityService(community.db_path, community.institutions).practice_progress("alice", year="2025")
    assert old["solved"] == 2 and len(old["calendar"]["days"]) == 365
    assert old["calendar"]["active_days"] == old["calendar"]["max_streak"] == 1
    assert old["calendar"]["end"] == "2025-12-31"
    assert old["calendar"]["days"][-1]["newly_solved"] == 1
    assert community.practice_progress("bob")["solved"] == 1
    payload = json.dumps(data)
    for private in ("private learner source", "report", "source_sha256", "research.edu", alice["id"], bob["id"]):
        assert private not in payload


def test_profile_visibility_is_enforced_on_server_and_owner_still_sees_progress(progress):
    community, _, accounts, alice, bob, submit = progress
    submit()
    accounts.update_profile(alice["id"], {"show_progress": False})
    assert community.practice_progress("alice") == {"visible": False}
    assert community.practice_progress("alice", viewer_id=bob["id"]) == {"visible": False}
    assert community.practice_progress("alice", viewer_id=alice["id"])["solved"] == 1
    accounts.update_profile(alice["id"], {"show_progress": True})
    assert community.practice_progress("alice")["solved"] == 1
    with pytest.raises(KeyError):
        community.practice_progress("missing")


def test_partial_mutation_and_malformed_runs_never_count_but_full_runtime_failure_does(progress):
    community, service, _, alice, _, submit = progress
    submit(layers=["T0"])
    submit(mutation={})
    submit(mutation={"m1": ["T0"]})
    for index, invalid in enumerate(("not JSON", "{}", '{"problem":"demo/core1","layers":["T0","T1","T2"]}')):
        with sqlite3.connect(service.db_path) as db:
            db.execute("INSERT INTO practice_runs VALUES(?,?,?,?,?,?,?,?)",
                       (f"invalid-{index}", alice["id"], "demo/core1", "private", invalid, "hash", 1, "2026-01-01T12:00:00Z"))
    crashed = {"problem": "demo/core1", "layers": ["T0", "T1", "T2"], "tests": [],
               "summary": {}, "error": "RuntimeError before tests"}
    service.save_run(alice["id"], problem("1"), "broken", crashed)
    data = community.practice_progress("alice")
    assert data["attempted"] == data["failed_submissions"] == data["total_submissions"] == 1
    assert data["solved"] == 0 and data["recent_accepted"] == []
    assert service.problem_stats()["demo/core1"]["total_submissions"] == 1


def test_timezone_midnight_and_leap_year_calendar(progress, monkeypatch):
    community, _, _, _, _, submit = progress
    monkeypatch.setattr(practice_progress, "_now", lambda: datetime(2024, 3, 1, 0, 30, tzinfo=timezone.utc))
    submit(1, "2024-02-29T00:30:00Z")
    submit(1, "2024-02-29T23:59:00Z")
    submit(2, "2024-03-01T00:10:00Z")
    ny = community.practice_progress("alice", timezone_name="America/New_York")
    assert ny["calendar"]["end"] == "2024-02-29" and len(ny["calendar"]["days"]) == 60
    assert ny["calendar"]["days"][-2]["submissions"] == 1
    assert ny["calendar"]["days"][-1] == {
        "date": "2024-02-29", "submissions": 2, "attempted": 2, "solved": 2, "newly_solved": 1}
    assert ny["calendar"]["current_streak"] == 2
    utc = community.practice_progress("alice")
    assert utc["calendar"]["end"] == "2024-03-01" and len(utc["calendar"]["days"]) == 61
    assert utc["calendar"]["days"][-1]["submissions"] == 1
    monkeypatch.setattr(practice_progress, "_now", lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    historic = community.practice_progress("alice", year=2024)
    assert len(historic["calendar"]["days"]) == 366


def test_current_streak_can_end_yesterday_but_expires_after_a_gap(progress):
    community, _, _, _, _, submit = progress
    submit(1, "2026-01-01T10:00:00Z")
    assert community.practice_progress("alice")["calendar"]["current_streak"] == 0
    submit(2, "2026-01-02T10:00:00Z")
    assert community.practice_progress("alice")["calendar"]["current_streak"] == 2


@pytest.mark.parametrize("dates,now,expected", [
    (["2026-03-08T04:30:00Z", "2026-03-08T07:30:00Z", "2026-03-09T04:30:00Z"],
     datetime(2026, 3, 9, 12, tzinfo=timezone.utc), [("2026-03-07", 1), ("2026-03-08", 1), ("2026-03-09", 1)]),
    (["2026-11-01T05:30:00Z", "2026-11-01T06:30:00Z", "2026-11-02T05:30:00Z"],
     datetime(2026, 11, 2, 12, tzinfo=timezone.utc), [("2026-11-01", 2), ("2026-11-02", 1)]),
])
def test_local_days_follow_dst_not_fixed_utc_offsets(progress, monkeypatch, dates, now, expected):
    community, _, _, _, _, submit = progress
    monkeypatch.setattr(practice_progress, "_now", lambda: now)
    for timestamp in dates:
        submit(1, timestamp)
    calendar = community.practice_progress("alice", timezone_name="America/New_York")["calendar"]
    assert [(day["date"], day["submissions"]) for day in calendar["days"] if day["submissions"]] == expected
    assert calendar["current_streak"] == len(expected)


def test_timezone_can_move_a_submission_and_current_year_to_previous_year(progress, monkeypatch):
    community, _, _, _, _, submit = progress
    monkeypatch.setattr(practice_progress, "_now", lambda: datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))
    submit(1, "2026-01-01T00:10:00Z")
    data = community.practice_progress("alice", timezone_name="America/New_York")
    assert data["year"] == 2025 and data["years"] == [2025]
    assert data["calendar"]["days"][-1]["date"] == "2025-12-31"
    assert data["calendar"]["days"][-1]["newly_solved"] == 1


def test_recent_acceptances_are_bounded_unique_and_have_safe_metadata_fallbacks(progress):
    community, _, _, _, _, submit = progress
    directory = problem("1")["directory"].parent
    for number in range(4, 15):
        folder = directory / f"core{number}"
        folder.mkdir()
        (folder / "problem.yaml").write_text(yaml.safe_dump({
            "id": f"demo/core{number}", "number": number, "difficulty": "M",
            "tests": {"T0": 1, "T1": 1, "T2": 1},
        }))
    for number in range(1, 15):
        submit(number, f"2026-01-02T10:{number:02d}:00Z")
    submit(14, "2026-01-02T11:00:00Z")
    data = community.practice_progress("alice")
    assert len(data["recent_accepted"]) == 10
    assert [item["number"] for item in data["recent_accepted"]] == list(range(14, 4, -1))
    assert data["recent_accepted"][0]["title_en"] == data["recent_accepted"][0]["title"] == "Problem 14"
    assert data["solved"] == 14 and data["total_submissions"] == 15


@pytest.mark.parametrize("options", [
    {"year": True}, {"year": "2026 OR 1=1"}, {"year": "2027"}, {"year": 1969},
    {"year": "２026"}, {"timezone_name": "../../etc/passwd"}, {"timezone_name": "No/Such_Zone"},
    {"timezone_name": ""}, {"timezone_name": None},
])
def test_calendar_options_validate_before_querying_runs(progress, options):
    community, _, _, _, _, _ = progress
    with pytest.raises(ValueError):
        community.practice_progress("alice", **options)


def test_empty_legacy_account_database_does_not_need_schema_write(progress):
    community, service, _, _, _, _ = progress
    with sqlite3.connect(service.db_path) as db:
        db.execute("DROP TABLE practice_drafts")
        db.execute("DROP TABLE practice_runs")
    data = community.practice_progress("alice")
    assert data["solved"] == data["attempted"] == data["total_submissions"] == 0
    assert data["calendar"]["active_days"] == data["calendar"]["max_streak"] == 0
    with sqlite3.connect(service.db_path) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='practice_runs'").fetchone()


def test_profile_aggregate_never_loads_code_or_private_account_tables(progress, monkeypatch):
    community, _, _, _, _, submit = progress
    submit()
    original = community.connection

    @contextmanager
    def guarded():
        with original() as db:
            def authorize(action, table, column, database, context):
                if action == sqlite3.SQLITE_READ and ((table == "practice_runs" and column == "code")
                        or table in ("accounts_private", "account_sessions", "test_accounts")):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
            yield db

    monkeypatch.setattr(community, "connection", guarded)
    assert community.practice_progress("alice")["solved"] == 1
