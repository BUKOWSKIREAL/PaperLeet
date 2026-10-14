import copy
import hashlib
import json
import sqlite3

import pytest
import yaml

from paperleet.accounts import AccountService
from paperleet.catalog import problem
from paperleet.practice import MAX_CODE_BYTES, PracticeConflict, PracticeService


@pytest.fixture
def practice(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "local")
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    folder = tmp_path / "problems" / "demo" / "core"
    folder.mkdir(parents=True)
    (folder / "problem.yaml").write_text(yaml.safe_dump({
        "id": "demo/core", "number": 1, "short_id": "X1",
        "tests": {"T0": 1, "T1": 1, "T2": 1, "T3": 0},
    }))
    institutions = tmp_path / "institutions.yaml"
    institutions.write_text("institutions:\n  - id: institute\n    name: Research University\n    domains: [research.edu]\n")

    class Provider:
        available = True

        def verify_code(self, email, code):
            return {"id": email, "email": email, "email_confirmed_at": "2026-01-01T00:00:00Z"}

    database = tmp_path / "community.sqlite3"
    accounts = AccountService(database, institutions, Provider())
    _, alice = accounts.verify_code("alice@research.edu", "123456")
    _, bob = accounts.verify_code("bob@research.edu", "123456")
    return PracticeService(database), alice["id"], bob["id"], problem("1")


def report(layers=None, failed=False):
    selected = layers or ["T0", "T1", "T2", "T3"]
    rows = [{"id": "test_" + layer, "nodeid": "tests/test_core.py::test_" + layer,
             "layer": layer, "passed": not failed} for layer in selected if layer != "T3"]
    return {"problem": "demo/core", "layers": selected, "tests": rows,
            "summary": {layer: [sum(row["passed"] for row in rows if row["layer"] == layer),
                                sum(row["layer"] == layer for row in rows)]
                        for layer in ("T0", "T1", "T2", "T3")}, "passed": not failed}


def test_private_drafts_and_runs_survive_restart_without_cross_user_data(practice):
    service, alice, bob, entry = practice
    alice_draft = service.save_draft(alice, "1", "alice_code", expected_updated="")
    service.save_draft(bob, "X1", "bob_code")
    run = service.save_run(alice, entry, "alice_code", report())
    restarted = PracticeService(service.db_path)
    assert restarted.workspace(alice)["demo/core"]["code"] == "alice_code"
    assert restarted.workspace(bob)["demo/core"] == {"code": "bob_code", "updated": restarted.workspace(bob)["demo/core"]["updated"]}
    assert restarted.workspace(alice)["demo/core"]["updated"] == alice_draft["updated"]
    assert restarted.list_runs(alice)[0]["id"] == run["id"]
    assert restarted.list_runs(bob) == []
    assert "acceptedReport" not in restarted.workspace(bob)["demo/core"]


def test_accepted_result_survives_later_failure_and_source_history_is_immutable(practice):
    service, alice, _, entry = practice
    passing = report()
    first = service.save_run(alice, entry, "correct_code", passing)
    passing["tests"][0]["passed"] = False
    second = service.save_run(alice, entry, "broken_code", report(failed=True))
    workspace = service.workspace(alice)[entry["id"]]
    assert first["accepted"] is True and second["accepted"] is False
    assert workspace["report"]["passed"] is False
    assert workspace["acceptedReport"]["passed"] is True
    history = service.list_runs(alice)
    assert [item["code"] for item in history] == ["broken_code", "correct_code"]
    assert history[1]["report"]["tests"][0]["passed"] is True
    assert history[1]["source_sha256"] == hashlib.sha256(b"correct_code").hexdigest()


def test_old_run_cannot_overwrite_a_newer_draft_or_its_revision(practice):
    service, alice, _, entry = practice
    first = service.save_draft(alice, entry["id"], "old_source")
    updated = service.save_draft(alice, entry["id"], "new_source", expected_updated=first["updated"])
    service.save_run(alice, entry, "old_source", report())
    workspace = service.workspace(alice)[entry["id"]]
    assert workspace["code"] == "new_source"
    assert workspace["updated"] == updated["updated"]
    assert workspace["report"]["source_sha256"] == hashlib.sha256(b"old_source").hexdigest()
    assert service.list_runs(alice)[0]["code"] == "old_source"


def test_optimistic_draft_save_rejects_stale_and_concurrent_first_writes(practice):
    service, alice, _, entry = practice
    initial = service.save_draft(alice, entry["id"], "one", expected_updated="")
    with pytest.raises(PracticeConflict):
        service.save_draft(alice, entry["id"], "stale first edit", expected_updated="")
    latest = service.save_draft(alice, entry["id"], "two", expected_updated=initial["updated"])
    with pytest.raises(PracticeConflict):
        service.save_draft(alice, entry["id"], "stale edit", expected_updated=initial["updated"])
    assert service.workspace(alice)[entry["id"]] == latest


def test_acceptance_requires_complete_canonical_suite_not_report_assertions(practice):
    service, alice, _, entry = practice
    partial = report(layers=["T0"])
    partial["passed"] = True
    assert service.save_run(alice, {**entry, "tests": {"T0": 1}}, "code", partial)["accepted"] is False
    insufficient = report(layers=["T0"])
    insufficient["layers"] = ["T0", "T1", "T2", "T3"]
    assert service.save_run(alice, entry, "code", insufficient)["accepted"] is False
    assert "acceptedReport" not in service.workspace(alice)[entry["id"]]
    full = report(layers=["T0", "T1", "T2"])
    full["passed"] = False  # Server derives verdict from cases and required counts.
    assert service.save_run(alice, entry, "code", full)["accepted"] is True


@pytest.mark.parametrize("field,value", [("error", "Judge crashed"), ("mutation", {}), ("mutation", {"mutant": ["test_T0"]})])
def test_incomplete_or_mutation_reports_do_not_mark_completed(practice, field, value):
    service, alice, _, entry = practice
    candidate = report()
    candidate[field] = value
    assert service.save_run(alice, entry, "code", candidate)["accepted"] is False
    assert "acceptedReport" not in service.workspace(alice)[entry["id"]]


def test_zero_completed_tests_is_recorded_but_never_accepted(practice):
    service, alice, _, entry = practice
    candidate = report(layers=["T3"])
    candidate["error"] = "The judge timed out before collecting a result"
    run = service.save_run(alice, entry, "code", candidate)
    assert run["accepted"] is False
    assert len(service.list_runs(alice)) == 1


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(problem="other/core"),
    lambda value: value.update(source_sha256="0" * 64),
    lambda value: value.update(layers=["T0"]),
    lambda value: value.pop("layers"),
    lambda value: value["summary"].update(T0=[100, 100]),
    lambda value: value["tests"].append(copy.deepcopy(value["tests"][0])),
    lambda value: value["tests"][0].update(passed=1),
    lambda value: value["tests"][0].update(artifacts={"loss": float("nan")}),
])
def test_malformed_or_mismatched_reports_are_not_persisted(practice, mutation):
    service, alice, _, entry = practice
    candidate = report()
    mutation(candidate)
    with pytest.raises(ValueError):
        service.save_run(alice, entry, "code", candidate)
    assert service.list_runs(alice) == []
    assert service.workspace(alice) == {}


def test_source_hash_is_bound_to_the_saved_snapshot(practice):
    service, alice, _, entry = practice
    candidate = report()
    candidate["source_sha256"] = hashlib.sha256(b"correct").hexdigest()
    run = service.save_run(alice, entry, "correct", candidate)
    assert run["source_sha256"] == candidate["source_sha256"]
    with pytest.raises(ValueError, match="source hash"):
        service.save_run(alice, entry, "different", candidate)


def test_inputs_are_bounded_and_unknown_or_inactive_users_are_denied(practice):
    service, alice, _, entry = practice
    for action in (lambda: service.workspace("unknown"),
                   lambda: service.list_runs("unknown"),
                   lambda: service.save_draft("unknown", entry["id"], "code"),
                   lambda: service.save_run("unknown", entry, "code", report())):
        with pytest.raises(PermissionError):
            action()
    with pytest.raises(ValueError):
        service.save_draft(alice, "../../etc/passwd", "code")
    with pytest.raises(ValueError):
        service.save_draft(alice, entry["id"], "a" * (MAX_CODE_BYTES + 1))
    with pytest.raises(ValueError):
        service.save_draft(alice, entry["id"], "文" * (MAX_CODE_BYTES // 2))
    with pytest.raises(ValueError):
        service.list_runs(alice, limit=101)
    with pytest.raises(ValueError):
        service.list_runs(alice, offset=-1)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE users SET status='suspended' WHERE id=?", (alice,))
    with pytest.raises(PermissionError):
        service.save_draft(alice, entry["id"], "code")
    with pytest.raises(PermissionError):
        service.workspace(alice)


def test_problem_stats_count_distinct_people_and_every_full_submission(practice):
    service, alice, bob, entry = practice
    empty = {"attempted_users": 0, "solved_users": 0, "total_submissions": 0,
             "accepted_submissions": 0, "failed_submissions": 0, "acceptance_rate": None}
    assert service.problem_stats() == {entry["id"]: empty}
    service.save_draft(bob, entry["id"], "drafts do not count")
    for _ in range(2):
        service.save_run(alice, entry, "correct", report())
    service.save_run(alice, entry, "later failure", report(failed=True))
    service.save_run(bob, entry, "wrong", report(failed=True))
    assert service.problem_stats()[entry["id"]] == {
        "attempted_users": 2, "solved_users": 1, "total_submissions": 4,
        "accepted_submissions": 2, "failed_submissions": 2, "acceptance_rate": 50.0,
    }
    # T3 is not required for this module; selected required layers are a full run.
    service.save_run(bob, entry, "correct", report(layers=["T0", "T1", "T2"]))
    assert service.problem_stats()[entry["id"]] == {
        "attempted_users": 2, "solved_users": 2, "total_submissions": 5,
        "accepted_submissions": 3, "failed_submissions": 2, "acceptance_rate": 60.0,
    }


def test_problem_stats_exclude_partial_checks_and_mutations_but_count_full_runtime_failures(practice):
    service, alice, bob, entry = practice
    service.save_run(alice, entry, "partial success", report(layers=["T0"]))
    service.save_run(alice, entry, "partial failure", report(layers=["T0"], failed=True))
    for mutation in ({}, {"mutant": ["test_T0"]}):
        candidate = report()
        candidate["mutation"] = mutation
        service.save_run(alice, entry, "mutation run", candidate)
    assert service.problem_stats()[entry["id"]]["total_submissions"] == 0
    failed = report(layers=["T3"])
    failed.update(layers=["T0", "T1", "T2", "T3"], error="RuntimeError before the first test completed")
    service.save_run(bob, entry, "crashing source", failed)
    assert service.problem_stats()[entry["id"]] == {
        "attempted_users": 1, "solved_users": 0, "total_submissions": 1,
        "accepted_submissions": 0, "failed_submissions": 1, "acceptance_rate": 0.0,
    }


def test_problem_stats_include_historical_records_without_rewriting_or_reading_private_code(practice, monkeypatch):
    from contextlib import contextmanager
    service, alice, bob, entry = practice
    full, partial = report(), report(layers=["T0"])
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("DROP INDEX practice_run_problem_stats")
        history = [("historical-full", alice, entry["id"], "private historical source", json.dumps(full), "old-hash", 1, "2026-01-01"),
                   ("historical-partial", bob, entry["id"], "private partial source", json.dumps(partial), "old-hash", 0, "2026-01-02"),
                   ("malformed-legacy", alice, entry["id"], "unread source", "not json", "old-hash", 0, "2026-01-03"),
                   ("legacy-mutation", bob, entry["id"], "private mutant source", json.dumps({**full, "mutation": {}}), "old-hash", 1, "2026-01-04"),
                   ("invalid-mutation", bob, entry["id"], "private source", json.dumps({**full, "mutation": False}), "old-hash", 1, "2026-01-05")]
        conn.executemany("INSERT INTO practice_runs VALUES (?,?,?,?,?,?,?,?)", history)
    restarted = PracticeService(service.db_path)
    original_connection = restarted._connect

    @contextmanager
    def privacy_guard():
        with original_connection() as conn:
            def authorize(action, table, column, database, context):
                if action == sqlite3.SQLITE_READ and ((table == "practice_runs" and column == "code")
                                                     or table in ("accounts_private", "account_sessions", "test_accounts")):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            conn.set_authorizer(authorize)
            yield conn

    monkeypatch.setattr(restarted, "_connect", privacy_guard)
    assert restarted.problem_stats()[entry["id"]] == {
        "attempted_users": 1, "solved_users": 1, "total_submissions": 1,
        "accepted_submissions": 1, "failed_submissions": 0, "acceptance_rate": 100.0,
    }
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT * FROM practice_runs ORDER BY created_at").fetchall() == history
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='practice_run_problem_stats'").fetchone()


def test_concurrent_full_submissions_have_consistent_aggregate_counts(practice):
    from concurrent.futures import ThreadPoolExecutor
    service, alice, bob, entry = practice

    def submit(index):
        return service.save_run(alice if index % 2 == 0 else bob, entry,
                                f"source {index}", report(failed=index % 4 >= 2))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(submit, index) for index in range(12)]
        snapshot = service.problem_stats()[entry["id"]]
        assert snapshot["total_submissions"] == snapshot["accepted_submissions"] + snapshot["failed_submissions"]
        assert snapshot["solved_users"] <= snapshot["attempted_users"] <= 2
        for future in futures:
            future.result()
    assert PracticeService(service.db_path).problem_stats()[entry["id"]] == {
        "attempted_users": 2, "solved_users": 2, "total_submissions": 12,
        "accepted_submissions": 6, "failed_submissions": 6, "acceptance_rate": 50.0,
    }
