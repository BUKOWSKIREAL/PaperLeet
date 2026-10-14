"""Private practice drafts and immutable server-judged run history.

The HTTP adapter supplies the authenticated user ID. Reports enter this service
only from the server judge, never from a client upload or localStorage migration.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from paperleet.catalog import problem, problems


LAYERS = ("T0", "T1", "T2", "T3")
MAX_CODE_BYTES = 128 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024


class PracticeConflict(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _code(value):
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Code must be text no larger than 128 KB")
    return value


def _validate_report(entry, code, report):
    if not isinstance(report, dict) or report.get("problem") != entry["id"]:
        raise ValueError("Judge report does not match the canonical problem")
    try:
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        raise ValueError("Judge report must contain finite JSON data") from None
    if len(encoded.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Judge report exceeds the 2 MB limit")
    result = json.loads(encoded)
    expected = entry.get("tests")
    if (not isinstance(expected, dict) or not expected or any(
            layer not in LAYERS or type(count) is not int or count < 0
            for layer, count in expected.items())):
        raise ValueError("Problem metadata must specify valid required test counts")
    layers = result.get("layers")
    if (not isinstance(layers, list) or not layers or any(layer not in LAYERS for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("Judge report must identify the selected test layers")
    tests = result.get("tests")
    if not isinstance(tests, list) or len(tests) > 10000:
        raise ValueError("Judge report must contain a bounded list of tests")
    seen = set()
    summary = {layer: [0, 0] for layer in LAYERS}
    for row in tests:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip()
                or row.get("layer") not in layers or type(row.get("passed")) is not bool):
            raise ValueError("Judge report contains an invalid test result")
        nodeid = row.get("nodeid", row["id"])
        if not isinstance(nodeid, str) or not nodeid.strip():
            raise ValueError("Judge report contains an invalid test identity")
        key = (row["layer"], nodeid)
        if key in seen:
            raise ValueError("Judge report contains duplicate test cases")
        seen.add(key)
        summary[row["layer"]][0] += int(row["passed"])
        summary[row["layer"]][1] += 1
    provided_summary = result.get("summary")
    if (not isinstance(provided_summary, dict) or any(layer not in LAYERS for layer in provided_summary)
            or any(provided_summary.get(layer, [0, 0]) != pair for layer, pair in summary.items())):
        raise ValueError("Judge report summary does not match its test cases")
    if "error" in result and not isinstance(result["error"], str):
        raise ValueError("Judge error must be text")
    mutation = result.get("mutation")
    if mutation is not None and (not isinstance(mutation, dict) or any(
            not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            for value in mutation.values())):
        raise ValueError("Judge mutation results must be a mapping or null")
    source_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if "source_sha256" in result and result["source_sha256"] != source_hash:
        raise ValueError("Judge report source hash does not match the submitted code")
    result["source_sha256"] = source_hash
    result["summary"] = summary
    result["passed"] = bool(tests) and all(row["passed"] for row in tests) and not result.get("error")
    accepted = bool(result["passed"] and mutation is None and any(expected.values())
                    and all(count == 0 or (layer in layers and summary[layer][0] >= count)
                            for layer, count in expected.items()))
    return result, accepted, source_hash


class PracticeService:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS practice_runs (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    problem_id TEXT NOT NULL, code TEXT NOT NULL, report_json TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, accepted INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS practice_run_history
                    ON practice_runs(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS practice_run_problem_stats
                    ON practice_runs(problem_id, user_id, accepted);
                CREATE TABLE IF NOT EXISTS practice_drafts (
                    user_id TEXT NOT NULL REFERENCES users(id), problem_id TEXT NOT NULL,
                    code TEXT NOT NULL, updated TEXT NOT NULL,
                    latest_run_id TEXT REFERENCES practice_runs(id),
                    accepted_run_id TEXT REFERENCES practice_runs(id),
                    PRIMARY KEY(user_id, problem_id)
                );
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _require_user(self, conn, user_id):
        if not isinstance(user_id, str) or not conn.execute(
                "SELECT id FROM users WHERE id=? AND status='active'", (user_id,)).fetchone():
            raise PermissionError("An active signed-in account is required")

    def _workspace(self, conn, user_id):
        rows = conn.execute("""
            SELECT d.problem_id,d.code,d.updated,r.report_json,a.report_json AS accepted_json
            FROM practice_drafts d
            LEFT JOIN practice_runs r ON r.id=d.latest_run_id AND r.user_id=d.user_id
            LEFT JOIN practice_runs a ON a.id=d.accepted_run_id AND a.user_id=d.user_id
            WHERE d.user_id=? ORDER BY d.problem_id
        """, (user_id,)).fetchall()
        result = {}
        for row in rows:
            saved = {"code": row["code"], "updated": row["updated"]}
            if row["report_json"] is not None:
                saved["report"] = json.loads(row["report_json"])
            if row["accepted_json"] is not None:
                saved["acceptedReport"] = json.loads(row["accepted_json"])
            result[row["problem_id"]] = saved
        return result

    def workspace(self, user_id):
        with self._connect() as conn:
            self._require_user(conn, user_id)
            return self._workspace(conn, user_id)

    def save_draft(self, user_id, problem_id, code, expected_updated=None):
        entry = problem(problem_id)
        code = _code(code)
        if expected_updated is not None and not isinstance(expected_updated, str):
            raise ValueError("Expected draft version must be a timestamp string")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_user(conn, user_id)
            existing = conn.execute("SELECT updated FROM practice_drafts WHERE user_id=? AND problem_id=?", (user_id, entry["id"])).fetchone()
            current_updated = existing["updated"] if existing else ""
            if expected_updated is not None and current_updated != expected_updated:
                raise PracticeConflict("This draft changed on another device. Reload the latest draft before saving.")
            conn.execute("""
                INSERT INTO practice_drafts (user_id,problem_id,code,updated) VALUES (?,?,?,?)
                ON CONFLICT(user_id,problem_id) DO UPDATE SET code=excluded.code,updated=excluded.updated
            """, (user_id, entry["id"], code, _now()))
            return self._workspace(conn, user_id)[entry["id"]]

    def save_run(self, user_id, entry, code, report):
        if not isinstance(entry, dict):
            raise ValueError("A canonical problem is required")
        entry = problem(entry.get("id"))
        code = _code(code)
        report, accepted, source_hash = _validate_report(entry, code, report)
        identifier = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_user(conn, user_id)
            now = _now()
            conn.execute("INSERT INTO practice_runs VALUES (?,?,?,?,?,?,?,?)",
                         (identifier, user_id, entry["id"], code, json.dumps(report, ensure_ascii=False, allow_nan=False), source_hash, int(accepted), now))
            # A completed run must not roll back code edited while it was running.
            conn.execute("""
                INSERT INTO practice_drafts (user_id,problem_id,code,updated,latest_run_id,accepted_run_id)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(user_id,problem_id) DO UPDATE SET latest_run_id=excluded.latest_run_id,
                    accepted_run_id=COALESCE(excluded.accepted_run_id,practice_drafts.accepted_run_id)
            """, (user_id, entry["id"], code, now, identifier, identifier if accepted else None))
        return {"id": identifier, "problem": entry["id"], "created_at": now, "accepted": accepted, "source_sha256": source_hash}

    def list_runs(self, user_id, limit=30, offset=0):
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 100000:
            raise ValueError("Run history requires limit 1–100 and offset 0–100000")
        with self._connect() as conn:
            self._require_user(conn, user_id)
            rows = conn.execute("SELECT * FROM practice_runs WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?", (user_id, limit, offset)).fetchall()
            return [{"id": row["id"], "problem": row["problem_id"], "created_at": row["created_at"],
                     "accepted": bool(row["accepted"]), "source_sha256": row["source_sha256"],
                     "code": row["code"], "report": json.loads(row["report_json"])} for row in rows]

    def problem_stats(self):
        """Aggregate trusted full submissions without loading any source code.

        Existing run history is classified on read, so old rows need no rewrite
        and a single SQLite query gives consistent counts during concurrent runs.
        Required layers come from the canonical catalog, including modules that
        intentionally have no T3 tests. A full run that crashes remains a failed
        submission; a partial-layer check never enters these denominators.
        """
        catalog = problems()
        required = {entry["id"]: [layer for layer, count in entry.get("tests", {}).items()
                                   if type(count) is int and count > 0 and layer in LAYERS]
                    for entry in catalog}
        result = {entry["id"]: {"attempted_users": 0, "solved_users": 0,
                                 "total_submissions": 0, "accepted_submissions": 0,
                                 "failed_submissions": 0, "acceptance_rate": None}
                  for entry in catalog}
        with self._connect() as conn:
            rows = conn.execute("""
                WITH candidates AS (
                    SELECT problem_id,user_id,accepted,
                           CASE WHEN json_valid(report_json) THEN report_json ELSE '{}' END AS report
                    FROM practice_runs
                )
                SELECT r.problem_id, COUNT(DISTINCT r.user_id) AS attempted_users,
                       COUNT(DISTINCT CASE WHEN r.accepted=1 THEN r.user_id END) AS solved_users,
                       COUNT(*) AS total_submissions,
                       SUM(CASE WHEN r.accepted=1 THEN 1 ELSE 0 END) AS accepted_submissions
                FROM json_each(?) AS specification
                JOIN candidates r ON r.problem_id=specification.key
                WHERE json_array_length(specification.value)>0
                  AND json_extract(r.report,'$.problem')=r.problem_id
                  AND json_type(r.report,'$.layers')='array'
                  AND (json_type(r.report,'$.mutation') IS NULL OR json_type(r.report,'$.mutation')='null')
                  AND NOT EXISTS (
                      SELECT 1 FROM json_each(specification.value) AS required_layer
                      WHERE NOT EXISTS (
                          SELECT 1 FROM json_each(r.report,'$.layers') AS selected_layer
                          WHERE selected_layer.type='text' AND selected_layer.value=required_layer.value
                      )
                  )
                GROUP BY r.problem_id
            """, (json.dumps(required),)).fetchall()
        for row in rows:
            total, accepted = row["total_submissions"], row["accepted_submissions"]
            result[row["problem_id"]] = {"attempted_users": row["attempted_users"],
                                         "solved_users": row["solved_users"],
                                         "total_submissions": total,
                                         "accepted_submissions": accepted,
                                         "failed_submissions": total - accepted,
                                         "acceptance_rate": 100.0 * accepted / total}
        return result
