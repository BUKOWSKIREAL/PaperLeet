"""Local HTTP contracts with a fake judge; these tests never execute learner code."""
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from paperleet import server
from paperleet.catalog import problem
from paperleet.judge.jobs_api import execute
from paperleet.judge.queue import JudgeQueue
from paperleet.workspace import save_code


@pytest.fixture
def local_api(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    monkeypatch.setenv("PAPERLEET_DATA_DIR", str(tmp_path / ".paperleet"))
    folder = tmp_path / "problems/demo/add"
    folder.mkdir(parents=True)
    (folder / "problem.yaml").write_text("id: demo/add\nnumber: 1\ntitle: Add\ndifficulty: E\ntests: {T0: 1}\n")
    (folder / "starter.py").write_text("def add(a,b): pass\n")
    calls = []

    def judge(identifier, source, layers=None, **kwargs):
        code = Path(source).read_text()
        calls.append(code)
        return {"problem": identifier, "passed": True, "layers": layers,
                "summary": {"T0": [1, 1]}, "tests": [],
                "source_sha256": hashlib.sha256(code.encode()).hexdigest()}

    monkeypatch.setattr(server, "run_problem", judge)
    bridge = server.BoundedHTTPServer(("127.0.0.1", 0), server.JudgeHandler)
    thread = threading.Thread(target=bridge.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()

    def request(method, path, data=None, headers=None, raw=None):
        connection = HTTPConnection("127.0.0.1", bridge.server_port, timeout=3)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        body = raw if raw is not None else json.dumps(data).encode() if data is not None else None
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            payload = json.loads(response.read())
            return response.status, payload, dict(response.getheaders())
        finally:
            connection.close()

    yield SimpleNamespace(root=tmp_path, server=bridge, request=request, calls=calls, judge=judge)
    bridge.shutdown()
    thread.join(2)
    bridge.server_close()


def test_health_catalog_and_problem_are_local_without_auth(local_api):
    status, health, headers = local_api.request("GET", "/api/health")
    assert status == 200 and health["mode"] == "local" and health["edition"] == "offline"
    assert health["auth_required"] is False and health["judge"]["ready"] is True
    assert headers["Cache-Control"] == "no-store"
    status, catalog, _ = local_api.request("GET", "/api/catalog")
    assert status == 200 and catalog["problems"][0]["id"] == "demo/add"
    assert "browser_judge" not in catalog["problems"][0]
    status, entry, _ = local_api.request("GET", "/api/problems/demo%2Fadd")
    assert status == 200 and entry["starter"] == "def add(a,b): pass\n"
    assert not local_api.calls


def test_removed_online_endpoints_return_404(local_api):
    for path in ("/api/me", "/api/auth/github", "/api/accounts", "/api/community",
                 "/api/community/posts", "/api/submissions", "/api/submissions/capabilities",
                 "/api/presence", "/api/practice/draft", "/api/practice/stats", "/api/avatars/123/456.jpg"):
        for method in ("GET", "POST"):
            assert local_api.request(method, path, {})[0] == 404, (method, path)
    assert not local_api.calls


def test_other_origins_and_public_hostnames_are_rejected(local_api):
    for headers in ({"Origin": "https://evil.example"}, {"Host": "paperleet.com"},
                    {"Origin": "null"}, {"Host": "["}, {"Origin": "http://evil@localhost"}):
        assert local_api.request("GET", "/api/health", headers=headers)[0] == 403
        assert local_api.request("POST", "/api/test", {}, headers=headers)[0] == 403
    assert local_api.request("GET", "/api/health", headers={"Origin": "http://localhost:5173"})[0] == 200


def test_workspace_files_are_explicit_local_persistence(local_api):
    get = lambda: local_api.request("GET", "/api/workspace?problem=demo%2Fadd")
    assert get()[1] == {"problem": "demo/add", "code": None, "report": None}
    status, payload, _ = local_api.request("POST", "/api/workspace", {"problem": "1", "code": "saved draft"})
    assert status == 200 and payload["ok"] is True
    assert (local_api.root / "workspace/demo/add.py").read_text() == "saved draft"
    assert get()[1]["code"] == "saved draft"
    local_api.request("POST", "/api/workspace", {"problem": "1", "code": ""})
    assert get()[1]["code"] == ""
    assert local_api.request("GET", "/api/workspace?problem=../../etc/passwd")[0] == 400
    assert local_api.request("GET", "/api/workspace")[0] == 400
    assert local_api.request("POST", "/api/workspace", {"problem": "../outside", "code": "x"})[0] == 400
    assert not local_api.calls


def test_test_endpoint_saves_report_and_preserves_ide_edits_during_run(local_api, monkeypatch):
    path = local_api.root / "workspace/demo/add.py"

    def judge(*args, **kwargs):
        path.write_text("new IDE edit")
        return local_api.judge(*args, **kwargs)

    monkeypatch.setattr(server, "run_problem", judge)
    status, report, _ = local_api.request("POST", "/api/test", {"problem": "1", "code": "submitted", "layers": ["T0"]})
    assert status == 200 and report["passed"]
    assert local_api.calls == ["submitted"] and path.read_text() == "new IDE edit"
    assert report["impl"] == str(path)
    assert local_api.request("GET", "/api/reports")[1]["reports"] == {"demo/add": report}
    assert local_api.request("GET", "/api/workspace?problem=1")[1]["report"] == report
    (local_api.root / "reports/demo-add.json").write_text("invalid interrupted JSON")
    assert local_api.request("GET", "/api/reports")[1] == {"reports": {}}


def test_invalid_inputs_and_busy_judge_do_not_write_workspace(local_api):
    for payload in ({}, {"code": "valid but missing problem"}, {"problem": "1", "code": ""}, {"problem": "1", "code": []},
                    {"problem": "1", "code": "x", "layers": "T9"},
                    {"problem": "1", "code": "x" * (128 * 1024 + 1)}):
        assert local_api.request("POST", "/api/test", payload)[0] == 400
    assert local_api.request("POST", "/api/test", raw=b"[]")[0] == 400
    assert local_api.request("POST", "/api/test", raw=b"{invalid")[0] == 400
    assert local_api.request("POST", "/api/test", raw=b"x" * (256 * 1024 + 1))[0] == 413
    assert local_api.request("POST", "/api/test", {}, headers={"Content-Type": "text/plain"})[0] == 415
    with server._judge_lock:
        assert local_api.request("POST", "/api/test", {"problem": "1", "code": "x"})[0] == 409
    assert not (local_api.root / "workspace").exists() and not local_api.calls


def wait_for_report(local_api, identifier, headers):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, payload, _ = local_api.request("GET", f"/api/judge/jobs/{identifier}", headers=headers)
        if payload.get("status") not in ("queued", "running"):
            assert status == 200, payload
            return payload
        time.sleep(.01)
    pytest.fail("Mock judge did not finish")


def test_queued_native_run_retries_reconnects_and_keeps_browser_ownership(local_api):
    headers = {"X-PaperLeet-Client-ID": "a" * 32}
    payload = {"id": "1" * 32, "problem": "1", "code": "saved queued solution", "layers": ["T0"]}
    assert local_api.request("POST", "/api/judge/jobs", payload)[0] == 400
    status, queued, _ = local_api.request("POST", "/api/judge/jobs", payload, headers=headers)
    assert status == 202 and queued["id"] == payload["id"]
    result = wait_for_report(local_api, payload["id"], headers)
    assert result["status"] == "succeeded" and result["report"]["passed"] is True
    path = local_api.root / "workspace/demo/add.py"
    assert path.read_text() == payload["code"]
    path.write_text("later IDE edit")
    status, retry, _ = local_api.request("POST", "/api/judge/jobs", payload, headers=headers)
    assert status == 202 and retry == result
    assert path.read_text() == "later IDE edit" and local_api.calls == [payload["code"]]
    assert local_api.request("GET", f"/api/judge/jobs/{payload['id']}", headers={"X-PaperLeet-Client-ID": "b" * 32})[0] == 404
    assert local_api.request("POST", "/api/judge/jobs", {**payload, "code": "different"}, headers=headers)[0] == 409
    assert (local_api.root / ".paperleet/judge.sqlite3").exists()


def test_queue_cancellation_and_invalid_ids_do_not_run_code(local_api):
    queue = JudgeQueue(local_api.root / "queue.sqlite3", execute, start=False)
    local_api.server.judge_queue = queue
    headers = {"X-PaperLeet-Client-ID": "a" * 32}
    payload = {"id": "1" * 32, "problem": "1", "code": "not executed"}
    for invalid in (None, [], "bad"):
        assert local_api.request("POST", "/api/judge/jobs", {**payload, "id": invalid}, headers=headers)[0] == 400
    assert local_api.request("POST", "/api/judge/jobs", payload, headers=headers)[0] == 202
    assert local_api.request("POST", f"/api/judge/jobs/{payload['id']}/cancel", {}, headers=headers)[1]["status"] == "cancelled"
    queue.thread.start()
    assert not local_api.calls
    assert not (local_api.root / "workspace").exists()


@pytest.mark.parametrize("initial", [None, "old IDE source"])
def test_queue_keeps_newer_ide_file_and_saves_submitted_solution_separately(local_api, initial):
    queue = JudgeQueue(local_api.root / "queue.sqlite3", execute, start=False)
    local_api.server.judge_queue = queue
    if initial is not None:
        save_code(problem("1"), initial)
    headers = {"X-PaperLeet-Client-ID": "a" * 32}
    payload = {"id": "1" * 32, "problem": "1", "code": "queued solution"}
    assert local_api.request("POST", "/api/judge/jobs", payload, headers=headers)[0] == 202
    path = save_code(problem("1"), "new IDE edit while waiting")
    # Reconnect/retry must retain the original disk version observation.
    assert local_api.request("POST", "/api/judge/jobs", payload, headers=headers)[0] == 202
    queue.thread.start()
    report = wait_for_report(local_api, payload["id"], headers)["report"]
    assert report["workspace_preserved"] is True and report["workspace_warning"]
    assert Path(report["source_path"]).read_text() == payload["code"]
    assert path.read_text() == "new IDE edit while waiting"
    assert local_api.calls == [payload["code"]]
    assert json.loads((local_api.root / "reports/demo-add.json").read_text()) == report
