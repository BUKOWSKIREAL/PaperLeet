"""Workflow regressions: model output never becomes a public problem without human review."""
import copy
import json
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

from paperleet.submissions import Conflict, SubmissionStore
from paperleet.submissions.checks import package_hash, static_checks
from paperleet.submissions.model import ModelUnavailable, OpenAIProvider
from paperleet.submissions.sandbox import OutputLimitError, Sandbox, _run_bounded, run_published


@pytest.fixture
def candidate():
    return {"title": "求和", "title_en": "Addition", "difficulty": "E", "summary": "Add scalars.",
            "readme": "实现 add(a, b)，返回两数之和。", "readme_en": "Implement add(a, b), returning their sum.",
            "starter": "def add(a, b):\n    raise NotImplementedError\n", "solution": "def add(a, b):\n    return a + b\n",
            "notes": "注意符号。", "notes_en": "Check the sign.", "functions": [{"name": "add", "signature": "add(a, b)"}],
            "evidence": [{"claim": "The test contract adds two scalars.", "source_url": "https://example.com/paper", "locator": "Equation 1"}],
            "tests": [{"name": f"test_layer_{i}", "layer": f"T{i}", "code": f"def test_layer_{i}(impl):\n    assert impl.add({i}, 2) == {i + 2}\n"} for i in range(4)],
            "mutants": [{"name": "wrong_sign", "bug": "Subtracts.", "code": "def add(a, b):\n    return a - b\n"},
                        {"name": "extra_one", "bug": "Adds one.", "code": "def add(a, b):\n    return a + b + 1\n"}]}


class FakeProvider:
    def __init__(self, candidate):
        self.candidate = candidate
        self.last_input = None

    def capability(self):
        return {"ready": True, "model": "test-model", "provider": "mock"}

    def generate(self, submission):
        self.last_input = copy.deepcopy(submission)
        return copy.deepcopy(self.candidate), {"independent_test_generation": True, "sources": []}


class FakeSandbox(Sandbox):
    """Only adapter responses are mocked; the real validation policy remains under test."""
    def capability(self):
        return {"ready": True, "image": "test-image", "image_id": "test-image-sha"}

    def run(self, candidate, implementation, layers=None):
        passing = implementation == candidate["solution"]
        rows = [{"id": row["name"], "nodeid": row["name"], "layer": row["layer"], "passed": passing,
                 "outcome": "passed" if passing else "failed", "error_type": "NotImplementedError" if implementation == candidate["starter"] else "AssertionError",
                 "duration_s": 0, "message": "", "why": "Check addition.", "artifacts": {}} for row in candidate["tests"] if not layers or row["layer"] in layers]
        return {"exit_code": 0 if passing else 1, "tests": rows, "output": ""}


@pytest.fixture
def workflow(tmp_path, monkeypatch, candidate):
    monkeypatch.setenv("PAPERLEET_ROOT", str(tmp_path))
    folder = tmp_path / "problems" / "trusted" / "existing"
    folder.mkdir(parents=True)
    (folder / "problem.yaml").write_text(yaml.safe_dump({"id": "trusted/existing", "number": 10, "short_id": "10", "title": "Existing"}))
    return SubmissionStore(tmp_path, provider=FakeProvider(candidate), sandbox=FakeSandbox())


def wait_for_job(store, identifier):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        result = store.get(identifier)
        if identifier not in store.jobs:
            return result
        time.sleep(.01)
    raise AssertionError("Background job did not finish")


def generated(store):
    submission = store.create({"title": "Arithmetic paper", "paper_url": "https://example.com/paper", "focus": "Addition"})
    store.start(submission["id"], "generate")
    return wait_for_job(store, submission["id"])


def review_payload(submission, decision="approve"):
    return {"decision": decision, "reviewer": "Test human reviewer", "notes": "Checked equations, examples and independent oracles.",
            "package_hash": submission["package_hash"], "checks": {k: True for k in ("paper_fidelity", "scope_clarity", "test_independence", "license_checked")}}


def test_submission_does_not_publish_until_validated_and_human_approved(workflow):
    record = generated(workflow)
    assert record["status"] == "validation_pending"
    with pytest.raises(Conflict, match="isolated validation"):
        workflow.review(record["id"], review_payload(record))
    assert not (workflow.base / "problems/community").exists()
    workflow.start(record["id"], "validate")
    record = wait_for_job(workflow, record["id"])
    assert record["status"] == "needs_review"
    assert not (workflow.base / "problems/community").exists()
    published = workflow.review(record["id"], review_payload(record))
    assert published["status"] == "published"
    assert published["published_problem"]["number"] == 11
    package = workflow.base / "problems" / published["published_problem"]["id"]
    assert json.loads((package / "candidate.json").read_text()) == record["candidate"]
    metadata = yaml.safe_load((package / "problem.yaml").read_text())
    assert metadata["execution"] == "sandbox" and metadata["package_hash"] == record["package_hash"]
    public = json.loads((workflow.base / "web/public/data/catalog.json").read_text())
    assert len(public["problems"]) == 2
    assert public["problems"][-1]["translations"]["en"]["readme"]
    with pytest.raises(Conflict):
        workflow.review(record["id"], review_payload(record))


def test_stale_hash_and_incomplete_human_evaluation_are_rejected(workflow):
    record = generated(workflow)
    workflow.start(record["id"], "validate")
    record = wait_for_job(workflow, record["id"])
    payload = review_payload(record)
    payload["package_hash"] = "0" * 64
    with pytest.raises(Conflict, match="changed"):
        workflow.review(record["id"], payload)
    payload = review_payload(record)
    payload["checks"]["license_checked"] = False
    with pytest.raises(ValueError, match="human must confirm"):
        workflow.review(record["id"], payload)
    record["candidate"]["notes"] = "Changed after validation"
    record["package_hash"] = package_hash(record["candidate"])
    workflow._save(record)
    with pytest.raises(Conflict, match="exact candidate"):
        workflow.review(record["id"], review_payload(record))


def test_revision_preserves_feedback_and_invalidates_prior_validation(workflow):
    record = generated(workflow)
    old_hash = record["package_hash"]
    record = workflow.review(record["id"], review_payload(record, "changes_requested"))
    workflow.provider.candidate["notes"] = "Revised explanation"
    workflow.start(record["id"], "generate")
    result = wait_for_job(workflow, record["id"])
    assert result["status"] == "validation_pending" and not result["validation"]["passed"]
    assert result["package_hash"] != old_hash
    assert result["reviews"][0]["package_hash"] == old_hash
    assert workflow.provider.last_input["reviews"][0]["decision"] == "changes_requested"
    assert (workflow.directory / (record["id"] + "-revisions") / (old_hash + ".json")).exists()


def test_missing_sandbox_never_enters_review(workflow):
    workflow.sandbox.capability = lambda: {"ready": False, "reason": "Docker is unavailable", "image": "test"}
    record = generated(workflow)
    workflow.start(record["id"], "validate")
    record = wait_for_job(workflow, record["id"])
    assert record["status"] == "validation_pending"
    assert record["validation"]["status"] == "unavailable"
    with pytest.raises(Conflict):
        workflow.review(record["id"], review_payload(record))


def test_missing_api_configuration_retains_submitted_paper(workflow, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PAPERLEET_MODEL", raising=False)
    workflow.provider = OpenAIProvider()
    record = workflow.create({"title": "Paper", "paper_url": "https://example.com/paper"})
    with pytest.raises(ModelUnavailable, match="PAPERLEET_MODEL"):
        workflow.start(record["id"], "generate")
    assert workflow.get(record["id"])["status"] == "submitted"
    assert not workflow.provider.capability()["ready"]


def test_failed_generation_and_restart_do_not_unlock_review(workflow):
    def fail(_):
        raise ModelUnavailable("Provider timed out")
    workflow.provider.generate = fail
    record = generated(workflow)
    assert record["status"] == "generation_failed"
    assert record["error"] == "Provider timed out"
    record["status"] = "validating"
    workflow._save(record)
    recovered = SubmissionStore(workflow.base, workflow.provider, workflow.sandbox).get(record["id"])
    assert recovered["status"] == "validation_failed"


@pytest.mark.parametrize("url", ["http://example.com/paper", "https://localhost/secret", "https://127.0.0.1/a", "https://user:password@example.com/paper", "https://10.0.0.1/paper"])
def test_source_input_rejects_credentials_and_private_urls(workflow, url):
    with pytest.raises(ValueError):
        workflow.create({"title": "Paper", "paper_url": url})


def test_static_policy_catches_imports_nonasserting_tests_and_wrong_starter(candidate):
    assert all(c["passed"] for c in static_checks(candidate))
    cases = [("solution", "import os\ndef add(a, b): return a + b\n"),
             ("starter", "def add(a, b): raise ValueError\n"),
             ("solution", "def add(a, c): return a + c\n")]
    for key, source in cases:
        changed = {**candidate, key: source}
        assert not all(c["passed"] for c in static_checks(changed))
    candidate["tests"][0]["code"] = "def test_layer_0(impl):\n    return impl.add(1, 2)\n"
    assert not all(c["passed"] for c in static_checks(candidate))


def test_mutant_runtime_errors_do_not_count_as_kills(candidate):
    sandbox = FakeSandbox()
    original = sandbox.run
    def invalid_mutants(package, source, layers=None):
        result = original(package, source, layers)
        if source in [m["code"] for m in package["mutants"]]:
            for row in result["tests"]:
                row["error_type"] = "NotImplementedError"
        return result
    sandbox.run = invalid_mutants
    report = sandbox.validate(candidate)
    assert not report["passed"]
    assert all(not c["passed"] for c in report["checks"] if c["name"].startswith("mutant:"))


def test_incomplete_reference_test_set_cannot_pass(candidate):
    sandbox = FakeSandbox()
    original = sandbox.run
    sandbox.run = lambda *args: {**original(*args), "tests": original(*args)["tests"][:-1]}
    assert not sandbox.validate(candidate)["passed"]


def test_output_capture_is_bounded_and_times_out():
    assert _run_bounded([sys.executable, "-c", "print('small')"])[1].strip() == "small"
    with pytest.raises(OutputLimitError):
        _run_bounded([sys.executable, "-c", "print('x' * 200000)"], max_bytes=1024)
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], timeout=.05)


def test_published_judge_checks_immutable_manifest_and_uses_sandbox(workflow, monkeypatch, tmp_path):
    record = generated(workflow)
    workflow.start(record["id"], "validate")
    record = wait_for_job(workflow, record["id"])
    published = workflow.review(record["id"], review_payload(record))
    from paperleet.catalog import problem
    entry = problem(published["published_problem"]["id"])
    monkeypatch.setattr("paperleet.submissions.sandbox.Sandbox", FakeSandbox)
    source = tmp_path / "learner.py"
    source.write_text(record["candidate"]["solution"])
    assert run_published(entry, source)["passed"]
    manifest = entry["directory"] / "candidate.json"
    data = json.loads(manifest.read_text())
    data["solution"] += "\n# changed"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        run_published(entry, source)


def test_model_tests_are_generated_without_reference_source(candidate, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("PAPERLEET_MODEL", "test-model")
    provider = OpenAIProvider()
    calls = []
    def request(prompt, schema=None, search=False):
        calls.append((prompt, search))
        if search:
            value = "Equation 1 adds two scalars."
        elif len(calls) == 2:
            value = {k: v for k, v in candidate.items() if k not in ("tests", "mutants")}
        else:
            value = {k: candidate[k] for k in ("tests", "mutants")}
        return {"value": value, "sources": [{"url": "https://example.com/paper"}], "response_id": "test"}
    provider.request = request
    output, provenance = provider.generate({"title": "Paper", "paper_url": "https://example.com/paper", "reviews": [{"decision": "changes_requested", "notes": "Add boundary examples"}]})
    assert output == candidate and provenance["independent_test_generation"]
    assert calls[0][1]
    assert "Add boundary examples" in calls[1][0]
    assert candidate["solution"] not in calls[2][0]
    assert '"solution"' not in calls[2][0]


def test_http_submission_endpoints_and_origin_boundary(workflow, monkeypatch):
    from http.server import ThreadingHTTPServer
    from paperleet.server import JudgeHandler
    monkeypatch.setattr("paperleet.server.store", lambda: workflow)
    server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(address + "/api/submissions", data=json.dumps({"title": "Paper", "paper_url": "https://example.com/paper"}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            record = json.load(response)
            assert response.status == 201
        with urlopen(address + "/api/submissions/" + record["id"]) as response:
            assert json.load(response)["status"] == "submitted"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(address + "/api/submissions", headers={"Origin": "https://example.com"}))
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(address + "/api/submissions", data=b"[]", headers={"Content-Type": "application/json"}))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_docker_execution_uses_offline_bounded_nonroot_snapshot(candidate, monkeypatch):
    sandbox = Sandbox()
    sandbox.capability = lambda: {"ready": True, "image": "test", "image_id": "sha256:pinned"}
    captured = {}
    def execute(args):
        captured["args"] = args
        mount = args[args.index("--mount") + 1]
        from pathlib import Path
        directory = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
        assert (directory / "implementation.py").read_text() == candidate["solution"]
        assert len(list((directory / "tests").glob("test_T*_*.py"))) == 4
        assert not (directory / "solution.py").exists()
        assert not (directory / "mutants").exists()
        return 0, 'PAPERLEET_RESULT={"exit_code":0,"tests":[]}\n'
    monkeypatch.setattr("paperleet.submissions.sandbox._run_bounded", execute)
    monkeypatch.setattr("paperleet.submissions.sandbox.subprocess.run", lambda *args, **kwargs: None)
    sandbox.run(candidate, candidate["solution"])
    args = captured["args"]
    assert "--network=none" in args and "--read-only" in args and "--cap-drop=ALL" in args
    assert "--security-opt=no-new-privileges" in args and "--memory=1g" in args and "--pids-limit=128" in args
    assert args[args.index("--user") + 1] == "65534:65534"
    assert args[-1] == "sha256:pinned"
    assert "--env" not in args and "--env-file" not in args


def test_openai_transport_uses_structured_outputs_without_storing_or_exposing_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setenv("PAPERLEET_MODEL", "configured-model")
    captured = {}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            return json.dumps({"status": "completed", "id": "r1", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}).encode()
    def transport(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["url"] = request.full_url
        assert request.get_header("Authorization") == "Bearer secret-test-key"
        return Response()
    monkeypatch.setattr("paperleet.submissions.model.urlopen", transport)
    provider = OpenAIProvider()
    result = provider.request("Test", {"type": "object", "properties": {}, "required": [], "additionalProperties": False})
    assert result["value"] == {"ok": True}
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["body"]["model"] == "configured-model" and captured["body"]["store"] is False
    assert captured["body"]["text"]["format"]["strict"] is True
    assert "secret-test-key" not in json.dumps(provider.capability())


def test_model_cannot_generate_without_retrieved_or_supplied_evidence(monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODEL", "test")
    provider = OpenAIProvider()
    provider.request = lambda *args, **kwargs: {"value": "Unavailable", "sources": [], "response_id": "test"}
    with pytest.raises(ModelUnavailable, match="No accessible"):
        provider.generate({"title": "Paper", "paper_url": "https://example.com/paper"})
