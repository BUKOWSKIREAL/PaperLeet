"""Workflow regressions: model output never becomes a public problem without human review."""
import copy
import json
import os
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
from paperleet.submissions.artifacts import serialize_artifact, validate_report_artifacts


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


@pytest.fixture
def toy_candidate(candidate):
    candidate["toy"] = {"kind": "teaching_adaptation", "source_url": "https://example.com/paper",
                        "locator": "Equation 1; synthetic test fixture", "adaptation": "Eight scalar addition inputs; no paper benchmark claims."}
    candidate["tests"][-1]["code"] = '''import torch
def test_layer_3(impl, artifact):
    """Measure the submitted function on a deterministic synthetic sequence."""
    inputs = torch.arange(8, dtype=torch.float32)
    output = impl.add(inputs, 2)
    artifact('addition_toy', {'kind': 'toy_experiment', 'title': 'Synthetic addition',
                             'samples': torch.stack((inputs, output), dim=-1),
                             'metrics': {'mean': output.mean()}})
    torch.testing.assert_close(output, inputs + 2)
'''
    return candidate


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


def test_optional_artifact_contract_and_provenance(toy_candidate):
    assert all(row['passed'] for row in static_checks(toy_candidate))
    legacy = copy.deepcopy(toy_candidate)
    legacy.pop('toy')
    legacy['tests'][-1]['code'] = 'def test_layer_3(impl):\n    assert impl.add(3, 2) == 5\n'
    assert all(row['passed'] for row in static_checks(legacy))
    missing = copy.deepcopy(toy_candidate)
    missing.pop('toy')
    assert not all(row['passed'] for row in static_checks(missing))
    unknown = copy.deepcopy(toy_candidate)
    unknown['toy']['kind'] = 'official-looking'
    assert not all(row['passed'] for row in static_checks(unknown))
    uncited = copy.deepcopy(toy_candidate)
    uncited['toy']['source_url'] = 'https://example.com/other-paper'
    assert not all(row['passed'] for row in static_checks(uncited))
    no_measurement = copy.deepcopy(toy_candidate)
    no_measurement['tests'][-1]['code'] = legacy['tests'][-1]['code']
    assert not all(row['passed'] for row in static_checks(no_measurement))


@pytest.mark.parametrize('parameters', ['impl, request', 'impl, artifact, request', 'impl, artifact=None', 'impl, *args', 'impl, /', 'impl, **kwargs'])
def test_generated_tests_cannot_request_other_fixtures(toy_candidate, parameters):
    toy_candidate['tests'][-1]['code'] = f'def test_layer_3({parameters}):\n    assert impl.add(3, 2) == 5\n'
    assert not all(row['passed'] for row in static_checks(toy_candidate))


def test_tensor_artifacts_are_finite_detached_copies(toy_candidate):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from paperleet.submissions.worker import Reporter
    reporter = Reporter(toy_candidate['toy'])
    item = SimpleNamespace()
    points = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    value = {'kind': 'toy_experiment', 'title': 'Measured output', 'samples': points,
             'metrics': {'mean': points.mean(), 'count': np.int64(2)},
             'loss_curve': np.array([[0., 2.], [1., .5]]),
             'provenance': {'kind': 'invented'}}
    reporter.save_artifact(item, 'actual_output', value)
    artifact = item._paperleet_artifacts['actual_output']
    assert artifact['samples'] == [[1., 2.], [3., 4.]]
    assert artifact['metrics'] == {'mean': 2.5, 'count': 2}
    assert artifact['provenance']['kind'] == 'teaching_adaptation'
    assert artifact['provenance']['source'] == toy_candidate['toy']['source_url']
    assert points.grad is None
    with torch.no_grad():
        points[0, 0] = 99
    assert artifact['samples'][0][0] == 1
    validate_report_artifacts([{'artifacts': item._paperleet_artifacts}])
    with pytest.raises(ValueError, match='unique'):
        reporter.save_artifact(item, 'actual_output', 1)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {'x': float('-inf')}, 2**60,
                                  {1: 'non-string key'}, {'key': 'x' * 4097}, list(range(4097))])
def test_artifact_json_rejects_nonfinite_or_unbounded_data(value):
    with pytest.raises(ValueError):
        serialize_artifact(value)


def test_artifact_depth_cycles_array_size_and_unsupported_objects():
    import numpy as np
    import torch
    from paperleet.submissions.worker import numerical_artifact
    cyclic = []; cyclic.append(cyclic)
    deep = 1
    for _ in range(14):
        deep = [deep]
    values = [cyclic, deep, object(), np.array([object()], dtype=object),
              np.zeros(4097), torch.zeros(4097), torch.tensor([complex(1, 2)]), torch.tensor([float('nan')])]
    for value in values:
        with pytest.raises(ValueError):
            serialize_artifact(value, numerical_artifact)


def test_artifact_limits_apply_per_test_and_whole_report(toy_candidate):
    from types import SimpleNamespace
    from paperleet.submissions.worker import Reporter
    reporter = Reporter(toy_candidate['toy'])
    item = SimpleNamespace()
    for i in range(4):
        reporter.save_artifact(item, f'metric_{i}', i)
    with pytest.raises(ValueError, match='four'):
        reporter.save_artifact(item, 'fifth', 5)
    large = {f'chunk_{i}': 'x' * 4000 for i in range(8)}
    reporter.save_artifact(SimpleNamespace(), 'large', large)
    combined = SimpleNamespace()
    reporter.save_artifact(combined, 'first', large)
    with pytest.raises(ValueError, match='32 KB'):
        reporter.save_artifact(combined, 'second', large)
    bounded_reporter = Reporter(toy_candidate['toy'])
    for _ in range(8):
        bounded_reporter.save_artifact(SimpleNamespace(), 'large', large)
    with pytest.raises(ValueError, match='256 KB'):
        bounded_reporter.save_artifact(SimpleNamespace(), 'large', large)
    row = {'artifacts': {'large': large}}
    with pytest.raises(ValueError, match='32 KB'):
        validate_report_artifacts([{'artifacts': {'first': large, 'second': large}}])
    with pytest.raises(ValueError, match='256 KB'):
        validate_report_artifacts([copy.deepcopy(row) for _ in range(9)])
    with pytest.raises(ValueError, match='provenance'):
        Reporter().save_artifact(SimpleNamespace(), 'missing_provenance', 1)


@pytest.mark.parametrize('failure', ['missing', 'empty', 'wrong_provenance'])
def test_declared_toy_without_measured_reference_artifact_cannot_pass(toy_candidate, failure):
    sandbox = FakeSandbox()
    original = sandbox.run
    def run(*args):
        result = original(*args)
        if failure != 'missing':
            toy = toy_candidate['toy']
            provenance = {'kind': toy['kind'], 'source': toy['source_url'],
                          'locator': toy['locator'], 'adaptation': toy['adaptation']}
            value = {'kind': 'toy_experiment', 'provenance': provenance}
            if failure == 'wrong_provenance':
                value['metrics'] = {'mean': 5.5}
                value['provenance']['kind'] = 'author_toy'
            next(row for row in result['tests'] if row['layer'] == 'T3')['artifacts'] = {'empty': value}
        return result
    sandbox.run = run
    report = sandbox.validate(toy_candidate)
    assert not report['passed']
    assert all(not row['passed'] for row in report['checks'] if row['name'] in ('reference', 'reference_repeat'))
    assert all('missing a measured T3' in row['detail'] for row in report['checks'] if row['name'] in ('reference', 'reference_repeat'))


def test_artifact_context_is_snapshotted_and_host_rejects_invalid_report(toy_candidate, monkeypatch):
    sandbox = Sandbox()
    sandbox.capability = lambda: {'ready': True, 'image_id': 'sha256:pinned'}
    def execute(args):
        from pathlib import Path
        mount = args[args.index('--mount') + 1]
        directory = Path(mount.split('source=', 1)[1].split(',target=', 1)[0])
        assert json.loads((directory / 'artifact_context.json').read_text()) == toy_candidate['toy']
        return 0, 'PAPERLEET_RESULT={"exit_code":0,"tests":[{"artifacts":{"bad":NaN}}]}\n'
    monkeypatch.setattr('paperleet.submissions.sandbox._run_bounded', execute)
    monkeypatch.setattr('paperleet.submissions.sandbox.subprocess.run', lambda *args, **kwargs: None)
    result = sandbox.run(toy_candidate, toy_candidate['solution'])
    assert result['exit_code'] == 125 and result['tests'] == [] and 'error' in result


def test_model_toy_prompt_keeps_test_generation_independent(toy_candidate):
    provider, prompts = OpenAIProvider(), []
    def request(prompt, schema=None, search=False):
        prompts.append(prompt)
        if search:
            value = 'Equation 1; no author toy available. A synthetic sequence is a teaching adaptation.'
        elif len(prompts) == 2:
            value = {k: v for k, v in toy_candidate.items() if k not in ('tests', 'mutants')}
            assert schema['properties']['toy']['properties']['kind']['enum'] == ['author_toy', 'paper_derived', 'teaching_adaptation']
        else:
            value = {k: toy_candidate[k] for k in ('tests', 'mutants')}
        return {'value': value, 'sources': [{'url': 'https://example.com/paper'}], 'response_id': 'fixture'}
    provider.request = request
    candidate, _ = provider.generate({'title': 'Synthetic fixture', 'paper_url': 'https://example.com/paper'})
    assert candidate == toy_candidate
    assert 'original toy experiments' in prompts[0]
    assert 'teaching_adaptation' in prompts[1]
    assert 'real measured results' in prompts[2] and '(impl, artifact)' in prompts[2]
    assert toy_candidate['solution'] not in prompts[2] and '"solution"' not in prompts[2]


@pytest.mark.skipif(os.environ.get('PAPERLEET_DOCKER_HARNESS_TESTS') != '1', reason='Set PAPERLEET_DOCKER_HARNESS_TESTS=1 after building the submission image')
def test_real_docker_toy_validation_and_published_artifact(toy_candidate, tmp_path):
    sandbox = Sandbox()
    assert sandbox.capability()['ready']
    result = sandbox.validate(toy_candidate)
    assert result['passed'], result
    artifact = next(row for row in result['runs']['reference']['tests'] if row['layer'] == 'T3')['artifacts']['addition_toy']
    assert artifact['samples'] == [[float(i), float(i + 2)] for i in range(8)]
    assert artifact['metrics']['mean'] == 5.5
    assert artifact['provenance']['kind'] == 'teaching_adaptation'
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'candidate.json').write_text(json.dumps(toy_candidate))
    implementation = tmp_path / 'learner.py'
    implementation.write_text(toy_candidate['solution'])
    entry = {'directory': package, 'id': 'community/toy', 'package_hash': package_hash(toy_candidate)}
    report = run_published(entry, implementation, ['T3'])
    assert report['passed'] and report['tests'][0]['artifacts']['addition_toy'] == artifact
    changed = copy.deepcopy(toy_candidate)
    changed['toy']['adaptation'] = 'Changed after review'
    (package / 'candidate.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed'):
        run_published(entry, implementation, ['T3'])
