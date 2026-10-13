"""Trusted container entrypoint; never run generated sources outside the container."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pytest
import torch

if __package__:
    from .artifacts import (MAX_ARTIFACTS_PER_TEST, MAX_ARTIFACT_BYTES_PER_TEST,
                            MAX_ARTIFACT_BYTES_PER_REPORT, MAX_ARTIFACT_NODES,
                            artifact_name, serialize_artifact)
else:
    from artifacts import (MAX_ARTIFACTS_PER_TEST, MAX_ARTIFACT_BYTES_PER_TEST,
                           MAX_ARTIFACT_BYTES_PER_REPORT, MAX_ARTIFACT_NODES,
                           artifact_name, serialize_artifact)


def numerical_artifact(value):
    """Detach small numeric arrays; check size before materializing Python lists."""
    if isinstance(value, torch.Tensor):
        if value.numel() > MAX_ARTIFACT_NODES or value.layout != torch.strided or value.is_complex():
            raise ValueError("Artifact tensors must be small, dense, real tensors")
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size > MAX_ARTIFACT_NODES or value.dtype.kind not in "biuf":
            raise ValueError("Artifact arrays must be small real numeric arrays")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Reporter:
    def __init__(self, toy=None):
        self.rows = {}
        self.toy = toy
        self.artifact_bytes = 0

    def save_artifact(self, item, name, value):
        """Artifacts are measured test outputs; provenance is bound to the candidate."""
        artifact_name(name)
        if not self.toy:
            raise ValueError("Artifact tests require declared toy provenance in the candidate")
        if not hasattr(item, "_paperleet_artifacts"):
            item._paperleet_artifacts = {}
            item._paperleet_artifact_bytes = 0
        if name in item._paperleet_artifacts or len(item._paperleet_artifacts) >= MAX_ARTIFACTS_PER_TEST:
            raise ValueError("Use at most four unique artifact names per test")
        if isinstance(value, dict) and value.get("kind") == "toy_experiment":
            value = {**value, "provenance": {"kind": self.toy["kind"], "source": self.toy["source_url"],
                      "locator": self.toy["locator"], "adaptation": self.toy["adaptation"]}}
        output, size = serialize_artifact(value, numerical_artifact)
        if item._paperleet_artifact_bytes + size > MAX_ARTIFACT_BYTES_PER_TEST:
            raise ValueError("Artifacts exceed the 32 KB per-test budget")
        if self.artifact_bytes + size > MAX_ARTIFACT_BYTES_PER_REPORT:
            raise ValueError("Artifacts exceed the 256 KB report budget")
        item._paperleet_artifacts[name] = output
        item._paperleet_artifact_bytes += size
        self.artifact_bytes += size

    @pytest.fixture
    def artifact(self, request):
        return lambda name, value: self.save_artifact(request.node, name, value)

    @pytest.fixture
    def impl(self):
        spec = importlib.util.spec_from_file_location("learner_implementation", "/candidate/implementation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.fixture(autouse=True)
    def deterministic(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        layer = Path(str(item.path)).stem.split("_", 2)[1]
        row = self.rows.setdefault(item.nodeid, {"id": item.name, "nodeid": item.nodeid, "layer": layer,
            "passed": True, "outcome": "passed", "message": "", "duration_s": 0,
            "why": (inspect.getdoc(item.obj) or "Check the published contract.").splitlines()[0], "artifacts": {}})
        row["duration_s"] += report.duration
        if report.failed or report.skipped:
            error_type = call.excinfo.type.__name__ if call.excinfo else ""
            row.update(passed=False, outcome="failed" if report.failed and report.when == "call" and error_type in ("AssertionError", "NotImplementedError") else "error", error_type=error_type)
            row["message"] += str(report.longrepr)[-8000:]
        row["artifacts"] = getattr(item, "_paperleet_artifacts", {})


if __name__ == "__main__":
    context = Path("/candidate/artifact_context.json")
    reporter = Reporter(json.loads(context.read_text()) if context.is_file() else None)
    started = time.monotonic()
    code = pytest.main(["/candidate/tests", "-q", "--tb=short", "-p", "pytest_timeout", "--timeout=15", "--confcutdir=/candidate/tests", "--import-mode=importlib", "-o", "cache_dir=/tmp/pytest-cache"], plugins=[reporter])
    print("PAPERLEET_RESULT=" + json.dumps({"exit_code": int(code), "tests": list(reporter.rows.values()), "duration_s": time.monotonic() - started}), flush=True)
