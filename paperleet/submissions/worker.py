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


class Reporter:
    def __init__(self):
        self.rows = {}

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


if __name__ == "__main__":
    reporter = Reporter()
    started = time.monotonic()
    code = pytest.main(["/candidate/tests", "-q", "--tb=short", "-p", "pytest_timeout", "--timeout=15", "--confcutdir=/candidate/tests", "--import-mode=importlib", "-o", "cache_dir=/tmp/pytest-cache"], plugins=[reporter])
    print("PAPERLEET_RESULT=" + json.dumps({"exit_code": int(code), "tests": list(reporter.rows.values()), "duration_s": time.monotonic() - started}), flush=True)
