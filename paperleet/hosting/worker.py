"""Trusted container entrypoint. This module must never run in the public app."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

from paperleet.hosting.judge_client import MAX_REQUEST_BYTES, MAX_REPORT_BYTES, strict_json, validate_job, validate_report
from paperleet.judge.runner import run_problem


def main():
    # A second deadline inside PID 1 also ends the container if the manager's
    # Docker connection disappears while a learner process is still running.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(140)
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        return 2
    try:
        job = validate_job(strict_json(raw))
        with tempfile.TemporaryDirectory(prefix="paperleet-job-") as directory:
            source = Path(directory) / "submission.py"
            source.write_text(job["code"], encoding="utf-8")
            report = run_problem(job["problem"], source, layers=job["layers"])
            report["source_sha256"] = hashlib.sha256(job["code"].encode()).hexdigest()
            report = validate_report(report, job)
        output = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(output) > MAX_REPORT_BYTES:
            return 2
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        # Never turn worker/protocol failure into an accepted submission.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
