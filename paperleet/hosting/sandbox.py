"""Private judge manager: bounded Docker execution with no host bind mounts."""
import json
import os
import re
import selectors
import shutil
import subprocess
import time
import uuid

from .judge_client import JOB_TIMEOUT, MAX_REPORT_BYTES, JudgeUnavailable, strict_json, validate_job, validate_report


class OutputLimitError(RuntimeError):
    pass


def _docker_env():
    # Credentials for email, accounts and model providers never enter the Docker
    # client or worker. Docker context configuration is manager-local only.
    return {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"}}


def _run_bounded(args, payload, timeout=JOB_TIMEOUT, max_bytes=MAX_REPORT_BYTES):
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_docker_env())
    selector = selectors.DefaultSelector()
    streams = (process.stdin, process.stdout, process.stderr)
    for stream in streams:
        os.set_blocking(stream.fileno(), False)
    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    deadline, length, sent = time.monotonic() + timeout, 0, 0
    output = {"stdout": [], "stderr": []}
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            for key, _ in selector.select(min(remaining, .2)):
                if key.data == "stdin":
                    try:
                        sent += os.write(key.fd, payload[sent:sent + 16384])
                    except BrokenPipeError:
                        sent = len(payload)
                    if sent >= len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    chunk = os.read(key.fd, 16384)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    length += len(chunk)
                    if length > max_bytes:
                        raise OutputLimitError("Judge exceeded the output limit")
                    output[key.data].append(chunk)
        return process.wait(timeout=max(.01, deadline - time.monotonic())), b"".join(output["stdout"]), b"".join(output["stderr"])
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in streams:
            stream.close()


class JudgeSandbox:
    def __init__(self, image=None):
        self.image = image or os.environ.get("PAPERLEET_JUDGE_IMAGE", "paperleet-worker:local")

    def capability(self):
        if not shutil.which("docker"):
            return {"ready": False, "reason": "The Docker client is unavailable"}
        try:
            result = subprocess.run(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"],
                capture_output=True, text=True, timeout=4, env=_docker_env())
            image_id = result.stdout.strip()
            if result.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                return {"ready": False, "reason": "Build the isolated judge worker image and start Docker"}
            return {"ready": True, "image_id": image_id}
        except (OSError, subprocess.TimeoutExpired):
            return {"ready": False, "reason": "The Docker daemon is unavailable"}

    def run(self, payload):
        job = validate_job(payload)
        capability = self.capability()
        if not capability["ready"]:
            raise JudgeUnavailable(capability["reason"])
        name = "paperleet-job-" + uuid.uuid4().hex
        args = ["docker", "run", "-i", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=128", "--memory=2g", "--memory-swap=2g", "--cpus=2",
            "--ulimit", "nofile=128:128", "--ulimit", "fsize=16777216:16777216", "--log-driver=none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777", "--user", "65534:65534",
            capability["image_id"]]
        try:
            returncode, raw, _stderr = _run_bounded(args, json.dumps(job, allow_nan=False).encode())
            if returncode != 0:
                raise JudgeUnavailable("The isolated judge did not complete. Check your code or retry.")
            return validate_report(strict_json(raw), job)
        except subprocess.TimeoutExpired:
            raise JudgeUnavailable("The isolated judge exceeded its 150-second limit. Check for an infinite loop and retry.") from None
        except OutputLimitError:
            raise JudgeUnavailable("The isolated judge exceeded its output limit. Remove excessive output and retry.") from None
        except (OSError, ValueError):
            raise JudgeUnavailable("The isolated judge returned no valid report.") from None
        finally:
            # Stopping a Docker CLI does not stop its container. Always remove
            # the exact manager-generated job name, including on timeout.
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=8, env=_docker_env())
            except (OSError, subprocess.TimeoutExpired):
                pass
