"""Queued local judging with browser client isolation and bounded requests."""
import re
import logging
import threading
import json

from paperleet.config import data_directory
from paperleet.judge.runner import normalize_layers
from paperleet.workspace import local_problem as problem, validate_code, workspace_digest
from .queue import JudgeQueue, JobConflict, QueueFull, WorkerBusy

_creation_lock = threading.Lock()
_submission_lock = threading.Lock()
_route = re.compile(r"/api/judge/jobs(?:/([0-9a-f]{32})(/cancel)?)?")


def is_job_path(path):
    return bool(_route.fullmatch(path))


def queue_for(handler):
    return start_queue(handler.server)


def start_queue(server):
    with _creation_lock:
        if not hasattr(server, "judge_queue"):
            server.judge_queue = JudgeQueue(data_directory() / "judge.sqlite3", execute,
                                            scope="local")
        return server.judge_queue


def execute(owner, job):
    entry = problem(job["problem"])
    # Share the execution lock with the legacy synchronous local endpoint.
    from paperleet import server
    if not server._judge_lock.acquire(blocking=False):
        raise WorkerBusy()
    try:
        return server.run_code(entry, job["code"], job["layers"], job["workspace_sha256"])
    finally:
        server._judge_lock.release()


def dispatch(handler, method, payload=None):
    payload = payload or {}
    match = _route.fullmatch(handler.path)
    if not match:
        return handler.respond(404, {"error": "Unknown job endpoint"})
    identifier, cancellation = match.groups()
    if (method == "GET" and (not identifier or cancellation)) or (method == "POST" and identifier and not cancellation):
        return handler.respond(404, {"error": "Unknown job endpoint"})
    try:
        client = handler.headers.get("X-PaperLeet-Client-ID", "")
        if not re.fullmatch(r"[0-9a-f]{32}", client):
            raise ValueError("A local browser client ID is required")
        owner = "local:" + client
        if method == "GET":
            return handler.respond(200, queue_for(handler).get(identifier, owner))
        if cancellation:
            return handler.respond(200, queue_for(handler).cancel(identifier, owner))
        request_id = payload.get("id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("A 32-character request ID is required")
        entry = problem(payload.get("problem"))
        code = validate_code(payload.get("code"))
        job = {"problem": entry["id"], "code": code, "layers": normalize_layers(payload.get("layers"))}
        queue = queue_for(handler)
        # Retried requests reuse the original disk observation. A retry after
        # execution must not create a conflict or overwrite a newer IDE edit.
        with _submission_lock:
            with queue.connect() as db:
                row = db.execute("SELECT payload FROM judge_jobs WHERE id=? AND scope=? AND owner=?",
                                 (request_id, queue.scope, owner)).fetchone()
            previous = json.loads(row["payload"]) if row else None
            if previous is not None and all(previous.get(key) == value for key, value in job.items()):
                job = previous
            else:
                job["workspace_sha256"] = workspace_digest(entry)
            result = queue.submit(request_id, owner, job)
        return handler.respond(202, result)
    except QueueFull as exc:
        return handler.respond(429, {"error": str(exc), "code": "queue_full"}, {"Retry-After": "5"})
    except JobConflict as exc:
        return handler.respond(409, {"error": str(exc)})
    except KeyError:
        return handler.respond(404, {"error": "This job does not exist or has expired. Your draft is preserved."})
    except (TypeError, ValueError) as exc:
        return handler.respond(400, {"error": str(exc)})
    except Exception:
        logging.exception("Judge jobs API unavailable")
        return handler.respond(503, {"error": "The judge job service is temporarily unavailable. Your draft is preserved."})
