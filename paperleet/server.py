"""Offline editor bridge: native CPU judging and files on this computer."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

from paperleet.config import validate_local_settings
from paperleet.judge.runner import normalize_layers, run_problem, write_report
from paperleet.judge.jobs_api import is_job_path, dispatch as dispatch_jobs
from paperleet.workspace import (local_problem as problem, read_reports, read_workspace, report_path,
                                 save_code, save_submission, validate_code, workspace_digest)


_judge_lock = threading.Lock()
_unobserved = object()


def run_code(entry, code, layers, expected_workspace=_unobserved):
    """Save the submitted solution, then judge an immutable private snapshot.

    An IDE save while tests run stays in workspace and cannot alter this run.
    The caller owns _judge_lock, shared by synchronous and queued requests.
    """
    if expected_workspace is _unobserved:
        expected_workspace = workspace_digest(entry)
    preserved = workspace_digest(entry) != expected_workspace
    path = save_submission(entry, code) if preserved else save_code(entry, code)
    with tempfile.TemporaryDirectory(prefix="paperleet-run-") as directory:
        snapshot = Path(directory) / path.name
        snapshot.write_text(code, encoding="utf-8")
        report = run_problem(entry["id"], snapshot, layers=layers)
    report["impl"] = str(path)
    if preserved:
        report.update(workspace_preserved=True, source_path=str(path),
                      workspace_warning="The local workspace changed while this run was queued. "
                                        "That file was preserved; the submitted solution was saved separately.")
    write_report(report, report_path(entry))
    return report


class JudgeHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass

    def respond(self, status, payload, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def valid_origin(self):
        allowed = ("127.0.0.1", "localhost", "::1")
        try:
            host = urlparse("http://" + self.headers.get("Host", ""))
            origin = self.headers.get("Origin")
            if host.hostname not in allowed or host.username is not None:
                return False
            if not origin:
                return True
            parsed = urlparse(origin)
            return (parsed.scheme in ("http", "https") and parsed.hostname in allowed
                    and parsed.username is None)
        except ValueError:
            return False

    def catalog(self):
        # Export is metadata-only: no reference execution, Torch or downloads.
        # One export per local bridge also picks up a freshly synced curriculum.
        with self.server.catalog_lock:
            if self.server.catalog_data is None:
                from paperleet.export import export_catalog
                self.server.catalog_data = json.loads(export_catalog().read_text(encoding="utf-8"))
            return self.server.catalog_data

    def do_GET(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "This request origin is not allowed"})
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            return self.respond(200, {"ok": True, "mode": "local", "edition": "offline",
                                      "version": "0.1.0", "auth_required": False,
                                      "judge": {"ready": True}})
        if is_job_path(self.path):
            return dispatch_jobs(self, "GET")
        try:
            if parsed.path == "/api/catalog":
                return self.respond(200, self.catalog())
            if parsed.path.startswith("/api/problems/"):
                entry = problem(unquote(parsed.path[len("/api/problems/"):]))
                data = next(row for row in self.catalog()["problems"] if row["id"] == entry["id"])
                return self.respond(200, data)
            if parsed.path == "/api/workspace":
                identifier = parse_qs(parsed.query).get("problem", [None])[0]
                return self.respond(200, read_workspace(problem(identifier)))
            if parsed.path == "/api/reports":
                return self.respond(200, {"reports": read_reports()})
        except ValueError as exc:
            return self.respond(400, {"error": str(exc)})
        except OSError:
            return self.respond(500, {"error": "Could not read local practice files"})
        return self.respond(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "This request origin is not allowed"})
        if self.path not in ("/api/test", "/api/workspace") and not is_job_path(self.path):
            return self.respond(404, {"error": "Unknown endpoint"})
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Expected application/json"})
        if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            return self.respond(400, {"error": "Expected one Content-Length header"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 256 * 1024:
                return self.respond(413, {"error": "Request must be between 1 byte and 256 KB"})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if is_job_path(self.path):
                return dispatch_jobs(self, "POST", data)
            entry = problem(data.get("problem"))
            code = validate_code(data.get("code"), allow_empty=self.path == "/api/workspace")
            if self.path == "/api/workspace":
                save_code(entry, code)
                return self.respond(200, {"ok": True, "problem": entry["id"]})
            layers = normalize_layers(data.get("layers"))
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            return self.respond(400, {"error": str(exc)})
        except OSError:
            return self.respond(500, {"error": "Could not save local practice files"})
        if not _judge_lock.acquire(blocking=False):
            return self.respond(409, {"error": "A judge run is already in progress. Please wait."})
        try:
            return self.respond(200, run_code(entry, code, layers))
        except Exception:
            return self.respond(500, {"error": "The local judge could not complete this run"})
        finally:
            _judge_lock.release()


class BoundedHTTPServer(ThreadingHTTPServer):
    """Limit concurrent connections; the bridge only binds loopback."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(32)
        self.catalog_lock = threading.Lock()
        self.catalog_data = None
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def server_close(self):
        if hasattr(self, "judge_queue"):
            self.judge_queue.close()
        super().server_close()


def serve(port=8765):
    validate_local_settings()
    address = "127.0.0.1"
    server = BoundedHTTPServer((address, port), JudgeHandler)
    from paperleet.judge.jobs_api import start_queue
    start_queue(server)
    print(f"PaperLeet local CPU judge: http://{address}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
