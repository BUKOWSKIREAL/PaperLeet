"""Loopback-only bridge for the local code editor and real pytest judge."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import re
from urllib.parse import urlparse

from paperleet.catalog import problem, root, workspace_path
from paperleet.judge.runner import normalize_layers, run_problem
from paperleet.submissions import Conflict, store
from paperleet.submissions.model import ModelUnavailable


_judge_lock = threading.Lock()


class JudgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def respond(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def valid_origin(self):
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        origin = self.headers.get("Origin")
        return host in ("127.0.0.1", "localhost", "::1") and (not origin or urlparse(origin).hostname in ("127.0.0.1", "localhost", "::1"))

    def do_GET(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "Only local requests are accepted"})
        if self.path == "/api/health":
            return self.respond(200, {"ok": True, "mode": "local", "version": "0.1.0"})
        try:
            if self.path == "/api/submissions/capabilities":
                return self.respond(200, store().capabilities())
            if self.path == "/api/submissions":
                return self.respond(200, {"submissions": store().list()})
            match = re.fullmatch(r"/api/submissions/([0-9a-f]{32})", self.path)
            if match:
                return self.respond(200, store().get(match[1]))
        except KeyError:
            return self.respond(404, {"error": "Unknown submission"})
        except ValueError as exc:
            return self.respond(400, {"error": str(exc)})
        return self.respond(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "Only local requests are accepted"})
        submission_action = re.fullmatch(r"/api/submissions/([0-9a-f]{32})/(generate|validate|review)", self.path)
        if self.path not in ("/api/test", "/api/submissions") and not submission_action:
            return self.respond(404, {"error": "Unknown endpoint"})
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Expected application/json"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 256 * 1024:
                return self.respond(413, {"error": "Request must be between 1 byte and 256 KB"})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/submissions":
                return self.respond(201, store().create(data))
            if submission_action:
                identifier, action = submission_action.groups()
                if action == "review":
                    return self.respond(200, store().review(identifier, data))
                return self.respond(202, store().start(identifier, action))
            entry = problem(data["problem"])
            code = data["code"]
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Python code cannot be empty")
            layers = normalize_layers(data.get("layers"))
        except ModelUnavailable as exc:
            return self.respond(503, {"error": str(exc)})
        except Conflict as exc:
            return self.respond(409, {"error": str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            return self.respond(400, {"error": str(exc)})
        if not _judge_lock.acquire(blocking=False):
            return self.respond(409, {"error": "A judge run is already in progress. Please wait."})
        try:
            path = workspace_path(entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            destination = root() / "reports" / (entry["id"].replace("/", "-") + ".json")
            report = run_problem(entry["id"], path, layers=layers, report_path=destination)
            self.respond(200, report)
        except Exception as exc:
            self.respond(500, {"error": str(exc)})
        finally:
            _judge_lock.release()


def serve(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), JudgeHandler)
    print(f"PaperLeet local judge: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
