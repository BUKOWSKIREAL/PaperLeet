"""Internal-only judge HTTP service; do not publish this port to the Internet."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
import socket
import threading

from .judge_client import MAX_REQUEST_BYTES, JudgeUnavailable, strict_json, validate_job
from .sandbox import JudgeSandbox


class BoundedJudgeServer(ThreadingHTTPServer):
    """Bound even unauthenticated/slow HTTP connections before making threads."""
    def __init__(self, *args, **kwargs):
        self.connections = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def handler_for(sandbox, token, concurrency=1):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii():
        raise ValueError("PAPERLEET_JUDGE_TOKEN must contain at least 32 ASCII characters")
    if type(concurrency) is not int or not 1 <= concurrency <= 2:
        raise ValueError("Judge concurrency must be 1 or 2")
    slots = threading.BoundedSemaphore(concurrency)

    class JudgeServiceHandler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def respond(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            return supplied.isascii() and hmac.compare_digest(supplied, "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "Unauthorized"})
            if self.path != "/health":
                return self.respond(404, {"error": "Unknown endpoint"})
            result = sandbox.capability()
            return self.respond(200 if result["ready"] else 503, {"ready": result["ready"]})

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "Unauthorized"})
            if self.path != "/run":
                return self.respond(404, {"error": "Unknown endpoint"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "Expected application/json"})
            if self.headers.get("Transfer-Encoding"):
                return self.respond(400, {"error": "Chunked requests are not accepted"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    return self.respond(413, {"error": "Judge request exceeds the size limit"})
                job = validate_job(strict_json(self.rfile.read(size)))
            except (ValueError, TypeError, socket.timeout):
                return self.respond(400, {"error": "Invalid judge request"})
            if not slots.acquire(blocking=False):
                return self.respond(429, {"error": "Judge is busy. Try again shortly."})
            try:
                return self.respond(200, sandbox.run(job))
            except JudgeUnavailable as exc:
                return self.respond(503, {"error": str(exc)})
            except Exception:
                return self.respond(503, {"error": "The isolated judge is unavailable"})
            finally:
                slots.release()

    return JudgeServiceHandler


def main():
    token = os.environ.get("PAPERLEET_JUDGE_TOKEN", "")
    concurrency = int(os.environ.get("PAPERLEET_JUDGE_CONCURRENCY", "1"))
    handler = handler_for(JudgeSandbox(), token, concurrency)
    server = BoundedJudgeServer((os.environ.get("PAPERLEET_JUDGE_BIND", "0.0.0.0"), int(os.environ.get("PAPERLEET_JUDGE_PORT", "8766"))), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
