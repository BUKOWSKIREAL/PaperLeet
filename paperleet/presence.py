"""Short-lived, aggregate visitor presence for a single API server process."""
import re
import threading
import time


WINDOW_SECONDS = 90
HEARTBEAT_SECONDS = 25
MAX_VISITORS = 10_000
MAX_BODY_BYTES = 512
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


class PresenceUnavailable(Exception):
    """Presence cannot accept more live visitors right now."""


class PresenceTracker:
    """Deduplicate browser IDs and signed-in accounts without recording history.

    Browser IDs are untrusted, anonymous deduplication hints. Account identities
    must come from the server's validated session, never the request body.
    """

    def __init__(self, *, clock=time.monotonic, capacity=MAX_VISITORS):
        if capacity < 1:
            raise ValueError("Presence capacity must be positive")
        self._clock = clock
        self._capacity = capacity
        self._visitors = {}
        self._lock = threading.Lock()

    def _expire(self, now):
        for visitor in [key for key, (seen, _) in self._visitors.items()
                        if now - seen >= WINDOW_SECONDS]:
            del self._visitors[visitor]

    def _snapshot(self):
        identities = {("account", account) if account else ("visitor", visitor)
                      for visitor, (_, account) in self._visitors.items()}
        return {"online": len(identities), "window_seconds": WINDOW_SECONDS,
                "heartbeat_seconds": HEARTBEAT_SECONDS}

    def heartbeat(self, visitor_id, account_id=None):
        with self._lock:
            now = self._clock()
            self._expire(now)
            if visitor_id not in self._visitors and len(self._visitors) >= self._capacity:
                raise PresenceUnavailable("Presence capacity reached")
            # Overwrite this browser's old identity on login, logout, or switch.
            self._visitors[visitor_id] = (now, account_id)
            return self._snapshot()

    def snapshot(self):
        with self._lock:
            self._expire(self._clock())
            return self._snapshot()


def visitor_from_payload(data):
    if not isinstance(data, dict) or set(data) != {"visitor_id"}:
        raise ValueError("Expected only visitor_id")
    visitor = data["visitor_id"]
    if not isinstance(visitor, str) or not _UUID.fullmatch(visitor):
        raise ValueError("visitor_id must be a UUID")
    return visitor.lower()


def dispatch(handler, method, data=None):
    """Called after the HTTP adapter's origin and bounded JSON-body checks."""
    visitor = visitor_from_payload(data) if method == "POST" else None
    try:
        tracker = handler.server.presence
        if method == "GET":
            payload = tracker.snapshot()
        else:
            from paperleet.social_api import services, session_token
            token = session_token(handler)
            # Guest traffic must not open or create an account database.
            user = services()[0].get_session(token) if token else None
            payload = tracker.heartbeat(visitor, user["id"] if user else None)
    except Exception:
        # An unavailable tracker/session store is not evidence of zero visitors.
        return handler.respond(503, {"error": "Online presence is temporarily unavailable."})
    return handler.respond(200, payload)
