"""Presence uses real HTTP/session boundaries without invoking the judge."""
from http.client import HTTPConnection
import json
import threading
from uuid import uuid4

import pytest

from paperleet import server as api_server, social_api
from paperleet.presence import PresenceTracker


PUBLIC_ORIGIN = "https://paperleet.test"


@pytest.fixture
def presence_api(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    monkeypatch.setenv("PAPERLEET_PUBLIC_URL", PUBLIC_ORIGIN)
    monkeypatch.setenv("PAPERLEET_DATA_DIR", str(tmp_path / "data"))

    def forbidden_judge(*args, **kwargs):
        raise AssertionError("Presence must not invoke the judge")

    monkeypatch.setattr(api_server, "run_problem", forbidden_judge)
    server = api_server.BoundedHTTPServer(("127.0.0.1", 0), api_server.JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path="presence", body=None, *, cookie="", origin=PUBLIC_ORIGIN,
                host="paperleet.test", content_type="application/json", raw=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        headers = {"Content-Type": content_type, "Host": host}
        if origin is not None:
            headers["Origin"] = origin
        if cookie:
            headers["Cookie"] = cookie
        encoded = raw if raw is not None else json.dumps(body) if body is not None else None
        connection.request("POST" if encoded is not None else "GET", "/api/" + path, encoded, headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read()), dict(response.getheaders())
        connection.close()
        return result

    request.server = server
    request.database = tmp_path / "data" / "community.sqlite3"
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    social_api._services.pop(request.database, None)


def test_guest_presence_is_public_aggregate_only_and_does_not_create_database(presence_api, monkeypatch):
    def forbidden_services():
        raise AssertionError("Guests must not initialize the account service")

    monkeypatch.setattr(social_api, "services", forbidden_services)
    api = presence_api
    assert api(origin=None)[1] == {"online": 0, "window_seconds": 90, "heartbeat_seconds": 25}
    visitor = str(uuid4())
    status, payload, headers = api("presence/heartbeat", {"visitor_id": visitor})
    assert status == 200
    assert payload == {"online": 1, "window_seconds": 90, "heartbeat_seconds": 25}
    assert headers["Cache-Control"] == "no-store"
    assert visitor not in json.dumps(payload)
    assert api("presence/heartbeat", {"visitor_id": visitor})[1]["online"] == 1
    assert api()[1]["online"] == 1
    assert not api.database.exists()


def test_real_sessions_deduplicate_and_logout_replaces_identity(presence_api):
    api = presence_api
    accounts = social_api.services()[0]
    credentials = accounts.create_test_account("presence_tester")
    tokens = [accounts.test_login(credentials["username"], credentials["access_code"])[0] for _ in range(2)]
    cookies = [social_api.COOKIE + "=" + token for token in tokens]
    visitors = [str(uuid4()), str(uuid4())]
    assert api("presence/heartbeat", {"visitor_id": visitors[0]})[1]["online"] == 1
    assert api("presence/heartbeat", {"visitor_id": visitors[0]}, cookie=cookies[0])[1]["online"] == 1
    assert api("presence/heartbeat", {"visitor_id": visitors[1]}, cookie=cookies[1])[1]["online"] == 1
    accounts.logout(tokens[0])
    assert api("presence/heartbeat", {"visitor_id": visitors[0]}, cookie=cookies[0])[1]["online"] == 2
    assert api("presence/heartbeat", {"visitor_id": visitors[0]}, cookie=cookies[1])[1]["online"] == 1


@pytest.mark.parametrize("origin", [None, "https://attacker.example", "http://paperleet.test", "null"])
def test_heartbeat_requires_same_origin(presence_api, origin):
    assert presence_api("presence/heartbeat", {"visitor_id": str(uuid4())}, origin=origin)[0] == 403
    assert presence_api()[1]["online"] == 0


def test_presence_rejects_wrong_host_and_cross_origin_reads(presence_api):
    assert presence_api(host="localhost")[0] == 403
    assert presence_api(origin="https://attacker.example")[0] == 403


@pytest.mark.parametrize("body", [{}, {"visitor_id": "bad"}, {"visitor_id": str(uuid4()), "user_id": "victim"},
                                   {"visitor_id": str(uuid4()), "online": 100}, []])
def test_invalid_or_spoofed_heartbeat_does_not_change_count(presence_api, body):
    assert presence_api("presence/heartbeat", body)[0] == 400
    assert presence_api()[1]["online"] == 0


def test_presence_has_small_body_limit_and_requires_json(presence_api):
    assert presence_api("presence/heartbeat", {"visitor_id": "a" * 600})[0] == 413
    assert presence_api("presence/heartbeat", {"visitor_id": str(uuid4())}, content_type="text/plain")[0] == 415
    assert presence_api("presence/heartbeat", raw="{")[0] == 400
    assert presence_api()[1]["online"] == 0


def test_capacity_and_account_store_failures_return_unavailable_not_zero(presence_api, monkeypatch):
    api = presence_api
    api.server.presence = PresenceTracker(capacity=1)
    assert api("presence/heartbeat", {"visitor_id": str(uuid4())})[0] == 200
    status, payload, _ = api("presence/heartbeat", {"visitor_id": str(uuid4())})
    assert status == 503 and "online" not in payload

    def unavailable():
        raise OSError("private database path should not leak")

    monkeypatch.setattr(social_api, "services", unavailable)
    status, payload, _ = api("presence/heartbeat", {"visitor_id": str(uuid4())}, cookie=social_api.COOKIE + "=" + "a" * 64)
    assert status == 503 and "online" not in payload
    assert "private" not in json.dumps(payload)
    assert api()[1]["online"] == 1


def test_http_snapshot_expires_heartbeats_without_counting_readers(presence_api):
    now = [0.0]
    presence_api.server.presence = PresenceTracker(clock=lambda: now[0])
    assert presence_api("presence/heartbeat", {"visitor_id": str(uuid4())})[1]["online"] == 1
    now[0] = 90
    assert presence_api()[1]["online"] == 0
    assert presence_api()[1]["online"] == 0
