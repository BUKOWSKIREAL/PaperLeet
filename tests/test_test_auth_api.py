"""Explicit test-account mode over real HTTP, with no email provider access."""
import hashlib
from http.client import HTTPConnection
from http.cookies import SimpleCookie
import json
import subprocess
import sys
import threading

import pytest

from paperleet import server as api_server, social_api
from paperleet.accounts import AccountService
from paperleet.catalog import root
from paperleet.community import CommunityService
from paperleet.config import validate_hosted_settings
from paperleet.hosting import judge_client


ORIGIN = "https://test-accounts.paperleet.test"


@pytest.fixture
def test_api(tmp_path, monkeypatch):
    settings = {"PAPERLEET_MODE": "hosted", "PAPERLEET_AUTH_MODE": "test",
        "PAPERLEET_PUBLIC_URL": ORIGIN, "PAPERLEET_DATA_DIR": str(tmp_path / "private-data"),
        "PAPERLEET_JUDGE_URL": "http://judge:8766", "PAPERLEET_JUDGE_TOKEN": "test-internal-key-" * 3}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY", "PAPERLEET_INVITE_EMAILS", "PAPERLEET_ADMIN_EMAILS", "PAPERLEET_TRUST_PROXY"):
        monkeypatch.delenv(key, raising=False)
    validate_hosted_settings()

    class NoMail:
        available = False
        def request_code(self, *args):
            pytest.fail("Test-account mode must not send verification email")
        def verify_code(self, *args):
            pytest.fail("Test-account mode must not contact an email provider")

    accounts = AccountService(tmp_path / "private-data" / "community.sqlite3", provider=NoMail())
    community = CommunityService(accounts.db_path, accounts.institutions)
    monkeypatch.setattr(social_api, "services", lambda: (accounts, community))
    social_api._requests.clear()
    executions = []

    def isolated_run(entry, code, layers=None):
        layers = layers or ["T0", "T1", "T2", "T3"]
        executions.append({"problem": entry["id"], "code": code})
        rows = [{"id": name, "nodeid": f"tests::{name}[{index}]", "layer": layer,
                 "passed": True, "outcome": "passed", "duration_s": 0, "why": "Test fixture", "message": "", "artifacts": {}}
                for (name, layer), count in judge_client.expected_tests(entry, layers).items() for index in range(count)]
        return {"problem": entry["id"], "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "created_at": "2026-10-13T00:00:00Z", "layers": layers, "tests": rows, "passed": True,
                "summary": {layer: [sum(row["layer"] == layer for row in rows)] * 2 for layer in ("T0", "T1", "T2", "T3")}}

    monkeypatch.setattr(judge_client, "run_hosted_problem", isolated_run)
    monkeypatch.setattr(api_server, "run_problem", lambda *args, **kwargs: pytest.fail("No host Python execution"))
    server = api_server.BoundedHTTPServer(("127.0.0.1", 0), api_server.JudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, data=None, cookie="", origin=ORIGIN):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = {"Content-Type": "application/json", "Host": "test-accounts.paperleet.test"}
        if origin is not None:
            headers["Origin"] = origin
        if cookie:
            headers["Cookie"] = cookie
        connection.request("GET" if data is None else "POST", "/api/" + path,
                           json.dumps(data) if data is not None else None, headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read()), dict(response.getheaders())
        connection.close()
        return result

    def provision(username, *extra):
        result = subprocess.run([sys.executable, "-m", "paperleet.test_accounts", "create", username,
            "--name", username.title(), *extra], cwd=root(), capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    request.accounts, request.provision, request.executions = accounts, provision, executions
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def login(api, credentials):
    status, payload, headers = api("account/test-login", credentials)
    assert status == 200, payload
    cookie = SimpleCookie(headers["Set-Cookie"])[social_api.COOKIE]
    assert cookie["secure"] and cookie["httponly"] and cookie["samesite"] == "Lax"
    return payload["user"], social_api.COOKIE + "=" + cookie.value


def test_cli_preprovisioned_login_needs_no_email_configuration_or_otp(test_api):
    capabilities = test_api("account/me")[1]["capabilities"]
    assert capabilities["auth_mode"] == "test" and capabilities["auth_available"] is True
    assert test_api("practice/workspace")[0] == 401
    credentials = test_api.provision("tester_alice")
    user, cookie = login(test_api, credentials)
    assert user["is_test_user"] is True and user["email"] == ""
    assert user["institution"] is None and user["institution_verified"] is False
    assert user["display_name"] and user["avatar_color"]
    assert test_api("account/me", cookie=cookie)[1]["user"]["id"] == user["id"]
    assert test_api("account/send-code", {"email": "alice@mit.edu"})[0] != 200
    assert test_api("account/verify", {"email": "alice@mit.edu", "code": "123456"})[0] != 200
    assert credentials["access_code"] not in json.dumps(test_api("account/me", cookie=cookie)[1])


def test_unknown_or_wrong_codes_do_not_create_users_or_sessions(test_api):
    credentials = test_api.provision("tester_alice")
    bad = test_api("account/test-login", {**credentials, "access_code": "not-the-real-code"})
    missing = test_api("account/test-login", {"username": "unprovisioned", "access_code": "not-the-real-code"})
    assert bad[0] == missing[0] == 401
    assert bad[1] == missing[1]
    assert "Set-Cookie" not in bad[2] and "Set-Cookie" not in missing[2]
    assert test_api("account/test-register", {"username": "new_account", "access_code": "choose-my-own"})[0] != 200
    with test_api.accounts._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0] == 0


def test_test_login_is_default_off_and_mode_flip_invalidates_its_sessions(test_api, monkeypatch):
    credentials = test_api.provision("tester_alice")
    _, cookie = login(test_api, credentials)
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "alice@mit.edu")
    monkeypatch.delenv("PAPERLEET_AUTH_MODE")
    assert test_api("account/me")[1]["capabilities"]["auth_mode"] == "institutional"
    assert test_api("account/me", cookie=cookie)[1]["user"] is None
    assert test_api("account/test-login", credentials)[0] != 200
    assert test_api("practice/workspace", cookie=cookie)[0] == 401
    assert test_api("community/posts", {"title": "Wrong mode", "body": "No access", "category": "papers"}, cookie)[0] == 401


def test_institutional_sessions_cannot_be_used_in_test_mode(test_api, monkeypatch):
    class VerifiedMail:
        available = True
        def verify_code(self, email, code):
            return {"id": "institutional-fixture", "email": email, "email_confirmed_at": "2026-10-13T00:00:00Z"}
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "institutional")
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "alice@mit.edu")
    test_api.accounts.provider = VerifiedMail()
    token, _ = test_api.accounts.verify_code("alice@mit.edu", "123456")
    cookie = social_api.COOKIE + "=" + token
    assert test_api("account/me", cookie=cookie)[1]["user"] is not None
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    assert test_api("account/me", cookie=cookie)[1]["user"] is None
    assert test_api("practice/workspace", cookie=cookie)[0] == 401


def test_two_test_accounts_keep_practice_private_and_community_authors_real(test_api):
    alice_credentials = test_api.provision("tester_alice")
    bob_credentials = test_api.provision("tester_bob")
    alice, cookie_a = login(test_api, alice_credentials)
    bob, cookie_b = login(test_api, bob_credentials)
    body = {"problem": "1", "code": "private alice draft", "expected_user_id": alice["id"], "expected_updated": "", "user_id": bob["id"]}
    assert test_api("practice/draft", body, cookie_a)[0] == 200
    assert test_api("test", {"problem": "1", "code": "private run", "expected_user_id": alice["id"]}, cookie_a)[0] == 200
    assert test_api("practice/workspace", cookie=cookie_b)[1]["workspace"] == {}
    assert test_api("practice/runs", cookie=cookie_b)[1]["runs"] == []
    assert test_api("practice/draft", body, cookie_b)[0] == 409
    assert len(test_api("practice/runs", cookie=cookie_a)[1]["runs"]) == 1
    status, post, _ = test_api("community/posts", {"title": "A test-user discussion", "body": "Public body", "category": "papers", "user_id": bob["id"]}, cookie_a)
    assert status == 201 and post["author"]["id"] == alice["id"]
    assert post["author"]["is_test_user"] is True
    assert "institution" not in post["author"] and "email" not in post["author"]
    assert test_api("community/posts/" + post["id"] + "/delete", {}, cookie_b)[0] == 403
    assert test_api("community/follow/" + alice["username"], {"active": True}, cookie_b)[0] == 200
    profile = test_api("community/profiles/" + alice["username"])[1]
    assert profile["is_test_user"] is True and profile["counts"]["followers"] == 1
    public = json.dumps(profile)
    assert alice_credentials["access_code"] not in public and "private alice draft" not in public
    assert test_api("account/profile", {"institution_id": "mit", "is_test_user": False}, cookie_a)[0] == 400


def test_code_reset_and_disable_revoke_existing_cookie(test_api):
    credentials = test_api.provision("tester_alice")
    _, cookie = login(test_api, credentials)
    command = [sys.executable, "-m", "paperleet.test_accounts"]
    reset = subprocess.run(command + ["reset-code", credentials["username"]], cwd=root(), capture_output=True, text=True, timeout=15)
    assert reset.returncode == 0, reset.stderr
    replacement = json.loads(reset.stdout)
    assert replacement["access_code"] != credentials["access_code"]
    assert test_api("account/me", cookie=cookie)[1]["user"] is None
    assert test_api("account/test-login", credentials)[0] == 401
    _, new_cookie = login(test_api, replacement)
    disabled = subprocess.run(command + ["disable", credentials["username"]], cwd=root(), capture_output=True, text=True, timeout=15)
    assert disabled.returncode == 0, disabled.stderr
    assert test_api("account/me", cookie=new_cookie)[1]["user"] is None
    assert test_api("account/test-login", replacement)[0] == 401


def test_test_login_keeps_hosted_csrf_boundary(test_api):
    credentials = test_api.provision("tester_alice")
    for origin in (None, "null", "https://attacker.test"):
        status, _, headers = test_api("account/test-login", credentials, origin=origin)
        assert status == 403 and "Set-Cookie" not in headers


@pytest.mark.parametrize("missing", ["PAPERLEET_PUBLIC_URL", "PAPERLEET_DATA_DIR", "PAPERLEET_JUDGE_URL", "PAPERLEET_JUDGE_TOKEN"])
def test_test_auth_does_not_remove_hosted_execution_requirements(test_api, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="Hosted mode requires"):
        validate_hosted_settings()
