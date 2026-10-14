"""Opt-in local Tunnel-stack smoke; no cloudflared or public ports are started."""
import json
import os
import shutil
import subprocess
import textwrap
import uuid

import pytest

from paperleet.catalog import root


pytestmark = pytest.mark.skipif(os.environ.get("PAPERLEET_TEST_TUNNEL") != "1",
    reason="Opt in with PAPERLEET_TEST_TUNNEL=1; requires Docker and a built worker image")


def test_tunnel_stack_serves_authenticated_practice_without_public_ports(tmp_path):
    """Exercise proxy, test login, persistent practice and the actual Docker judge."""
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_", "TUNNEL_"))}
    assert subprocess.run(["docker", "info"], env=environment, capture_output=True, timeout=15).returncode == 0, "Start Docker before running the Tunnel smoke"
    assert subprocess.run(["docker", "image", "inspect", "paperleet-worker:local"], env=environment,
                          capture_output=True, timeout=10).returncode == 0, "Build paperleet-worker:local before running the Tunnel smoke"
    project = "paperleet-tunnel-test-" + uuid.uuid4().hex[:12]
    settings = {
        "PAPERLEET_DOMAIN": "paperleet.example.test",
        "PAPERLEET_AUTH_MODE": "test",
        "PAPERLEET_JUDGE_TOKEN": "fake-test-only-judge-token-" + "0" * 40,
        "TUNNEL_TOKEN": "fake-test-only-token-never-connected",
    }
    env_file = tmp_path / "fake.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    compose = ["docker", "compose", "--project-name", project, "--env-file", str(env_file),
        "-f", str(root() / "deploy/compose.yaml"), "-f", str(root() / "deploy/compose.tunnel.yaml")]

    def command(*arguments, timeout=30, source=None):
        result = subprocess.run([*compose, *arguments], env=environment, input=source,
                                capture_output=True, text=True, timeout=timeout)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    document = json.loads(command("config", "--format", "json"))
    assert not any(service.get("ports") for service in document["services"].values())
    try:
        # Build application targets only. This must never start the tunnel or
        # load a developer's deploy/.env, local accounts, or default project.
        command("build", "api", "judge", "web", timeout=600)
        command("up", "-d", "--wait", "--wait-timeout", "120", "api", "judge", "web", timeout=150)
        identifiers = command("ps", "-q").split()
        assert len(identifiers) == 3
        inspected = subprocess.run(["docker", "inspect", *identifiers], env=environment,
                                   capture_output=True, text=True, timeout=15)
        assert inspected.returncode == 0, inspected.stderr
        containers = json.loads(inspected.stdout)
        assert {container["Config"]["Labels"]["com.docker.compose.service"] for container in containers} == {"api", "judge", "web"}
        assert all(not container["HostConfig"]["PortBindings"] for container in containers)
        source = textwrap.dedent('''\
            import json
            from http.cookies import SimpleCookie
            import subprocess
            import sys
            from urllib.error import HTTPError
            from urllib.request import ProxyHandler, Request, build_opener

            from paperleet.catalog import problem

            domain = "paperleet.example.test"
            origin = "https://" + domain
            opener = build_opener(ProxyHandler({}))

            def request(path, payload=None, *, cookie="", host=domain, supplied_origin=origin):
                headers = {"Host": host, "Origin": supplied_origin, "CF-Connecting-IP": "203.0.113.29"}
                if cookie:
                    headers["Cookie"] = cookie
                data = None
                if payload is not None:
                    headers["Content-Type"] = "application/json"
                    data = json.dumps(payload).encode()
                outgoing = Request("http://web:80" + path, data=data, headers=headers)
                try:
                    response = opener.open(outgoing, timeout=120)
                except HTTPError as error:
                    response = error
                with response:
                    raw = response.read()
                    body = json.loads(raw) if response.headers.get_content_type() == "application/json" else raw.decode()
                    return response.status, response.headers, body

            def no_cache(headers):
                assert "no-store" in headers.get("Cache-Control", ""), dict(headers)

            status, headers, page = request("/")
            assert status == 200 and "PaperLeet" in page and 'id="root"' in page
            status, headers, health = request("/api/health")
            assert status == 200 and health["ok"] and health["mode"] == "hosted" and health["auth_required"]
            assert health["judge"]["ready"] and health["judge"]["mode"] == "isolated"
            no_cache(headers)
            status, headers, account = request("/api/account/me")
            assert status == 200 and account["user"] is None
            assert account["capabilities"]["auth_mode"] == "test" and account["capabilities"]["test_login_available"]
            no_cache(headers)
            status, headers, denied = request("/api/test", {"problem": "rl/gae", "code": "pass"})
            assert status == 401 and "test account" in denied["error"]
            no_cache(headers)
            status, _, _ = request("/api/health", host="untrusted.example.test")
            assert status in (400, 403, 404, 421), status
            status, _, _ = request("/api/test", {"problem": "rl/gae", "code": "pass"}, supplied_origin="https://untrusted.example.test")
            assert status == 403

            operator = subprocess.run([sys.executable, "-m", "paperleet.test_accounts", "create", "tunneltester", "--name", "Tunnel Tester"], capture_output=True, text=True)
            assert operator.returncode == 0, operator.stderr
            credentials = json.loads(operator.stdout)
            status, headers, login = request("/api/account/test-login", credentials)
            assert status == 200 and login["user"]["is_test_user"]
            no_cache(headers)
            cookies = SimpleCookie()
            cookies.load(headers["Set-Cookie"])
            session = cookies["paperleet_session"]
            assert session["secure"] and session["httponly"] and session["samesite"] == "Lax" and session["path"] == "/api"
            # The external browser sends Secure cookies over HTTPS to
            # Cloudflare; this internal HTTP hop intentionally sends it by hand.
            cookie = "paperleet_session=" + session.value
            status, _, current = request("/api/account/me", cookie=cookie)
            assert status == 200 and current["user"]["id"] == login["user"]["id"]
            entry = problem("rl/gae")
            payload = {"problem": "rl/gae", "code": (entry["directory"] / "solution.py").read_text(),
                       "layers": ["T0", "T1", "T2", "T3"], "expected_user_id": current["user"]["id"]}
            status, headers, report = request("/api/test", payload, cookie=cookie)
            assert status == 200 and report["passed"] and report["run_id"], report
            no_cache(headers)
            status, headers, history = request("/api/practice/runs", cookie=cookie)
            assert status == 200 and len(history["runs"]) == 1
            assert history["runs"][0]["id"] == report["run_id"]
            no_cache(headers)
            print(json.dumps({"root": 200, "auth": "test", "cookie_secure": True,
                              "judge_passed": True, "saved_runs": 1, "wrong_host_rejected": True}))
        ''')
        result = json.loads(command("exec", "-T", "api", "python", "-", source=source, timeout=150))
        assert result["judge_passed"] and result["wrong_host_rejected"]
    finally:
        # The random project prefix scopes deletion to this test's containers,
        # networks and temporary data volume. Real account volumes are untouched.
        command("down", "--volumes", "--remove-orphans", timeout=60)
