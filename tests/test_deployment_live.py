"""Opt-in real Compose smoke: PAPERLEET_TEST_DEPLOYMENT=1 pytest this file.

Build all four deployment images first. The test uses its own loopback HTTPS
port and temporary project/volumes. Institutional mode stubs mail identity;
test mode provisions real test accounts and uses their HTTP login endpoint.
No external credentials or outgoing email are used. Only test resources are deleted.
"""
import io
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import subprocess
import tarfile
import urllib.error
import urllib.request
import uuid

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("PAPERLEET_TEST_DEPLOYMENT") != "1",
                                reason="Opt-in: builds must exist and Docker must be running")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("auth_mode", ["institutional", "test"])
def test_https_judging_persistence_and_backup_restore(tmp_path, auth_mode):
    clean_env = {key: value for key, value in os.environ.items()
                 if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_"))}
    project = "paperleet-smoke-" + uuid.uuid4().hex[:10]
    restored_volume = project + "-restore"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    origin = f"https://localhost:{port}"
    env = tmp_path / "settings.env"
    auth_settings = ("SUPABASE_URL=https://test.supabase.co\n"
                     "SUPABASE_PUBLISHABLE_KEY=not-a-real-key\n"
                     "PAPERLEET_INVITE_EMAILS=alice@mit.edu,bob@mit.edu\n"
                     "PAPERLEET_ADMIN_EMAILS=alice@mit.edu\n") if auth_mode == "institutional" else (
                     "SUPABASE_URL=\nSUPABASE_PUBLISHABLE_KEY=\nPAPERLEET_INVITE_EMAILS=\nPAPERLEET_ADMIN_EMAILS=\n")
    env.write_text("PAPERLEET_DOMAIN=localhost\nPAPERLEET_AUTH_MODE=" + auth_mode + "\n" + auth_settings +
                   "PAPERLEET_JUDGE_TOKEN=" + secrets.token_hex(32) + "\n")
    env.chmod(0o600)
    override = tmp_path / "compose.override.yaml"
    override.write_text('services:\n  web:\n    ports: !override\n'
                        f'      - "127.0.0.1:{port}:443"\n'
                        f'  api:\n    environment:\n      PAPERLEET_PUBLIC_URL: {origin}\n')
    command = ["docker", "compose", "--env-file", str(env), "-p", project,
               "-f", str(ROOT / "deploy/compose.yaml"), "-f", str(override)]

    def compose(*args, data=None, timeout=180):
        return subprocess.run(command + list(args), input=data, capture_output=True,
                              check=True, timeout=timeout, env=clean_env).stdout

    try:
        compose("up", "-d", "--wait", "--wait-timeout", "90")
        ca = compose("exec", "-T", "web", "cat", "/data/caddy/pki/authorities/local/root.crt").decode()
        context = ssl.create_default_context(cadata=ca)

        def api(path, data=None, person=None, include_headers=False):
            headers = {"Origin": origin, "Content-Type": "application/json"}
            if person:
                headers["Cookie"] = "paperleet_session=" + person["cookie"]
            request = urllib.request.Request(origin + "/api/" + path, headers=headers,
                                             data=json.dumps(data).encode() if data is not None else None)
            try:
                with urllib.request.urlopen(request, context=context, timeout=160) as response:
                    result = response.status, json.load(response)
                    return (*result, dict(response.headers)) if include_headers else result
            except urllib.error.HTTPError as exc:
                result = exc.code, json.load(exc)
                return (*result, dict(exc.headers)) if include_headers else result

        assert api("health")[1]["judge"]["ready"]
        with urllib.request.urlopen(origin, context=context) as response:
            assert b'<div id="root">' in response.read()
        assert api("test", {"problem": "1", "code": "pass"})[0] == 401
        if auth_mode == "institutional":
            seed = """import json
from paperleet.social_api import services
class TestMail:
 available=True
 def verify_code(self,email,code):
  return {'id':'smoke-'+email,'email':email,'email_confirmed_at':'2026-10-13T00:00:00Z'}
a,_=services(); a.provider=TestMail(); people=[]
for email in ('alice@mit.edu','bob@mit.edu'):
 token,user=a.verify_code(email,'123456'); people.append({'cookie':token,'id':user['id']})
print(json.dumps(people))
"""
            alice, bob = json.loads(compose("exec", "-T", "api", "python", "-c", seed))
        else:
            assert api("account/me")[1]["capabilities"]["auth_mode"] == "test"
            assert api("account/send-code", {"email": "alice@mit.edu"})[0] != 200
            people = []
            for login_name in ("tester_alice", "tester_bob"):
                credentials = json.loads(compose("exec", "-T", "api", "python", "-m", "paperleet.test_accounts",
                                                "create", login_name, "--name", login_name.title()))
                status, payload, headers = api("account/test-login", credentials, include_headers=True)
                assert status == 200, payload
                user = payload["user"]
                assert user["is_test_user"] is True and user["institution"] is None and user["email"] == ""
                cookie = SimpleCookie(headers["Set-Cookie"])["paperleet_session"]
                assert cookie["secure"] and cookie["httponly"]
                people.append({"cookie": cookie.value, "id": user["id"]})
                public = api("community/profiles/" + user["username"])[1]
                assert public["is_test_user"] is True and "institution" not in public
            alice, bob = people
        code = (ROOT / "problems/rl/gae/solution.py").read_text()
        status, draft = api("practice/draft", {"problem": "1", "code": code,
                            "expected_user_id": alice["id"], "expected_updated": ""}, alice)
        assert status == 200, draft
        status, report = api("test", {"problem": "1", "code": code, "expected_user_id": alice["id"]}, alice)
        assert status == 200 and report["passed"], report
        assert api("practice/workspace", person=bob)[1]["workspace"] == {}
        assert api("practice/draft", {"problem": "1", "code": "pass", "expected_user_id": alice["id"]}, bob)[0] == 409
        assert len(api("practice/runs", person=alice)[1]["runs"]) == 1
        compose("restart", "api")
        compose("up", "-d", "--wait", "--wait-timeout", "90")
        saved = api("practice/workspace", person=alice)[1]["workspace"]["rl/gae"]
        assert saved["code"] == code and saved["acceptedReport"]["passed"]
        backup = compose("exec", "-T", "api", "python", "-m", "paperleet.backup", "backup", "--stdout")
        with tarfile.open(fileobj=io.BytesIO(backup), mode="r:gz") as archive:
            assert any(name.endswith("community.sqlite3") for name in archive.getnames())
        subprocess.run(["docker", "volume", "create", restored_volume], check=True, capture_output=True)
        restore_command = ["docker", "run", "--rm", "-i", "--read-only", "--network=none",
                           "-v", restored_volume + ":/var/lib/paperleet",
                           "-e", "PAPERLEET_DATA_DIR=/var/lib/paperleet/data", "paperleet-api:local"]
        subprocess.run(restore_command + ["python", "-m", "paperleet.backup", "restore", "--stdin"],
                       input=backup, check=True, capture_output=True, timeout=60)
        verify = """import sqlite3
c=sqlite3.connect('/var/lib/paperleet/data/community.sqlite3')
assert c.execute('SELECT COUNT(*) FROM users').fetchone()[0]==2
assert c.execute('SELECT COUNT(*) FROM practice_runs WHERE accepted=1').fetchone()[0]==1
assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
"""
        subprocess.run(restore_command + ["python", "-c", verify], check=True, capture_output=True, timeout=30)
        refused = subprocess.run(restore_command + ["python", "-m", "paperleet.backup", "restore", "--stdin"],
                                 input=backup, capture_output=True, timeout=60)
        assert refused.returncode != 0, "Restore must not replace existing user data"
    finally:
        subprocess.run(command + ["down", "--volumes"], capture_output=True, timeout=60, env=clean_env)
        subprocess.run(["docker", "volume", "rm", restored_volume], capture_output=True, timeout=30)
