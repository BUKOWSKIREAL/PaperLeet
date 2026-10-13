import json
import os
import sqlite3
import subprocess
import sys

import pytest

from paperleet.accounts import AccountError, AccountService, AuthenticationError, AuthUnavailable, RateLimited
from paperleet.community import CommunityService
from paperleet.practice import PracticeService


class NoMailProvider:
    available = False

    def request_code(self, email):
        raise AssertionError("Test accounts must never send mail")

    def verify_code(self, email, code):
        raise AssertionError("Test accounts must never call the institutional provider")


@pytest.fixture
def test_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    monkeypatch.setenv("PAPERLEET_MODE", "hosted")
    monkeypatch.setenv("PAPERLEET_DATA_DIR", str(tmp_path / "data"))
    for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY", "PAPERLEET_INVITE_EMAILS", "PAPERLEET_ADMIN_EMAILS"):
        monkeypatch.delenv(key, raising=False)
    return AccountService(tmp_path / "data/community.sqlite3", provider=NoMailProvider())


def provision(accounts, login="tester1", **kwargs):
    credentials = accounts.create_test_account(login, **kwargs)
    token, user = accounts.test_login(credentials["username"], credentials["access_code"])
    return credentials, token, user


def test_test_mode_works_without_email_provider_and_never_claims_verification(test_accounts):
    accounts = test_accounts
    capabilities = accounts.capabilities()
    assert capabilities["auth_mode"] == "test" and capabilities["auth_available"] is True
    assert capabilities["test_login_available"] is True
    assert capabilities["institution_verification_required"] is False and capabilities["institutions"] == []
    assert capabilities["invite_only"] is False
    credentials, token, user = provision(accounts, display_name="Test Researcher")
    assert user["display_name"] == "Test Researcher"
    assert user["email"] == "" and user["institution"] is None
    assert user["institution_verified"] is False and user["is_test_user"] is True
    assert user["login_username"] == "tester1"
    assert accounts.get_session(token)["id"] == user["id"]
    public = accounts.public_user(user["id"])
    assert public["is_test_user"] is True
    assert "email" not in public and "institution" not in public and "login_username" not in public
    accounts.update_profile(user["id"], {"show_institution": True})
    assert "institution" not in accounts.public_user(user["id"])
    with accounts._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM accounts_private").fetchone()[0] == 0
    for action in (lambda: accounts.request_code("alice@mit.edu"),
                   lambda: accounts.verify_code("alice@mit.edu", "123456"),
                   lambda: accounts.request_institution({"name": "Institute"})):
        with pytest.raises(AuthUnavailable, match="test-account"):
            action()
    assert credentials["access_code"] not in json.dumps(user)


def test_codes_are_individual_salted_hashes_and_list_never_discloses_them(test_accounts):
    accounts = test_accounts
    first = accounts.create_test_account("first")
    second = accounts.create_test_account("second")
    assert len(first["access_code"]) >= 32 and first["access_code"] != second["access_code"]
    with accounts._connect() as conn:
        rows = conn.execute("SELECT * FROM test_accounts ORDER BY login_username").fetchall()
        assert rows[0]["code_salt"] != rows[1]["code_salt"]
        assert rows[0]["code_hash"] != rows[1]["code_hash"]
        assert rows[0]["code_iterations"] >= 600000
        for row in rows:
            assert first["access_code"] not in tuple(row) and second["access_code"] not in tuple(row)
    listing = accounts.list_test_accounts()
    assert [item["username"] for item in listing] == ["first", "second"]
    assert "access_code" not in json.dumps(listing) and "code_hash" not in json.dumps(listing)
    with pytest.raises(AuthenticationError):
        accounts.test_login("first", second["access_code"])
    with pytest.raises(AccountError, match="already exists"):
        accounts.create_test_account("FIRST")


def test_public_username_changes_do_not_change_stable_login(test_accounts):
    accounts = test_accounts
    credentials, _, user = provision(accounts)
    updated = accounts.update_profile(user["id"], {"username": "new_public_name", "display_name": "New Name"})
    assert updated["username"] == "new_public_name" and updated["login_username"] == "tester1"
    assert accounts.test_login("TESTER1", credentials["access_code"])[1]["id"] == user["id"]
    with pytest.raises(AuthenticationError):
        accounts.test_login("new_public_name", credentials["access_code"])
    for spoof in ({"is_test_user": False}, {"institution_id": "mit"}, {"institution_verified": True}, {"login_username": "newlogin"}):
        with pytest.raises(AccountError):
            accounts.update_profile(user["id"], spoof)


def test_rotating_code_revokes_sessions_and_disabling_blocks_all_access(test_accounts):
    accounts = test_accounts
    old, token, user = provision(accounts)
    rotated = accounts.reset_test_access_code(old["username"])
    assert rotated["access_code"] != old["access_code"]
    assert accounts.get_session(token) is None
    with pytest.raises(AuthenticationError):
        accounts.test_login(old["username"], old["access_code"])
    current_token, current = accounts.test_login(rotated["username"], rotated["access_code"])
    assert current["id"] == user["id"]
    assert accounts.disable_test_account("tester1")["status"] == "disabled"
    assert accounts.get_session(current_token) is None
    with pytest.raises(AuthenticationError):
        accounts.test_login(rotated["username"], rotated["access_code"])
    again = accounts.reset_test_access_code("tester1")
    with pytest.raises(AuthenticationError):
        accounts.test_login("tester1", again["access_code"])
    assert accounts.list_test_accounts()[0]["status"] == "disabled"


def test_mode_switches_invalidate_sessions_without_turning_test_users_into_verified_users(test_accounts, monkeypatch):
    accounts = test_accounts
    credentials, token, user = provision(accounts)
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "institutional")
    assert accounts.get_session(token) is None
    for action in (lambda: accounts.test_login("tester1", credentials["access_code"]),
                   lambda: accounts.create_test_account("another"),
                   lambda: accounts.reset_test_access_code("tester1"),
                   lambda: accounts.disable_test_account("tester1"),
                   lambda: accounts.list_test_accounts()):
        with pytest.raises(AuthUnavailable, match="AUTH_MODE=test"):
            action()
    assert accounts.get_user(user["id"])["institution_verified"] is False
    assert accounts.capabilities()["test_login_available"] is False


def test_institutional_session_is_not_accepted_in_test_mode(test_accounts, monkeypatch):
    class InstitutionalProvider:
        available = True

        def verify_code(self, email, code):
            return {"id": "verified-person", "email": email, "email_confirmed_at": "2026-01-01T00:00:00Z"}

    accounts = test_accounts
    accounts.provider = InstitutionalProvider()
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "institutional")
    monkeypatch.setenv("PAPERLEET_INVITE_EMAILS", "person@mit.edu")
    token, user = accounts.verify_code("person@mit.edu", "123456")
    assert user["is_test_user"] is False and accounts.get_session(token)
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    assert accounts.get_session(token) is None


def test_login_is_rate_limited_before_hashing(test_accounts, monkeypatch):
    credentials = test_accounts.create_test_account("tester1")
    calls = []
    original_hash = test_accounts._test_hash

    def counted(*args):
        calls.append(True)
        return original_hash(*args)

    monkeypatch.setattr(test_accounts, "_test_hash", counted)
    for _ in range(10):
        with pytest.raises(AuthenticationError):
            test_accounts.test_login("tester1", "wrong code")
    assert len(calls) == 10
    with pytest.raises(RateLimited):
        test_accounts.test_login("tester1", credentials["access_code"])
    assert len(calls) == 10


@pytest.mark.parametrize("action", ["reset", "disable"])
def test_concurrent_credential_change_cannot_issue_session_from_old_code(test_accounts, monkeypatch, action):
    accounts = test_accounts
    credentials = accounts.create_test_account("tester1")
    original_hash = accounts._test_hash
    changed = False

    def change_during_check(*args):
        nonlocal changed
        digest = original_hash(*args)
        if not changed:
            changed = True
            if action == "reset":
                accounts.reset_test_access_code("tester1")
            else:
                accounts.disable_test_account("tester1")
        return digest

    monkeypatch.setattr(accounts, "_test_hash", change_during_check)
    with pytest.raises(AuthenticationError):
        accounts.test_login("tester1", credentials["access_code"])
    with accounts._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0] == 0


def test_individual_test_users_keep_private_practice_and_public_authorship_separate(test_accounts):
    accounts = test_accounts
    _, _, first = provision(accounts, "alice")
    _, _, second = provision(accounts, "bobby")
    practice = PracticeService(accounts.db_path)
    practice.save_draft(first["id"], "rl/gae", "alice implementation")
    assert practice.workspace(second["id"]) == {}
    community = CommunityService(accounts.db_path, accounts.institutions)
    post = community.create_post(first["id"], {"title": "A test discussion", "body": "Core implementation", "category": "papers"})
    assert post["author"]["is_test_user"] is True and post["author"]["id"] == first["id"]
    assert "institution" not in post["author"]
    assert "login_username" not in json.dumps(community.profile("alice"))


def test_admin_role_requires_explicit_operator_provisioning(test_accounts):
    _, _, member = provision(test_accounts, "member")
    _, _, admin = provision(test_accounts, "admin", admin=True)
    assert member["role"] == "member" and admin["role"] == "admin"
    with pytest.raises(AccountError):
        test_accounts.update_profile(member["id"], {"role": "admin"})


def test_users_schema_migration_preserves_existing_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    database = tmp_path / "old.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute("""CREATE TABLE users (id TEXT PRIMARY KEY,username TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,avatar_color TEXT NOT NULL,bio TEXT DEFAULT '',institution_id TEXT,
            country TEXT DEFAULT '',show_institution INTEGER DEFAULT 0,show_country INTEGER DEFAULT 0,
            show_progress INTEGER DEFAULT 1,role TEXT DEFAULT 'member',status TEXT DEFAULT 'active',created_at TEXT NOT NULL)""")
        conn.execute("INSERT INTO users (id,username,display_name,avatar_color,created_at) VALUES ('old','existing','Existing','#123456','now')")
    accounts = AccountService(database, provider=NoMailProvider())
    assert accounts.public_user("old")["is_test_user"] is False
    accounts.create_test_account("newuser")
    with accounts._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2


def test_operator_cli_returns_code_once_and_requires_explicit_mode(tmp_path):
    env = dict(os.environ, PAPERLEET_AUTH_MODE="test", PAPERLEET_DATA_DIR=str(tmp_path / "data"))

    def command(*arguments, mode="test"):
        return subprocess.run([sys.executable, "-m", "paperleet.test_accounts", *arguments],
                              env={**env, "PAPERLEET_AUTH_MODE": mode}, capture_output=True, text=True)

    disabled = command("create", "tester1", mode="institutional")
    assert disabled.returncode == 1 and disabled.stdout == ""
    assert not (tmp_path / "data/community.sqlite3").exists()
    created = command("create", "tester1", "--name", "Tester One", "--admin")
    assert created.returncode == 0, created.stderr
    credentials = json.loads(created.stdout)
    assert set(credentials) == {"username", "access_code"}
    listed = command("list")
    rows = json.loads(listed.stdout)
    assert rows[0]["display_name"] == "Tester One" and rows[0]["role"] == "admin"
    assert credentials["access_code"] not in listed.stdout
    rotated = json.loads(command("reset-code", "tester1").stdout)
    assert rotated["access_code"] != credentials["access_code"]
    assert json.loads(command("disable", "tester1").stdout)["status"] == "disabled"
    assert "access_code" not in command("list").stdout
