"""Institutional mailbox verification, private sessions, and public profiles.

The authentication provider verifies mailbox control. An exact local domain
allowlist determines eligibility; neither suffixes nor profile fields grant it.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

import yaml


class AccountError(ValueError):
    pass


class AuthUnavailable(AccountError):
    pass


class AuthenticationError(AccountError):
    pass


class RateLimited(AccountError):
    pass


COUNTRIES = frozenset("""AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE
BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN
CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA
GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM
IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS
LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA
NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE
RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF
TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS
YE YT ZA ZM ZW""".split())
SESSION_TTL = 7 * 24 * 60 * 60
DEFAULT_INSTITUTIONS = Path(__file__).resolve().parents[1] / "institutions.yaml"


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _domain(value):
    if not isinstance(value, str):
        raise AccountError("A valid institutional email domain is required")
    domain = value.strip().lower()
    if len(domain) > 253 or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}", domain):
        raise AccountError("A valid institutional email domain is required")
    return domain


def _email(value):
    if not isinstance(value, str) or len(value) > 254:
        raise AccountError("A valid institutional email address is required")
    email = value.strip().lower()
    parts = email.split("@")
    if len(parts) != 2 or not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}", parts[0]):
        raise AccountError("A valid institutional email address is required")
    if parts[0].startswith(".") or parts[0].endswith(".") or ".." in parts[0]:
        raise AccountError("A valid institutional email address is required")
    return parts[0] + "@" + _domain(parts[1])


def load_institutions(path=None):
    document = yaml.safe_load(Path(path or DEFAULT_INSTITUTIONS).read_text(encoding="utf-8")) or {}
    records = document.get("institutions", [])
    result, domains = {}, set()
    if not isinstance(records, list):
        raise AccountError("Invalid institutions configuration")
    for item in records:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9-]{1,80}", identifier) or identifier in result:
            raise AccountError("Invalid or duplicate institution id")
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            raise AccountError("Institution name is required")
        allowed = [_domain(value) for value in item.get("domains", [])]
        if not allowed or domains.intersection(allowed):
            raise AccountError("Institution email domains must be unique")
        domains.update(allowed)
        result[identifier] = {**item, "domains": allowed}
    return result


def _institution_public(institution):
    return {key: institution[key] for key in ("id", "name", "name_en") if key in institution}


def public_user(conn, row, institutions=None):
    """Single safe projection used by profiles and every community author card."""
    if row is None:
        raise KeyError("Unknown user")
    institutions = load_institutions() if institutions is None else institutions
    result = {key: row[key] for key in ("id", "username", "display_name", "avatar_color", "bio", "created_at")}
    result["show_progress"] = bool(row["show_progress"])
    institution = institutions.get(row["institution_id"])
    if row["show_institution"] and institution:
        result["institution"] = _institution_public(institution)
    if row["show_country"] and row["country"]:
        result["country"] = row["country"]
    return result


class SupabaseEmailProvider:
    def __init__(self, url=None, key=None):
        self.url = (url if url is not None else os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key if key is not None else (os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY", ""))
        parsed = urlparse(self.url)
        self.available = bool(self.key and parsed.hostname and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"))))

    def _request(self, endpoint, payload):
        if not self.available:
            raise AuthUnavailable("Institutional email sign-in is not configured")
        request = Request(self.url + "/auth/v1/" + endpoint,
                          data=json.dumps(payload).encode(), method="POST",
                          headers={"apikey": self.key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read(128 * 1024 + 1)
                if len(body) > 128 * 1024:
                    raise AuthUnavailable("The email verification service returned an invalid response")
                return json.loads(body)
        except HTTPError as exc:
            if exc.code == 429:
                raise RateLimited("Too many email verification attempts; please try again later") from None
            if exc.code in (400, 401, 403, 422):
                raise AuthenticationError("Email verification failed; request a new code and try again") from None
            raise AuthUnavailable("The email verification service is temporarily unavailable") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise AuthUnavailable("The email verification service is temporarily unavailable") from None

    def request_code(self, email):
        self._request("otp", {"email": email, "create_user": True})

    def verify_code(self, email, code):
        response = self._request("verify", {"email": email, "token": code, "type": "email"})
        return response.get("user") if isinstance(response, dict) else None


class AccountService:
    def __init__(self, db_path, institutions_path=None, provider=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.institutions = load_institutions(institutions_path)
        self.provider = provider if provider is not None else SupabaseEmailProvider()
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL, avatar_color TEXT NOT NULL,
                    bio TEXT NOT NULL DEFAULT '', institution_id TEXT,
                    country TEXT NOT NULL DEFAULT '',
                    show_institution INTEGER NOT NULL DEFAULT 0,
                    show_country INTEGER NOT NULL DEFAULT 0,
                    show_progress INTEGER NOT NULL DEFAULT 1,
                    role TEXT NOT NULL DEFAULT 'member', status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS accounts_private (
                    user_id TEXT PRIMARY KEY REFERENCES users(id),
                    email TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL UNIQUE,
                    verified_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user ON account_sessions(user_id);
                CREATE TABLE IF NOT EXISTS account_rate_events (
                    bucket TEXT NOT NULL, occurred_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS account_rate_bucket ON account_rate_events(bucket, occurred_at);
                CREATE TABLE IF NOT EXISTS institution_requests (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, website TEXT NOT NULL,
                    email_domain TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def capabilities(self):
        available = bool(self.provider.available)
        result = {"auth_available": available, "institution_verification_required": True,
                  "invite_only": os.environ.get("PAPERLEET_MODE", "local") == "hosted",
                  "institutions": [{**_institution_public(item), "domains": item["domains"]}
                                   for item in self.institutions.values()]}
        if not available:
            result["reason"] = "Institutional email sign-in is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY."
        try:
            self._invites()
        except AuthUnavailable as exc:
            result.update(auth_available=False, reason=str(exc))
        return result

    def _invites(self):
        # Read on each operation, including session lookup: removing an invite
        # revokes access without relying on session expiry or another sign-in.
        if os.environ.get("PAPERLEET_MODE", "local") != "hosted":
            return None
        values = [value.strip() for value in os.environ.get("PAPERLEET_INVITE_EMAILS", "").split(",") if value.strip()]
        if not values:
            raise AuthUnavailable("Hosted sign-in requires a nonempty PAPERLEET_INVITE_EMAILS allowlist")
        if any("*" in value for value in values):
            raise AuthUnavailable("PAPERLEET_INVITE_EMAILS must list individual email addresses, not wildcards")
        try:
            return frozenset(_email(value) for value in values)
        except AccountError:
            raise AuthUnavailable("PAPERLEET_INVITE_EMAILS must contain valid individual email addresses") from None

    def _check_invitation(self, email):
        invited = self._invites()
        if invited is not None and email not in invited:
            raise AuthenticationError("This hosted beta is invite-only. Ask the owner to invite your institutional email.")

    def _ready(self):
        if not self.provider.available:
            raise AuthUnavailable("Institutional email sign-in is not configured")

    def _institution(self, email):
        domain = email.rsplit("@", 1)[1]
        for item in self.institutions.values():
            if domain in item["domains"]:
                return item
        raise AccountError("This institution's email domain is not approved yet. Submit an institution request for review.")

    def _limit(self, action, key, maximum, seconds, cooldown=0, global_maximum=100):
        bucket = action + ":" + hashlib.sha256(key.encode()).hexdigest()
        global_bucket = action + ":global"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            conn.execute("DELETE FROM account_rate_events WHERE occurred_at < ?", (now - 86400,))
            count, last = conn.execute("SELECT COUNT(*), MAX(occurred_at) FROM account_rate_events WHERE bucket=? AND occurred_at>?", (bucket, now - seconds)).fetchone()
            total = conn.execute("SELECT COUNT(*) FROM account_rate_events WHERE bucket=? AND occurred_at>?", (global_bucket, now - seconds)).fetchone()[0]
            if count >= maximum or (cooldown > 0 and last is not None and now - last < cooldown) or total >= global_maximum:
                raise RateLimited("Too many requests; please try again later")
            conn.executemany("INSERT INTO account_rate_events VALUES (?, ?)", [(bucket, now), (global_bucket, now)])

    def request_code(self, email):
        self._ready()
        email = _email(email)
        self._check_invitation(email)
        self._institution(email)
        self._limit("email", email, maximum=5, seconds=3600, cooldown=60, global_maximum=60)
        self.provider.request_code(email)
        return {"sent": True, "message": "Check your institutional email for a one-time sign-in code."}

    def verify_code(self, email, code):
        self._ready()
        email = _email(email)
        self._check_invitation(email)
        institution = self._institution(email)
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6,10}", code.strip()):
            raise AccountError("Enter the verification code from your email")
        self._limit("verify", email, maximum=10, seconds=900, global_maximum=500)
        provider_user = self.provider.verify_code(email, code.strip())
        if (not isinstance(provider_user, dict) or not provider_user.get("email_confirmed_at")
                or not isinstance(provider_user.get("id"), str) or not provider_user["id"]
                or provider_user.get("is_anonymous") is True):
            raise AuthenticationError("An email-verified account is required")
        try:
            verified_email = _email(provider_user.get("email"))
        except AccountError:
            raise AuthenticationError("The verified email does not match this sign-in request") from None
        if verified_email != email:
            raise AuthenticationError("The verified email does not match this sign-in request")
        now = _now()
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        admin_emails = {item.strip().lower() for item in os.environ.get("PAPERLEET_ADMIN_EMAILS", "").split(",") if item.strip()}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM accounts_private WHERE provider_id=? OR email=?", (provider_user["id"], email)).fetchall()
            if existing:
                if len(existing) != 1 or existing[0]["email"] != email or existing[0]["provider_id"] != provider_user["id"]:
                    raise AuthenticationError("This account's verified identity changed; contact an administrator")
                user_id = existing[0]["user_id"]
                status = conn.execute("SELECT status FROM users WHERE id=?", (user_id,)).fetchone()[0]
                if status != "active":
                    raise AuthenticationError("This account is not active")
                conn.execute("UPDATE accounts_private SET verified_at=? WHERE user_id=?", (now, user_id))
                conn.execute("UPDATE users SET institution_id=?, role=? WHERE id=?", (institution["id"], "admin" if email in admin_emails else "member", user_id))
            else:
                user_id = uuid.uuid4().hex
                conn.execute("INSERT INTO users (id,username,display_name,avatar_color,institution_id,role,created_at) VALUES (?,?,?,?,?,?,?)",
                             (user_id, "member_" + user_id[:12], "Researcher " + user_id[:6], "#" + user_id[:6], institution["id"], "admin" if email in admin_emails else "member", now))
                conn.execute("INSERT INTO accounts_private VALUES (?,?,?,?)", (user_id, email, provider_user["id"], now))
            conn.execute("DELETE FROM account_sessions WHERE expires_at <= ?", (time.time(),))
            conn.execute("INSERT INTO account_sessions VALUES (?,?,?,?)", (digest, user_id, time.time() + SESSION_TTL, now))
        return token, self.get_user(user_id)

    def get_user(self, user_id):
        with self._connect() as conn:
            row = conn.execute("SELECT u.*, a.email FROM users u JOIN accounts_private a ON a.user_id=u.id WHERE u.id=?", (user_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown user")
            result = public_user(conn, row, self.institutions)
            result.update({key: row[key] for key in ("email", "role", "country")})
            result.update({key: bool(row[key]) for key in ("show_institution", "show_country", "show_progress")})
            institution = self.institutions.get(row["institution_id"])
            result["institution"] = _institution_public(institution) if institution else None
            result["institution_verified"] = bool(institution and row["email"].rsplit("@", 1)[1] in institution["domains"])
            return result

    def public_user(self, user_id, viewer_id=None):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id=? AND status='active'", (user_id,)).fetchone()
            return public_user(conn, row, self.institutions)

    def get_session(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("SELECT u.id FROM account_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>? AND u.status='active'", (digest, time.time())).fetchone()
        if row is None:
            return None
        user = self.get_user(row["id"])
        try:
            self._check_invitation(user["email"])
        except AccountError:
            return None
        return user if user["institution_verified"] else None

    def logout(self, token):
        if isinstance(token, str):
            with self._connect() as conn:
                conn.execute("DELETE FROM account_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def update_profile(self, user_id, payload):
        allowed = {"username", "display_name", "avatar_color", "bio", "country", "show_institution", "show_country", "show_progress"}
        if not isinstance(payload, dict) or not payload or payload.keys() - allowed:
            raise AccountError("Only editable profile fields may be changed")
        changes = {}
        for field, value in payload.items():
            if field.startswith("show_"):
                if not isinstance(value, bool):
                    raise AccountError("Privacy settings must be true or false")
                changes[field] = int(value)
                continue
            if not isinstance(value, str):
                raise AccountError("Profile text fields must be strings")
            value = value.strip()
            if field == "username":
                value = value.lower()
                if not re.fullmatch(r"[a-z0-9_]{3,24}", value):
                    raise AccountError("Username must contain 3–24 letters, numbers, or underscores")
            elif field == "display_name" and not 1 <= len(value) <= 40:
                raise AccountError("Display name must contain 1–40 characters and is always public")
            elif field == "avatar_color" and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                raise AccountError("Choose a valid avatar color; your avatar is always public")
            elif field == "bio" and len(value) > 500:
                raise AccountError("Bio must be at most 500 characters")
            elif field == "country":
                value = value.upper()
                if value and value not in COUNTRIES:
                    raise AccountError("Choose a valid two-letter country or region code")
            changes[field] = value
        with self._connect() as conn:
            if not conn.execute("SELECT id FROM users WHERE id=? AND status='active'", (user_id,)).fetchone():
                raise KeyError("Unknown user")
            try:
                columns = ", ".join(field + "=?" for field in changes)
                conn.execute("UPDATE users SET " + columns + " WHERE id=?", (*changes.values(), user_id))
            except sqlite3.IntegrityError:
                raise AccountError("This username is already taken") from None
        return self.get_user(user_id)

    def request_institution(self, payload):
        if not isinstance(payload, dict):
            raise AccountError("Expected institution name, website, and email domain")
        name, website = payload.get("name"), payload.get("website")
        domain = _domain(payload.get("email_domain"))
        if not isinstance(name, str) or not 2 <= len(name.strip()) <= 160:
            raise AccountError("Institution name must contain 2–160 characters")
        if not isinstance(website, str) or len(website) > 500:
            raise AccountError("An official institution website is required")
        parsed = urlparse(website)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise AccountError("An official HTTPS institution website is required")
        if any(domain in item["domains"] for item in self.institutions.values()):
            raise AccountError("This domain is already approved; verify your institutional email")
        self._limit("institution", domain, maximum=3, seconds=86400, global_maximum=30)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id,status FROM institution_requests WHERE email_domain=?", (domain,)).fetchone()
            if row:
                return {"id": row["id"], "status": row["status"], "email_domain": domain}
            identifier = uuid.uuid4().hex
            conn.execute("INSERT INTO institution_requests (id,name,website,email_domain,created_at) VALUES (?,?,?,?,?)", (identifier, name.strip(), website, domain, _now()))
        return {"id": identifier, "status": "pending", "email_domain": domain}


def main(argv=None):
    """Read-only maintenance entry point; approvals are reviewed config edits."""
    import argparse
    from paperleet.config import data_directory
    parser = argparse.ArgumentParser(description="Inspect PaperLeet institutional account requests")
    parser.add_argument("command", choices=["institution-requests"])
    parser.add_argument("--database", type=Path, default=data_directory() / "community.sqlite3")
    args = parser.parse_args(argv)
    if not args.database.exists():
        print("[]")
        return
    institutions = load_institutions()
    approved_domains = {domain for item in institutions.values() for domain in item["domains"]}
    conn = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id,name,website,email_domain,status,created_at FROM institution_requests ORDER BY created_at DESC").fetchall()
        # Requests retain their intake status; exact config domains determine approval.
        result = [{**dict(row), "domain_approved": row["email_domain"] in approved_domains} for row in rows]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
