"""Deployment boundaries: configuration, actual Compose parsing and proxy syntax."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from paperleet.catalog import root
from paperleet.config import validate_hosted_settings


DEPLOY = root() / "deploy"
FAKE_SETTINGS = {
    "PAPERLEET_DOMAIN": "paperleet.example.test",
    "PAPERLEET_MODE": "hosted",
    "PAPERLEET_AUTH_MODE": "institutional",
    "PAPERLEET_PUBLIC_URL": "https://paperleet.example.test",
    "PAPERLEET_DATA_DIR": "/var/lib/paperleet",
    "PAPERLEET_INVITE_EMAILS": "tester@mit.edu",
    "PAPERLEET_ADMIN_EMAILS": "tester@mit.edu",
    "SUPABASE_URL": "https://example-test.supabase.co",
    "SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test_only_not_a_credential",
    "PAPERLEET_JUDGE_URL": "http://judge:8766",
    "PAPERLEET_JUDGE_TOKEN": "test-only-internal-token-" + "0" * 40,
}


def compose_document():
    return yaml.safe_load((DEPLOY / "compose.yaml").read_text())


def test_only_proxy_publishes_ports_and_only_manager_gets_docker_socket():
    document = compose_document()
    services = document["services"]
    assert set(services) == {"web", "api", "judge"}
    assert set(services["web"]["ports"]) == {"80:80", "443:443"}
    assert not services["api"].get("ports") and not services["judge"].get("ports")
    mounts = {name: service.get("volumes", []) for name, service in services.items()}
    assert not any("docker.sock" in volume for name, volumes in mounts.items() if name != "judge" for volume in volumes)
    assert mounts["judge"] == ["/var/run/docker.sock:/var/run/docker.sock"]
    assert services["judge"]["networks"] == ["judge_internal"]
    assert document["networks"]["judge_internal"]["internal"] is True
    assert "app_data:/var/lib/paperleet" in mounts["api"]
    for name in ("api", "judge"):
        assert services[name]["read_only"] is True
        assert services[name]["cap_drop"] == ["ALL"]
        assert services[name]["mem_limit"] and services[name]["pids_limit"]


def test_secrets_and_configuration_are_scoped_to_the_correct_service():
    services = compose_document()["services"]
    assert set(services["web"]["environment"]) == {"PAPERLEET_DOMAIN"}
    judge_env = services["judge"]["environment"]
    assert not any(key.startswith("SUPABASE") or key in ("OPENAI_API_KEY", "PAPERLEET_INVITE_EMAILS", "PAPERLEET_ADMIN_EMAILS") for key in judge_env)
    api_env = services["api"]["environment"]
    assert api_env["PAPERLEET_MODE"] == "hosted"
    # Atomic restore stages beside this directory. It cannot replace the mount
    # point itself or create staging files under the read-only container root.
    assert Path(api_env["PAPERLEET_DATA_DIR"]).parent == Path("/var/lib/paperleet")
    assert api_env["PAPERLEET_TRUST_PROXY"] == "1"
    assert api_env["PAPERLEET_JUDGE_URL"] == "http://judge:8766"
    assert api_env["PAPERLEET_JUDGE_TOKEN"] == judge_env["PAPERLEET_JUDGE_TOKEN"]
    proxy = (DEPLOY / "Caddyfile").read_text()
    assert "header_up X-PaperLeet-Client-IP {remote_host}" in proxy
    assert "response_header_timeout 180s" in proxy
    assert "max_size 256KB" in proxy


@pytest.mark.parametrize("missing", ["PAPERLEET_PUBLIC_URL", "PAPERLEET_DATA_DIR", "PAPERLEET_INVITE_EMAILS", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "PAPERLEET_JUDGE_URL", "PAPERLEET_JUDGE_TOKEN"])
def test_hosted_boot_fails_closed_without_required_settings(monkeypatch, missing):
    for key, value in FAKE_SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="Hosted mode requires"):
        validate_hosted_settings()


@pytest.mark.parametrize("key,value", [("PAPERLEET_MODE", "production"),
    ("PAPERLEET_PUBLIC_URL", "http://paperleet.example.test"),
    ("PAPERLEET_PUBLIC_URL", "https://example.test/secret/path"),
    ("PAPERLEET_DATA_DIR", "relative/data"), ("PAPERLEET_JUDGE_TOKEN", "short"),
    ("PAPERLEET_JUDGE_TOKEN", "x" * 40 + "\n"),
    ("PAPERLEET_JUDGE_URL", "http://user:password@judge:8766")])
def test_hosted_configuration_rejects_ambiguous_or_insecure_values(monkeypatch, key, value):
    for setting, fake in FAKE_SETTINGS.items():
        monkeypatch.setenv(setting, fake)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        validate_hosted_settings()


def test_valid_hosted_settings_and_local_mode_are_distinct(monkeypatch):
    for key, value in FAKE_SETTINGS.items():
        monkeypatch.setenv(key, value)
    validate_hosted_settings()
    for key in FAKE_SETTINGS:
        monkeypatch.delenv(key, raising=False)
    validate_hosted_settings()


def test_test_auth_needs_no_email_but_preserves_hosted_boundaries(monkeypatch):
    for key, value in FAKE_SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY", "PAPERLEET_INVITE_EMAILS"):
        monkeypatch.delenv(key, raising=False)
    validate_hosted_settings()
    monkeypatch.setenv("PAPERLEET_PUBLIC_URL", "http://paperleet.example.test")
    with pytest.raises(ValueError, match="HTTPS"):
        validate_hosted_settings()
    monkeypatch.setenv("PAPERLEET_PUBLIC_URL", FAKE_SETTINGS["PAPERLEET_PUBLIC_URL"])
    monkeypatch.delenv("PAPERLEET_JUDGE_TOKEN")
    with pytest.raises(ValueError, match="PAPERLEET_JUDGE_TOKEN"):
        validate_hosted_settings()


@pytest.mark.parametrize("mode", ["local", "hosted"])
def test_unknown_auth_mode_fails_closed(monkeypatch, mode):
    monkeypatch.setenv("PAPERLEET_MODE", mode)
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "demo")
    with pytest.raises(ValueError, match="PAPERLEET_AUTH_MODE"):
        validate_hosted_settings()


def test_manage_script_parses_and_refuses_unconfigured_start(tmp_path):
    script = DEPLOY / "manage.sh"
    result = subprocess.run(["sh", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    isolated = tmp_path / "deploy"
    isolated.mkdir()
    shutil.copyfile(script, isolated / "manage.sh")
    result = subprocess.run(["sh", str(isolated / "manage.sh"), "up"], capture_output=True, text=True)
    assert result.returncode != 0 and "configure it first" in result.stderr


def test_build_context_excludes_runtime_data_and_secrets():
    patterns = set((root() / ".dockerignore").read_text().splitlines())
    assert {".git", ".env", "**/.env", "**/.env.*", ".paperleet", "deploy/backups", "**/*.sqlite3*", "workspace", "reports"} <= patterns
    dockerfile = (DEPLOY / "Dockerfile").read_text()
    assert "USER 10001:10001" in dockerfile
    assert not any(line.startswith("COPY . ") for line in dockerfile.splitlines())
    result = subprocess.run(["git", "check-ignore", "deploy/.env", "deploy/backups/check.tar.gz"], cwd=root(), capture_output=True, text=True)
    assert set(result.stdout.splitlines()) == {"deploy/.env", "deploy/backups/check.tar.gz"}


def test_actual_compose_config_resolves_with_only_fake_credentials(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in FAKE_SETTINGS.items()))
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_"))}
    version = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=10, env=env)
    if version.returncode:
        pytest.skip("Docker Compose is unavailable")
    result = subprocess.run(["docker", "compose", "--env-file", str(env_file), "-f", str(DEPLOY / "compose.yaml"), "config", "--format", "json"], capture_output=True, text=True, timeout=15, env=env)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["services"]["api"]["environment"]["PAPERLEET_JUDGE_TOKEN"] == FAKE_SETTINGS["PAPERLEET_JUDGE_TOKEN"]
    assert all(not document["services"][name].get("ports") for name in ("api", "judge"))
    assert document["services"]["api"]["volumes"][0]["target"] == "/var/lib/paperleet"
    assert document["services"]["judge"]["networks"] == {"judge_internal": None}


def test_actual_caddy_configuration_validates_offline():
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    image = subprocess.run(["docker", "image", "inspect", "paperleet-web:local"], capture_output=True, timeout=10)
    if image.returncode:
        pytest.skip("Build paperleet-web:local to validate Caddy")
    result = subprocess.run(["docker", "run", "--rm", "--network=none", "--entrypoint", "caddy", "-e", "PAPERLEET_DOMAIN=paperleet.example.test", "paperleet-web:local", "validate", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_test_account_compose_configuration_does_not_require_supabase(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    settings = {key: FAKE_SETTINGS[key] for key in ("PAPERLEET_DOMAIN", "PAPERLEET_JUDGE_TOKEN")}
    settings["PAPERLEET_AUTH_MODE"] = "test"
    env_file = tmp_path / "test.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_"))}
    if subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=10, env=env).returncode:
        pytest.skip("Docker Compose is unavailable")
    result = subprocess.run(["docker", "compose", "--env-file", str(env_file), "-f", str(DEPLOY / "compose.yaml"),
                             "config", "--format", "json"], capture_output=True, text=True, timeout=15, env=env)
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    configured = services["api"]["environment"]
    assert configured["PAPERLEET_AUTH_MODE"] == "test"
    assert configured["SUPABASE_URL"] == configured["SUPABASE_PUBLISHABLE_KEY"] == configured["PAPERLEET_INVITE_EMAILS"] == ""
    assert "PAPERLEET_AUTH_MODE" not in services["judge"]["environment"]
