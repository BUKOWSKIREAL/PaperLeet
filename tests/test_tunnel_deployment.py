"""Exercise the merged tunnel deployment without connecting a public tunnel."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from paperleet.catalog import root


DEPLOY = root() / "deploy"
FAKE_SETTINGS = {
    "PAPERLEET_DOMAIN": "paperleet.example.test",
    "PAPERLEET_AUTH_MODE": "test",
    "PAPERLEET_JUDGE_TOKEN": "test-only-internal-token-" + "0" * 40,
    "TUNNEL_TOKEN": "test-only-tunnel-token-not-a-real-credential",
}


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("PAPERLEET_", "SUPABASE_", "COMPOSE_", "TUNNEL_"))}


def compose_config(tmp_path, *, token=True):
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    env = clean_environment()
    if subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=10, env=env).returncode:
        pytest.skip("Docker Compose is unavailable")
    settings = {key: value for key, value in FAKE_SETTINGS.items() if token or key != "TUNNEL_TOKEN"}
    env_file = tmp_path / "test.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    return subprocess.run(["docker", "compose", "--env-file", str(env_file),
        "-f", str(DEPLOY / "compose.yaml"), "-f", str(DEPLOY / "compose.tunnel.yaml"),
        "config", "--format", "json"], capture_output=True, text=True, timeout=15, env=env)


def test_merged_tunnel_has_no_host_ports_and_preserves_judge_isolation(tmp_path):
    result = compose_config(tmp_path)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    services = document["services"]
    assert set(services) == {"web", "api", "judge", "tunnel"}
    assert all(not service.get("ports") for service in services.values())
    assert all(service.get("network_mode") != "host" for service in services.values())
    assert set(services["tunnel"]["networks"]) == {"tunnel_ingress", "tunnel_egress"}
    assert set(services["web"]["networks"]) == {"frontend", "tunnel_ingress"}
    assert set(services["api"]["networks"]) == {"frontend", "judge_internal"}
    assert set(services["judge"]["networks"]) == {"judge_internal"}
    assert document["networks"]["tunnel_ingress"]["internal"] is True
    assert document["networks"]["judge_internal"]["internal"] is True
    mounts = {name: service.get("volumes", []) for name, service in services.items()}
    assert not any("docker.sock" in mount["target"]
                   for name, volumes in mounts.items() if name != "judge" for mount in volumes)
    assert mounts["judge"][0]["target"] == "/var/run/docker.sock"
    assert mounts["api"][0]["target"] == "/var/lib/paperleet"
    proxy = next(mount for mount in mounts["web"] if mount["target"] == "/etc/caddy/Caddyfile")
    assert Path(proxy["source"]) == DEPLOY / "Caddyfile.tunnel"
    assert proxy["read_only"] is True
    assert not mounts["tunnel"]
    for name in ("api", "judge", "tunnel"):
        assert services[name]["read_only"] is True
        assert services[name]["cap_drop"] == ["ALL"]
        assert services[name]["security_opt"] == ["no-new-privileges:true"]


def test_tunnel_token_is_connector_only_and_auth_remains_https(tmp_path):
    result = compose_config(tmp_path)
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    assert services["tunnel"]["environment"] == {"TUNNEL_TOKEN": FAKE_SETTINGS["TUNNEL_TOKEN"]}
    for name in ("web", "api", "judge"):
        assert "TUNNEL_TOKEN" not in services[name]["environment"]
    assert services["api"]["environment"]["PAPERLEET_MODE"] == "hosted"
    assert services["api"]["environment"]["PAPERLEET_AUTH_MODE"] == "test"
    assert services["api"]["environment"]["PAPERLEET_PUBLIC_URL"] == "https://paperleet.example.test"
    for name in ("api", "judge"):
        assert services[name]["environment"]["PAPERLEET_JUDGE_TIMEOUT_SECONDS"] == "90"
        assert services[name]["environment"]["PAPERLEET_JUDGE_TOKEN"] == FAKE_SETTINGS["PAPERLEET_JUDGE_TOKEN"]
    assert services["tunnel"]["command"] == ["tunnel", "--no-autoupdate", "run"]


def test_tunnel_configuration_requires_a_token(tmp_path):
    result = compose_config(tmp_path, token=False)
    assert result.returncode != 0
    assert "Set your Cloudflare tunnel token" in result.stderr


@pytest.mark.parametrize("project", [None, "paperleet-tunnel-restore"])
def test_tunnel_wrapper_selects_overlay_and_separate_project(tmp_path, project):
    deploy = tmp_path / "space in path" / "deploy"
    deploy.mkdir(parents=True)
    for filename in ("manage.sh", "tunnel.sh"):
        shutil.copyfile(DEPLOY / filename, deploy / filename)
        (deploy / filename).chmod(0o755)
    (deploy / ".env").write_text("# Deliberately fake configuration for command routing only.\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                      "print(json.dumps({'args': sys.argv[1:], 'project': os.environ.get('COMPOSE_PROJECT_NAME')}))\n")
    docker.chmod(0o755)
    env = clean_environment()
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    if project:
        env["COMPOSE_PROJECT_NAME"] = project
    result = subprocess.run([str(deploy / "tunnel.sh"), "status"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["args"] == ["compose", "--env-file", "deploy/.env", "-f", "deploy/compose.yaml",
                              "-f", "deploy/compose.tunnel.yaml", "ps"]
    assert output["project"] == (project or "paperleet-tunnel")


def test_tunnel_caddy_config_preserves_public_host_and_cloudflare_client_ip():
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    available_image = None
    for candidate in ("paperleet-web:local", "caddy:2-alpine"):
        image = subprocess.run(["docker", "image", "inspect", candidate], capture_output=True, timeout=10)
        if image.returncode == 0:
            available_image = candidate
            break
    if available_image is None:
        pytest.skip("Build paperleet-web:local or pull caddy:2-alpine to validate Caddy")
    result = subprocess.run(["docker", "run", "--rm", "--network=none", "--entrypoint", "caddy",
        "-e", "PAPERLEET_DOMAIN=paperleet.example.test",
        "-v", str(DEPLOY / "Caddyfile.tunnel") + ":/etc/caddy/Caddyfile:ro", available_image,
        "adapt", "--validate", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    servers = document["apps"]["http"]["servers"]
    assert len(servers) == 1
    server = next(iter(servers.values()))
    assert server["listen"] == [":80"]
    assert server["routes"][0]["match"] == [{"host": ["paperleet.example.test"]}]
    rendered = json.dumps(server)
    assert '"X-Paperleet-Client-Ip": ["{http.request.header.CF-Connecting-IP}"]' in rendered
    assert '"Cache-Control": ["no-store"]' in rendered
    assert '"response_header_timeout": 115000000000' in rendered
    # Public direct-hosting must continue using its own TCP client address.
    assert "header_up X-PaperLeet-Client-IP {remote_host}" in (DEPLOY / "Caddyfile").read_text()
