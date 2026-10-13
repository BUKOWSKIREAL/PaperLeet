# Hosted beta deployment validation

Validated locally on 2026-10-13. This records the deployment package, not a public launch. No server was provisioned and no real verification email was sent.

## Results

| Check | Result |
| --- | --- |
| Framework, account, practice, backup, deployment and hosted HTTP tests | 260 passed; 15 opt-in Docker checks skipped by the default command |
| Isolated worker checks with Docker enabled | 13 passed: all ten maintained reference implementations, an unfinished starter, isolation probes, and timeout cleanup |
| Complete Compose deployment | Both institutional and test-account modes passed with real loopback HTTPS, API, manager and disposable judge containers |
| Persistence and recovery | Accepted report and draft survived API restart; online SQLite backup restored into a new volume; a second restore refused to overwrite it |
| Frontend checks and build | Passed, including account isolation, optimistic draft conflicts and accepted-result preservation |
| Browser interaction | Verified English/Chinese login gates, invitation copy, autosave/reload, both conflict choices, cross-tab logout and proposal-only submission |

The institutional deployment test seeds disposable identities through a maintenance process with a fake email provider. The test-account deployment uses the real operator CLI and access-code login API with no Supabase settings. Both require actual session authentication. Caddy uses a local CA for these tests; the client trusts that CA explicitly. Temporary deployment containers and volumes are removed after validation.

Test-account checks cover unique hashed access codes, a fixed private login name independent of the public username, no institution-verification claim, mode isolation, session revocation after reset/disable, and credential-change races. This login mode is disabled by default and requires `PAPERLEET_AUTH_MODE=test`. See the [test-account guide](../test-accounts.md).

The container probes confirmed UID 65534, dropped capabilities, no external network, no application credentials/database/Docker socket, and a read-only root filesystem. A timed-out job's exact container was removed. The API has no Docker socket; only the private manager has host Docker control.

One initial framework run during the original deployment image builds failed an existing mutation test. The targeted rerun and subsequent full runs passed, including the current 260-test run above.

## Reproduce

From the repository root with the Python environment and Docker running:

```sh
.venv/bin/python -m pytest tests -q
npm --prefix web run check
npm --prefix web run build
PAPERLEET_DOCKER_SMOKE=1 .venv/bin/python -m pytest tests/test_hosted_judge.py -k live_docker -q
PAPERLEET_TEST_DEPLOYMENT=1 .venv/bin/python -m pytest tests/test_deployment_live.py -q
```

Build the four images first using the [deployment guide](../hosting.md). The final command creates its own project, credentials, loopback port and volumes; it does not use or publish a production deployment.

## Deployment scope

The beta supports invited institutional accounts or explicitly enabled operator-created test accounts, community, private cloud practice, and ten maintained modules. Paper proposals are private to the author and administrators; generation, validation and publication remain offline maintainer work. Public learning-progress statistics, durable judge queues and anti-cheating grading are not included. Workers contain readable tests and references.

Before inviting users, verify public DNS, trusted HTTPS and off-server backup restoration on the actual host. Institutional mode additionally needs real Supabase email OTP/custom SMTP and institution/invitation checks; test-account mode needs separately issued credentials instead. External email delivery and public DNS were not exercised by the local package tests.
