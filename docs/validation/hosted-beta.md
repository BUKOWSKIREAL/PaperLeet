# Hosted beta deployment validation

Validated locally on 2026-10-13. This records the deployment package, not a public launch. No server was provisioned and no real verification email was sent.

## Results

| Check | Result |
| --- | --- |
| Framework, account, practice, backup, deployment and hosted HTTP tests | 232 passed; 14 opt-in Docker checks skipped by the default command |
| Isolated worker checks with Docker enabled | 13 passed: all ten maintained reference implementations, an unfinished starter, isolation probes, and timeout cleanup |
| Complete Compose deployment | Passed with real loopback HTTPS, API, manager and disposable judge containers |
| Persistence and recovery | Accepted report and draft survived API restart; online SQLite backup restored into a new volume; a second restore refused to overwrite it |
| Frontend checks and build | Passed, including account isolation, optimistic draft conflicts and accepted-result preservation |
| Browser interaction | Verified English/Chinese login gates, invitation copy, autosave/reload, both conflict choices, cross-tab logout and proposal-only submission |

The full deployment test seeds disposable identities through a test-only maintenance process with a fake email provider. It adds no sign-in bypass to the application. Caddy uses a local CA for this test; the client trusts that CA explicitly. Temporary containers, volumes, API fixture and browser tabs were removed after validation.

The container probes confirmed UID 65534, dropped capabilities, no external network, no application credentials/database/Docker socket, and a read-only root filesystem. A timed-out job's exact container was removed. The API has no Docker socket; only the private manager has host Docker control.

One initial framework run during image builds failed an existing mutation test. The targeted rerun and subsequent full runs passed, including the final 232-test run above.

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

The beta supports invited institutional accounts, community, private cloud practice, and ten maintained modules. Paper proposals are private to the author and administrators; generation, validation and publication remain offline maintainer work. Public learning-progress statistics, durable judge queues and anti-cheating grading are not included. Workers contain readable tests and references.

Before inviting real users, verify public DNS, trusted HTTPS, Supabase email OTP/custom SMTP, the institution and invitation lists, and off-server backup restoration on the actual host. These external configurations were not exercised by the local package tests.
