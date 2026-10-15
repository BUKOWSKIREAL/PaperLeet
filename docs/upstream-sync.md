# Updating the offline library

`PaperLeet/paperleet.github.io` is the source of the maintained problem library.
This repository owns the offline application. The daily **Sync offline problem
library** workflow imports a pinned source snapshot, runs the offline checks,
and opens or refreshes one pull request on `sync/upstream`. A maintainer reviews
and merges that PR; the workflow never pushes directly to `main`.

## What follows upstream

The exact allowlist is [sync/upstream.json](../sync/upstream.json):

- `problems/**` and `papers.yaml`;
- the catalog and paper registry, `paperleet/common/**`, shared judge loading,
  fixtures, mutation checks and queue utilities, and `conftest.py`;
- the precomputed `web/public/data/<category>-<module>.json` visualization for
  each imported problem;
- `web/src/paperCollections.ts`, the shared curriculum/collection membership
  data, so new core exercises also appear in the appropriate learning paths.

The offline CLI, server, judge runner, job adapter, exporter, configuration and
frontend application are maintained here; the curriculum data file listed above
is the sole frontend source exception. In particular, the offline judge runner
removes the upstream community submission sandbox path. Changes to those
upstream adapters require a separate reviewed adaptation here. Browser/WASM
execution, account/community interfaces and online deployment are not imported.

The exporter rebuilds catalog/index data locally; generated bundles, deployment
files, environment files, user workspace files and reports are never copied.

## Immutable revisions and local changes

[sync/upstream.lock.json](../sync/upstream.lock.json) records the last imported
content revision and each managed file's SHA-256 and executable mode. The
workflow resolves `main` once, then reads Git objects from that exact commit.
It does not run upstream scripts during synchronization.

If a later upstream commit only changes the online site, the lock stays at the
last content revision and no PR is created. Repeating the same import is also
a no-op.

Before writing anything, synchronization checks all previously managed files.
A local edit, deletion, new-file collision or symlink stops the import and lists
the conflicts. Upstream deletions remove only unchanged files already recorded
in the lock. Other files are preserved.

Submit shared problem or helper fixes to the backbone first. When a managed file
needs a permanent offline variation, review a policy/lock migration and make it
an explicitly offline-owned adapter. Do not overwrite local changes or edit
manifest hashes merely to silence a conflict. Human commits on `sync/upstream`
also stop automatic branch replacement; move those edits to their own PR before
rerunning synchronization.

## Run an update

In GitHub Actions, select **Sync offline problem library → Run workflow**. Leave
the revision empty for the latest upstream `main`, or provide a complete
40-character commit SHA. The workflow installs CPU-only dependencies and runs
`make ci PYTHON=python` before proposing any changes.

For a local preview, from a clean checkout:

```sh
python scripts/sync_upstream.py --dry-run
python scripts/sync_upstream.py
make ci PYTHON=python
python scripts/sync_upstream.py --verify-lock
```

Use `--ref <full-commit-sha>` to select a specific revision. `--source /path/to/the/backbone`
can read a local checkout whose `origin` is the fixed upstream repository;
working-tree changes there are ignored. `--bootstrap` is only for the initial,
reviewed import or an explicit ownership migration: it can replace existing
managed files only when they still match the destination's committed version.

## GitHub permissions

The upstream repository is public, so no cross-repository secret is needed.
Validation runs with read-only repository permissions. A separate publishing
job receives `contents: write` and `pull-requests: write`, verifies the validated
base and managed patch, and uses this repository's `GITHUB_TOKEN` for the PR.
It never runs imported tests with that write token.

The repository setting **Allow GitHub Actions to create and approve pull
requests** must be enabled; the default workflow permission can remain read-only.
GitHub may require a maintainer to select **Approve workflows to run** for the
bot-created PR's checks. This workflow does not approve or merge its own PR.
