"""Import a pinned, allowlisted backbone snapshot without overwriting local work.

Only Git objects are read from upstream: its scripts, workflows and checkout
hooks never run here. The offline repository owns the synchronizer and policy.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile


REPOSITORY = "PaperLeet/paperleet.github.io"
URL = "https://github.com/" + REPOSITORY + ".git"
CONFIG = "sync/upstream.json"
LOCK = "sync/upstream.lock.json"
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MODULE = re.compile(r"problems/([a-z0-9_]+)/([a-z0-9_]+)/problem\.yaml")
CORE = frozenset({"papers.yaml", "conftest.py", "paperleet/__init__.py", "paperleet/catalog.py",
                  "paperleet/paper_catalog.py", "paperleet/judge/__init__.py", "paperleet/judge/loader.py",
                  "paperleet/judge/pytest_plugin.py", "paperleet/judge/mutation.py", "paperleet/judge/queue.py",
                  "web/src/paperCollections.ts"})
FORBIDDEN = {"deploy", "workspace", "reports", "node_modules", "__pycache__", "paperleet_cloud"}


class SyncError(ValueError):
    pass


def git(repository, *arguments, input=None):
    result = subprocess.run(["git", "-C", str(repository), *arguments], input=input,
                            capture_output=True, check=False)
    if result.returncode:
        raise SyncError("Git operation failed: " + " ".join(arguments[:2]) + "\n" + result.stderr.decode(errors="replace"))
    return result.stdout


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) < 32 for c in value):
        raise SyncError("Invalid repository path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.startswith(".") or part in FORBIDDEN for part in parts):
        raise SyncError("Forbidden repository path: " + value)
    if PurePosixPath(value).is_absolute():
        raise SyncError("Absolute repository paths are not allowed")
    return value


def load_config(root):
    local_record(root, CONFIG)
    config = json.loads((root / CONFIG).read_text())
    if config.get("version") != 1 or config.get("repository") != REPOSITORY:
        raise SyncError("The synchronizer only accepts the configured PaperLeet backbone")
    files = config.get("files")
    directories = config.get("directories")
    if not isinstance(files, list) or len(files) != len(set(files)) or not set(files) <= CORE:
        raise SyncError("Invalid core file allowlist")
    if not isinstance(directories, list) or not set(directories) <= {"problems/", "paperleet/common/"}:
        raise SyncError("Invalid directory allowlist")
    if config.get("per_problem_visualizations") is not True:
        raise SyncError("Per-problem visualization policy is required")
    return config


def visualizations(paths):
    return {"web/public/data/" + match[1] + "-" + match[2] + ".json"
            for path in paths if (match := MODULE.fullmatch(path))}


def allowed(path, config, viz):
    safe_path(path)
    if path in config["files"] or path in viz:
        return True
    if "problems/" in config["directories"] and re.fullmatch(r"problems/[a-z0-9_]+/[a-z0-9_]+/.+", path):
        return True
    return "paperleet/common/" in config["directories"] and path.startswith("paperleet/common/") and path.endswith(".py")


def tree(repository, ref):
    entries = {}
    for record in git(repository, "ls-tree", "-rz", "--full-tree", ref).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        entries[raw_path.decode("utf-8")] = (mode, kind, oid)
    return entries


def blobs(repository, entries):
    if not entries:
        return {}
    requested = "".join(item[2] + "\n" for item in entries.values()).encode()
    headers = git(repository, "cat-file", "--batch-check", input=requested).decode().splitlines()
    total = 0
    for (path, (_, _, oid)), line in zip(entries.items(), headers):
        header = line.split()
        if len(header) != 3 or header[:2] != [oid, "blob"]:
            raise SyncError("Invalid Git blob response")
        size = int(header[2])
        total += size
        if size > 8 * 1024 * 1024 or total > 64 * 1024 * 1024:
            raise SyncError("Managed snapshot exceeds its file or total size budget: " + path)
    if len(headers) != len(entries):
        raise SyncError("Incomplete Git blob metadata")
    output = git(repository, "cat-file", "--batch", input=requested)
    result, position = {}, 0
    for path, (mode, _, oid) in entries.items():
        end = output.index(b"\n", position)
        header = output[position:end].decode().split()
        if len(header) != 3 or header[:2] != [oid, "blob"]:
            raise SyncError("Invalid Git blob response")
        size = int(header[2])
        if size > 8 * 1024 * 1024:
            raise SyncError("Managed file exceeds 8 MiB: " + path)
        raw = output[end + 1:end + 1 + size]
        if len(raw) != size:
            raise SyncError("Truncated Git blob")
        result[path] = (mode, raw)
        position = end + size + 2
    return result


def snapshot(repository, commit, config, content_only=False):
    if not SHA.fullmatch(commit):
        raise SyncError("Upstream revision must be a full 40-character commit SHA")
    if git(repository, "rev-parse", "--verify", commit + "^{commit}").decode().strip() != commit:
        raise SyncError("Upstream revision is not a commit")
    entries = tree(repository, commit)
    viz = visualizations(entries)
    selected = {}
    for path, entry in entries.items():
        # Unmanaged upstream dotfiles are deliberately never imported.
        if not (path in config["files"] or path in viz or any(path.startswith(p) for p in config["directories"])):
            continue
        if entry[0] not in {"100644", "100755"} or entry[1] != "blob":
            raise SyncError("Symlinks and submodules cannot be synchronized: " + path)
        if not allowed(path, config, viz):
            continue
        if content_only and not (path.startswith("problems/") or path in viz or path == "papers.yaml"):
            continue
        selected[path] = entry
    required = {"papers.yaml"} if content_only else set(config["files"])
    if not required <= selected.keys():
        raise SyncError("Required upstream files disappeared: " + ", ".join(sorted(required - selected.keys())))
    if not any(MODULE.fullmatch(path) for path in selected):
        raise SyncError("Upstream snapshot has no problem packages")
    return blobs(repository, dict(sorted(selected.items())))


def record(mode, data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "mode": mode}


def local_record(root, relative):
    safe_path(relative)
    path = root
    for index, part in enumerate(relative.split("/")):
        path = path / part
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise SyncError("Local symlink would escape managed paths: " + relative)
        if index < len(relative.split("/")) - 1 and not stat.S_ISDIR(info.st_mode):
            raise SyncError("Managed parent is not a directory: " + relative)
    if not stat.S_ISREG(info.st_mode):
        raise SyncError("Managed path is not a regular file: " + relative)
    return record("100755" if info.st_mode & stat.S_IXUSR else "100644", path.read_bytes())


def read_lock(root):
    path = root / LOCK
    if local_record(root, LOCK) is None:
        return None
    lock = json.loads(path.read_text())
    if lock.get("version") != 1 or lock.get("repository") != REPOSITORY or not SHA.fullmatch(lock.get("commit", "")):
        raise SyncError("Invalid upstream lock identity")
    if not isinstance(lock.get("files"), dict):
        raise SyncError("Invalid upstream file manifest")
    for name, value in lock["files"].items():
        safe_path(name)
        if not isinstance(value, dict) or not DIGEST.fullmatch(value.get("sha256", "")) or value.get("mode") not in {"100644", "100755"}:
            raise SyncError("Invalid managed file record: " + name)
    return lock


def synchronize(root, source, commit, *, bootstrap=False, dry_run=False, content_only=False):
    root = root.resolve()
    config, previous = load_config(root), read_lock(root)
    if previous is None and not bootstrap:
        raise SyncError("No upstream lock exists; use --bootstrap once on a reviewed clean checkout")
    desired = snapshot(source, commit, config, content_only)
    old = previous["files"] if previous else {}
    old_viz = visualizations(old)
    # Shrinking the policy never silently deletes a now locally owned file.
    released = [path for path in old if not allowed(path, config, old_viz)]
    if released:
        raise SyncError("Explicit lock migration required for released managed files: " + ", ".join(released))
    if content_only and previous:
        raise SyncError("--content-only is only available for the initial bootstrap")
    baseline = blobs(root, {path: entry for path, entry in tree(root, "HEAD").items()
                            if path in desired and entry[0] in {"100644", "100755"} and entry[1] == "blob"}) if bootstrap else {}
    conflicts, updates, removals = [], [], []
    desired_records = {path: record(*value) for path, value in desired.items()}
    for path in sorted(set(old) | set(desired)):
        current = local_record(root, path)
        target = desired_records.get(path)
        if path in old:
            if current != old[path]:
                conflicts.append(path)
                continue
        elif current is not None and current != target:
            if path not in baseline or current != record(*baseline[path]):
                conflicts.append(path)
                continue
        if target is None:
            removals.append(path)
        elif current != target:
            updates.append(path)
    if conflicts:
        raise SyncError("Local changes conflict with upstream; nothing was written:\n" + "\n".join(conflicts))
    changed = previous is None or old != desired_records
    report = {"changed": changed, "repository": REPOSITORY, "commit": commit,
              "previous_commit": previous["commit"] if previous else None,
              "updated": updates, "removed": removals, "managed_files": len(desired)}
    if dry_run or not changed:
        return report
    # All conflicts and path types have been checked before the first mutation.
    for path in updates:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode, data = desired[path]
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o755 if mode == "100755" else 0o644)
        temporary_path.replace(destination)
    for path in removals:
        (root / path).unlink()
    lock = {"version": 1, "repository": REPOSITORY, "commit": commit, "files": desired_records}
    local_record(root, LOCK)
    (root / LOCK).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=(root / LOCK).parent, delete=False) as temporary:
        json.dump(lock, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(root / LOCK)
    return report


def verify_lock(root, staged=False):
    config, lock = load_config(root), read_lock(root)
    if lock is None:
        raise SyncError("Missing upstream lock")
    if not set(config["files"]) <= lock["files"].keys() or not any(MODULE.fullmatch(path) for path in lock["files"]):
        raise SyncError("The manifest omits required managed files")
    viz = visualizations(lock["files"])
    for path, expected in lock["files"].items():
        if not allowed(path, config, viz) or local_record(root, path) != expected:
            raise SyncError("Managed file does not match its lock: " + path)
    if staged:
        paths = git(root, "diff", "--cached", "--name-only", "-z").decode().split("\0")
        original = json.loads(git(root, "show", "HEAD:" + LOCK))
        for path in filter(None, paths):
            if path == LOCK:
                continue
            if path in lock["files"] and allowed(path, config, viz):
                continue
            if path in original["files"] and local_record(root, path) is None and allowed(path, config, visualizations(original["files"])):
                continue
            raise SyncError("Patch changed an unmanaged file: " + path)
    return lock


def write_patch(root, destination):
    lock = verify_lock(root)
    original = json.loads(git(root, "show", "HEAD:" + LOCK))
    paths = sorted(set(lock["files"]) | set(original["files"]) | {LOCK})
    git(root, "add", "-A", "--", *paths)
    verify_lock(root, staged=True)
    destination.write_bytes(git(root, "diff", "--cached", "--binary", "HEAD", "--", *paths))
    return lock


@contextmanager
def source_repository(source, commit):
    if source:
        origin = git(source, "remote", "get-url", "origin").decode().strip().removesuffix(".git")
        if origin != URL.removesuffix(".git"):
            raise SyncError("Local source origin must be " + URL)
        commit = commit or git(source, "rev-parse", "refs/heads/main").decode().strip()
        yield Path(source), commit
        return
    if commit is None:
        remote = subprocess.run(["git", "ls-remote", URL, "refs/heads/main"], capture_output=True, check=False)
        if remote.returncode or not remote.stdout.strip():
            raise SyncError("Cannot resolve upstream main")
        commit = remote.stdout.decode().split()[0]
    if not SHA.fullmatch(commit):
        raise SyncError("Upstream revision must be a full 40-character commit SHA")
    with tempfile.TemporaryDirectory(prefix="paperleet-upstream-") as folder:
        subprocess.run(["git", "init", "--quiet", folder], check=True)
        git(folder, "fetch", "--quiet", "--depth=1", "--no-tags", URL, commit)
        yield Path(folder), commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, help="Read an existing checkout of the fixed upstream repository")
    parser.add_argument("--ref", help="Full upstream commit SHA (default: resolve main once)")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--content-only", action="store_true", help="Initial topic/viz import before core integration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--apply-patch", type=Path, help="Apply a validated CI patch and verify its managed boundary")
    parser.add_argument("--write-patch", type=Path, help="Stage only managed files and write their validated binary diff")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.apply_patch:
            # This process loaded its trusted code before applying any patch;
            # no updated upstream code is executed by the publishing job.
            before_config = (args.root / CONFIG).read_bytes()
            before_script = Path(__file__).read_bytes()
            git(args.root, "apply", "--index", str(args.apply_patch.resolve()))
            if (args.root / CONFIG).read_bytes() != before_config or Path(__file__).read_bytes() != before_script:
                raise SyncError("Patch cannot alter the synchronizer or policy")
            lock = verify_lock(args.root, staged=True)
            report = {"changed": True, "commit": lock["commit"]}
        elif args.write_patch:
            lock = write_patch(args.root, args.write_patch)
            report = {"changed": True, "commit": lock["commit"]}
        elif args.verify_lock:
            lock = verify_lock(args.root)
            report = {"verified": True, "commit": lock["commit"], "managed_files": len(lock["files"])}
        else:
            with source_repository(args.source, args.ref) as (source, commit):
                report = synchronize(args.root, source, commit, bootstrap=args.bootstrap,
                                     dry_run=args.dry_run, content_only=args.content_only)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        if args.github_output:
            with args.github_output.open("a") as output:
                output.write("changed=" + str(report.get("changed", False)).lower() + "\n")
                output.write("commit=" + report["commit"] + "\n")
        print(json.dumps(report, indent=2))
        return 0
    except (SyncError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Upstream sync stopped: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
