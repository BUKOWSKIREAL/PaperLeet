"""Portable backups of private application data with safe, empty-only restore."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import sys
import tarfile
import tempfile

from paperleet.config import data_directory


MAX_FILES = 100000
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MANIFEST_NAME = "paperleet-backup.json"
DATABASE_NAME = "community.sqlite3"
DATABASE_JOURNALS = {DATABASE_NAME + suffix for suffix in ("-wal", "-shm", "-journal")}


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Backup contains an invalid path")
    parts = value.split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError("Backup contains an unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("Backup contains an absolute path")
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_file(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _stream_identity(stream):
    try:
        info = os.fstat(stream.fileno())
        if stat.S_ISREG(info.st_mode):
            os.fchmod(stream.fileno(), 0o600)
            return info.st_dev, info.st_ino
    except (AttributeError, OSError):
        pass
    return None


def _snapshot(source, target, output_identity, staging_root):
    files, directories = {}, []
    total = 0
    for current, dirnames, filenames in os.walk(source, followlinks=False):
        dirnames[:] = [name for name in dirnames if Path(current) / name != staging_root]
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            path = Path(current) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("Application data contains a symlink or unsupported directory")
            relative = path.relative_to(source).as_posix()
            _relative(relative)
            destination = target / relative
            destination.mkdir(mode=0o700)
            directories.append(relative)
            if len(directories) + len(files) > MAX_FILES:
                raise ValueError("Application backup contains too many paths")
        for name in filenames:
            path = Path(current) / name
            relative = path.relative_to(source).as_posix()
            _relative(relative)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Application data contains a symlink or special file")
            if (info.st_dev, info.st_ino) == output_identity:
                continue
            if relative in DATABASE_JOURNALS:
                continue
            if total + info.st_size > MAX_TOTAL_BYTES:
                raise ValueError("Application backup exceeds the supported size")
            destination = target / relative
            if relative == DATABASE_NAME:
                # SQLite's online backup includes committed WAL transactions;
                # copying the database/WAL files independently would not.
                with _private_file(destination):
                    pass
                original = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                snapshot = sqlite3.connect(destination)
                try:
                    page_size = original.execute("PRAGMA page_size").fetchone()[0]

                    def bounded_snapshot(status, remaining, pages):
                        if total + pages * page_size > MAX_TOTAL_BYTES:
                            raise ValueError("SQLite backup exceeds the supported size")

                    original.backup(snapshot, pages=256, progress=bounded_snapshot)
                finally:
                    snapshot.close()
                    original.close()
            else:
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as original, _private_file(destination) as snapshot:
                    if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
                        raise ValueError("Application data changed to an unsupported file")
                    copied = 0
                    for chunk in iter(lambda: original.read(1024 * 1024), b""):
                        copied += len(chunk)
                        if total + copied > MAX_TOTAL_BYTES:
                            raise ValueError("Application backup exceeds the supported size")
                        snapshot.write(chunk)
            size = destination.stat().st_size
            total += size
            if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                raise ValueError("Application backup exceeds the supported size or file count")
            files[relative] = {"size": size, "sha256": _hash(destination)}
    return files, directories


def backup(output, source=None):
    """Write one self-contained tar.gz; SQLite and regular files are snapshotted."""
    source = Path(source).absolute() if source is not None else data_directory()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("The application data directory must exist and be a real directory")
    identity = _stream_identity(output)
    # Use the persistent volume: the application's read-only container has only
    # a small /tmp tmpfs, and SQLite snapshots may exceed that memory budget.
    with tempfile.TemporaryDirectory(prefix=".paperleet-backup-", dir=source.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir(mode=0o700)
        files, directories = _snapshot(source, staging, identity, Path(temporary))
        manifest = {"format": "paperleet-backup", "version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
        raw_manifest = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        if len(raw_manifest) > MAX_MANIFEST_BYTES:
            raise ValueError("Backup manifest is too large")
        import io
        with tarfile.open(fileobj=output, mode="w|gz", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size, info.mode = len(raw_manifest), 0o600
            archive.addfile(info, io.BytesIO(raw_manifest))
            for relative in ["", *directories]:
                info = tarfile.TarInfo("data" + ("/" + relative if relative else ""))
                info.type, info.mode = tarfile.DIRTYPE, 0o700
                archive.addfile(info)
            for relative, metadata in files.items():
                info = tarfile.TarInfo("data/" + relative)
                info.size, info.mode = metadata["size"], 0o600
                with (staging / relative).open("rb") as stream:
                    archive.addfile(info, stream)
    return {"files": len(files), "bytes": sum(item["size"] for item in files.values())}


def _empty_target(destination):
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("Restore requires an empty data directory and never overwrites existing data")


def _manifest(archive, member):
    if member is None or member.name != MANIFEST_NAME or not member.isreg() or member.linkname or not 0 < member.size <= MAX_MANIFEST_BYTES:
        raise ValueError("A valid PaperLeet backup manifest must be the first archive entry")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("Backup manifest is missing")
    try:
        document = json.loads(stream.read(MAX_MANIFEST_BYTES + 1))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Backup manifest is not valid JSON") from None
    if not isinstance(document, dict) or document.get("format") != "paperleet-backup" or document.get("version") != 1:
        raise ValueError("Unsupported PaperLeet backup format")
    files = document.get("files")
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("Backup manifest contains too many files")
    total = 0
    for name, metadata in files.items():
        _relative(name)
        if name in DATABASE_JOURNALS:
            raise ValueError("A backup cannot restore SQLite journal sidecars")
        if (not isinstance(metadata, dict) or type(metadata.get("size")) is not int or metadata["size"] < 0
                or not isinstance(metadata.get("sha256"), str) or len(metadata["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in metadata["sha256"])):
            raise ValueError("Backup manifest has invalid file metadata")
        total += metadata["size"]
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Backup exceeds the supported restored size")
    return files


def restore(input_stream, destination=None):
    """Validate into a private staging directory, then install into empty data."""
    destination = Path(destination).absolute() if destination is not None else data_directory()
    _empty_target(destination)
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paperleet-restore-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir(mode=0o700)
        with tarfile.open(fileobj=input_stream, mode="r|gz") as archive:
            expected = _manifest(archive, archive.next())
            seen, extracted, count = {MANIFEST_NAME}, set(), 0
            for member in archive:
                # Streaming TarFile iteration may revisit its already-read first member.
                if member.name == MANIFEST_NAME and count == 0 and member.offset == 0:
                    continue
                count += 1
                if count > MAX_FILES * 2 + 1:
                    raise ValueError("Backup contains too many entries")
                name = str(_relative(member.name))
                if name in seen:
                    raise ValueError("Backup contains duplicate paths")
                seen.add(name)
                if (member.linkname or member.issym() or member.islnk() or member.issparse()
                        or not (member.isdir() or member.isreg())):
                    raise ValueError("Backup links and special files cannot be restored")
                if name != "data" and not name.startswith("data/"):
                    raise ValueError("Backup entries must be inside the data directory")
                relative = name[5:] if name.startswith("data/") else ""
                target = staging / relative
                if member.isdir():
                    target.mkdir(parents=True, mode=0o700, exist_ok=True)
                    target.chmod(0o700)
                    continue
                metadata = expected.get(relative)
                if metadata is None or member.size != metadata["size"]:
                    raise ValueError("Backup file does not match its manifest")
                target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Backup file is unreadable")
                with _private_file(target) as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                if target.stat().st_size != metadata["size"] or _hash(target) != metadata["sha256"]:
                    raise ValueError("Backup file integrity check failed")
                extracted.add(relative)
            if extracted != set(expected):
                raise ValueError("Backup is missing files listed in its manifest")
        for current, _, _ in os.walk(staging):
            Path(current).chmod(0o700)
        database = staging / DATABASE_NAME
        if database.exists():
            connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            try:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("The restored SQLite snapshot failed its integrity check")
            finally:
                connection.close()
        _empty_target(destination)
        # DATA_DIR is a child of the mounted volume so the parent is writable.
        # Rename refuses a nonempty raced destination instead of overwriting it.
        os.rename(staging, destination)
    return {"files": len(extracted), "bytes": sum(item["size"] for item in expected.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Back up or restore private PaperLeet application data")
    commands = parser.add_subparsers(dest="command", required=True)
    backup_parser = commands.add_parser("backup")
    backup_parser.add_argument("--stdout", action="store_true", required=True, help="Write a tar.gz archive to stdout")
    restore_parser = commands.add_parser("restore")
    restore_parser.add_argument("--stdin", action="store_true", required=True, help="Read a tar.gz archive from stdin into an empty data directory")
    args = parser.parse_args(argv)
    try:
        result = backup(sys.stdout.buffer) if args.command == "backup" else restore(sys.stdin.buffer)
    except (OSError, EOFError, ValueError, sqlite3.Error, tarfile.TarError) as exc:
        print("PaperLeet backup error: " + str(exc), file=sys.stderr)
        return 1
    print(f"{args.command.capitalize()} complete: {result['files']} files, {result['bytes']} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
