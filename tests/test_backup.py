import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tarfile

import pytest

from paperleet.backup import backup, restore, MANIFEST_NAME


@pytest.fixture
def live_data(tmp_path):
    data = tmp_path / "volume" / "data"
    (data / "submissions" / "revisions").mkdir(parents=True)
    (data / "submissions" / "paper.json").write_text('{"title":"论文","status":"submitted"}', encoding="utf-8")
    database = sqlite3.connect(data / "community.sqlite3")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT)")
    database.execute("INSERT INTO users VALUES ('alice','alice@mit.edu')")
    database.commit()
    assert (data / "community.sqlite3-wal").exists()
    yield data, database
    database.close()


def saved_archive(data):
    stream = io.BytesIO()
    backup(stream, data)
    stream.seek(0)
    return stream


def crafted_archive(entries, files=None):
    """entries: (name, bytes, optional tar type, optional link target)."""
    if files is None:
        files = {name[5:]: {"size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
                 for name, contents, *rest in entries if name.startswith("data/") and not rest}
    manifest = json.dumps({"format": "paperleet-backup", "version": 1, "files": files}).encode()
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        item = tarfile.TarInfo(MANIFEST_NAME)
        item.size = len(manifest)
        archive.addfile(item, io.BytesIO(manifest))
        for name, contents, *options in entries:
            item = tarfile.TarInfo(name)
            item.type = options[0] if options else tarfile.REGTYPE
            item.linkname = options[1] if len(options) > 1 else ""
            item.size = len(contents) if item.isreg() else 0
            archive.addfile(item, io.BytesIO(contents) if item.isreg() else None)
    result.seek(0)
    return result


def test_online_sqlite_backup_includes_committed_wal_and_round_trips_private_files(live_data, tmp_path):
    data, database = live_data
    database.execute("INSERT INTO users VALUES ('bob','bob@mit.edu')")
    database.commit()
    stream = saved_archive(data)
    with tarfile.open(fileobj=stream, mode="r:gz") as archive:
        names = archive.getnames()
        assert "data/community.sqlite3" in names
        assert not any(name.endswith(("-wal", "-shm", "-journal")) for name in names)
        assert not any(".paperleet-backup-" in name for name in names)
    stream.seek(0)
    target = tmp_path / "new-volume" / "data"
    result = restore(stream, target)
    assert result["files"] == 2
    assert (target / "submissions/paper.json").read_text() == (data / "submissions/paper.json").read_text()
    assert (target / "submissions/revisions").is_dir()
    with sqlite3.connect(target / "community.sqlite3") as restored:
        assert restored.execute("SELECT id FROM users ORDER BY id").fetchall() == [("alice",), ("bob",)]
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    for path in target.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_backup_excludes_its_own_redirected_archive_and_sets_private_permissions(live_data):
    data, _ = live_data
    archive_path = data / "own-backup.tar.gz"
    with archive_path.open("wb") as stream:
        backup(stream, data)
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "data/own-backup.tar.gz" not in archive.getnames()


def test_restore_can_replace_existing_empty_child_directory(live_data, tmp_path):
    data, _ = live_data
    target = tmp_path / "target-volume/data"
    target.mkdir(parents=True)
    restore(saved_archive(data), target)
    assert (target / "community.sqlite3").is_file()


def test_restore_never_overwrites_existing_data(live_data, tmp_path):
    data, _ = live_data
    target = tmp_path / "target"
    target.mkdir()
    existing = target / "keep.txt"
    existing.write_text("keep this")
    with pytest.raises(ValueError, match="empty data directory"):
        restore(saved_archive(data), target)
    assert existing.read_text() == "keep this"
    assert list(target.iterdir()) == [existing]


def test_restore_refuses_a_raced_destination_without_overwriting_files(tmp_path, monkeypatch):
    target = tmp_path / "data"
    original_rename = os.rename

    def racing_writer(source, destination):
        destination = Path(destination)
        destination.mkdir(exist_ok=True)
        (destination / "concurrent.txt").write_text("new data")
        return original_rename(source, destination)

    monkeypatch.setattr("paperleet.backup.os.rename", racing_writer)
    with pytest.raises(OSError):
        restore(crafted_archive([("data/backup.txt", b"old data")]), target)
    assert (target / "concurrent.txt").read_text() == "new data"
    assert not (target / "backup.txt").exists()


@pytest.mark.parametrize("name", ["../outside", "data/../../outside", "/absolute", "data//double", "data/./dot", "data/C:\\outside", "other/file"])
def test_restore_rejects_traversal_and_outside_paths_without_partial_data(tmp_path, name):
    target = tmp_path / "target"
    stream = crafted_archive([("data/safe.txt", b"safe"), (name, b"attack")], files={"safe.txt": {"size": 4, "sha256": hashlib.sha256(b"safe").hexdigest()}})
    with pytest.raises(ValueError):
        restore(stream, target)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".paperleet-restore-*"))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_restore_rejects_links_and_special_files(tmp_path, kind):
    stream = crafted_archive([("data/link", b"", kind, "../../outside")], files={})
    with pytest.raises(ValueError, match="links and special"):
        restore(stream, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_restore_rejects_duplicate_and_corrupted_files(tmp_path):
    duplicate = crafted_archive([("data/item", b"one"), ("data/item", b"one")])
    with pytest.raises(ValueError, match="duplicate"):
        restore(duplicate, tmp_path / "duplicate")
    wrong_hash = {"item": {"size": 3, "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="integrity"):
        restore(crafted_archive([("data/item", b"one")], wrong_hash), tmp_path / "corrupt")
    missing = {"item": {"size": 3, "sha256": hashlib.sha256(b"one").hexdigest()}}
    with pytest.raises(ValueError, match="missing files"):
        restore(crafted_archive([], missing), tmp_path / "missing")


def test_restore_rejects_invalid_sqlite_snapshot_and_retains_empty_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(sqlite3.DatabaseError):
        restore(crafted_archive([("data/community.sqlite3", b"not a database")]), target)
    assert list(target.iterdir()) == []


def test_source_and_destination_symlinks_are_rejected(live_data, tmp_path):
    data, _ = live_data
    (data / "external-link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        backup(io.BytesIO(), data)
    (data / "external-link").unlink()
    target = tmp_path / "symlink-target"
    real = tmp_path / "real"
    real.mkdir()
    target.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="empty data directory"):
        restore(saved_archive(data), target)
    assert list(real.iterdir()) == []


def test_restore_limits_expansion_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.setattr("paperleet.backup.MAX_TOTAL_BYTES", 2)
    with pytest.raises(ValueError, match="restored size"):
        restore(crafted_archive([("data/item", b"large")]), tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_implicit_parent_directories_are_private(tmp_path):
    target = tmp_path / "data"
    restore(crafted_archive([("data/a/b/item", b"secret")]), target)
    for path in [target, target / "a", target / "a/b"]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_cli_streams_only_archive_bytes_to_stdout_and_restores_from_stdin(live_data, tmp_path):
    data, _ = live_data
    environment = dict(os.environ, PAPERLEET_DATA_DIR=str(data))
    exported = subprocess.run([sys.executable, "-m", "paperleet.backup", "backup", "--stdout"],
                              env=environment, capture_output=True, check=True)
    assert exported.stdout.startswith(b"\x1f\x8b")
    assert b"Backup complete" in exported.stderr
    target = tmp_path / "restored-volume/data"
    environment["PAPERLEET_DATA_DIR"] = str(target)
    imported = subprocess.run([sys.executable, "-m", "paperleet.backup", "restore", "--stdin"],
                              env=environment, input=exported.stdout, capture_output=True, check=True)
    assert imported.stdout == b""
    assert b"Restore complete" in imported.stderr
    assert (target / "community.sqlite3").exists()
    rejected = subprocess.run([sys.executable, "-m", "paperleet.backup", "restore", "--stdin"],
                             env=environment, input=exported.stdout, capture_output=True)
    assert rejected.returncode == 1 and rejected.stdout == b""
    assert b"never overwrites existing data" in rejected.stderr
