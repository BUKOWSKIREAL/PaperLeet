"""Synchronizer checks use tiny Git fixtures, never the judge or network."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("sync_upstream", Path(__file__).resolve().parents[1] / "scripts/sync_upstream.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def command(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


def put(root, path, value):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value)


def commit(root):
    command(root, "add", "-A")
    command(root, "commit", "-qm", "fixture")
    return command(root, "rev-parse", "HEAD")


class SynchronizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "offline"
        self.source = Path(self.temporary.name) / "upstream"
        for root in (self.root, self.source):
            root.mkdir()
            command(root, "init", "-q", "-b", "main")
            command(root, "config", "user.name", "Fixture")
            command(root, "config", "user.email", "fixture@example.test")
        command(self.source, "remote", "add", "origin", sync.URL)
        self.config = {"version": 1, "repository": sync.REPOSITORY, "files": ["papers.yaml"],
                       "directories": ["problems/", "paperleet/common/"], "per_problem_visualizations": True}
        put(self.root, sync.CONFIG, json.dumps(self.config))
        put(self.root, "README.md", "Offline owned")
        commit(self.root)
        put(self.source, "papers.yaml", "papers: []\n")
        put(self.source, "problems/demo/one/problem.yaml", "id: demo/one\nnumber: 1\n")
        put(self.source, "problems/demo/one/starter.py", "raise NotImplementedError\n")
        put(self.source, "web/public/data/demo-one.json", '{"points": []}')
        put(self.source, "paperleet/common/asserts.py", "def assert_equal(a, b): assert a == b\n")
        self.revision = commit(self.source)

    def initial(self):
        result = sync.synchronize(self.root, self.source, self.revision, bootstrap=True)
        commit(self.root)
        return result

    def test_allowlist_omits_online_private_and_generated_artifacts(self):
        for path in ["web/src/App.tsx", "web/public/data/catalog.json", "web/public/data/index.json",
                     "web/public/data/modules/one.json", "web/public/data/browser-judge/one.json",
                     "web/public/data/unrelated-example.json", "deploy/config.txt", ".env", "workspace/one.py",
                     "reports/one.json", "paperleet/accounts.py", "paperleet/server.py"]:
            put(self.source, path, "not for the offline edition")
        self.revision = commit(self.source)
        report = self.initial()
        self.assertEqual(report["managed_files"], 5)
        self.assertFalse((self.root / "deploy").exists())
        self.assertEqual((self.root / "README.md").read_text(), "Offline owned")
        self.assertEqual(set(sync.read_lock(self.root)["files"]), {
            "papers.yaml", "problems/demo/one/problem.yaml", "problems/demo/one/starter.py",
            "web/public/data/demo-one.json", "paperleet/common/asserts.py"})

    def test_same_snapshot_and_unrelated_upstream_commit_are_noops(self):
        self.initial()
        before = (self.root / sync.LOCK).read_bytes()
        self.assertFalse(sync.synchronize(self.root, self.source, self.revision)["changed"])
        put(self.source, "web/src/App.tsx", "online-only update")
        newer = commit(self.source)
        self.assertFalse(sync.synchronize(self.root, self.source, newer)["changed"])
        self.assertEqual(before, (self.root / sync.LOCK).read_bytes())

    def test_fixed_commit_ignores_later_source_worktree_and_head_changes(self):
        put(self.source, "papers.yaml", "different revision")
        commit(self.source)
        self.initial()
        self.assertEqual((self.root / "papers.yaml").read_text(), "papers: []\n")

    def test_curriculum_is_an_explicit_frontend_exception(self):
        self.config["files"].append("web/src/paperCollections.ts")
        put(self.root, sync.CONFIG, json.dumps(self.config))
        put(self.source, "web/src/paperCollections.ts", "export const collections = ['gqa'];\n")
        put(self.source, "web/src/App.tsx", "online application")
        self.revision = commit(self.source)
        self.initial()
        self.assertIn("web/src/paperCollections.ts", sync.read_lock(self.root)["files"])
        self.assertFalse((self.root / "web/src/App.tsx").exists())

    def test_local_edits_fail_before_any_other_file_is_written(self):
        self.initial()
        before = (self.root / sync.LOCK).read_bytes()
        put(self.root, "problems/demo/one/starter.py", "my maintained correction")
        put(self.source, "papers.yaml", "updated papers")
        newer = commit(self.source)
        with self.assertRaisesRegex(sync.SyncError, "Local changes conflict"):
            sync.synchronize(self.root, self.source, newer)
        self.assertEqual((self.root / "papers.yaml").read_text(), "papers: []\n")
        self.assertEqual((self.root / sync.LOCK).read_bytes(), before)

    def test_managed_upstream_deletion_preserves_unmanaged_files(self):
        self.initial()
        put(self.root, "problems/demo/one/local-notes.md", "mine")
        (self.source / "problems/demo/one/starter.py").unlink()
        report = sync.synchronize(self.root, self.source, commit(self.source))
        self.assertEqual(report["removed"], ["problems/demo/one/starter.py"])
        self.assertTrue((self.root / "problems/demo/one/local-notes.md").exists())

    def test_upstream_addition_cannot_clobber_an_unmanaged_local_file(self):
        self.initial()
        put(self.root, "problems/demo/one/notes.md", "mine")
        put(self.source, "problems/demo/one/notes.md", "upstream")
        with self.assertRaisesRegex(sync.SyncError, "Local changes conflict"):
            sync.synchronize(self.root, self.source, commit(self.source))
        self.assertEqual((self.root / "problems/demo/one/notes.md").read_text(), "mine")

    def test_bootstrap_only_replaces_existing_tracked_clean_files(self):
        put(self.root, "papers.yaml", "older tracked version")
        commit(self.root)
        put(self.root, "papers.yaml", "uncommitted work")
        with self.assertRaisesRegex(sync.SyncError, "Local changes conflict"):
            sync.synchronize(self.root, self.source, self.revision, bootstrap=True)
        self.assertFalse((self.root / sync.LOCK).exists())

    def test_source_and_destination_symlinks_are_rejected(self):
        target = self.source / "problems/demo/one/starter.py"
        target.unlink()
        target.symlink_to("../../../papers.yaml")
        with self.assertRaisesRegex(sync.SyncError, "Symlinks"):
            sync.synchronize(self.root, self.source, commit(self.source), bootstrap=True)
        command(self.source, "reset", "--hard", self.revision)
        (self.root / "problems").symlink_to(self.source / "problems")
        with self.assertRaisesRegex(sync.SyncError, "symlink"):
            sync.synchronize(self.root, self.source, self.revision, bootstrap=True)

    def test_broken_lock_symlink_and_traversal_are_rejected(self):
        (self.root / sync.LOCK).symlink_to(self.root / "missing")
        with self.assertRaisesRegex(sync.SyncError, "symlink"):
            sync.synchronize(self.root, self.source, self.revision, bootstrap=True)
        for path in ["../outside", "/absolute", "problems/a/b/../../x", "a\\b", "problems/a/b/.env", "reports/x"]:
            with self.subTest(path=path), self.assertRaises(sync.SyncError):
                sync.safe_path(path)

    def test_dry_run_and_manifest_validation(self):
        report = sync.synchronize(self.root, self.source, self.revision, bootstrap=True, dry_run=True)
        self.assertTrue(report["changed"])
        self.assertFalse((self.root / sync.LOCK).exists())
        self.initial()
        sync.verify_lock(self.root)
        put(self.root, "papers.yaml", "changed")
        with self.assertRaisesRegex(sync.SyncError, "does not match"):
            sync.verify_lock(self.root)

    def test_policy_narrowing_requires_explicit_manifest_migration(self):
        self.initial()
        self.config["directories"].remove("paperleet/common/")
        put(self.root, sync.CONFIG, json.dumps(self.config))
        with self.assertRaisesRegex(sync.SyncError, "lock migration"):
            sync.synchronize(self.root, self.source, self.revision)
        self.assertTrue((self.root / "paperleet/common/asserts.py").exists())

    def test_patch_contains_only_managed_changes_and_replays(self):
        self.initial()
        before = command(self.root, "rev-parse", "HEAD")
        put(self.source, "papers.yaml", "next papers")
        put(self.source, "problems/demo/one/notes.md", "new notes")
        sync.synchronize(self.root, self.source, commit(self.source))
        put(self.root, "README.md", "not part of the synchronization")
        patch = Path(self.temporary.name) / "upstream.patch"
        sync.write_patch(self.root, patch)
        self.assertNotIn("README.md", patch.read_text())
        command(self.root, "reset", "--hard", before)
        command(self.root, "clean", "-fd")
        sync.git(self.root, "apply", "--index", str(patch))
        sync.verify_lock(self.root, staged=True)
        self.assertEqual((self.root / "papers.yaml").read_text(), "next papers")

    def test_publishing_rejects_unmanaged_and_unrecorded_patch_paths(self):
        self.initial()
        for path in ["README.md", "problems/demo/one/unrecorded.py"]:
            with self.subTest(path=path):
                put(self.root, path, "must not be published")
                command(self.root, "add", path)
                with self.assertRaisesRegex(sync.SyncError, "unmanaged file"):
                    sync.verify_lock(self.root, staged=True)
                command(self.root, "reset", "--hard", "HEAD")
                command(self.root, "clean", "-fd")

    def test_manifest_cannot_drop_a_required_core_file(self):
        self.initial()
        lock = sync.read_lock(self.root)
        del lock["files"]["papers.yaml"]
        put(self.root, sync.LOCK, json.dumps(lock))
        with self.assertRaisesRegex(sync.SyncError, "required managed files"):
            sync.verify_lock(self.root)


if __name__ == "__main__":
    unittest.main()
