"""Restore a nested adaptation without losing local changes or requiring a network."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.shared.robosuite_patch import export_patch, restore_patch


class RobosuitePatchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.git(self.repo, "init", "-q")
        (self.repo / "registered.py").write_text("original\n")
        self.git(self.repo, "add", "registered.py")
        self.git(self.repo, "-c", "user.name=Patch Test", "-c", "user.email=patch@test.invalid",
                 "commit", "-qm", "base")
        self.fresh = self.root / "clone"
        subprocess.run(["git", "clone", "-q", str(self.repo), str(self.fresh)], check=True)
        (self.repo / "registered.py").write_text("original\nadaptation\n")
        (self.repo / "model.bin").write_bytes(bytes(range(256)) * 2)
        self.patch = self.root / "patches/adaptation.patch"
        export_patch(self.repo, self.patch)

    def git(self, repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    def test_clean_clone_restores_sources_and_binary_files_idempotently(self):
        self.assertTrue(restore_patch(apply=True, repo=self.fresh, patch=self.patch)["applied"])
        for name in ("registered.py", "model.bin"):
            self.assertEqual((self.fresh / name).read_bytes(), (self.repo / name).read_bytes())
        self.assertFalse(restore_patch(apply=True, repo=self.fresh, patch=self.patch)["applied"])

    def test_conflicting_local_edit_is_preserved(self):
        file = self.fresh / "registered.py"
        file.write_text("local edit\n")
        with self.assertRaises(RuntimeError):
            restore_patch(apply=True, repo=self.fresh, patch=self.patch)
        self.assertEqual(file.read_text(), "local edit\n")
        self.assertFalse((self.fresh / "model.bin").exists())

    def test_damaged_patch_is_rejected_before_modification(self):
        self.patch.write_bytes(self.patch.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            restore_patch(apply=True, repo=self.fresh, patch=self.patch)
        self.assertEqual((self.fresh / "registered.py").read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
