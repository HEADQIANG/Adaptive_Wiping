"""Package identity and unambiguous source evidence across the package rename."""

import ast
import importlib.util
import json
import unittest

from scripts.shared.artifacts import file_hash
from scripts.shared.paths import (
    ARCHIVE,
    ROOT,
    historical_file_records,
    project_source_files,
    read_path,
    writable_path,
)


class PackageRenameTests(unittest.TestCase):
    def test_new_package_only_and_no_stale_imports(self):
        self.assertIsNone(importlib.util.find_spec("adaptive_wiping"))
        self.assertEqual(
            importlib.util.find_spec("scripts").origin, str(ROOT / "scripts/__init__.py")
        )
        for folder in (ROOT / "scripts", ROOT / "tests"):
            for path in project_source_files(folder):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]
                    self.assertFalse(
                        any(name.split(".")[0] == "adaptive_wiping" for name in names), path
                    )

    def test_all_rename_snapshot_files_are_preserved(self):
        records = json.loads((ARCHIVE / "_migration/package_rename.json").read_text())
        all_records = historical_file_records()
        for record in records:
            path = ROOT / record["new"]
            self.assertEqual(path.stat().st_size, record["size"])
            self.assertEqual(file_hash(path), record["sha256"])
            self.assertIn(record, all_records)

    def test_hash_selects_the_correct_generation_without_rewriting_old_hashes(self):
        from scripts.sim_pretrain.experiments.vae_ablation import verify_snapshot

        key = "adaptive_wiping/real_training/training.py"
        versions = [record for record in historical_file_records() if record["old"] == key]
        self.assertGreaterEqual(len({record["sha256"] for record in versions}), 2)
        paths = set()
        for record in versions:
            path = read_path(key, historical=True, source_sha256=record["sha256"])
            self.assertEqual(file_hash(path), record["sha256"])
            self.assertTrue(verify_snapshot({key: record["sha256"]}, historical=True)["unchanged"])
            paths.add(path)
        self.assertGreaterEqual(len(paths), 2)
        self.assertTrue(
            read_path(key, historical=True).is_relative_to(ARCHIVE / "_migration/source_snapshot")
        )
        self.assertEqual(read_path(key, historical=True, source_sha256="0" * 64), ROOT / key)
        self.assertFalse(verify_snapshot({key: "0" * 64}, historical=True)["unchanged"])
        with self.assertRaises(PermissionError):
            writable_path(key)


if __name__ == "__main__":
    unittest.main()
