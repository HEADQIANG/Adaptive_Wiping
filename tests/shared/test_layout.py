"""Layout, read-only history and optional dependency boundaries."""

import ast
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.shared.artifacts import verify
from scripts.shared.paths import ARCHIVE, ROOT, project_source_files, read_path, writable_path


class LayoutTests(unittest.TestCase):
    def test_five_help_entrypoints_without_optional_dependencies(self):
        for module in (
            "sim_pretrain",
            "force_sensor",
            "real_training",
            "real_deploy",
            "robot_control",
        ):
            code = f"""
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'mujoco', 'robosuite', 'arm_sdk', 'serial'):
            raise RuntimeError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from scripts.{module}.__main__ import main
main(['--help'])
"""
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_motion_and_calibration_do_not_import_training(self):
        code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'mujoco', 'robosuite', 'arm_sdk') or fullname.startswith('scripts.real_training'):
            raise RuntimeError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import scripts.robot_control.motion
import scripts.shared.airbot_calibration
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_historical_lookup_and_source_snapshot_are_distinct(self):
        path = read_path("outputs/normal_mu1p2_pretraining_v1/normal/encoder.pt")
        self.assertTrue(path.is_file())
        self.assertTrue(path.is_relative_to(ARCHIVE / "sim_pretrain"))
        old_source = read_path("adaptive_wiping/real_training/training.py", historical=True)
        self.assertTrue(old_source.is_relative_to(ARCHIVE / "_migration/source_snapshot"))
        self.assertNotEqual(old_source, read_path("scripts/real_training/training.py"))
        with self.assertRaises(ValueError):
            read_path("outputs/real_robot", historical=True)

    def test_archive_and_old_output_writes_rejected(self):
        for name in (
            "archive/sim_pretrain/new.bin",
            "outputs/new.bin",
            "data/new.bin",
            "logs/new.bin",
        ):
            with self.assertRaises(PermissionError):
                writable_path(name)
        with self.assertRaises(PermissionError):
            (ARCHIVE / "must_not_be_created.txt").write_text("forbidden")
        self.assertFalse((ARCHIVE / "must_not_be_created.txt").exists())

    def test_archive_migration_can_resume_after_interrupted_rename(self):
        spec = importlib.util.spec_from_file_location(
            "layout_migration", ARCHIVE / "_migration/migrate_layout.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "outputs/normal_run").mkdir(parents=True)
            (root / "outputs/normal_run/a.bin").write_bytes(b"first")
            (root / "outputs/normal_run/b.bin").write_bytes(b"second")
            rename = Path.rename

            def interrupted(path, destination):
                if path.name == "b.bin":
                    raise OSError("interrupted")
                return rename(path, destination)

            with (
                patch.object(migration, "ROOT", root),
                patch.object(migration, "META", root / "archive/_migration"),
            ):
                with (
                    patch.object(Path, "rename", interrupted),
                    self.assertRaisesRegex(OSError, "interrupted"),
                ):
                    migration.archive()
                records = migration.archive()
                self.assertEqual(len(records), 2)
                self.assertFalse((root / "outputs").exists())
                for r in records:
                    self.assertEqual(migration.sha(root / r["new"]), r["sha256"])

    def test_archive_verifier_detects_missing_or_changed_files(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "a"
            p.write_bytes(b"original")
            from scripts.shared.artifacts import file_hash

            record = dict(new=str(p), size=8, sha256=file_hash(p))
            self.assertTrue(verify([record])["passed"])
            p.write_bytes(b"changed!")
            self.assertFalse(verify([record])["passed"])
            p.unlink()
            self.assertEqual(verify([record])["errors"][0]["error"], "missing")

    def test_no_old_application_entrypoints_remain(self):
        self.assertFalse((ROOT / "adaptive_wiping").exists())
        self.assertTrue((ROOT / "scripts/__init__.py").is_file())
        self.assertFalse((ROOT / "scripts/pretrain.py").exists())
        self.assertFalse((ROOT / "scripts/kw_ft").exists())

    def test_static_source_sibling_references_exist(self):
        for path in project_source_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "with_name"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    continue
                if isinstance(node.func.value, ast.Call) and any(
                    isinstance(a, ast.Name) and a.id == "__file__" for a in node.func.value.args
                ):
                    self.assertTrue(
                        path.with_name(node.args[0].value).exists(),
                        f"Missing sibling in {path}: {node.args[0].value}",
                    )


if __name__ == "__main__":
    unittest.main()
