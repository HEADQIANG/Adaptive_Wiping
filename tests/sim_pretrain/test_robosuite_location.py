"""Nested vendor imports, source mapping and ownership boundaries."""

import json
import unittest
from pathlib import Path

from scripts.shared.assets import verify
from scripts.shared.common import file_digest, provenance
from scripts.shared.paths import (
    ARCHIVE,
    ROOT,
    ROBOSUITE_PACKAGE,
    ROBOSUITE_REPO,
    project_source_files,
    read_path,
    writable_path,
)
from scripts.sim_pretrain.experiments.vae_ablation import verify_snapshot

import robosuite


class RobosuiteLocationTests(unittest.TestCase):
    def test_nested_vendor_is_loaded_and_old_root_is_absent(self):
        self.assertFalse((ROOT / "robosuite").exists())
        self.assertEqual(Path(robosuite.__file__).resolve(), ROBOSUITE_PACKAGE / "__init__.py")
        self.assertTrue((ROBOSUITE_REPO / ".git/HEAD").is_file())
        self.assertTrue(verify(sources=True)["passed"])

    def test_old_source_paths_relocate_without_bypassing_hash_checks(self):
        old = "robosuite/robosuite/controllers/config/robots/default_airbot_play.json"
        new = ROBOSUITE_PACKAGE / "controllers/config/robots/default_airbot_play.json"
        self.assertEqual(read_path(old), new)
        self.assertEqual(read_path(old, historical=True), new)
        self.assertEqual(read_path("robosuite"), ROBOSUITE_REPO)
        self.assertTrue(
            verify_snapshot({str(ROOT / old): file_digest(new)}, historical=True)["unchanged"]
        )
        self.assertFalse(verify_snapshot({str(ROOT / old): "0" * 64}, historical=True)["unchanged"])
        with self.assertRaises(PermissionError):
            writable_path(old)

    def test_source_collection_excludes_vendor_but_records_actual_dependencies(self):
        files = project_source_files()
        self.assertIn(ROOT / "scripts/sim_pretrain/simulation.py", files)
        self.assertFalse(any(p.is_relative_to(ROBOSUITE_REPO) for p in files))
        sources = provenance()["sources"]
        selected = (ROBOSUITE_PACKAGE / "models/robots/robot_model.py").relative_to(ROOT).as_posix()
        self.assertIn(selected, sources)
        self.assertFalse(any(key.startswith("robosuite/") for key in sources))
        self.assertFalse(any("robosuite/tests/" in key for key in sources))

    def test_packaging_excludes_nested_external_distribution(self):
        from setuptools import find_namespace_packages
        from setuptools.config.pyprojecttoml import read_configuration

        config = read_configuration(str(ROOT / "pyproject.toml"), expand=False)
        discovery = config["tool"]["setuptools"]["packages"]["find"]
        packages = find_namespace_packages(
            where=str(ROOT), include=discovery["include"], exclude=discovery["exclude"]
        )
        self.assertIn("scripts.sim_pretrain", packages)
        self.assertFalse(
            any(name.startswith("scripts.sim_pretrain.robosuite") for name in packages)
        )

    def test_migration_inventory_preserves_existing_airbot_adaptations(self):
        records = json.loads((ARCHIVE / "_migration/robosuite_move.json").read_text())
        for old in (
            "robosuite/robosuite/models/robots/manipulators/airbot_play_robot.py",
            "robosuite/robosuite/robots/__init__.py",
            "robosuite/docs_codex/airbot_play_adapt.md",
        ):
            record = next(r for r in records if r["old"] == old)
            path = ROOT / record["new"]
            self.assertEqual(path, read_path(old))
            self.assertEqual(path.stat().st_size, record["size"])
            self.assertEqual(file_digest(path), record["sha256"])


if __name__ == "__main__":
    unittest.main()
