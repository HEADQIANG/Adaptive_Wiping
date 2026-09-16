"""Exercise relocation, source bindings and split simulation storage without hardware."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.shared import paths
from scripts.shared.run_layout import relocated_run_path, sim_data_path


class RunsLayoutTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name, value in (("ROOT", self.root), ("RUNS", self.root / "runs")):
            replacement = patch.object(paths, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_saved_absolute_and_relative_paths_resolve_to_same_bytes(self):
        old = "runs/real_training/manual_demonstrations/0916_161723/session_001/session.json"
        target = self.root / "runs/real_demonstrations/manual/0916_161723/session_001/session.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b'original metadata')
        for value in (old, self.root / old):
            self.assertEqual(paths.read_path(value), target)
            self.assertEqual(paths.read_path(value).read_bytes(), b'original metadata')
        bindings = {str(self.root / old): "original-hash"}
        self.assertEqual(paths.recorded_source_hash(bindings, target), "original-hash")
        bindings[str(target)] = "different-hash"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            paths.recorded_source_hash(bindings, target)

    def test_simulation_run_keeps_model_and_data_categories_distinct(self):
        cases = {
            "sim_pretrain/explore_once_20260912/exploration.npz": "sim_data/explore_once_20260912/exploration.npz",
            "sim_pretrain/0916_170000/explore_once/ft.png": "sim_data/0916_170000/explore_once/ft.png",
            "sim_pretrain/pretrain_wide_1200_v1/dataset.h5": "sim_data/pretrain_wide_1200_v1/dataset.h5",
            "sim_pretrain/pretrain_wide_1200_v1/encoder.pt": "sim_training/pretrain_wide_1200_v1/encoder.pt",
            "robot_control/sdk_logs/all.log": "real_deploy/robot_control/sdk_logs/all.log",
        }
        for old, new in cases.items():
            self.assertEqual(paths.read_path("runs/" + old), self.root / "runs" / new)
            self.assertEqual(paths.writable_path("runs/" + old), self.root / "runs" / new)
        outside = self.root.parent / "unrelated/runs/sim_pretrain/dataset.h5"
        self.assertEqual(relocated_run_path(outside, self.root), outside)

    def test_atomic_data_writes_preserve_training_links(self):
        from scripts.shared.common import write_json
        import h5py

        out = self.root / "runs/sim_training/0916_180000/experiment"
        target = sim_data_path(out, "dataset.h5")
        with h5py.File(target, "w") as stream:
            stream.create_dataset("example", data=[1, 2, 3])
        self.assertEqual(target, self.root / "runs/sim_data/0916_180000/experiment/dataset.h5")
        with h5py.File(out / "dataset.h5", "r") as stream:
            self.assertEqual(stream["example"][:].tolist(), [1, 2, 3])
        info = sim_data_path(out, "collection.json")
        for count in (1, 2):
            write_json(out / "collection.json", {"count": count})
            self.assertEqual(json.loads(info.read_text())["count"], count)
            self.assertTrue((out / "collection.json").is_symlink())
        self.assertEqual(sim_data_path(out, "dataset.h5"), target)

    def test_existing_data_is_not_silently_moved_or_overwritten(self):
        out = self.root / "runs/sim_training/legacy"
        out.mkdir(parents=True)
        original = out / "dataset.h5"
        original.write_bytes(b"saved")
        with self.assertRaises(FileExistsError):
            sim_data_path(out, "dataset.h5")
        self.assertEqual(original.read_bytes(), b"saved")
        external = self.root / "custom"
        self.assertEqual(sim_data_path(external, "dataset.h5"), external / "dataset.h5")
        self.assertFalse(external.exists())

    def test_cleanup_preserves_completed_policy_with_handoff_fault(self):
        from scripts.shared import reorganize_runs as migration

        for name, events in (
            ("startup_fault", ["session_start", "fault"]),
            ("complete", ["session_start", "sample", "policy_complete", "fault"]),
        ):
            out = self.root / "runs/real_deploy" / name
            out.mkdir(parents=True)
            (out / "events.jsonl").write_text("\n".join(json.dumps({"event": e}) for e in events))
        with patch.object(migration, "ROOT", self.root):
            targets = migration.incomplete_targets()
        self.assertIn(self.root / "runs/real_deploy/startup_fault", targets)
        self.assertNotIn(self.root / "runs/real_deploy/complete", targets)


if __name__ == "__main__":
    unittest.main()
