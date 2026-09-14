"""Interrupted HDF5 recovery must not discard or change committed records."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.shared.common import file_digest, load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    collect_mode,
    initialize,
)
from scripts.sim_pretrain.experiments.recover_control_dataset import (
    read_text,
    rebuild,
    recover,
)
from tests.sim_pretrain.test_comparison_collection import FakeWipe


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wiping-recovery-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        self.cfg["dataset"] = {"train": 2, "validation": 1, "test": 1}
        self.manifest, self.configs = initialize(self.root, self.cfg)
        FakeWipe.fail_next = False
        collect_mode(
            self.configs["normal"], self.manifest, True, max_trajectories=1, factory=FakeWipe
        )
        self.path = self.root / "normal/dataset.h5"

    def test_backup_and_committed_data_preserved_and_resumable(self):
        original_hash = file_digest(self.path)
        report = recover(self.root, "normal", self.cfg)
        self.assertEqual(file_digest(report["backup"]), original_hash)
        self.assertTrue(report["committed_rows_verified"])
        self.assertEqual(report["unreadable_uncommitted_strings"], [])
        result = collect_mode(self.configs["normal"], self.manifest, True, factory=FakeWipe)
        self.assertTrue(result["complete"])
        with h5py.File(self.path, "r") as h5:
            np.testing.assert_array_equal(h5["train/attempts"][:], [1, 1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            recover(self.root, "normal", self.cfg)

    def test_only_uncommitted_unreadable_string_can_be_replaced(self):
        def damaged(dataset, index):
            if dataset.file.mode == "r" and dataset.name == "/train/acceptance_json" and index == 1:
                raise OSError("synthetic interrupted global heap")
            return read_text(dataset, index)

        with (
            h5py.File(self.path, "r") as source,
            h5py.File(self.root / "candidate.h5", "w") as dest,
        ):
            with patch(
                "scripts.sim_pretrain.experiments.recover_control_dataset.read_text",
                side_effect=damaged,
            ):
                errors = rebuild(source, dest, self.configs["normal"], self.manifest)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["index"], 1)
            self.assertEqual(dest["train/acceptance_json"].asstr()[1], "")
            np.testing.assert_array_equal(dest["train/valid"][:], [True, False])
            self.assertEqual(json.loads(dest["train/acceptance_json"].asstr()[0])["passed"], False)

    def test_committed_corruption_aborts_without_replacing_source(self):
        original_hash = file_digest(self.path)

        def damaged(dataset, index):
            if dataset.file.mode == "r" and dataset.name == "/train/acceptance_json" and index == 0:
                raise OSError("synthetic committed corruption")
            return read_text(dataset, index)

        with patch(
            "scripts.sim_pretrain.experiments.recover_control_dataset.read_text",
            side_effect=damaged,
        ):
            with self.assertRaisesRegex(ValueError, "Committed string"):
                recover(self.root, "normal", self.cfg)
        self.assertEqual(file_digest(self.path), original_hash)


if __name__ == "__main__":
    unittest.main()
