"""Explicit research authorization, paired assignments and resumable collection."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    EXTRA_FIELDS,
    MODES,
    CollectionWipe,
    collect_mode,
    initialize,
    verify,
)
from scripts.sim_pretrain.experiments.summarize_control_data import (
    split_statistics,
    summarize,
)
from scripts.sim_pretrain.learning import load_data
from scripts.sim_pretrain.simulation import exploration_offsets


class FakeWipe:
    fail_next = False

    def __init__(self, cfg, parameters, spec):
        self.parameters = parameters
        self.spec = spec

    def set_parameters(self, parameters):
        self.parameters = parameters

    def rollout(self):
        if FakeWipe.fail_next:
            FakeWipe.fail_next = False
            raise RuntimeError("synthetic numerical failure")
        data = {key: np.zeros(shape) for key, shape in {**FIELDS, **EXTRA_FIELDS}.items()}
        data["time"] = np.arange(1, 401) / 100
        data["target_position"] = exploration_offsets(data["time"])
        data["contact_count"][20:] = 1
        data["contact_count_loaded"][20:] = 1
        data["ft"][:] = float(self.parameters[0])
        data["parameters"] = np.asarray(self.parameters).copy()
        return data

    def unload(self, data):
        return [0] * 6

    def close(self):
        pass


class ComparisonCollectionTests(unittest.TestCase):
    def test_reused_environment_matches_fresh_second_assignment(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        for spec in MODES.values():
            reused = CollectionWipe(cfg, (0.9, 1000, 0.02), spec)
            fresh = None
            try:
                first = reused.rollout()
                reused.unload(first)
                parameters = (3.5, 0.5, 0.3)
                reused.set_parameters(parameters)
                actual = reused.rollout()
                fresh = CollectionWipe(cfg, parameters, spec)
                expected = fresh.rollout()
                for key in {**FIELDS, **EXTRA_FIELDS}:
                    np.testing.assert_array_equal(actual[key], expected[key])
            finally:
                reused.close()
                if fresh is not None:
                    fresh.close()

    def test_vectorized_checks_match_saved_pilot_dynamics(self):
        pilot = ROOT / "archive/sim_pretrain/two_control_pretraining_v1"
        if not (pilot / "manifest.json").exists():
            self.skipTest("Historical pilot is not available")
        manifest = json.loads((pilot / "manifest.json").read_text())
        for name, spec in MODES.items():
            env = CollectionWipe(manifest["base_config"], manifest["assignments"]["train"][0], spec)
            try:
                actual = env.rollout()
                env.unload(actual)
                with h5py.File(pilot / name / "dataset.h5", "r") as h5:
                    for key in {**FIELDS, **EXTRA_FIELDS}:
                        np.testing.assert_array_equal(actual[key], h5["train"][key][0])
                original_contacts = env.contacts()
                env.sim.data.qpos[env.robot.part_controllers["right"].qpos_index[0]] += 0.01
                env.sim.forward()
                from scripts.sim_pretrain.simulation import PretrainingWipe

                self.assertEqual(env.contacts(), PretrainingWipe.contacts(env))
                self.assertEqual(original_contacts, 0)
            finally:
                env.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wiping-comparison-")
        self.addCleanup(self.tmp.cleanup)
        self.cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        self.cfg["dataset"] = {"train": 2, "validation": 1, "test": 1}
        self.manifest, self.configs = initialize(self.tmp.name, self.cfg)
        FakeWipe.fail_next = False

    def test_explicit_authorization_required(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            collect_mode(self.configs["normal"], self.manifest, factory=FakeWipe)
        self.assertFalse((Path(self.tmp.name) / "normal/dataset.h5").exists())

    def test_summary_requires_complete_paired_data(self):
        for cfg in self.configs.values():
            collect_mode(cfg, self.manifest, True, max_trajectories=1, factory=FakeWipe)
        with self.assertRaisesRegex(ValueError, "both collections"):
            summarize(self.tmp.name)
        for cfg in self.configs.values():
            collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        result = summarize(self.tmp.name)
        self.assertEqual(result["unique_parameter_assignments"], 4)
        self.assertEqual(result["paired_target_max_difference_m"]["test"], 0)
        self.assertEqual(result["modes"]["normal"]["splits"]["train"]["count"], 2)
        self.assertTrue((Path(self.tmp.name) / "paired_sample.png").exists())

    def test_statistics_preserve_failed_motion_and_reject_nonfinite(self):
        cfg = self.configs["normal"]
        collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        with h5py.File(Path(cfg["output_dir"]) / "dataset.h5", "a") as h5:
            result = split_statistics(h5["train"])
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["motion_pass_count"], 0)
            self.assertEqual(result["failed_motion_check_counts"]["forward_travel"], 2)
            h5["train/ft"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "raw FT"):
                split_statistics(h5["train"])

    def test_failed_motion_retained_paired_and_loadable_for_training(self):
        for cfg in self.configs.values():
            result = collect_mode(cfg, self.manifest, True, factory=FakeWipe)
            self.assertTrue(result["complete"])
            with h5py.File(Path(cfg["output_dir"]) / "dataset.h5", "r") as h5:
                self.assertTrue(h5["train/valid"][:].all())
                self.assertFalse(h5["train/motion_passed"][:].any())
                self.assertIn("NOT", h5.attrs["valid_meaning"])
            raw, _ = load_data(cfg)
            self.assertEqual(raw["train"].shape, (2, 400, 6))
        self.assertTrue(verify(self.tmp.name, self.manifest, self.configs)["complete"])

    def test_partial_resume_preserves_committed_rows(self):
        cfg = self.configs["normal"]
        result = collect_mode(cfg, self.manifest, True, max_trajectories=1, factory=FakeWipe)
        self.assertFalse(result["complete"])
        with h5py.File(Path(cfg["output_dir"]) / "dataset.h5", "r") as h5:
            first = h5["train/ft"][0]
        result = collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        self.assertTrue(result["complete"])
        with h5py.File(Path(cfg["output_dir"]) / "dataset.h5", "r") as h5:
            np.testing.assert_array_equal(first, h5["train/ft"][0])
            np.testing.assert_array_equal(h5["train/attempts"][:], [1, 1])

    def test_failure_retry_keeps_same_parameters_and_history(self):
        cfg = self.configs["normal"]
        FakeWipe.fail_next = True
        result = collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        self.assertFalse(result["complete"])
        self.assertEqual(result["splits"]["train"]["failed"], 1)
        result = collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        self.assertFalse(result["complete"])
        result = collect_mode(cfg, self.manifest, True, retry_failed=True, factory=FakeWipe)
        self.assertTrue(result["complete"])
        with h5py.File(Path(cfg["output_dir"]) / "dataset.h5", "r") as h5:
            np.testing.assert_array_equal(
                h5["train/parameters"][:], self.manifest["assignments"]["train"]
            )
            self.assertEqual(h5["train/attempts"][0], 2)
        history = (Path(cfg["output_dir"]) / "failure_history.jsonl").read_text().splitlines()
        self.assertEqual(len(history), 1)
        self.assertEqual(
            json.loads(history[0])["parameters"], self.manifest["assignments"]["train"][0]
        )

    def test_changed_configuration_and_corrupted_trace_rejected(self):
        self.cfg["training"]["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "manifest"):
            initialize(self.tmp.name, self.cfg)
        for cfg in self.configs.values():
            collect_mode(cfg, self.manifest, True, factory=FakeWipe)
        with h5py.File(Path(self.tmp.name) / "normal/dataset.h5", "a") as h5:
            h5["train/ft"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "trajectory"):
            verify(self.tmp.name, self.manifest, self.configs)


if __name__ == "__main__":
    unittest.main()
