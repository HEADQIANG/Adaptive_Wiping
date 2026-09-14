"""Stiffness labels, parameter isolation and reuse of the validated FT recorder."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.shared.common import load_config, write_json
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    EXTRA_FIELDS,
)
from scripts.sim_pretrain.experiments.stiffness_ft_sweep import (
    DEFAULT_STIFFNESSES,
    run_stiffness_sweep,
    validate_stiffnesses,
)
from scripts.sim_pretrain.simulation import exploration_offsets


class FakeStiffnessWipe:
    calls = []
    closed = 0
    fail_k = None

    def __init__(self, cfg, parameters, spec):
        self.parameters = parameters
        self.calls.append((parameters, spec))

    def rollout(self):
        if self.parameters[1] == self.fail_k:
            raise RuntimeError("synthetic stiffness failure")
        data = {key: np.zeros(shape) for key, shape in {**FIELDS, **EXTRA_FIELDS}.items()}
        data["time"] = np.arange(1, 401) / 100.0
        data["target_position"] = exploration_offsets(data["time"])
        data["contact_count"][:] = 1
        data["normal_sum"][:] = 1
        data["parameters"] = np.asarray(self.parameters)
        data["ft"][:] = np.arange(6)[None] + self.parameters[1] / 1000
        return data

    def unload(self, data):
        return np.zeros(6)

    def close(self):
        type(self).closed += 1


class StiffnessSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="stiffness-ft-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "frozen/config.json"
        self.cfg = load_config(
            ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
        )
        self.cfg["output_dir"] = str(self.config.parent)
        write_json(self.config, self.cfg)
        FakeStiffnessWipe.calls, FakeStiffnessWipe.closed, FakeStiffnessWipe.fail_k = [], 0, None

    def test_fixed_friction_width_order_exact_export_and_filter(self):
        out = self.root / "sweep"
        report = run_stiffness_sweep(self.config, out, factory=FakeStiffnessWipe)
        self.assertEqual(report["state"], "completed")
        self.assertTrue(report["sources_and_inputs_unchanged"])
        self.assertEqual(FakeStiffnessWipe.closed, 6)
        self.assertEqual(
            FakeStiffnessWipe.calls, [((0.9, k, 0.02), None) for k in DEFAULT_STIFFNESSES]
        )
        with np.load(out / "sweep.npz", allow_pickle=False) as data:
            self.assertEqual(data["ft"].shape, (6, 400, 6))
            np.testing.assert_array_equal(data["parameters"][:, 1], DEFAULT_STIFFNESSES)
            np.testing.assert_array_equal(
                data["ft_filtered"], Preprocessor(**self.cfg["filter"]).filtered(data["ft"])
            )
            for index, row in enumerate(report["runs"]):
                self.assertFalse(row["acceptance"]["passed"])
                self.assertEqual(row["index"], index)
                k = DEFAULT_STIFFNESSES[index]
                np.testing.assert_array_equal(row["contact_solref"], [-k, -2 * np.sqrt(k)])
                with np.load(out / row["trajectory_path"], allow_pickle=False) as single:
                    for key in single.files:
                        np.testing.assert_array_equal(data[key][index], single[key])
        for name in (
            "ft_raw_comparison.png",
            "ft_filtered_comparison.png",
            "motion_contact.png",
            "k_0.5/ft.png",
            "k_1000/ft.png",
        ):
            self.assertGreater((out / name).stat().st_size, 10000)

    def test_validation_and_output_protection(self):
        for values in ([], [0], [-1], [np.nan], [np.inf], [1, 1], [[1, 2]], [1.0, 1.00000001]):
            with self.assertRaises(ValueError):
                validate_stiffnesses(values)
        out = self.root / "existing"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            run_stiffness_sweep(self.config, out, factory=FakeStiffnessWipe)
        with self.assertRaisesRegex(ValueError, "outside"):
            run_stiffness_sweep(self.config, self.config.parent / "new", factory=FakeStiffnessWipe)
        for mu, width in ((-1, 0.02), (np.nan, 0.02), (0.9, 0)):
            with self.assertRaises(ValueError):
                run_stiffness_sweep(
                    self.config,
                    self.root / "invalid",
                    mu=mu,
                    width=width,
                    factory=FakeStiffnessWipe,
                )
        self.assertFalse((self.root / "invalid").exists())
        self.assertEqual(FakeStiffnessWipe.calls, [])

    def test_failed_stiffness_is_preserved_and_remaining_cases_attempted(self):
        FakeStiffnessWipe.fail_k = 10.0
        out = self.root / "failed"
        result = run_stiffness_sweep(self.config, out, factory=FakeStiffnessWipe, make_plots=False)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(FakeStiffnessWipe.closed, 6)
        self.assertEqual(
            [row["state"] for row in result["runs"]],
            ["completed", "failed", "completed", "completed", "completed", "completed"],
        )
        self.assertEqual(result["runs"][1]["parameters"], [0.9, 10.0, 0.02])
        self.assertIn("synthetic stiffness failure", result["runs"][1]["error"])
        self.assertFalse((out / "sweep.npz").exists())


if __name__ == "__main__":
    unittest.main()
