"""Isolated single-factor sweep, physical units and output protection."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.shared.common import load_config, write_json
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.friction_ft_sweep import (
    DEFAULT_MUS,
    EXTRA_FIELDS,
    PANEL_CHANNELS,
    prepare_output,
    run_sweep,
    summarize_case,
)
from scripts.sim_pretrain.simulation import exploration_offsets


class FakeWipe:
    calls = []
    closed = 0
    fail_mu = None

    def __init__(self, cfg, parameters, spec):
        self.parameters = parameters
        self.calls.append((parameters, spec))

    def rollout(self):
        if self.parameters[0] == self.fail_mu:
            raise RuntimeError("synthetic failure")
        data = {key: np.zeros(shape) for key, shape in {**FIELDS, **EXTRA_FIELDS}.items()}
        data["time"] = np.arange(1, 401) / 100.0
        data["target_position"] = exploration_offsets(data["time"])
        data["contact_count"][:] = 1
        data["normal_sum"][:] = 1
        data["parameters"] = np.asarray(self.parameters)
        data["ft"][:] = np.arange(6)[None] + self.parameters[0]
        return data

    def unload(self, data):
        return np.zeros(6)

    def close(self):
        type(self).closed += 1


class FrictionSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="friction-ft-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "frozen/config.json"
        self.cfg = load_config(
            ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
        )
        self.cfg["output_dir"] = str(self.root / "frozen")
        write_json(self.config, self.cfg)
        FakeWipe.calls, FakeWipe.closed, FakeWipe.fail_mu = [], 0, None

    def test_single_factor_order_exact_filter_and_failed_motion_retained(self):
        out = self.root / "sweep"
        result = run_sweep(self.config, out, factory=FakeWipe)
        self.assertEqual(result["state"], "completed")
        self.assertTrue(result["sources_and_inputs_unchanged"])
        self.assertEqual(FakeWipe.closed, 5)
        self.assertEqual(FakeWipe.calls, [((mu, 1000.0, 0.02), None) for mu in DEFAULT_MUS])
        self.assertTrue(all(not row["acceptance"]["passed"] for row in result["runs"]))
        self.assertEqual(PANEL_CHANNELS, (0, 3, 1, 4, 2, 5))
        with np.load(out / "sweep.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["ft"].shape, (5, 400, 6))
            expected = Preprocessor(**self.cfg["filter"]).filtered(arrays["ft"])
            np.testing.assert_array_equal(arrays["ft_filtered"], expected)
            np.testing.assert_array_equal(arrays["parameters"][:, 0], DEFAULT_MUS)
        for name in (
            "ft_raw_comparison.png",
            "ft_filtered_comparison.png",
            "motion_contact.png",
            "mu_0.9/ft.png",
        ):
            self.assertGreater((out / name).stat().st_size, 10000)

    def test_output_guard_and_invalid_parameters(self):
        out = self.root / "existing"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            prepare_output(self.config, out, DEFAULT_MUS, 1000, 0.02)
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_output(self.config, self.config.parent / "new", DEFAULT_MUS, 1000, 0.02)
        for mus in ([], [0, 0], [-1], [float("nan")]):
            with self.assertRaisesRegex(ValueError, "finite"):
                prepare_output(self.config, self.root / "bad", mus, 1000, 0.02)
        self.assertFalse((self.root / "bad").exists())

    def test_phases_and_nonfinite_record_rejection(self):
        data = FakeWipe(self.cfg, (0, 1000, 0.02), None).rollout()
        for (start, stop), value in zip(((0, 200), (200, 300), (300, 400)), (1, 2, -3)):
            data["ft"][start:stop] = value
        result = summarize_case(data)
        for phase, value, count in (("press", 1, 200), ("forward", 2, 100), ("reverse", -3, 100)):
            self.assertEqual(result["phases"][phase]["sample_count"], count)
            np.testing.assert_array_equal(result["phases"][phase]["raw_mean"], [value] * 6)
            np.testing.assert_array_equal(result["phases"][phase]["raw_abs_peak"], [abs(value)] * 6)
        data["normal_sum"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "extra field"):
            summarize_case(data)

    def test_failed_case_recorded_without_substitution_or_success_summary(self):
        FakeWipe.fail_mu = 0.5
        result = run_sweep(self.config, self.root / "failed", factory=FakeWipe, make_plots=False)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(
            [r["state"] for r in result["runs"]],
            ["completed", "failed", "completed", "completed", "completed"],
        )
        self.assertIn("synthetic failure", result["runs"][1]["error"])
        self.assertEqual(FakeWipe.closed, 5)
        self.assertFalse((self.root / "failed/sweep.npz").exists())
        saved = json.loads((self.root / "failed/summary.json").read_text())
        self.assertEqual(saved["parameters"][1], [0.5, 1000.0, 0.02])


if __name__ == "__main__":
    unittest.main()
