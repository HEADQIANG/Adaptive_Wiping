"""Width-only assignment, exact export and isolated failure handling."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.shared.common import load_config, write_json
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.friction_ft_sweep import EXTRA_FIELDS
from scripts.sim_pretrain.experiments.width_ft_sweep import (
    DEFAULT_WIDTHS,
    run_width_sweep,
)
from scripts.sim_pretrain.simulation import exploration_offsets


class FakeWidthWipe:
    calls = []
    fail_width = None
    closed = 0

    def __init__(self, cfg, parameters, spec):
        self.parameters = parameters
        self.calls.append((parameters, spec))

    def rollout(self):
        if self.parameters[2] == self.fail_width:
            raise RuntimeError("synthetic width failure")
        data = {key: np.zeros(shape) for key, shape in {**FIELDS, **EXTRA_FIELDS}.items()}
        data["time"] = np.arange(1, 401) / 100.0
        data["target_position"] = exploration_offsets(data["time"])
        data["parameters"] = np.asarray(self.parameters)
        data["contact_count"][:] = 1
        data["ft"][:] = self.parameters[2] + np.arange(6)
        return data

    def unload(self, data):
        return np.zeros(6)

    def close(self):
        type(self).closed += 1


class WidthSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="width-ft-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "frozen/config.json"
        cfg = load_config(
            ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
        )
        cfg["output_dir"] = str(self.config.parent)
        write_json(self.config, cfg)
        FakeWidthWipe.calls, FakeWidthWipe.fail_width, FakeWidthWipe.closed = [], None, 0

    def test_exact_single_factor_export_and_plots(self):
        out = self.root / "sweep"
        result = run_width_sweep(self.config, out, factory=FakeWidthWipe)
        self.assertEqual(result["state"], "completed")
        self.assertTrue(result["sources_and_inputs_unchanged"])
        self.assertEqual(FakeWidthWipe.calls, [((0.9, 1000.0, w), None) for w in DEFAULT_WIDTHS])
        self.assertEqual(FakeWidthWipe.closed, 5)
        self.assertTrue(all(not r["acceptance"]["passed"] for r in result["runs"]))
        with np.load(out / "sweep.npz") as data:
            self.assertEqual(data["ft"].shape, (5, 400, 6))
            np.testing.assert_array_equal(data["parameters"][:, 2], DEFAULT_WIDTHS)
            np.testing.assert_array_equal(data["ft_filtered"], Preprocessor().filtered(data["ft"]))
            for i, w in enumerate(DEFAULT_WIDTHS):
                with np.load(out / f"width_{w:g}/mu_0.9/trajectory.npz") as single:
                    for key in single.files:
                        np.testing.assert_array_equal(data[key][i], single[key])
        for name in (
            "ft_raw_comparison.png",
            "ft_filtered_comparison.png",
            "motion_contact.png",
            "width_0.02/ft.png",
        ):
            self.assertGreater((out / name).stat().st_size, 10000)

    def test_invalid_widths_and_output_protection(self):
        for widths in ([], [0], [-0.1], [0.1, 0.1], [np.nan], [np.inf], [0.10000001, 0.10000002]):
            with self.assertRaises(ValueError):
                run_width_sweep(self.config, self.root / "invalid", widths, factory=FakeWidthWipe)
        self.assertFalse((self.root / "invalid").exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            run_width_sweep(self.config, self.config.parent / "new", factory=FakeWidthWipe)
        with self.assertRaises(FileExistsError):
            run_width_sweep(self.config, self.root, factory=FakeWidthWipe)
        self.assertEqual(FakeWidthWipe.calls, [])

    def test_failed_width_preserved_without_retry(self):
        FakeWidthWipe.fail_width = 0.1
        out = self.root / "failed"
        result = run_width_sweep(self.config, out, factory=FakeWidthWipe, make_plots=False)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(
            [r["state"] for r in result["runs"]],
            ["completed", "completed", "failed", "completed", "completed"],
        )
        self.assertIn("synthetic width failure", result["runs"][2]["error"])
        self.assertEqual(FakeWidthWipe.closed, 5)
        self.assertFalse((out / "sweep.npz").exists())


if __name__ == "__main__":
    unittest.main()
