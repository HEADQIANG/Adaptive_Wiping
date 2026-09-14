import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.real_training.tools.audit_real_input_alignment import (
    arc_resample,
    force_audit,
    loo_mean,
    onset,
)
from scripts.shared.preprocessing import Preprocessor


class InputAlignmentTests(unittest.TestCase):
    def test_leave_one_out_excludes_target_and_start_alignment(self):
        paths = np.repeat(np.array([0.0, 1.0, 2.0])[:, None, None], 5, axis=1)
        self.assertAlmostEqual(loo_mean(paths)["rmse_mm"], np.sqrt(1.5) * 1000)
        self.assertEqual(loo_mean(paths - paths[:, :1])["rmse_mm"], 0)

    def test_onset_requires_sustained_motion(self):
        xy = np.zeros((20, 2))
        xy[2, 0] = 0.01
        xy[10:, 0] = 0.003
        self.assertEqual(onset(np.arange(20) / 100, xy, xy[0]), 0.1)
        self.assertIsNone(onset(np.arange(20) / 100, xy, xy[0], 0.005))

    def test_arc_resampling_duplicates_and_time_warp(self):
        line = np.array([[0, 0], [0, 0], [1, 0], [2, 0]], dtype=float)
        resampled = arc_resample(line, 5)
        np.testing.assert_allclose(resampled[:, 0], np.linspace(0, 2, 5))
        with self.assertRaises(ValueError):
            arc_resample(np.zeros((5, 2)))

    def test_force_rotation_invariance_and_counterfactual_label(self):
        rng = np.random.default_rng(42)
        sim = rng.normal(size=(6, 400, 6)) * 0.1
        real = sim[:1] + np.array([10, 0, 0, 0, 0, 0])
        prep = Preprocessor().fit(sim)
        q = np.tile([0.0, 0.0, 0.0, 1.0], (400, 1))
        report = force_audit(sim, real, real[0, 0], prep, sim[:, 0], q, q[0])
        self.assertEqual(report["fraction_real_force_norm_above_simulation_train_max"], 1)
        self.assertFalse(report["bias_gravity_contact_separately_identified"])
        self.assertTrue(report["counterfactual_is_not_bias_compensation"])
        rot = Rotation.from_euler("xyz", [23, 41, 79], degrees=True)
        np.testing.assert_allclose(
            np.linalg.norm(rot.apply(real[0, :, :3]), axis=1),
            np.linalg.norm(real[0, :, :3], axis=1),
        )


if __name__ == "__main__":
    unittest.main()
