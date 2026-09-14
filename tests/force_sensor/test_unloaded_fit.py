import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.force_sensor.tools.fit_unloaded_calibration import (
    fit,
    groups,
    predict,
)


class UnloadedFitTests(unittest.TestCase):
    def test_known_bias_rotation_mass_and_com_both_signs(self):
        rng = np.random.default_rng(3)
        u = rng.normal(size=(20, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        for sign in (1, -1):
            model = {
                "force_sign_hypothesis": sign,
                "sensed_weight_N": 2.0,
                "candidate_sensor_from_sdk_rotation": Rotation.from_euler(
                    "xyz", [0.3, 0.5, 0.2]
                ).as_matrix(),
                "force_bias_N": [4.0, 5.0, 6.0],
                "candidate_com_sensor_m": [0.01, -0.02, 0.04],
                "torque_bias_Nm": [0.1, 0.2, 0.3],
            }
            y = predict(model, u)
            result = fit(u, y)
            np.testing.assert_allclose(predict(result, u), y, atol=1e-12)
            self.assertEqual(result["force_sign_hypothesis"], sign)
            np.testing.assert_allclose(result["force_bias_N"], model["force_bias_N"], atol=1e-12)
            np.testing.assert_allclose(
                result["candidate_com_sensor_m"], model["candidate_com_sensor_m"], atol=1e-12
            )

    def test_degenerate_orientations_rejected(self):
        with self.assertRaises(ValueError):
            fit(np.tile([0.0, 0.0, 1.0], (8, 1)), np.zeros((8, 6)))

    def test_near_repeats_grouped_together(self):
        u = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(groups(u), [[0, 2], [1]])


if __name__ == "__main__":
    unittest.main()
