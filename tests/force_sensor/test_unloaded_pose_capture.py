import copy
import sys
import unittest
from pathlib import Path

from scripts.force_sensor.tools.capture_unloaded_pose import assess


class CaptureTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "observation_perf_s": 10 + i * 0.05,
                "sensor_receive_perf_s": 10 + i * 0.05 - 0.01,
                "raw_sensor_wrench_si": [1, 2, 3, 0.01, 0.02, 0.03],
                "pose": {
                    "joint_velocity_rad_s": [0] * 6,
                    "joint_position_rad": [0] * 6,
                    "sdk_end_position_m": [0, 0, 0.2],
                    "sdk_end_orientation_xyzw": [0, 0, 0, 1],
                    "service_state": {"controller_state": "idle"},
                },
            }
            for i in range(61)
        ]

    def test_confirmation_never_inferred(self):
        self.assertFalse(assess(self.rows(), False)["eligible_unloaded_pose"])
        self.assertTrue(assess(self.rows(), True)["eligible_unloaded_pose"])
        self.assertFalse(assess(self.rows(), True)["complete_calibration"])

    def test_motion_stale_and_mode_change_rejected(self):
        for change in ("motion", "stale", "mode"):
            rows = copy.deepcopy(self.rows())
            if change == "motion":
                rows[-1]["pose"]["joint_velocity_rad_s"][0] = 0.2
            elif change == "stale":
                rows[-1]["sensor_receive_perf_s"] -= 1
            else:
                rows[-1]["pose"]["service_state"]["controller_state"] = "servo"
            self.assertFalse(assess(rows, True)["eligible_unloaded_pose"])

    def test_short_and_nonfinite_rejected(self):
        self.assertFalse(assess(self.rows()[:15], True)["eligible_unloaded_pose"])
        rows = self.rows()
        rows[-1]["raw_sensor_wrench_si"][0] = float("nan")
        with self.assertRaises(ValueError):
            assess(rows, True)


if __name__ == "__main__":
    unittest.main()
