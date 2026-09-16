"""Offline plotting checks; no robot or sensor connection."""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from scripts.real_deploy.plot_comparison import load_run, reference_lines


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "events.jsonl"
        self.rows = [
            {"event": "session_start", "shadow": False},
            {"event": "baseline_complete", "bias_si": [1] * 6},
        ]
        for tick in range(1001):
            self.rows.append({
                "event": "sample", "phase": "policy", "tick": tick,
                "due_perf_s": 100 + tick / 100,
                "pose": {"host_monotonic_s": 100.002 + tick / 100,
                         "sdk_end_position_m": [0.2, 0, 0.1]},
                "raw_ft": [3] * 6, "tared_ft": [2] * 6,
                "filtered_ft": [2] * 6, "target_sdk_m": [0.2, 0, 0.1],
                "tracking_target_sdk_m": [0.2, 0, 0.1],
            })
        self.rows.extend([{"event": "policy_complete"}, {"event": "session_complete"}])

    def load(self):
        self.path.write_text("\n".join(json.dumps(row) for row in self.rows))
        return load_run(self.path)

    def test_time_axes_and_tare(self):
        run, _ = self.load()
        np.testing.assert_allclose(run["time"], np.arange(1001) / 100)
        np.testing.assert_allclose(run["pose_time"] - run["time"], 0.002)
        np.testing.assert_allclose(run["tared_ft"], 2)

    def test_missing_tick_rejected(self):
        self.rows.pop(100)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.load()

    def test_wrong_baseline_rejected(self):
        self.rows[1]["bias_si"][0] = 0
        with self.assertRaises(AssertionError):
            self.load()

    def test_wrong_filter_rejected(self):
        self.rows[100]["filtered_ft"][0] = 3
        with self.assertRaises(AssertionError):
            self.load()

    def test_shadow_rejected(self):
        self.rows[0]["shadow"] = True
        with self.assertRaisesRegex(ValueError, "shadow"):
            self.load()

    def test_aborted_rejected(self):
        self.rows.append({"event": "aborted"})
        with self.assertRaisesRegex(ValueError, "completed"):
            self.load()

    def test_padding_dashed_after_measured_endpoint(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        demo = {"time": np.arange(1001) / 100, "duration": 6,
                "position": np.full((1001, 3), 0.2)}
        reference_lines(ax, [demo], "position", 0, 1000)
        measured, padded = ax.lines
        self.assertEqual(measured.get_linestyle(), "-")
        self.assertEqual(padded.get_linestyle(), "--")
        self.assertEqual(measured.get_xdata()[-1], 6)
        self.assertEqual(padded.get_xdata()[0], 6)
        np.testing.assert_allclose(measured.get_ydata(), 200)


if __name__ == "__main__":
    unittest.main()
