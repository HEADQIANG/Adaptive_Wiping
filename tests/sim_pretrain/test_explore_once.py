"""Single-rollout window routing and resource cleanup without a display."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.sim_pretrain import explore_once


class ExploreOnceTests(unittest.TestCase):
    def run_case(self, flags=(), fail=False):
        env = MagicMock()
        data = {"ft": np.zeros((400, 6))}

        def rollout(sample_callback=None):
            if sample_callback is not None:
                sample_callback(0.0, np.zeros(6))
                sample_callback(0.01, np.ones(6))
            return data

        env.rollout.side_effect = rollout
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "single"
            with (
                patch.object(explore_once, "PretrainingWipe", return_value=env),
                patch.object(explore_once, "LivePlot") as plot,
                patch.object(explore_once, "LiveViewer") as viewer,
                patch.object(explore_once, "rollout_metrics", return_value={}),
                patch("matplotlib.get_backend", return_value="TkAgg"),
            ):
                if fail:
                    viewer.return_value.update.side_effect = RuntimeError("viewer closed")
                    with self.assertRaisesRegex(RuntimeError, "viewer closed"):
                        explore_once.main(["--output", str(output), "--close-after-run", *flags])
                    self.assertFalse((output / "exploration.npz").exists())
                else:
                    self.assertEqual(explore_once.main(["--output", str(output), "--close-after-run", *flags]), 0)
                    with np.load(output / "exploration.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["ft"].shape, (400, 6))
                        metadata = json.loads(str(saved["metadata_json"]))
                        gain = float(flags[flags.index("--gain")+1]) if "--gain" in flags else 300
                        self.assertEqual(metadata["gain"], gain)
                        ramp = float(flags[flags.index("--ramp-s")+1]) if "--ramp-s" in flags else 0.2
                        self.assertEqual(metadata["config"]["simulation"]["exploration_ramp_s"], ramp)
                        self.assertEqual(metadata["trajectory_profile"],
                                         "smoothstep_velocity_v1" if ramp > 0 else "legacy_piecewise_linear")
                env.rollout.assert_called_once()
                env.close.assert_called_once()
                if "--no-plot" in flags:
                    plot.assert_not_called()
                    viewer.assert_not_called()
                else:
                    plot.return_value.plt.close.assert_called_once()
                    if "--no-viewer" in flags:
                        viewer.assert_not_called()
                    else:
                        viewer.assert_called_once_with(env)
                        viewer.return_value.close.assert_called_once()
                        self.assertEqual(viewer.return_value.update.call_count, 1 if fail else 2)

    def test_default_synchronizes_both_windows(self):
        self.run_case()

    def test_plot_only(self):
        self.run_case(["--no-viewer"])

    def test_headless_stays_headless(self):
        self.run_case(["--no-plot"])

    def test_early_viewer_close_cleans_up_without_complete_data(self):
        self.run_case(fail=True)

    def test_ramp_override_is_saved(self):
        self.run_case(["--no-plot", "--ramp-s", "0"])
        self.run_case(["--no-plot", "--ramp-s", "0.3"])

    def test_expanded_gain_is_accepted_and_saved(self):
        self.run_case(["--no-plot", "--gain", "5000"])

    def test_invalid_ramp_rejected_before_simulation(self):
        with patch.object(explore_once, "PretrainingWipe") as env:
            for value in ("-0.1", "0.6", "nan", "inf"):
                with self.assertRaises(SystemExit) as raised:
                    explore_once.main(["--no-plot", "--ramp-s", value])
                self.assertEqual(raised.exception.code, 2)
            env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
