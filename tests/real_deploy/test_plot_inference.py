"""Automatic plotting timing, reference identity and numerical correspondence; no hardware."""

import csv
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from scripts.real_deploy import plot_inference as plots
from scripts.shared.common import file_digest
from tests.real_deploy.test_plot_run_comparison import fixture


class PlotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.events = self.folder / "events.jsonl"
        self.output = self.folder / "plots"
        self.raw = self.folder / "raw.h5"
        with h5py.File(self.raw, "w") as source:
            source.attrs.update(complete=True, source_kind="real", metadata=json.dumps({
                "source_hashes": {"/recorded/session.json": "session-hash"},
                "derivation": "manual_recorded_baseline_10s_v1",
                "compensation": "recorded_unloaded_baseline_subtracted", "contains_derived_samples": False}))
            for i in range(8):
                group = source.create_group(f"demonstrations/demo_{i + 1}")
                group.attrs.update(start_time=100.0, complete=True, source_kind="real")
                ft = np.zeros((1001, 6))
                ft[:, 2] = -(i + 1)
                ft[-1, 2] = 1e6  # Right endpoint is excluded from the reference mean.
                group.create_dataset("ft_time", data=100 + np.arange(1001) / 100)
                group.create_dataset("ft", data=ft)
                group.create_dataset("ft_raw_before_baseline", data=ft + 2)
                group.create_dataset("recorded_unloaded_baseline", data=[2] * 6)
        self.spec = {"collection_session_sha256": "session-hash"}
        self.training = {"raw_data": str(self.raw)}
        self.bindings = {str(self.raw): file_digest(self.raw)}
        self.reference = plots.reference_from_training(self.spec, self.training, self.bindings)
        self.rows = fixture("manual", stationary=True)
        self.rows[0]["spec"].update(self.spec, plot_reference_fz=self.reference)
        self.rows[0]["bindings"] = self.bindings
        # Generation must work before g/session_complete, with a live partial trailing write.
        self.rows = [r for r in self.rows if r["event"] != "session_complete"]
        self.save_events()

    def save_events(self, suffix=""):
        self.events.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n" + suffix, encoding="utf-8")

    def test_reference_uses_bound_eight_demos_signed_mean_and_excludes_endpoint(self):
        self.assertEqual(self.reference["mean_fz_n"], -4.5)
        self.assertEqual([r["sample_count"] for r in self.reference["episodes"]], [1000] * 8)
        self.assertAlmostEqual(self.reference["rms_fz_n"], np.sqrt(np.mean(np.arange(1, 9) ** 2)))
        self.assertEqual(file_digest(self.raw), self.bindings[str(self.raw)])

    def test_reference_rejects_changed_data_and_unbound_session(self):
        with self.assertRaisesRegex(ValueError, "raw data changed"):
            plots.reference_from_training(self.spec, self.training, {str(self.raw): "wrong"})
        with self.assertRaisesRegex(ValueError, "bound"):
            plots.reference_from_training({"collection_session_sha256": "other"}, self.training, self.bindings)

    def test_plot_after_inference_before_exit_pairs_correct_predictions_and_ratios(self):
        self.save_events('{"event": "unfinished')
        before = file_digest(self.events)
        summary = plots.generate(self.events, self.output)
        self.assertEqual(summary["duration_s"], 12)
        self.assertEqual(summary["prediction_count"], 25)
        self.assertEqual(summary["reference"]["mean_fz_n"], -4.5)
        self.assertEqual(file_digest(self.events), before)
        with (self.output / "fz_height_predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 25)
        self.assertEqual(float(rows[0]["prediction_time_s"]), 2)
        self.assertEqual(float(rows[0]["endpoint_time_s"]), 2.4)
        self.assertAlmostEqual(float(rows[-1]["endpoint_time_s"]), 12)
        for row in rows:
            self.assertAlmostEqual(float(row["measured_anchor_z_m"]) + float(row["predicted_delta_h_m"]),
                                   float(row["predicted_next_z_m"]))
            self.assertEqual(row["history_fz_5_n"], row["filtered_fz_n"])
        with (self.output / "fz_reference_ratio.csv").open() as stream:
            ratio = list(csv.DictReader(stream))
        self.assertEqual(len(ratio), 1201)
        for row in ratio:
            self.assertAlmostEqual(float(row["ratio_percent"]), 100 * float(row["tared_fz_n"]) / -4.5)
        import matplotlib.pyplot as plt
        for name in summary["plots"]:
            pixels = plt.imread(self.output / name)
            self.assertGreater(pixels[:, :, :3].std(), 0.05)
        snapshot = (self.output / "inference_events.jsonl").read_text()
        self.assertNotIn("unfinished", snapshot)
        self.assertTrue(snapshot.rstrip().endswith('"policy_complete"}'))

    def test_zero_reference_produces_undefined_ratio_without_nan_json(self):
        reference = self.rows[0]["spec"]["plot_reference_fz"]
        reference["mean_fz_n"] = 0
        reference["absolute_mean_over_rms"] = 0
        for i, episode in enumerate(reference["episodes"]):
            episode["mean_fz_n"] = 4.5 if i % 2 else -4.5
        self.save_events()
        summary = plots.generate(self.events, self.output)
        self.assertFalse(summary["ratio_defined"])
        self.assertNotIn("NaN", (self.output / "summary.json").read_text())
        with (self.output / "fz_reference_ratio.csv").open() as stream:
            self.assertTrue(all(row["ratio_percent"] == "" for row in csv.DictReader(stream)))

    def test_wall_plots_and_execution_error_use_correct_normal_axis_and_sign(self):
        from scripts.real_deploy import plot_delta_h_error
        from tests.real_deploy.test_vertical_wiping import wall_log
        import copy

        original = copy.deepcopy(self.rows)
        for direction, sign in (("+x", -1), ("-x", 1), ("+y", -1), ("-y", 1)):
            with self.subTest(direction=direction):
                axis = direction[-1]
                self.rows = wall_log(copy.deepcopy(original), direction)
                self.rows.append({"event": "session_complete"})
                self.save_events()
                output = self.folder / direction
                summary = plots.generate(self.events, output)
                self.assertEqual(summary["wiping_frame"]["normal_axis_sdk"], axis.upper())
                with (output / "fz_height_predictions.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    self.assertNotIn("predicted_next_z_m", row)
                    self.assertEqual(float(row["delta_h_to_sdk_sign"]), sign)
                    self.assertAlmostEqual(float(row[f"measured_anchor_{axis}_m"]) + sign * float(row["predicted_delta_h_m"]),
                                           float(row[f"predicted_next_{axis}_m"]))
                error = plot_delta_h_error.generate(self.events, self.folder / (direction + "-error"))
                self.assertEqual(error["count"], 25)
                self.assertIn(f"measured_{axis.upper()}", error["error_definition"])
                self.assertAlmostEqual(error["mae_mm"], 0.2, places=8)

    def test_wall_log_rejects_wrong_feedback_axis_and_mismatched_mode(self):
        from scripts.real_deploy.plot_run_comparison import load_run
        from tests.real_deploy.test_vertical_wiping import wall_log

        self.rows = wall_log(self.rows, "+x")
        reference = next(row for row in self.rows if row["event"] == "reference_captured")
        reference["wiping_frame"]["delta_h_to_sdk_sign"] = 1
        self.save_events()
        with self.assertRaisesRegex(ValueError, "Captured wiping frame mismatch"):
            load_run(self.events, policy_only=True)
        reference["wiping_frame"]["delta_h_to_sdk_sign"] = -1
        sample = next(row for row in self.rows if row.get("tick") == 240)
        sample["target_sdk_m"][0] += 0.01
        self.save_events()
        with self.assertRaisesRegex(ValueError, "next height endpoint"):
            load_run(self.events, policy_only=True)

    def test_incomplete_or_faulted_inference_never_generates_complete_plots(self):
        self.rows = [r for r in self.rows if r["event"] != "policy_complete"]
        self.save_events()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            plots.generate(self.events, self.output)
        self.assertFalse(self.output.exists())
        self.rows.append({"event": "fault"})
        self.save_events()
        with self.assertRaisesRegex(ValueError, "did not complete"):
            plots.generate(self.events, self.output)
        self.assertFalse(self.output.exists())

    def test_background_writer_flush_is_waited_for(self):
        completed = self.events.read_bytes()
        self.rows.pop()  # policy_complete has not reached disk yet.
        self.save_events()
        with patch.object(plots.time, "sleep", side_effect=lambda _: self.events.write_bytes(completed)):
            self.assertEqual(plots.policy_snapshot(self.events, wait_seconds=1), completed)

    def test_reference_mismatch_is_rejected(self):
        self.rows[0]["spec"]["plot_reference_fz"]["mean_fz_n"] = -99
        self.save_events()
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            plots.generate(self.events, self.output)
        self.assertFalse((self.output / "summary.json").exists())

    def test_real_background_process_saves_before_session_exit(self):
        job = plots.AutomaticPlots(self.events)
        with redirect_stdout(io.StringIO()):
            job.start()
            self.assertTrue(job.finish(), job.log.read_text())
        self.assertEqual(job.process.returncode, 0)
        self.assertTrue((job.output / "fz_height.png").is_file())
        self.assertTrue((job.output / "fz_ratio.png").is_file())
        self.assertNotIn("session_complete", self.events.read_text())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            plots.generate(self.events, self.output)
        self.assertEqual(marker.read_text(), "original")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.job = plots.AutomaticPlots(Path(temp.name) / "events.jsonl")

    def test_start_is_nonblocking_and_uses_separate_process(self):
        with patch.object(plots.subprocess, "Popen") as popen, redirect_stdout(io.StringIO()):
            self.job.start()
        popen.return_value.wait.assert_not_called()
        args, kwargs = popen.call_args
        self.assertIn("scripts.real_deploy.plot_inference", args[0])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
        self.assertNotIn("PYTHONPATH", kwargs["env"])

    def test_start_failure_is_reported_without_raising_into_robot_loop(self):
        with patch.object(plots.subprocess, "Popen", side_effect=OSError("cannot launch")), redirect_stderr(io.StringIO()):
            self.job.start()
            self.assertFalse(self.job.finish())

    def test_child_failure_or_timeout_is_reported(self):
        self.job.process = Mock()
        self.job.process.wait.return_value = 2
        with redirect_stderr(io.StringIO()):
            self.assertFalse(self.job.finish())
            self.job.process.wait.side_effect = plots.subprocess.TimeoutExpired("plots", 30)
            self.assertFalse(self.job.finish())
        self.job.process.kill.assert_not_called()

    def test_aborted_before_start_requires_no_plot_wait(self):
        self.assertTrue(self.job.finish())


if __name__ == "__main__":
    unittest.main()
