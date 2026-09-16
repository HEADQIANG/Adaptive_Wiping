"""Two deployment log schemas and offline comparison output; no hardware."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_deploy import plot_run_comparison as plots
from scripts.shared.common import file_digest
from scripts.shared.real_preprocessing import CausalFTFilter


def fixture(kind, *, stationary=False):
    final_tick = 1200 if stationary else 1000
    prediction_ticks = np.arange(200, final_tick, 40)
    t = np.arange(final_tick + 1) / 100
    motion_t = np.maximum(0, t - 2) if stationary else t
    start_z = 0.16 if kind == "manual" else 0.178
    target = np.column_stack((0.2 + motion_t * 0.002, motion_t * 0.003, np.full(len(t), start_z)))
    if kind == "fixed":
        target[:201, 2] -= t[:201] * 0.005
    for tick in prediction_ticks:
        target[tick:tick + 41, 2] = np.linspace(target[tick, 2], target[tick, 2] - 0.0001, 41)
    position = target - [0.0001, 0.0001, 0.0002]
    tared = np.tile(np.sin(t)[:, None], (1, 6))
    filtered = CausalFTFilter().process(tared)
    spec = {"policy": "policy.pt", "policy_sha256": kind, "training_config": "training.yaml",
            "collection_session": kind, "initialization": "nominal_10mm_during_first_2s"}
    if kind == "manual":
        spec["workflow"] = "runtime_start_h_z_s_g_v1"
    else:
        spec["mode"] = "fixed_setup_tared_v1"
    rows = [{"event": "session_start", "shadow": False, "spec": spec}]
    if kind == "manual":
        rows.extend([
            {"event": "reference_captured", "initialization": plots.STATIONARY_INITIALIZATION if stationary else "hold_z_first_2s_no_automatic_press",
             "runtime_reference_pose": {"sdk_end_position_m": [0.2, 0, start_z]}},
            {"event": "tare_complete", "tare_bias_si": [1] * 6},
        ])
    else:
        rows.append({"event": "baseline_complete", "bias_si": [1] * 6})
    for tick in range(len(t)):
        rows.append({"event": "sample", "phase": "policy", "tick": tick,
                     "due_perf_s": 100 + t[tick],
                     "causal_sensor_receive_perf_s" if kind == "manual" else "sensor_receive_perf_s": 99.997 + t[tick],
                     "state" if kind == "manual" else "pose": {
                         "host_monotonic_s": 100.002 + t[tick], "sdk_end_position_m": position[tick].tolist()},
                     "raw_ft": (tared[tick] + 1).tolist(), "tared_ft": tared[tick].tolist(),
                     "filtered_ft": filtered[tick].tolist(), "target_sdk_m": target[tick].tolist(),
                     "delta_h_m": 0.0001 if tick in prediction_ticks else None})
        if stationary:
            rows[-1].update(control_phase="history_hold" if tick < 200 else "motion",
                            motion_tick=None if tick < 200 else tick - 200)
    rows.extend([{"event": "policy_complete"}, {"event": "session_complete"}])
    return rows


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.rows = fixture("manual")
        self.path = self.folder / "manual.jsonl"

    def write(self, path, rows):
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def load(self):
        self.write(self.path, self.rows)
        return plots.load_run(self.path)

    def sample(self, tick=100):
        return next(row for row in self.rows if row.get("event") == "sample" and row["tick"] == tick)

    def test_both_schemas_and_real_pose_time(self):
        for kind in ("manual", "fixed"):
            with self.subTest(kind=kind):
                self.rows = fixture(kind)
                run = self.load()
                self.assertEqual(run["kind"], kind)
                np.testing.assert_allclose(run["time"], np.arange(1001) / 100)
                np.testing.assert_allclose(run["pose_time"] - run["time"], 0.002)
                np.testing.assert_allclose(run["delta_h"], 0.0001)

    def test_logged_runtime_initialization_overrides_legacy_spec_label(self):
        run = self.load()
        self.assertEqual(run["initialization"], "hold_z_first_2s_no_automatic_press")
        np.testing.assert_allclose(run["target_sdk_m"][:201, 2], 0.16)

    def test_stationary_start_retains_all_twelve_seconds_and_predictions(self):
        self.rows = fixture("manual", stationary=True)
        run = self.load()
        self.assertEqual(len(run["time"]), 1201)
        result = plots.metrics(run)
        self.assertEqual(result["prediction_count"], 25)
        self.assertEqual(result["phases"]["later_feedback"]["time_s"], [2.4, 12])
        self.assertEqual(result["prediction_table"][-1]["prediction_time_s"], 11.6)
        self.assertEqual(result["prediction_table"][-1]["endpoint_time_s"], 12)
        self.assertEqual(result["height_decomposition_mm"]["feedback_time_s"], [2, 12])

    def test_stationary_start_rejects_early_xy_and_shortened_log(self):
        self.rows = fixture("manual", stationary=True)
        self.sample()["target_sdk_m"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "stationary initialization XYZ"):
            self.load()
        self.rows = fixture("manual", stationary=True)
        self.rows = [r for r in self.rows if r.get("tick", 0) <= 1000]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.load()

    def test_stationary_start_rejects_wrong_motion_tick(self):
        self.rows = fixture("manual", stationary=True)
        self.sample(200)["motion_tick"] = 200
        with self.assertRaisesRegex(ValueError, "hold / motion timeline"):
            self.load()

    def test_nonpolicy_samples_do_not_enter_comparison(self):
        self.rows.insert(1, {"event": "sample", "phase": "approach"})
        self.rows.append({"event": "return_target"})
        self.assertEqual(len(self.load()["time"]), 1001)

    def test_missing_duplicate_or_reordered_tick_rejected(self):
        for mutation in ("missing", "duplicate", "reordered"):
            with self.subTest(mutation=mutation):
                self.rows = fixture("manual")
                row = self.sample()
                if mutation == "missing":
                    self.rows.remove(row)
                elif mutation == "duplicate":
                    row["tick"] = 99
                else:
                    index = self.rows.index(row)
                    self.rows[index], self.rows[index + 1] = self.rows[index + 1], self.rows[index]
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    self.load()

    def test_bad_baseline_or_filter_rejected(self):
        self.rows[2]["tare_bias_si"][0] = 3
        with self.assertRaisesRegex(ValueError, "baseline subtraction"):
            self.load()
        self.rows = fixture("manual")
        self.sample()["filtered_ft"][0] = 3
        with self.assertRaisesRegex(ValueError, "online filter"):
            self.load()

    def test_shadow_fault_and_missing_completion_rejected(self):
        for mutation in ("shadow", "fault", "aborted", "missing_completion"):
            with self.subTest(mutation=mutation):
                self.rows = fixture("manual")
                if mutation == "shadow":
                    self.rows[0]["shadow"] = True
                elif mutation == "missing_completion":
                    self.rows.pop()
                else:
                    self.rows.append({"event": mutation})
                with self.assertRaises(ValueError):
                    self.load()

    def test_invalid_shapes_and_nonfinite_values_rejected(self):
        for value in ([0] * 5, [float("nan")] * 6):
            with self.subTest(value=value):
                self.rows = fixture("manual")
                self.sample()["raw_ft"] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_bad_timestamps_rejected(self):
        for key, value in (("due_perf_s", 300), ("causal_sensor_receive_perf_s", 300)):
            with self.subTest(key=key):
                self.rows = fixture("manual")
                self.sample()[key] = value
                with self.assertRaises(ValueError):
                    self.load()
        self.rows = fixture("manual")
        self.sample()["state"]["host_monotonic_s"] = 0
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            self.load()

    def test_prediction_schedule_and_endpoint_validation(self):
        self.sample(200)["delta_h_m"] = None
        with self.assertRaisesRegex(ValueError, "20 height predictions"):
            self.load()
        self.rows = fixture("manual")
        self.sample(200)["delta_h_m"] = 0.001
        with self.assertRaisesRegex(ValueError, "next height endpoint"):
            self.load()

    def test_stage_boundaries_and_start_offsets_are_preserved(self):
        result = plots.metrics(self.load())
        self.assertEqual([result["phases"][name]["prediction_delta_mm"]["count"]
                          for name, _, _ in plots.PHASES], [0, 1, 19])
        self.assertAlmostEqual(result["reference_xyz_mm"][2], 160)
        self.assertAlmostEqual(result["phases"]["whole_policy"]["measured_xyz_mm"]["Z"]["first"], 159.8)
        self.assertEqual(result["prediction_table"][0]["prediction_time_s"], 2)
        self.assertEqual(result["prediction_table"][0]["endpoint_time_s"], 2.4)

    def test_measured_reanchoring_explains_target_change(self):
        result = plots.metrics(self.load())["height_decomposition_mm"]
        self.assertAlmostEqual(result["sum_predicted_delta"], 2)
        self.assertAlmostEqual(result["sum_measured_reanchor_offset"], -4)
        self.assertAlmostEqual(result["target_change_2_to_10_s"], -2)

    def test_existing_output_not_modified(self):
        output = self.folder / "output"
        output.mkdir()
        marker = output / "marker"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            plots.generate(self.path, self.path, output)
        self.assertEqual(marker.read_text(), "keep")

    def test_full_artifacts_and_annotation_without_hardware_imports(self):
        self.check_full_artifacts()

    def test_stationary_start_artifacts_include_all_twelve_seconds(self):
        self.rows = fixture("manual", stationary=True)
        self.check_full_artifacts()
        output = self.folder / "output"
        report = (output / "analysis.md").read_text()
        self.assertIn("12 秒、1201 个策略周期和 25 次高度预测", report)
        self.assertIn("前 2 秒 XYZ 目标均保持起点", report)
        self.assertNotIn("XY 仍执行网络轨迹", report)

    def check_full_artifacts(self):
        import sys

        self.write(self.path, self.rows)
        reference = self.folder / "fixed.jsonl"
        self.write(reference, fixture("fixed"))
        inputs = {str(path.resolve()): file_digest(path) for path in (self.path, reference)}
        with patch.dict(sys.modules, {"arm_sdk": None, "serial": None}):
            result = plots.generate(self.path, reference, self.folder / "output", tare_contact="loaded")
        self.assertEqual(result["inputs_sha256"], inputs)
        for path in (self.path, reference):
            self.assertEqual(file_digest(path), inputs[str(path.resolve())])
        output = self.folder / "output"
        self.assertEqual({p.name for p in output.iterdir()}, set(plots.PLOTS) | {"analysis.md", "summary.json"})
        for name in plots.PLOTS:
            pixels = plots.plt.imread(output / name)
            self.assertGreater(pixels.shape[0], 1000)
            self.assertGreater(pixels[:, :, :3].std(), 0.05)
        report = (output / "analysis.md").read_text()
        self.assertIn("接触或受压状态", report)
        self.assertIn("不是网络", report)
        self.assertEqual(json.loads((output / "summary.json").read_text())["manual_tare_contact"]["state"], "loaded")


if __name__ == "__main__":
    unittest.main()
