"""Wall motion, measured feedback, entrypoint isolation and log analysis; no hardware."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

from scripts.real_deploy import airbot_deploy, manual_runtime as runtime, manual_setup
from scripts.real_deploy.plot_run_comparison import load_run, metrics
from scripts.real_deploy.wiping_frame import WipingFrame
from tests.real_deploy.test_fixed_setup import FakePolicy
from tests.real_training.test_manual_exploration import Fixture


class LoopTests(unittest.TestCase):
    def test_all_wall_directions_preserve_path_and_reanchor_measured_normal(self):
        start = np.array([0.31, -0.12, 0.74])
        original = np.column_stack((np.linspace(1, 1.046, 25), np.linspace(2, 2.183, 25)))
        raw = np.arange(1201 * 6).reshape(1201, 6) * 0.0001
        for direction, sign, up, axis in (("+x", -1, 1, 0), ("-x", 1, -1, 0),
                                          ("+y", -1, 1, 1), ("-y", 1, -1, 1)):
            with self.subTest(direction=direction):
                frame = WipingFrame("vertical", direction)
                policy = FakePolicy({"sdk_end_position_m": start})
                policy.predict_xy = lambda _: original[None]
                policy.delta = -0.0004
                loop, path = runtime.make_loop(policy, np.zeros((1, 5)), start, wiping_frame=frame)
                expected_ft = policy.make_ft_filter(100).process(raw)
                targets, predicted_endpoints = [], []
                for tick in range(1201):
                    # Dissimilar XYZ and tracking offset; never use target as measured feedback.
                    measured = start + np.array([tick * 1e-6, -0.003 - tick * 2e-6, tick * 1e-5])
                    filtered = loop.push(tick, raw[tick], measured)
                    np.testing.assert_allclose(filtered, expected_ft[tick], atol=1e-12)
                    if loop.last_delta is not None:
                        predicted_endpoints.append(measured[axis] + sign * policy.delta)
                        self.assertAlmostEqual(loop.segment_end[axis], predicted_endpoints[-1])
                        np.testing.assert_allclose(policy.histories[-1][0], expected_ft[np.arange(tick - 160, tick + 1, 40)])
                    targets.append(loop.target(tick))
                targets = np.array(targets)
                np.testing.assert_allclose(targets[:201], np.tile(start, (201, 1)))
                endpoints = targets[np.arange(240, 1201, 40)]
                np.testing.assert_allclose(endpoints[:, axis], predicted_endpoints)
                unchanged_axis = 1 - axis
                np.testing.assert_allclose(endpoints[:, unchanged_axis], start[unchanged_axis] + original[:, unchanged_axis] - original[0, unchanged_axis])
                np.testing.assert_allclose(endpoints[:, 2], start[2] + up * (original[:, axis] - original[0, axis]))
                self.assertEqual(len(policy.histories), 25)
                self.assertEqual(path["wiping_frame"]["normal_axis_sdk"], "XY"[axis])
                self.assertNotIn("translated_xy_sdk_m", path)
                self.assertAlmostEqual(np.linalg.det(frame.rotation), 1)
                pressing_direction = np.zeros(3)
                pressing_direction[axis] = -sign
                np.testing.assert_array_equal(frame.rotation @ [0, 0, -1], pressing_direction)
                np.testing.assert_array_equal(frame.rotation.T @ frame.rotation, np.eye(3))
                from scipy.spatial.transform import Rotation
                rotation_axis, angle = frame.tool_rotation_axis_angle
                np.testing.assert_allclose(frame.rotation, Rotation.from_euler(rotation_axis, angle, degrees=True).as_matrix(), atol=1e-15)

    def test_explicit_horizontal_matches_original_default_exactly(self):
        start = [0.21, 0.03, 0.64]
        policy = FakePolicy({"sdk_end_position_m": start})
        policy.delta = 0.0002
        default, path = runtime.make_loop(policy, np.zeros((1, 5)), start)
        explicit, _ = runtime.make_loop(policy, np.zeros((1, 5)), start, wiping_frame=WipingFrame())
        for tick in range(1201):
            measured = [0.31, -0.12, 0.75 + tick * 1e-6]
            for loop in (default, explicit):
                loop.push(tick, np.full(6, tick * 0.001), measured)
            np.testing.assert_array_equal(default.target(tick), explicit.target(tick))
            if default.last_delta is not None:
                self.assertAlmostEqual(default.segment_end[2], measured[2] + policy.delta)
        np.testing.assert_array_equal(path["translated_xy_sdk_m"], policy.predict_xy(None)[0])

    def test_invalid_modes_rejected_without_hardware(self):
        cases = (("vertical", None), ("vertical", "z"), ("horizontal", "+x"), ("unknown", None))
        for mode, direction in cases:
            with self.subTest(mode=mode, direction=direction), self.assertRaises(ValueError):
                WipingFrame(mode, direction)


class EntryTests(unittest.TestCase):
    def test_preflight_reports_wall_frame_without_connecting_or_legacy_gates(self):
        for direction, sign, axis in (("+x", -1, "X"), ("-x", 1, "X"),
                                      ("+y", -1, "Y"), ("-y", 1, "Y")):
            cfg = {}
            loaded = ({"workflow": runtime.WORKFLOW}, cfg, {}, Mock(warnings=[]), None, None, {}, {}, {})
            output = io.StringIO()
            with patch.object(manual_setup, "load_setup", return_value=loaded), patch.object(
                    manual_setup, "open_client") as connect, patch.object(
                        manual_setup, "blockers", side_effect=AssertionError("legacy gate")), redirect_stdout(output):
                code = airbot_deploy.main(["preflight", "--wiping-mode", "vertical", f"--wall-direction={direction}"])
            self.assertEqual(code, 0)
            connect.assert_not_called()
            report = json.loads(output.getvalue())
            self.assertEqual(report["wiping_frame"]["normal_axis_sdk"], axis)
            self.assertEqual(report["wiping_frame"]["delta_h_to_sdk_sign"], sign)
            self.assertEqual(cfg["wall_direction"], direction)

    def test_incompatible_cli_modes_fail_before_loading_or_connecting(self):
        for args in (["--wiping-mode", "vertical"], ["--wall-direction", "+x"],
                     ["--mode", "fixed-setup", "--wiping-mode", "vertical", "--wall-direction", "+x"],
                     ["replay", "--wiping-mode", "vertical", "--wall-direction", "+x"],
                     ["shadow", "--wiping-mode", "vertical", "--wall-direction", "+x"]):
            with self.subTest(args=args), patch.object(manual_setup, "load_setup") as load, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    airbot_deploy.main(args)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_legacy_manual_workflow_cannot_silently_ignore_vertical(self):
        loaded = ({}, {}, {}, Mock(warnings=[]), None, None, {}, {}, {})
        with patch.object(manual_setup, "load_setup", return_value=loaded), patch.object(
                manual_setup, "open_client") as connect, redirect_stderr(io.StringIO()):
            self.assertEqual(airbot_deploy.main(["preflight", "--wiping-mode", "vertical", "--wall-direction", "+x"]), 2)
        connect.assert_not_called()


class RuntimeTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg.update(wiping_mode="vertical", wall_direction="+x")
        self.sensor.flush_csv = Mock()
        self.sensor.latest_before = lambda due, net=False: (due, self.sensor.wrench)
        self.policy = FakePolicy(self.pose)
        self.policy.delta = -0.0001
        self.policy.predict_xy = lambda _: np.column_stack((np.linspace(1, 1.046, 25), np.linspace(2, 2.183, 25)))[None]

    def run_policy(self):
        runtime.execute(self.robot, self.sensor, self.policy, np.zeros((1, 5)), self.cfg, self.stream, self.keyboard)
        return self.events()

    def check_runtime(self, direction, sign):
        self.cfg["wall_direction"] = direction
        axis = 0 if direction.endswith("x") else 1
        quaternion = ([0, sign * np.sqrt(0.5), 0, np.sqrt(0.5)] if axis == 0
                      else [-sign * np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        self.robot.state["sdk_end_orientation_xyzw"] = quaternion
        self.sensor.wrench = [1, 2, 3, 0.1, 0.2, 0.3]
        self.keyboard.hook = lambda key: setattr(self.sensor, "wrench", [2, 4, 6, 0.2, 0.4, 0.6]) if key == "s" else None
        original_read, original_send = self.robot.read, self.robot.send
        orientations = []

        def read(mode):
            row = original_read(mode)
            row["host_monotonic_s"] = self.clock.now()
            return row

        def send(target, quat):
            orientations.append(list(quat))
            original_send(target, quat)

        self.robot.read, self.robot.send = read, send
        prompts = []
        original_wait = self.keyboard.wait

        def wait(key, monitor, prompt):
            prompts.append(prompt)
            return original_wait(key, monitor, prompt)

        self.keyboard.wait = wait
        frame = WipingFrame.from_settings(self.cfg)
        runtime.write_event(self.stream, {"event": "session_start", "shadow": False,
                            "spec": {"workflow": runtime.WORKFLOW, "wiping_mode": "vertical",
                                     "wall_direction": direction, "wiping_frame": frame.metadata()}})
        rows = self.run_policy()
        rotation_axis, angle = ("Y", sign * 90) if axis == 0 else ("X", -sign * 90)
        self.assertIn(f"{angle:+d} deg about SDK {rotation_axis}", prompts[0])
        self.assertEqual(self.keyboard.keys, ["h", "z", "s", "g"])
        self.assertEqual(self.robot.calls[-1], "gravity_comp")
        self.assertEqual(len(orientations), 1001)
        np.testing.assert_array_equal(orientations, np.tile(quaternion, (1001, 1)))
        samples = [r for r in rows if r["event"] == "sample"]
        np.testing.assert_allclose([r["tared_ft"] for r in samples], np.tile([1, 2, 3, 0.1, 0.2, 0.3], (1201, 1)))
        self.assertAlmostEqual(samples[240]["target_sdk_m"][axis], samples[200]["state"]["sdk_end_position_m"][axis] + sign * self.policy.delta)
        path = self.folder / "wall.jsonl"
        path.write_text(self.stream.getvalue(), encoding="utf-8")
        report = metrics(load_run(path))
        self.assertEqual(report["prediction_count"], 25)
        self.assertEqual(report["wiping_frame"]["wall_direction"], direction)
        self.assertAlmostEqual(report["phases"]["whole_policy"]["target_xyz_mm"]["Z"]["span"], 46 if axis == 0 else 183)
        self.assertIn(f"anchor_measured_{'xy'[axis]}_mm", report["prediction_table"][0])

    def test_positive_wall_runtime_and_log(self):
        self.check_runtime("+x", -1)

    def test_negative_wall_runtime_and_log(self):
        self.check_runtime("-x", 1)

    def test_positive_y_wall_runtime_and_log(self):
        self.check_runtime("+y", -1)

    def test_negative_y_wall_runtime_and_log(self):
        self.check_runtime("-y", 1)

    def test_vertical_stale_sensor_stops_before_motion(self):
        original = runtime.infer

        def infer(robot, sensor, *args):
            started = self.clock.now()
            sensor.latest_before = lambda due, net=False: (min(due, started + 1), sensor.wrench)
            return original(robot, sensor, *args)

        with patch.object(runtime, "infer", side_effect=infer), self.assertRaisesRegex(runtime.StateError, "fresh causal"):
            self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")


def wall_log(rows, direction):
    """Map an independent table-log fixture to a wall, without using runtime mapping code."""
    sign = -1 if direction.startswith("+") else 1
    frame = WipingFrame("vertical", direction)
    rows[0]["spec"].update(wiping_mode="vertical", wall_direction=direction, wiping_frame=frame.metadata())

    def convert(point):
        x, y, z = point
        return ([0.8 + sign * z, y, 1.2 - sign * x] if direction.endswith("x")
                else [x, 0.8 + sign * z, 1.2 - sign * y])

    for row in rows:
        if row["event"] == "reference_captured":
            row["wiping_frame"] = frame.metadata()
            pose = row["runtime_reference_pose"]
            pose["sdk_end_position_m"] = convert(pose["sdk_end_position_m"])
        if row["event"] == "sample":
            row["target_sdk_m"] = convert(row["target_sdk_m"])
            row["state"]["sdk_end_position_m"] = convert(row["state"]["sdk_end_position_m"])
    return rows
