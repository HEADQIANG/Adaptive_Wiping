"""Candidate path math, hardware isolation and manual-only start checks."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.real_deploy import fixed_setup as fixed, manual_setup as manual
from scripts.real_deploy.airbot_deploy import main, guard_segment
from scripts.real_deploy.manual_path import CandidatePolicy, plan_path
from scripts.shared.common import ROOT, file_digest
from scripts.robot_control.safety import read_force, torque_limit_enforced
from tests.real_deploy import test_fixed_setup as fixed_tests
from tests.real_training.test_airbot_programmed_demonstrations import Terminal


class PathTests(unittest.TestCase):
    def test_original_coordinates_and_timing_preserved_exactly(self):
        xy = np.column_stack((np.linspace(0.1, 0.15, 25), np.sin(np.arange(25)) * 0.1))
        original = xy.copy()
        start = np.array([0.2, 0.1, 0.15])
        candidate, report = plan_path(xy, start, 0.05)
        np.testing.assert_array_equal(xy, original)
        np.testing.assert_array_equal(candidate, original)
        self.assertEqual(report["uniform_scale"], 1)
        self.assertEqual(report["translation_sdk_m"], [0, 0])
        self.assertFalse(report["cartesian_speed_stop_enforced"])
        self.assertFalse(report["time_scaled"])
        self.assertFalse(report["runtime_clipping"])

    def test_entry_segment_included_and_stationary_path_supported(self):
        xy = np.tile([100, -100], (25, 1))
        start = np.array([0.2, 0.1, 0.15])
        candidate, report = plan_path(xy, start, 0.05)
        np.testing.assert_array_equal(candidate, xy)
        self.assertEqual(report["uniform_scale"], 1)
        self.assertGreater(report["candidate_max_xy_speed_including_entry_m_s"], 0.05)
        self.assertEqual(report["original_max_adjacent_xy_speed_m_s"], 0)

    def test_fast_zigzag_is_not_scaled_or_clipped(self):
        xy = np.array([[0.0001, (-1) ** i * 0.05] for i in range(25)])
        candidate, report = plan_path(xy, [0, 0, 0], 0.05)
        self.assertEqual(report["uniform_scale"], 1)
        np.testing.assert_array_equal(candidate, xy)

    def test_invalid_shapes_values_or_expanded_caps_rejected(self):
        for xy, start, speed in ((np.zeros((24, 2)), [0] * 3, 0.05),
                                 (np.full((25, 2), np.nan), [0] * 3, 0.05),
                                 (np.zeros((25, 2)), [0] * 2, 0.05),
                                 (np.zeros((25, 2)), [0] * 3, 0.051),
                                 (np.zeros((25, 2)), [0] * 3, float("nan"))):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                plan_path(xy, start, speed)

    def test_wrapper_preserves_feedback_and_binds_embedding(self):
        p = fixed_tests.FakePolicy({"sdk_end_position_m": [0, 0, 0]})
        p.delta = 0.001
        candidate = CandidatePolicy(p, np.zeros((1, 5)), [0, 0, 0], 0.05)
        self.assertEqual(candidate.predict_delta_h(np.zeros((1, 5)), np.zeros((1, 5, 6)))[0, 0], p.delta)
        with self.assertRaisesRegex(ValueError, "bound"):
            candidate.predict_xy(np.ones((1, 5)))

    def test_speed_record_only_is_scoped_and_height_guard_remains(self):
        loop = type("Loop", (), {"segment_start": np.zeros(3), "segment_end": np.array([1, 0, 0]), "last_delta": 0.0})()
        cfg = {"max_cartesian_speed_m_s": 0.05, "max_delta_h_m": 0.003}
        with self.assertRaises(fixed.StateError):
            guard_segment(loop, cfg)
        cfg.update(cartesian_speed_policy="record-only")
        with self.assertRaises(ValueError):
            guard_segment(loop, cfg)
        cfg["deployment_mode"] = manual.MODE
        guard_segment(loop, cfg)
        loop.last_delta = 0.004
        with self.assertRaises(fixed.StateError):
            guard_segment(loop, cfg)

    def test_manual_region_does_not_expand_legacy_region_or_depth(self):
        pose = {"sdk_end_position_m": [0, 0, 0.2]}
        with self.assertRaises(fixed.StateError):
            fixed.envelope([0.1, 0.1, 0.19], pose)
        pose.update(deployment_mode=manual.MODE, task_xy_bounds_sdk_m=[[-0.01, -0.2], [0.2, 0.2]])
        fixed.envelope([0.1, 0.1, 0.19], pose)
        for point in ([0.21, 0.1, 0.19], [0.1, 0.1, 0.17]):
            with self.assertRaises(fixed.StateError):
                fixed.envelope(point, pose)


class ManualExecutionTests(unittest.TestCase):
    setUp = fixed_tests.FixedExecutionTests.setUp

    def execute(self, *, shadow=False, commands=None):
        candidate = CandidatePolicy(self.policy, np.zeros((1, 5)), self.pose["sdk_end_position_m"], 0.05)
        fixed.execute(self.robot, self.sensor, candidate, np.zeros((1, 5)), self.cfg, self.pose,
                      self.stream, Terminal(commands or (["s"] if shadow else ["s", "IDLE"])),
                      shadow=shadow, manual_start=True)
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def test_no_automatic_approach_and_full_run_keeps_original_timing(self):
        with patch.object(fixed, "startup_approach", side_effect=AssertionError("No approach")):
            rows = self.execute()
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])
        samples = [r for r in rows if r.get("phase") == "policy"]
        self.assertEqual(len(samples), 1001)
        self.assertEqual([r["tick"] for r in samples if r["delta_h_m"] is not None], list(range(200, 1000, 40)))
        self.assertAlmostEqual(samples[200]["target_sdk_m"][2], self.pose["sdk_end_position_m"][2] - 0.01)
        self.assertTrue(any(r["event"] == "return_confirmed" for r in rows))

    def test_wrong_start_never_enters_servo_or_moves(self):
        self.robot.position[1] += 0.003
        with self.assertRaisesRegex(ValueError, "start pose"):
            self.execute()
        self.assertEqual(self.robot.calls, ["acquire"])
        self.assertFalse(self.robot.targets)

    def test_q_does_not_send_motion(self):
        self.execute(commands=["q", "IDLE"])
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])

    def test_shadow_never_acquires_switches_sends_or_stops(self):
        self.execute(shadow=True)
        self.assertFalse(self.robot.calls)
        self.assertFalse(self.robot.targets)

    def test_stale_force_aborts_without_retraction(self):
        self.sensor.latest_before = lambda due, net=False: (due - 0.021, [0] * 6)
        with self.assertRaisesRegex(fixed.StateError, "fresh causal"):
            self.execute()
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertNotIn("return_target", self.stream.getvalue())

    def test_excessive_height_not_scaled_or_clipped(self):
        self.policy.delta = 0.004
        with self.assertRaisesRegex(fixed.StateError, "vertical increment"):
            self.execute()
        self.assertEqual(self.robot.calls[-1], "abort")

    def test_original_fast_path_runs_without_project_speed_stop_in_fake_hardware(self):
        xy = np.column_stack((np.full(25, 0.25), np.linspace(0, 0.18, 25)))
        xy[0, 1] = 0.03
        self.policy.predict_xy = lambda embedding: xy[None].copy()
        self.cfg.update(deployment_mode=manual.MODE, cartesian_speed_policy="record-only")
        self.pose.update(deployment_mode=manual.MODE,
                         task_xy_bounds_sdk_m=[[0.245, -0.005], [0.255, 0.185]])
        rows = self.execute()
        samples = [r for r in rows if r.get("phase") == "policy"]
        self.assertGreater(max(r["command_segment_speed_m_s"] for r in samples), 0.05)
        self.assertTrue(all(r["cartesian_speed_policy"] == "record-only" for r in samples))
        np.testing.assert_allclose([samples[i * 40]["target_sdk_m"][:2] for i in range(1, 26)], xy)
        self.assertEqual(self.robot.calls[-1], "idle")

    def test_torque_record_only_applies_to_live_and_causal_samples(self):
        self.cfg.update(deployment_mode=manual.MODE, torque_limit_policy="record-only", max_torque_nm=None)
        original = self.sensor.latest
        self.sensor.latest = lambda net=False: (original(net)[0], [0, 0, 20, 10, 0, 0])
        self.sensor.latest_before = lambda due, net=False: (due, [0, 0, 20, 10, 0, 0])
        rows = self.execute()
        samples = [r for r in rows if r.get("phase") == "policy"]
        self.assertTrue(all(r["torque_limit_enforced"] is False for r in samples))
        self.assertTrue(all(r["raw_ft"][3] == 10 for r in samples))
        self.assertEqual(self.robot.calls[-1], "idle")

    def test_torque_opt_out_does_not_disable_force_or_finite_checks(self):
        self.cfg.update(deployment_mode=manual.MODE, torque_limit_policy="record-only", max_torque_nm=None)
        for wrench in ([0, 0, 41, 0, 0, 0], [0, 0, 20, float("nan"), 0, 0]):
            self.sensor.latest = lambda net=False: (self.clock.now, wrench)
            with self.assertRaises((ValueError, fixed.StateError)):
                read_force(self.sensor, self.cfg)
        self.cfg.pop("deployment_mode")
        with self.assertRaises(ValueError):
            torque_limit_enforced(self.cfg)
        self.assertTrue(torque_limit_enforced({}))


class GateTests(unittest.TestCase):
    def test_preflight_and_both_live_actions_do_not_connect_when_blocked(self):
        candidate = type("Candidate", (), {"path_report": {}})()
        policy = type("Policy", (), {"warnings": []})()
        loaded = ({}, {}, {}, policy, candidate, None, {}, {}, {})
        for action in ("preflight", "run", "shadow"):
            with self.subTest(action=action), patch.object(manual, "load_setup", return_value=loaded), patch.object(
                manual, "blockers", return_value=["not reviewed"]), patch.object(manual, "open_client") as connect, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([action, "--mode", "manual-tared", "--execute"]), 2)
                connect.assert_not_called()

    def test_config_or_approval_changes_invalidate_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            spec = {**dict.fromkeys(manual.CONFIRMATIONS, True), "replay_report": str(report)}
            self.assertTrue(manual.blockers(spec, {}))
            report.write_text(json.dumps({"mode": manual.MODE, "passed": True, "bindings": {"cfg": "a"}}))
            self.assertFalse(manual.blockers(spec, {"cfg": "a"}))
            self.assertTrue(manual.blockers(spec, {"cfg": "b"}))
            self.assertTrue(manual.blockers({**spec, "policy_path_confirmed": False}, {"cfg": "a"}))
            self.assertTrue(manual.blockers(spec, {"cfg": "a"}, {"demo_to_candidate_orientation_range_rad": [0.24, 1.68]}))


class CleanupTests(unittest.TestCase):
    def run_cli(self, *, failure=None, sensor_failure=None, client_failure=None, plots_success=True):
        from scripts.real_deploy import manual_runtime

        client, sensor = Mock(), Mock()
        sensor.stop.side_effect = sensor_failure
        client.close.side_effect = client_failure
        policy = type("Policy", (), {"warnings": []})()
        loaded = ({"workflow": manual_runtime.WORKFLOW}, {"sensor_port": "fake"}, {},
                  policy, None, None, {}, {}, {})
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(manual, "load_setup", return_value=loaded))
            stack.enter_context(patch.object(manual.fixed, "assert_unchanged"))
            stack.enter_context(patch.object(manual, "open_client", return_value=client))
            stack.enter_context(patch.object(manual, "DeadlineStub"))
            stack.enter_context(patch("scripts.robot_control.hardware.HardwareRobot"))
            stack.enter_context(patch("scripts.force_sensor.kwr75_reader.Kwr75Reader", return_value=sensor))
            execute = stack.enter_context(patch.object(manual_runtime, "execute", side_effect=failure))
            warmup = stack.enter_context(patch.object(manual_runtime, "warm_policy"))
            stack.enter_context(patch("scripts.real_deploy.plot_inference.reference_from_training", return_value={}))
            plots = stack.enter_context(patch("scripts.real_deploy.plot_inference.AutomaticPlots"))
            def finish_plots():
                sensor.stop.assert_called_once_with()
                client.close.assert_called_once_with()
                return plots_success

            plots.return_value.finish.side_effect = finish_plots
            stack.enter_context(patch.object(manual.sys.stdin, "isatty", return_value=True))
            stack.enter_context(patch.object(manual.time, "sleep"))
            stack.enter_context(patch.object(manual.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(output))
            code = main(["run", "--mode", "manual-tared", "--execute",
                         "--output", str(Path(temp) / "events.jsonl")])
        sensor.stop.assert_called_once_with()
        client.close.assert_called_once_with()
        warmup.assert_called_once_with(policy, None)
        sensor.start_csv.assert_called_once()
        self.assertTrue(sensor.start_csv.call_args.kwargs["background"])
        self.assertEqual(execute.call_args.kwargs["on_policy_complete"], plots.return_value.start)
        self.assertNotIn("Traceback", output.getvalue())
        return code, output.getvalue()

    def test_successful_cleanup_returns_success(self):
        code, output = self.run_cli()
        self.assertEqual(code, 0, output)

    def test_plot_failure_returns_nonzero_after_hardware_cleanup(self):
        code, output = self.run_cli(plots_success=False)
        self.assertEqual(code, 2, output)

    def test_recording_fault_and_repeated_close_fault_are_reported(self):
        error = RuntimeError("CSV queue full")
        code, output = self.run_cli(failure=error, sensor_failure=error)
        self.assertEqual(code, 2)
        self.assertIn("RuntimeError: CSV queue full", output)
        self.assertIn("Sensor cleanup failed; CSV may be incomplete", output)

    def test_cleanup_fault_does_not_mask_original_fault(self):
        code, output = self.run_cli(failure=ValueError("primary failure"),
                                    sensor_failure=OSError("disk full"),
                                    client_failure=RuntimeError("release failed"))
        self.assertEqual(code, 2)
        self.assertIn("ValueError: primary failure", output)
        self.assertIn("OSError: disk full", output)
        self.assertIn("Client cleanup failed; verify control release", output)

    def test_cleanup_only_fault_returns_failure(self):
        for resource in ("sensor_failure", "client_failure"):
            with self.subTest(resource=resource):
                code, output = self.run_cli(**{resource: OSError("cleanup failed")})
                self.assertEqual(code, 2, output)

    def test_interrupt_exit_code_is_preserved_with_cleanup_failure(self):
        code, output = self.run_cli(failure=KeyboardInterrupt(), sensor_failure=OSError("disk full"))
        self.assertEqual(code, 130)
        self.assertIn("Interrupted", output)
        self.assertIn("Sensor cleanup failed", output)


class RealManualAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/real_deploy/airbot_manual_tared.json"
        cls.spec = json.loads(cls.path.read_text())
        if not (ROOT / cls.spec["policy"]).exists():
            raise unittest.SkipTest("Real manual policy audit fixture not installed")

    def test_real_input_parity_and_candidate_replay(self):
        loaded = manual.load_setup(self.path)
        before = {path: file_digest(path) for path in loaded[-1]}
        with tempfile.TemporaryDirectory() as temp:
            report = manual.replay(*loaded, Path(temp) / "replay")
        self.assertTrue(report["path_limits_passed"], report["violations"][:5])
        self.assertTrue(report["recorded_force_guard_passed"])
        self.assertFalse(report["torque_limit_enforced"])
        self.assertFalse(report["candidate_orientation_supported_by_demo"])
        self.assertFalse(report["passed"])
        self.assertGreater(report["recorded_peak_raw_torque_nm"], 0.8)
        self.assertGreater(report["ideal_tracking_peak_cartesian_speed_m_s"], 0.05)
        self.assertTrue(report["original_xy_preserved_exactly"])
        self.assertFalse(report["cartesian_speed_stop_enforced"])
        np.testing.assert_array_equal(loaded[4].xy, loaded[3].predict_xy(loaded[5])[0])
        self.assertEqual(report["feedback_predictions"], 160)
        self.assertLessEqual(report["max_stream_batch_error_m"], 1e-8)
        self.assertFalse(report["contact_dynamics_simulated"])
        self.assertGreater(report["path_review"]["candidate_max_3d_speed_with_vertical_budget_m_s"], 0.05)
        self.assertEqual(before, {path: file_digest(path) for path in before})
        self.assertTrue(manual.blockers(loaded[0], loaded[-1]))

    def test_wrong_policy_identity_or_increased_limits_rejected(self):
        for key, value in (("policy_sha256", "wrong"), ("exploration_sha256", "wrong"),
                           ("collection_session_sha256", "wrong")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "config.json"
                path.write_text(json.dumps({**self.spec, key: value}))
                with self.assertRaises(ValueError):
                    manual.load_setup(path)
        for guard in manual.CAPS:
            spec = copy.deepcopy(self.spec)
            spec["guards"][guard] *= 1.01
            with self.subTest(guard=guard), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "config.json"
                path.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):
                    manual.load_setup(path)


if __name__ == "__main__":
    unittest.main()
