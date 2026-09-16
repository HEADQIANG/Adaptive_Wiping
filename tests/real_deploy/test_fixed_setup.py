"""Fixed-installation policy bindings and no-hardware execution tests."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.real_deploy import fixed_setup as fixed
from scripts.real_deploy.airbot_deploy import main, PolicyLoop
from scripts.real_training import airbot_programmed_demonstrations as program
from scripts.shared.common import ROOT
from scripts.shared.real_preprocessing import CausalFTFilter
from tests.real_training.test_airbot_programmed_demonstrations import Clock, Robot, Sensor, Terminal


class FakePolicy:
    def __init__(self, pose):
        self.pose, self.delta = pose, 0.0
        self.histories = []

    def predict_xy(self, embedding):
        return np.tile(self.pose["sdk_end_position_m"][:2], (1, 25, 1))

    def make_ft_filter(self, hz):
        return CausalFTFilter(hz)

    def predict_delta_h(self, embedding, ft):
        self.histories.append(ft.copy())
        return np.array([[self.delta]])


class FixedLoopTests(unittest.TestCase):
    def setUp(self):
        self.pose = {"sdk_end_position_m": [0.25, 0, 0.15],
                     "sdk_end_orientation_xyzw": [0, 0, 0, 1], "joint_position_rad": [0] * 6}
        self.policy = FakePolicy(self.pose)

    def test_run_and_shadow_reach_connection_without_consuming_start_key(self):
        self.policy.warnings = []
        loaded = ({}, {}, self.pose, self.policy, np.zeros((1, 5)), {}, {}, {})
        for action in ("run", "shadow"):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as folder,
                patch.object(fixed, "load_setup", return_value=loaded),
                patch.object(fixed, "run_blockers", return_value=[]),
                patch.object(fixed, "assert_unchanged"),
                patch.object(fixed.sys.stdin, "isatty", return_value=True),
                patch.object(program.Terminal, "ask", side_effect=AssertionError("Unexpected startup prompt")) as prompt,
                patch.object(fixed, "open_client", side_effect=RuntimeError("Test connection boundary")) as connect,
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(main([action, "--mode", "fixed-setup", "--execute",
                                       "--output", str(Path(folder) / "events.jsonl")]), 2)
                self.assertEqual(connect.call_count, 1, errors.getvalue())
                prompt.assert_not_called()

    def test_bootstrap_once_in_original_timeline(self):
        loop = fixed.FixedSetupLoop(self.policy, np.zeros((1, 5)), self.pose["sdk_end_position_m"])
        measured = list(self.pose["sdk_end_position_m"])
        predictions = []
        for tick in range(1001):
            loop.push(tick, np.zeros(6), measured)
            target = loop.target(tick)
            if tick <= 200:
                self.assertAlmostEqual(target[2], 0.15 - tick * 0.01 * 0.005)
            else:
                # Feedback is anchored to the measurement before this tick's send.
                self.assertLessEqual(abs(target[2] - 0.14), 0.00005 + 1e-12)
            if loop.last_delta is not None:
                predictions.append(tick)
                self.assertAlmostEqual(loop.segment_end[2], measured[2])
            measured = target.tolist()
        self.assertEqual(predictions, list(range(200, 1000, 40)))
        self.assertEqual(len(self.policy.histories), 20)

    def test_legacy_warmup_unchanged(self):
        loop = PolicyLoop(self.policy, np.zeros((1, 5)), self.pose["sdk_end_position_m"])
        for tick in range(200):
            loop.push(tick, np.zeros(6), self.pose["sdk_end_position_m"])
            self.assertEqual(loop.target(tick)[2], 0.15)

    def test_envelope_and_prediction_reject_not_clip(self):
        for point in ([0.25, 0, 0.129], [0.256, 0, 0.14], [0.25, 0.051, 0.14], [0.25, 0, 0.151]):
            with self.assertRaises(fixed.StateError):
                fixed.envelope(point, self.pose)


class FixedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _, _ = program.load_configuration(ROOT / "configs/real_training/airbot_programmed_demonstrations.json")
        self.cfg.update(joint_min_rad=[-4] * 6, joint_max_rad=[4] * 6,
                        max_cartesian_speed_m_s=0.05, max_delta_h_m=0.003,
                        max_tracking_error_m=0.005, orientation_error_limit_rad=0.02,
                        tracking_error_policy="stop", orientation_error_policy="stop")
        self.pose = {"sdk_end_position_m": [0.25, 0, 0.15], "sdk_end_orientation_xyzw": [0, 0, 0, 1],
                     "joint_position_rad": [0] * 6}
        self.clock = Clock()
        self.robot = Robot(self.clock, self.pose)
        self.sensor = Sensor(self.clock, self.robot)
        self.sensor.latest_before = lambda due, net=False: (due, [0, 0, 20 - 1000 * (0.15 - self.robot.position[2]), 0, 0, 0])
        self.policy = FakePolicy(self.pose)
        self.stream = io.StringIO()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(fixed.time, "perf_counter", side_effect=lambda: self.clock.now))
        self.stack.enter_context(patch.object(fixed.time, "sleep", side_effect=self.clock.sleep))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def run_policy(self, *, shadow=False, commands=None):
        fixed.execute(self.robot, self.sensor, self.policy, np.zeros((1, 5)), self.cfg, self.pose,
                      self.stream, Terminal(commands or (["s"] if shadow else ["s", "IDLE"])), shadow=shadow)
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def test_complete_baseline_policy_return_and_attended_idle(self):
        rows = self.run_policy()
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])
        np.testing.assert_allclose(self.robot.position, self.pose["sdk_end_position_m"])
        samples = [r for r in rows if r.get("phase") == "policy"]
        self.assertEqual(len(samples), 1001)
        self.assertEqual(samples[0]["tared_ft"], [0] * 6)
        self.assertAlmostEqual(samples[200]["target_sdk_m"][2], 0.14)
        self.assertTrue(any(r["event"] == "return_confirmed" for r in rows))
        self.assertEqual(rows[-1]["event"], "session_complete")

    def test_shadow_never_acquires_sends_switches_or_aborts(self):
        rows = self.run_policy(shadow=True)
        self.assertFalse(self.robot.calls)
        self.assertFalse(self.robot.targets)
        self.assertEqual(sum(r.get("phase") == "shadow" for r in rows), 1001)

    def test_q_does_not_start_policy(self):
        rows = self.run_policy(commands=["q", "IDLE"])
        self.assertFalse(any(r.get("phase") == "policy" for r in rows))
        self.assertEqual(self.robot.calls[-1], "idle")

    def test_causal_stale_fault_stops_without_return_or_idle(self):
        self.sensor.latest_before = lambda due, net=False: (due - 0.021, [0] * 6)
        with self.assertRaisesRegex(fixed.StateError, "fresh causal"):
            self.run_policy()
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertNotIn("return_target", self.stream.getvalue())
        self.assertNotIn("idle", self.robot.calls)

    def test_shadow_fault_does_not_request_stop(self):
        self.sensor.latest_before = lambda due, net=False: None
        with self.assertRaises(fixed.StateError):
            self.run_policy(shadow=True)
        self.assertFalse(self.robot.calls)
        self.assertFalse(self.robot.targets)

    def test_causal_raw_spike_is_checked_even_if_latest_is_safe(self):
        self.sensor.latest_before = lambda due, net=False: (due, [0, 0, 41, 0, 0, 0])
        with self.assertRaisesRegex(fixed.StateError, "Causal raw force"):
            self.run_policy()
        self.assertEqual(self.robot.calls[-1], "abort")

    def test_return_failure_never_releases_to_idle(self):
        with patch.object(fixed, "retract", side_effect=fixed.StateError("return timeout")):
            with self.assertRaisesRegex(fixed.StateError, "return timeout"):
                self.run_policy()
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertNotIn("idle", self.robot.calls)

    def test_return_checks_actual_pose_not_only_sent_commands(self):
        self.robot.position[2] -= 0.004
        self.robot.send = lambda *args: None
        with self.assertRaisesRegex(fixed.StateError, "return/stationary"):
            fixed.retract(self.robot, self.sensor, self.cfg, self.pose,
                          self.robot.read("servo"), self.stream)
        self.assertNotIn("return_confirmed", self.stream.getvalue())

    def test_unstable_baseline_is_fully_discarded_and_resampled(self):
        validate = fixed.validate_stationary
        started = self.clock.now
        count = 0
        def once_unstable(states, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise fixed.NotStationary("baseline disturbed")
            return validate(states, **kwargs)
        with patch.object(fixed, "validate_stationary", side_effect=once_unstable):
            bias = fixed.baseline(self.robot, self.sensor, self.cfg, self.pose, self.stream)
        rows = [json.loads(line) for line in self.stream.getvalue().splitlines()]
        self.assertEqual(sum(r["event"] == "baseline_discarded" for r in rows), 1)
        self.assertGreaterEqual(rows[-1]["distinct_samples"], 40)
        np.testing.assert_allclose(bias, [0, 0, 20, 0, 0, 0])
        self.assertLess(self.clock.now - started, 10)

    def test_large_feedback_stops_without_clipping(self):
        self.policy.delta = 0.004
        with self.assertRaisesRegex(fixed.StateError, "vertical increment"):
            self.run_policy()
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertNotIn("return_target", self.stream.getvalue())

    def test_raw_force_not_disabled_by_subtraction(self):
        self.sensor.bias = 100
        with self.assertRaisesRegex(fixed.StateError, "Force/torque"):
            self.run_policy()
        self.assertFalse(self.robot.targets)

    def test_runtime_faults_abort_no_recovery(self):
        for kind in ("lease", "rejected", "orientation", "depth", "deadline"):
            with self.subTest(kind=kind):
                self.robot = Robot(self.clock, self.pose)
                self.sensor.robot = self.robot
                original = self.robot.send
                def send(position, quaternion):
                    original(position, quaternion)
                    if len(self.robot.targets) == 20:
                        if kind == "lease":
                            self.robot.lease = False
                        elif kind == "rejected":
                            raise fixed.StateError("Command rejected")
                        elif kind == "orientation":
                            self.robot.angle = [0, 0, 1, 0]
                        elif kind == "depth":
                            self.robot.position[2] = 0.129
                        else:
                            self.clock.now += 0.025
                self.robot.send = send
                with self.assertRaises((fixed.StateError, ValueError)):
                    self.run_policy()
                self.assertEqual(self.robot.calls[-1], "abort")
                self.assertNotIn("idle", self.robot.calls)

    def record_only(self):
        self.cfg.update(tracking_error_policy="record-only", orientation_error_policy="record-only")

    def disturbed_send(self):
        original = self.robot.send
        def send(position, quaternion):
            original(position, quaternion)
            if 10 <= len(self.robot.targets) <= 20:
                self.robot.position[1] += 0.006
                self.robot.angle = [0, 0, np.sin(0.03), np.cos(0.03)]
            else:
                self.robot.angle = None
        self.robot.send = send

    def assert_recorded_deviations(self, rows):
        deviations = [r for r in rows if r.get("position_tracking_exceeds_reference")
                      and r.get("orientation_tracking_exceeds_reference")]
        self.assertTrue(deviations)
        for row in deviations:
            self.assertEqual(row["tracking_error_policy"], "record-only")
            self.assertEqual(row["orientation_error_policy"], "record-only")
            self.assertGreater(row["position_error_m"], 0.005)
            self.assertGreater(row["orientation_error_rad"], 0.02)
            self.assertEqual(len(row["tracking_target_sdk_m"]), 3)

    def test_record_only_policy_continues_and_logs_both_errors(self):
        self.record_only()
        self.disturbed_send()
        rows = self.run_policy()
        self.assert_recorded_deviations([r for r in rows if r.get("phase") == "policy"])
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])

    def test_record_only_direct_approach_continues_until_actual_start(self):
        self.record_only()
        self.robot.position[1] -= 0.015
        self.disturbed_send()
        rows = self.run_policy(commands=["q", "IDLE"])
        self.assert_recorded_deviations([r for r in rows if r.get("phase") == "approach"])
        np.testing.assert_allclose(self.robot.position, self.pose["sdk_end_position_m"])
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])

    def test_record_only_retract_continues_and_checks_final_return(self):
        self.record_only()
        self.robot.position[2] -= 0.01
        self.disturbed_send()
        fixed.retract(self.robot, self.sensor, self.cfg, self.pose, self.robot.read("servo"), self.stream)
        rows = [json.loads(line) for line in self.stream.getvalue().splitlines()]
        self.assert_recorded_deviations(rows)
        self.assertEqual(rows[-1]["event"], "return_confirmed")

    def test_record_only_does_not_disable_baseline_pose_qualification(self):
        self.record_only()
        self.robot.position[1] += 0.003
        with self.assertRaisesRegex(ValueError, "Return/start pose"):
            fixed.baseline(self.robot, self.sensor, self.cfg, self.pose, self.stream)
        self.assertNotIn("baseline_complete", self.stream.getvalue())

    def test_record_only_does_not_disable_other_guards(self):
        self.record_only()
        for fault in ("depth", "force", "speed", "invalid_orientation"):
            with self.subTest(fault=fault):
                self.robot.position = list(self.pose["sdk_end_position_m"])
                self.robot.speed, self.sensor.bias, self.robot.angle = 0, 0, None
                if fault == "depth":
                    self.robot.position[2] -= 0.021
                elif fault == "force":
                    self.sensor.bias = 100
                elif fault == "speed":
                    self.robot.speed = 1.3
                else:
                    self.robot.angle = [0, 0, float("nan"), 1]
                with self.assertRaises((ValueError, fixed.StateError)):
                    fixed.observation(self.robot, self.sensor, self.cfg, self.pose,
                                      self.pose["sdk_end_position_m"])


class FixedBindingTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs/real_deploy/airbot_fixed_setup.json"
        self.spec = json.loads(self.path.read_text())
        if not (ROOT / self.spec["policy"]).exists():
            self.skipTest("Reviewed fixed-setup policy fixture not installed")

    def test_exact_policy_inputs_load_but_unconfirmed_run_cannot_connect(self):
        # Historical policy sources predate current code. Isolate the confirmation
        # gate here; changed-source rejection is exercised separately below.
        with patch.object(fixed, "assert_unchanged") as source_check:
            loaded = fixed.load_setup(self.path)
        source_check.assert_called_once()
        self.assertEqual(loaded[0]["scope"], fixed.SCOPE)
        self.assertEqual(loaded[1]["tracking_error_policy"], "record-only")
        self.assertEqual(loaded[1]["orientation_error_policy"], "record-only")
        unconfirmed = {**loaded[0], "motion_limits_confirmed": False}
        self.assertTrue(fixed.run_blockers(unconfirmed, loaded[-1]))
        with patch.object(fixed, "load_setup", return_value=(unconfirmed, *loaded[1:])), \
                patch.object(fixed, "open_client") as client, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--mode", "fixed-setup", "--execute"]), 2)
        client.assert_not_called()

    def test_changed_source_provenance_still_prevents_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.py"
            source.write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                fixed.assert_unchanged({str(source): "0" * 64})
        with patch.object(fixed, "assert_unchanged", side_effect=ValueError("inputs changed")), \
                patch.object(fixed, "open_client") as client, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--mode", "fixed-setup", "--execute"]), 2)
        client.assert_not_called()

    def test_invalid_tracking_policies_rejected(self):
        for key in ("tracking_error_policy", "orientation_error_policy"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "config.json"
                path.write_text(json.dumps({**self.spec, key: "disabled"}))
                with self.assertRaisesRegex(ValueError, "stop or record-only"):
                    fixed.load_setup(path)

    def test_exception_is_bound_to_exact_model_encoder_and_session(self):
        for key in ("policy_sha256", "encoder_sha256", "collection_session_sha256", "scope"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                spec = copy.deepcopy(self.spec)
                spec["encoder_quality_exception"][key] = "wrong"
                path = Path(temp) / "config.json"
                path.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError, "exception"):
                    fixed.load_setup(path)

    def test_policy_hash_or_hardware_confirmation_change_rejected(self):
        for key, value in (("policy_sha256", "bad"), ("same_hardware_and_sponge_confirmed", False)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                spec = {**self.spec, key: value}
                path = Path(temp) / "config.json"
                path.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):
                    fixed.load_setup(path)

    def test_replay_requires_current_bindings_and_passed_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            spec = {**self.spec, **dict.fromkeys(fixed.CONFIRMATIONS, True), "replay_report": str(path)}
            path.write_text(json.dumps({"mode": fixed.MODE, "passed": True, "bindings": {"a": "b"}}))
            self.assertFalse(fixed.run_blockers(spec, {"a": "b"}))
            self.assertTrue(fixed.run_blockers(spec, {"a": "changed"}))
