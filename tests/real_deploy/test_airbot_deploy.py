"""No-hardware deployment scheduling, frame transforms and failure paths."""

import copy
import io
import json
import threading
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.force_sensor.kwr75_reader import Kwr75Reader
from scripts.real_deploy.airbot_deploy import (
    Frames,
    PolicyLoop,
    deployment_blockers,
    execute,
    guard_segment,
    main,
    rigid,
    validate_setup,
)
from scripts.robot_control.client import StateError
from scripts.shared.common import ROOT
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_preprocessing import CausalFTFilter


class FakePolicy:
    def predict_xy(self, z):
        return np.tile([0.1, 0.2], (1, 25, 1))

    def make_ft_filter(self, hz):
        return CausalFTFilter(hz)

    def predict_delta_h(self, z, ft):
        return np.array([[0.0001]])


class FakeRobot:
    def __init__(self):
        self.sent, self.stopped, self.idled = [], False, False
        self.position = [0.1, 0.2, 0.3]

    def acquire(self):
        pass

    def owned(self):
        pass

    def enter(self):
        pass

    def read(self, controller):
        return {
            "sdk_end_position_m": list(self.position),
            "sdk_end_orientation_xyzw": [0, 0, 0, 1],
            "joint_position_rad": [0] * 6,
            "joint_velocity_rad_s": [0] * 6,
        }

    def send(self, position, quaternion):
        self.sent.append(position)
        self.position = list(position)

    def abort(self):
        self.stopped = True

    def idle(self):
        self.idled = True


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def sleep(self, duration):
        self.value += duration


class FakeSensor:
    def __init__(self, clock, failure=None):
        self.clock, self.failure = clock, failure
        self.calls = 0

    def latest(self, net=False):
        return self.clock.value, np.zeros(6)

    def latest_before(self, due, net=False):
        self.calls += 1
        if self.failure and self.calls == 42:
            if isinstance(self.failure, BaseException):
                raise self.failure
            return due - 0.021, np.zeros(6)
        return due, np.zeros(6)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "base_from_sdk": np.eye(4).tolist(),
            "end_from_tcp": np.eye(4).tolist(),
            "sensor_to_ft_frame": np.eye(4).tolist(),
            "sensor_bias_si": [0] * 6,
            "joint_min_rad": [-3] * 6,
            "joint_max_rad": [3] * 6,
            "joint_speed_limit_rad_s": 0.1,
            "measured_joint_speed_stop_rad_s": 0.2,
            "max_force_n": 5,
            "max_initial_force_n": 5,
            "max_torque_nm": 0.5,
            "max_cartesian_speed_m_s": 0.05,
            "max_delta_h_m": 0.001,
            "max_tracking_error_m": 0.002,
            "max_start_error_m": 0.001,
            "orientation_error_limit_rad": 0.02,
            "initial_tcp_position_m": [0.1, 0.2, 0.3],
            "sdk_end_orientation_xyzw": [0, 0, 0, 1],
        }

    def test_frame_roundtrip_with_rotation_and_offset(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["base_from_sdk"][:3] = np.c_[
            Rotation.from_euler("z", 40, degrees=True).as_matrix(), [1, 2, 3]
        ].tolist()
        cfg["end_from_tcp"][0][3] = 0.03
        frames = Frames(cfg)
        q = Rotation.from_euler("x", 30, degrees=True).as_quat()
        state = {"sdk_end_position_m": [0.1, 0.2, 0.3], "sdk_end_orientation_xyzw": q}
        np.testing.assert_allclose(
            frames.to_end(frames.to_tcp(state), q), state["sdk_end_position_m"], atol=1e-12
        )

    def test_wrench_translation_includes_lever_arm_and_bias(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["sensor_to_ft_frame"][0][3] = 0.1
        cfg["sensor_bias_si"] = [1, 0, 0, 0, 0, 0]
        np.testing.assert_allclose(Frames(cfg).wrench([1, 2, 0, 0, 0, 0]), [0, 2, 0, 0, 0, 0.2])
        with self.assertRaises(ValueError):
            Frames(cfg).wrench([1])

    def test_nonrigid_transform_rejected(self):
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            rigid(bad, "reflection")

    def test_warmup_and_twenty_next_step_predictions(self):
        loop = PolicyLoop(FakePolicy(), np.zeros((1, 5)), [0.1, 0.2, 0.3])
        predictions = []
        for tick in range(1001):
            loop.push(tick, np.zeros(6), [0.1, 0.2, 0.3])
            if tick < 200:
                self.assertAlmostEqual(loop.target(tick)[2], 0.3)
            if loop.last_delta is not None:
                predictions.append(tick)
        self.assertEqual(predictions, list(range(200, 1000, 40)))
        self.assertAlmostEqual(loop.target(1000)[2], 0.3001)
        with self.assertRaises(ValueError):
            loop.push(1002, np.zeros(6), [0.1, 0.2, 0.3])

    def test_interpolation_continuity_at_first_boundary(self):
        policy = FakePolicy()
        policy.predict_xy = lambda z: np.array([[[0.1 + i * 0.001, 0.2] for i in range(25)]])
        loop = PolicyLoop(policy, np.zeros((1, 5)), [0.09, 0.2, 0.3])
        for tick in range(41):
            loop.push(tick, np.zeros(6), [0.1, 0.2, 0.3])
        self.assertAlmostEqual(loop.target(40)[0], 0.1)
        self.assertAlmostEqual(loop.target(60)[0], 0.1005)

    def test_latest_before_never_uses_future_and_selects_last_in_batch(self):
        sensor = Kwr75Reader.__new__(Kwr75Reader)
        sensor._lock = threading.Lock()
        sensor._bias = np.ones(6)
        sensor._ring = deque([(1.0, np.ones(6)), (1.0, np.ones(6) * 2), (1.02, np.ones(6) * 3)])
        np.testing.assert_array_equal(sensor.latest_before(1.01)[1], np.ones(6) * 2)
        np.testing.assert_array_equal(sensor.latest_before(1.01, net=True)[1], np.ones(6))
        self.assertIsNone(sensor.latest_before(0.99))

    def test_predictions_are_rejected_not_clipped(self):
        loop = PolicyLoop(FakePolicy(), np.zeros((1, 5)), [0.1, 0.2, 0.3])
        loop.last_delta = 0.002
        with self.assertRaisesRegex(StateError, "vertical increment"):
            guard_segment(loop, self.cfg)
        loop.last_delta = None
        loop.segment_end[0] = 0.9
        with self.assertRaisesRegex(StateError, "Cartesian speed"):
            guard_segment(loop, self.cfg)

    def test_deployment_ignores_legacy_workspace_bounds(self):
        cfg = {**self.cfg, "workspace_min_m": [0] * 3, "workspace_max_m": [0.01] * 3}
        robot, events = self.run_fake(cfg=cfg)
        self.assertTrue(robot.idled)
        self.assertEqual(len(robot.sent), 1001)
        self.assertEqual(events[-1]["event"], "session_complete")

    def run_fake(self, failure=None, cfg=None, robot=None):
        clock, robot = FakeClock(), robot or FakeRobot()
        sensor = FakeSensor(clock, failure)
        log = io.StringIO()
        cfg = cfg or self.cfg
        with (
            patch(
                "scripts.real_deploy.airbot_deploy.time.perf_counter",
                side_effect=lambda: clock.value,
            ),
            patch("scripts.real_deploy.airbot_deploy.time.sleep", side_effect=clock.sleep),
        ):
            execute(
                robot,
                sensor,
                FakePolicy(),
                np.zeros((1, 5)),
                cfg,
                Frames(cfg),
                log,
                finish=lambda check: check(),
            )
        return robot, [json.loads(line) for line in log.getvalue().splitlines()]

    def test_complete_mock_execution_and_idle_handoff(self):
        robot, rows = self.run_fake()
        self.assertEqual(len(robot.sent), 1001)
        self.assertFalse(robot.stopped)
        self.assertTrue(robot.idled)
        self.assertEqual(rows[-1]["event"], "session_complete")

    def test_stale_sensor_aborts_without_retract_or_idle(self):
        robot = FakeRobot()
        with self.assertRaisesRegex(StateError, "fresh causal"):
            self.run_fake("stale", robot=robot)
        self.assertTrue(robot.stopped)
        self.assertFalse(robot.idled)
        self.assertEqual(len(robot.sent), 41)

    def test_interrupt_requests_stop(self):
        robot = FakeRobot()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(KeyboardInterrupt(), robot=robot)
        self.assertTrue(robot.stopped)
        self.assertFalse(robot.idled)

    def test_start_mismatch_sends_no_commands(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["initial_tcp_position_m"] = [0.3, 0.2, 0.3]
        robot = FakeRobot()
        with self.assertRaisesRegex(StateError, "Manually position"):
            self.run_fake(cfg=cfg, robot=robot)
        self.assertEqual(robot.sent, [])
        self.assertFalse(robot.stopped)

    def test_servo_read_failure_requests_stop(self):
        robot = FakeRobot()
        original = robot.read

        def fail(controller):
            if controller == "servo":
                raise StateError("RPC unavailable")
            return original(controller)

        robot.read = fail
        with self.assertRaisesRegex(StateError, "RPC unavailable"):
            self.run_fake(robot=robot)
        self.assertTrue(robot.stopped)
        self.assertEqual(robot.sent, [])

    def test_overforce_preflight_sends_no_commands(self):
        robot = FakeRobot()
        with patch.object(FakeSensor, "latest", return_value=(0.0, np.ones(6) * 10)):
            with self.assertRaisesRegex(StateError, "Force/torque"):
                self.run_fake(robot=robot)
        self.assertEqual(robot.sent, [])

    def test_missed_tick_does_not_catch_up(self):
        robot = FakeRobot()
        original = FakeClock.sleep

        def delayed(clock, duration):
            original(clock, duration)
            if duration < 0.02:
                clock.value += 0.006

        with patch.object(FakeClock, "sleep", delayed):
            with self.assertRaisesRegex(StateError, "deadline"):
                self.run_fake(robot=robot)
        self.assertTrue(robot.stopped)
        self.assertEqual(robot.sent, [])

    def test_unreviewed_native_policy_cannot_connect(self):
        path = ROOT / "archive/real_training/real_training_airbot_native_v1/policy.pt"
        if not path.exists():
            self.skipTest("Trained real policy is not installed")
        cfg = json.loads((ROOT / "configs/real_deploy/airbot_deployment.json").read_text())
        policy = OfflinePolicy(path)
        blockers = deployment_blockers(policy, cfg)
        self.assertTrue(any("Native-frame" in item for item in blockers))
        self.assertTrue(any("outside the encoder" in item for item in blockers))
        with patch("scripts.real_deploy.airbot_deploy.open_client") as connect:
            self.assertEqual(main(["run", "--execute"]), 2)
            connect.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Deployment blocked"):
            validate_setup(policy, cfg)


if __name__ == "__main__":
    unittest.main()
