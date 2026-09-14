"""Software-only motion contracts. No device connections or CAN traffic."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.robot_control.__main__ import (
    interact,
    main,
    save_pose,
    to_sdk_pose,
)
from scripts.robot_control.client import StateError
from scripts.robot_control.motion import (
    FakeRobot,
    MotionSession,
    fake_config,
    validate_config,
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class BasicControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.robot = FakeRobot()
        self.events = []
        self.session = MotionSession(
            self.robot,
            fake_config(),
            emit=self.events.append,
            clock=self.clock.now,
            sleep=self.clock.sleep,
        )
        self.session.begin()

    def test_modes_seed_hold_from_measured_joints(self):
        self.session.switch("zero-gravity")
        self.robot.state["joint_position_rad"] = [0.1] * 6
        self.session.hold()
        self.assertEqual(self.session.hold_target, ("joint", [0.1] * 6))
        self.assertEqual(self.robot.calls[:3], ["acquire", "gravity_comp", "servo"])
        self.session.finish(supported=True)
        self.assertEqual(self.robot.mode, "idle")

    def test_joint_and_pose_moves_and_velocity_bound(self):
        self.session.move("joint", [0.01] * 6)
        q = np.array([e["target"] for e in self.events if e["event"] == "target"])
        self.assertLessEqual(np.max(np.abs(np.diff(q, axis=0))) / 0.01, 0.1 + 1e-12)
        np.testing.assert_allclose(q[-1], [0.01] * 6)
        self.session.move("pose", ([0.251, 0.001, 0.301], [0, 0, 0, -1]))
        np.testing.assert_allclose(self.robot.state["sdk_end_position_m"], [0.251, 0.001, 0.301])
        self.assertTrue(any(e["event"] == "target-reached" for e in self.events))

    def test_keyboard_all_axes_and_no_repeat_queue(self):
        self.session.hold()
        for key in "1+2+3+4+5+6+xXyYzZrRpPwW":
            self.session.jog(key)
        np.testing.assert_allclose(self.robot.state["joint_position_rad"], [0.01] * 6)
        np.testing.assert_allclose(self.robot.state["sdk_end_position_m"], [0.25, 0.0, 0.3])

    def test_invalid_targets_never_enter_servo(self):
        for kind, target in (
            ("joint", [9.0] * 6),
            ("joint", [float("nan")] * 6),
            ("pose", ([2.0, 0.0, 0.3], [0, 0, 0, 1])),
            ("pose", ([0.25, 0.0, 0.3], [0, 0, 0, 0])),
        ):
            with self.assertRaises((ValueError, StateError)):
                self.session.move(kind, target)
        self.assertEqual(self.robot.calls, ["acquire"])

    def test_stop_latches_and_no_future_targets(self):
        self.session.hold()
        self.assertFalse(self.session.move("joint", [0.01] * 6, stop_requested=lambda: True))
        before = list(self.robot.calls)
        with self.assertRaises(StateError):
            self.session.hold()
        with self.assertRaises(StateError):
            self.session.move("joint", [0.0] * 6)
        with self.assertRaises(StateError):
            self.session.switch("gravity-comp")
        self.session.finish(supported=True)
        self.assertEqual(self.robot.calls, before)

    def test_control_loss_never_reacquires(self):
        self.robot.lease = False
        with self.assertRaises(StateError):
            self.session.hold()
        self.assertEqual(self.robot.calls.count("acquire"), 1)

    def test_feedback_bounds_and_tracking_rejected(self):
        self.session.hold()
        self.robot.state["joint_velocity_rad_s"][0] = 0.21
        with self.assertRaises(StateError):
            self.session.read()
        self.robot.state["joint_velocity_rad_s"][0] = 0
        self.robot.state["joint_position_rad"][0] = 0.2
        with self.assertRaises(StateError):
            self.session.maintain()
        self.robot.state["sdk_end_position_m"][2] = -0.1
        self.session.read()
        self.robot.state["sdk_end_position_m"][2] = float("nan")
        with self.assertRaises(StateError):
            self.session.read()

    def test_no_workspace_fields_required_and_legacy_xyz_bounds_ignored(self):
        cfg = fake_config()
        self.assertNotIn("workspace_min_m", cfg)
        validate_config(cfg)
        cfg.update(workspace_min_m=[0, 0, 0], workspace_max_m=[0.01] * 3)
        validate_config(cfg)
        self.session.cfg = cfg
        self.session.read()
        self.session.move("pose", ([0.251, 0.001, 0.301], [0, 0, 0, 1]))
        np.testing.assert_allclose(self.robot.state["sdk_end_position_m"], [0.251, 0.001, 0.301])

    def test_stop_failure_exposed_and_not_cleared(self):
        with patch.object(self.robot, "abort", side_effect=StateError("stop refused")):
            with self.assertRaisesRegex(StateError, "stop refused"):
                self.session.stop()
        self.assertTrue(self.session.faulted)
        self.assertNotIn("idle", self.robot.calls)

    def test_idle_needs_support_confirmation(self):
        with self.assertRaises(StateError):
            self.session.finish(supported=False)
        self.assertNotIn("idle", self.robot.calls)

    def test_pose_saved_exclusively_with_synthetic_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            save_pose(self.session, path)
            self.assertEqual(json.loads(path.read_text())["metadata"]["source_kind"], "synthetic")
            with self.assertRaises(FileExistsError):
                save_pose(self.session, path)

    def test_hardware_gates_before_connection(self):
        with (
            patch("scripts.robot_control.client.open_client") as connect,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(["hold", "--execute"]), 2)
        connect.assert_not_called()
        with self.assertRaises(ValueError):
            validate_config(fake_config(), hardware=True)

    def test_cli_offline_defaults_and_no_hardware_import(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("scripts.robot_control.client.open_client") as connect:
                result = main(
                    ["move-joint", "--joints", *([".001"] * 6), "--log", directory + "/log.jsonl"]
                )
            connect.assert_not_called()
            self.assertEqual(result, 0)
            records = [
                json.loads(line) for line in Path(directory + "/log.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[-1]["event"], "idle-confirmed")
            self.assertTrue(all(r["source_kind"] == "synthetic" for r in records))

    def test_motion_timeout_and_missed_tick(self):
        self.session.hold()
        with patch.object(self.robot, "send_joint"):
            with self.assertRaisesRegex(StateError, "settle"):
                self.session.move("joint", [0.01] * 6)
        self.robot.hardware = True
        original_read = self.robot.read
        calls = 0

        def delayed_read(expected=None):
            nonlocal calls
            calls += 1
            if calls > 1:
                self.clock.sleep(0.06)
            return original_read(expected)

        with patch.object(self.robot, "read", side_effect=delayed_read):
            with self.assertRaisesRegex(StateError, "tick missed"):
                self.session.move("joint", [0.01] * 6)

    def test_failed_mode_confirmation_and_empty_input(self):
        self.session.jog("")
        with patch.object(self.robot, "switch"):
            with self.assertRaisesRegex(StateError, "controller"):
                self.session.switch("servo")
        self.assertNotIn("joint", self.robot.calls)

    def test_console_requires_confirmation_and_disables_jog_after_drag(self):
        keys = iter(["k", "n", "k", "Y", "g", "Y", "x", "q", "Y"])
        with (
            patch("scripts.robot_control.__main__.terminal", return_value=nullcontext()),
            patch(
                "scripts.robot_control.__main__.key_available",
                side_effect=lambda *a: next(keys),
            ),
            redirect_stdout(io.StringIO()),
        ):
            interact(self.session)
        self.assertEqual(self.robot.calls.count("servo"), 1)
        self.assertEqual(self.robot.calls.count("gravity_comp"), 1)
        self.assertNotIn("pose", self.robot.calls)
        self.assertEqual(self.robot.mode, "idle")

    def test_tcp_target_roundtrip(self):
        from types import SimpleNamespace

        from scipy.spatial.transform import Rotation

        from scripts.shared.airbot_calibration import Frames

        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
        base[:3, 3] = [0.1, 0.2, 0.3]
        tcp = np.eye(4)
        tcp[:3, 3] = [0.01, 0, 0.02]
        tcp[:3, :3] = Rotation.from_euler("x", 0.2).as_matrix()
        frames = Frames(
            dict(
                base_from_sdk=base,
                end_from_tcp=tcp,
                sensor_to_ft_frame=np.eye(4),
                sensor_bias_si=[0] * 6,
            )
        )
        q = Rotation.from_euler("y", 0.3).as_quat()
        position, orientation = to_sdk_pose([0.2, 0.3, 0.4], q, SimpleNamespace(frames=frames))
        state = dict(sdk_end_position_m=position, sdk_end_orientation_xyzw=orientation)
        np.testing.assert_allclose(frames.to_tcp(state), [0.2, 0.3, 0.4], atol=1e-12)
        np.testing.assert_allclose(frames.tcp_quaternion(orientation), q, atol=1e-12)

    def test_cli_eof_interrupt_and_partial_mode_failure_request_stop(self):
        from scripts.robot_control.motion import FLAGS

        for failure in (EOFError("closed"), KeyboardInterrupt(), StateError("read failed")):
            cfg = fake_config()
            cfg.update(source_kind="real", **{key: True for key in FLAGS})
            robot = FakeRobot()
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.json"
                config.write_text(json.dumps(cfg))
                client = MagicMock()
                with (
                    patch("scripts.robot_control.client.open_client", return_value=client),
                    patch(
                        "scripts.robot_control.hardware.HardwareRobot", return_value=robot
                    ),
                    patch("scripts.robot_control.__main__.interact", side_effect=failure),
                    patch("sys.stdin.isatty", return_value=True),
                    patch("builtins.input", return_value="CONTROL"),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    code = main(
                        [
                            "hold",
                            "--execute",
                            "--config",
                            str(config),
                            "--log",
                            directory + "/events.jsonl",
                        ]
                    )
                self.assertIn(code, (2, 130))
                self.assertIn("stop", robot.calls)
                self.assertNotIn("idle", robot.calls)
                client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
