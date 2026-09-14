"""No hardware connections; test teaching with a strictly limited fake client."""

import copy
import io
import json
import math
import os
import pty
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.robot_control.airbot_initial_pose as recorder


class ExplorationConfigTests(unittest.TestCase):
    def test_updates_only_pose_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pose = root / "start.json"
            pose.write_text("{}")
            config = root / "config.json"
            config.write_text(json.dumps({"initial_pose_file": "old.json", "force": {"limit": 40}}))
            with patch.object(recorder, "ROOT", root):
                recorder.update_exploration_config(pose, config)
            self.assertEqual(json.loads(config.read_text()),
                             {"initial_pose_file": "start.json", "force": {"limit": 40}})

    def test_failed_replace_preserves_config_and_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pose = root / "start.json"
            pose.write_text("{}")
            config = root / "config.json"
            config.write_text('{"initial_pose_file": "old.json"}')
            before = config.read_bytes()
            with patch.object(recorder.os, "replace", side_effect=OSError("test failure")):
                with self.assertRaises(OSError):
                    recorder.update_exploration_config(pose, config)
            self.assertEqual(config.read_bytes(), before)
            self.assertTrue(pose.exists())
            self.assertEqual(list(root.glob(".initial-pose-*")), [])


@dataclass
class Service:
    service_state: bool = True
    valid: bool = True
    fsm_state: str = "running"
    controller_state: str = "idle"


class FakeClient:
    def __init__(self):
        self.state = Service()
        self.lease = False
        self.acquire_ok = True
        self.enter_ok = True
        self.idle_ok = True
        self.calls = []
        self.errors = [0] * 6
        self.joints = SimpleNamespace(angles=[0.1] * 6, velocities=[0.0] * 6)
        self.pose = SimpleNamespace(position=[0.25, 0.02, 0.15], orientation=[0, 0, 0, 1])

    def get_service_state(self):
        return self.state

    def get_arm_motor_state(self):
        return SimpleNamespace(error_ids=self.errors)

    def get_arm_joint_state(self):
        return self.joints

    def get_end_pose(self):
        return self.pose

    def get_firmware_info(self):
        return None

    def acquire_control(self):
        self.calls.append("acquire")
        self.lease = self.acquire_ok
        return self.acquire_ok

    def has_control(self):
        return self.lease

    def enter_gravity_compensation_mode(self):
        self.calls.append("gravity")
        if self.enter_ok:
            self.state.controller_state = "gravity_comp"
        return self.enter_ok

    def switch_controller(self, controller):
        self.calls.append(controller)
        if self.idle_ok:
            self.state.controller_state = controller
        return self.idle_ok

    def close(self):
        self.calls.append("close")


class InitialPoseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "initial.json"
        self.client = FakeClient()
        self.output = io.StringIO()
        for context in (
            redirect_stdout(self.output),
            redirect_stderr(self.output),
            patch.object(recorder.time, "sleep"),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def run_teach(self, commands):
        iterator = iter(commands)
        return recorder.teach(
            self.client,
            "idle",
            self.path,
            {"label": "test"},
            command_reader=lambda _: next(iterator),
        )

    def samples(self):
        return [recorder.snapshot(self.client) for _ in range(recorder.SAMPLE_COUNT)]

    def test_single_key_save_exits_without_motion(self):
        self.assertTrue(self.run_teach(["s"]))
        self.assertEqual(self.client.calls, ["acquire", "gravity", "idle"])
        record = json.loads(self.path.read_text())
        self.assertEqual(record["sdk_version"], "5.2.2")
        self.assertEqual(record["initial_pose"]["joint_position_rad"], [0.1] * 6)
        self.assertEqual(record["initial_pose"]["sdk_end_orientation_xyzw"], [0, 0, 0, 1])
        self.assertEqual(len(record["samples"]), 11)
        self.assertFalse(record["coordinate_semantics"]["sponge_tcp_calibrated"])

    def test_q_is_not_an_exit_option_and_uppercase_s_saves(self):
        self.assertTrue(self.run_teach(["q", "S"]))
        self.assertTrue(self.path.exists())
        self.assertIn("Press S to save and exit", self.output.getvalue())

    def test_idle_capture_never_acquires_or_switches(self):
        samples = recorder.capture_idle(self.client)
        self.assertEqual(len(samples), recorder.SAMPLE_COUNT)
        self.assertEqual(self.client.calls, [])

    def test_idle_capture_rejects_motion_and_other_controllers(self):
        self.client.joints.velocities[0] = 0.06
        with self.assertRaises(recorder.NotStationary):
            recorder.capture_idle(self.client)
        self.client.state.controller_state = "servo"
        with self.assertRaises(recorder.StateError):
            recorder.capture_idle(self.client)
        self.assertEqual(self.client.calls, [])

    def test_idle_capture_rejects_controller_change_during_sampling(self):
        def change(_):
            self.client.state.controller_state = "gravity_comp"

        with patch.object(recorder.time, "sleep", side_effect=change):
            with self.assertRaises(recorder.StateError):
                recorder.capture_idle(self.client)
        self.assertEqual(self.client.calls, [])

    def test_idle_cli_saves_without_control(self):
        self.client.get_firmware_info = lambda: Service()
        with (
            patch.object(recorder, "open_client", return_value=self.client),
            patch.dict("sys.modules", {"arm_sdk": None}),
        ):
            result = recorder.main(
                ["capture-idle", "--output", str(self.path), "--label", "air_motion_start"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(self.client.calls, ["close"])
        self.assertEqual(
            json.loads(self.path.read_text())["metadata"]["capture_method"], "idle_read_only"
        )

    def test_refused_lease_never_switches_mode(self):
        self.client.acquire_ok = False
        with self.assertRaises(recorder.StateError):
            self.run_teach(["s"])
        self.assertEqual(self.client.calls, ["acquire"])

    def test_existing_active_controller_not_taken_over(self):
        self.client.state.controller_state = "servo"
        with self.assertRaises(recorder.StateError):
            self.run_teach(["s"])
        self.assertEqual(self.client.calls, [])

    def test_failed_entry_requests_idle_without_saving(self):
        self.client.enter_ok = False
        with self.assertRaises(recorder.StateError):
            self.run_teach(["s"])
        self.assertEqual(self.client.calls, ["acquire", "gravity", "idle"])
        self.assertFalse(self.path.exists())

    def test_interrupt_and_eof_cleanup(self):
        for error in (KeyboardInterrupt, EOFError):
            self.client = FakeClient()
            with self.subTest(error=error), self.assertRaises(error):
                recorder.teach(
                    self.client,
                    "idle",
                    self.path,
                    {},
                    command_reader=lambda _: (_ for _ in ()).throw(error()),
                )
            self.assertEqual(self.client.calls[-1], "idle")

    def test_idle_failure_is_not_reported_as_success(self):
        self.client.idle_ok = False
        with self.assertRaises(recorder.StateError):
            self.run_teach(["s"])
        self.assertIn("idle NOT confirmed", self.output.getvalue())
        self.assertTrue(self.path.exists())

    def test_lease_loss_never_reacquires(self):
        def lost_lease(_):
            self.client.lease = False
            return "s"

        with self.assertRaises(recorder.StateError):
            recorder.teach(self.client, "idle", self.path, {}, command_reader=lost_lease)
        self.assertEqual(self.client.calls, ["acquire", "gravity"])
        self.assertFalse(self.path.exists())

    def test_moving_arm_rejected_and_can_retry(self):
        self.client.joints.velocities[0] = 0.06

        def commands(_):
            if "Not saved" in self.output.getvalue():
                self.client.joints.velocities[0] = 0
                return "s"
            return "s"

        self.assertTrue(recorder.teach(self.client, "idle", self.path, {}, command_reader=commands))
        self.assertIn("Not saved", self.output.getvalue())

    def test_stable_pose_with_speed_noise_saves_current_thresholds(self):
        self.client.joints.velocities = [0, 0, 0, -0.037, 0.022, -0.022]
        self.assertTrue(self.run_teach(["s"]))
        record = json.loads(self.path.read_text())
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(
            record["stationarity_check"],
            {
                "sample_count": 11,
                "max_joint_speed_rad_s": 0.05,
                "max_joint_span_rad": 0.01,
                "max_position_span_m": 0.002,
                "max_orientation_span_rad": 0.02,
            },
        )
        self.assertEqual(recorder.SAMPLE_INTERVAL_S, 0.05)
        self.assertEqual(record["initial_pose"]["joint_velocity_rad_s"][3], -0.037)

    def test_speed_boundary_in_both_directions_for_every_joint(self):
        for axis in range(6):
            for speed in (-0.05, 0.05):
                samples = self.samples()
                samples[5]["joint_velocity_rad_s"][axis] = speed
                with self.subTest(axis=axis, speed=speed):
                    recorder.validate_stationary(samples)
            for speed in (-0.050001, 0.050001):
                samples = self.samples()
                samples[5]["joint_velocity_rad_s"][axis] = speed
                with (
                    self.subTest(axis=axis, speed=speed),
                    self.assertRaises(recorder.NotStationary) as error,
                ):
                    recorder.validate_stationary(samples)
                self.assertIn(f"J{axis + 1} peak=0.050001 rad/s", str(error.exception))
                self.assertIn("limit=0.05 rad/s", str(error.exception))

    def test_speed_diagnostic_reports_all_exceeded_joints_and_peaks(self):
        samples = self.samples()
        samples[3]["joint_velocity_rad_s"][3] = -0.07
        samples[5]["joint_velocity_rad_s"][3] = 0.06
        samples[7]["joint_velocity_rad_s"][5] = 0.08
        with self.assertRaises(recorder.NotStationary) as error:
            recorder.validate_stationary(samples)
        message = str(error.exception)
        self.assertIn("J4 peak=0.07 rad/s", message)
        self.assertIn("J6 peak=0.08 rad/s", message)
        self.assertNotIn("J5", message)

    def test_stationary_check_positions_and_quaternion_sign(self):
        self.client.joints.velocities = [0, 0, 0, -0.037, 0.022, -0.022]
        samples = self.samples()
        samples[3]["sdk_end_orientation_xyzw"] = [0, 0, 0, -1]
        recorder.validate_stationary(samples)
        for key, index, delta in (
            ("joint_position_rad", 0, 0.02),
            ("sdk_end_position_m", 2, 0.003),
        ):
            changed = copy.deepcopy(samples)
            changed[5][key][index] += delta
            with self.subTest(key=key), self.assertRaises(recorder.NotStationary):
                recorder.validate_stationary(changed)
        samples[5]["sdk_end_orientation_xyzw"] = [0, 0, math.sin(0.02), math.cos(0.02)]
        with self.assertRaises(recorder.NotStationary):
            recorder.validate_stationary(samples)

    def test_invalid_and_faulted_states(self):
        for field, value in (
            ("position", [math.nan, 0, 0]),
            ("position", [0, 0]),
            ("orientation", [0, 0, 0, 0]),
        ):
            self.client = FakeClient()
            setattr(self.client.pose, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(recorder.StateError):
                recorder.snapshot(self.client)
        self.client = FakeClient()
        self.client.state.valid = False
        with self.assertRaises(recorder.StateError):
            recorder.snapshot(self.client)
        self.client.state.valid = True
        self.client.errors[2] = 8
        with self.assertRaises(recorder.StateError):
            recorder.snapshot(self.client)
        self.client.errors = [0] * 6
        self.client.joints = None
        with self.assertRaises(recorder.StateError):
            recorder.snapshot(self.client)

    def test_no_overwrite_and_cleanup_on_save_failure(self):
        self.path.write_text("existing pose")
        with self.assertRaises(FileExistsError):
            self.run_teach(["s"])
        self.assertEqual(self.path.read_text(), "existing pose")
        self.assertEqual(self.client.calls[-1], "idle")

    def test_od_dm_no_error_codes_are_preserved_and_can_be_taught(self):
        self.client.errors = [0, 0, 0, 1, 1, 1]
        state = recorder.snapshot(self.client)
        self.assertEqual(state["service_state"]["motor_status_codes"], self.client.errors)
        self.assertTrue(self.run_teach(["s"]))
        record = json.loads(self.path.read_text())
        self.assertEqual(
            record["initial_pose"]["service_state"]["motor_status_codes"], [0, 0, 0, 1, 1, 1]
        )

    def test_other_motor_codes_still_block_control_acquisition(self):
        for joint in range(6):
            for code in (-1, *range(2, 256)):
                self.client = FakeClient()
                self.client.errors[joint] = code
                with self.subTest(joint=joint, code=code), self.assertRaises(recorder.StateError):
                    self.run_teach(["s"])
                self.assertEqual(self.client.calls, [])
        self.assertFalse(self.path.exists())

    def test_inspect_accepts_normal_dm_status_without_acquiring_control(self):
        self.client.errors = [0, 0, 0, 1, 1, 1]
        with patch.object(recorder, "open_client", return_value=self.client):
            self.assertEqual(recorder.main(["inspect"]), 0)
        self.assertEqual(self.client.calls, ["close"])

    def test_execute_and_terminal_gates_before_connect(self):
        with patch.object(recorder, "open_client") as connect:
            with self.assertRaises(SystemExit):
                recorder.main(["teach"])
            with patch.object(recorder.sys.stdin, "isatty", return_value=False):
                with self.assertRaises(SystemExit):
                    recorder.main(["teach", "--execute", "--output", str(self.path)])
            connect.assert_not_called()

    def test_cancel_does_not_connect(self):
        with (
            patch.object(recorder, "open_client") as connect,
            patch.object(recorder.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", return_value="no"),
        ):
            self.assertEqual(recorder.main(["teach", "--execute", "--output", str(self.path)]), 0)
            connect.assert_not_called()

    def test_inspect_is_read_only_and_closes(self):
        with patch.object(recorder, "open_client", return_value=self.client):
            self.assertEqual(recorder.main(["inspect"]), 0)
        self.assertEqual(self.client.calls, ["close"])
        self.assertFalse(self.path.exists())

    def test_reject_other_sdk_before_import_or_connection(self):
        with patch("scripts.robot_control.client.version", return_value="5.1.6"):
            with self.assertRaisesRegex(RuntimeError, "Requires arm-sdk 5.2.2"):
                recorder.open_client("127.0.0.1", 50051)


class SingleKeyTerminalTests(unittest.TestCase):
    def setUp(self):
        self.master, self.slave = pty.openpty()
        self.addCleanup(os.close, self.master)
        self.addCleanup(os.close, self.slave)
        self.stdin = os.fdopen(os.dup(self.slave), "r")
        self.addCleanup(self.stdin.close)
        self.original = recorder.termios.tcgetattr(self.slave)
        self.client = FakeClient()
        self.client.lease = True
        self.client.state.controller_state = "gravity_comp"
        self.select = recorder.select.select

    def send_key(self, key):
        def ready(readers, writers, errors, timeout):
            flags = recorder.termios.tcgetattr(self.slave)[3]
            self.assertFalse(flags & recorder.termios.ICANON)
            self.assertFalse(flags & recorder.termios.ECHO)
            self.assertTrue(flags & recorder.termios.ISIG)
            os.write(self.master, key)
            result = self.select(readers, writers, errors, 1)
            self.assertTrue(result[0], "Single key must be readable without a newline")
            return result
        return ready

    def test_s_key_without_enter_and_terminal_restored(self):
        for key in (b"s", b"S"):
            with (
                self.subTest(key=key),
                patch.object(recorder.sys, "stdin", self.stdin),
                patch.object(recorder.select, "select", side_effect=self.send_key(key)),
            ):
                self.assertEqual(recorder.read_command(self.client), "s")
            self.assertEqual(recorder.termios.tcgetattr(self.slave), self.original)

    def test_ctrl_d_and_interrupt_restore_terminal(self):
        with (
            patch.object(recorder.sys, "stdin", self.stdin),
            patch.object(recorder.select, "select", side_effect=self.send_key(b"\x04")),
            self.assertRaises(EOFError),
        ):
            recorder.read_command(self.client)
        self.assertEqual(recorder.termios.tcgetattr(self.slave), self.original)
        with (
            patch.object(recorder.sys, "stdin", self.stdin),
            patch.object(recorder.select, "select", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            recorder.read_command(self.client)
        self.assertEqual(recorder.termios.tcgetattr(self.slave), self.original)

    def test_lost_lease_and_health_failure_restore_terminal(self):
        for lease, valid in ((False, True), (True, False)):
            self.client.lease = lease
            self.client.state.valid = valid
            with patch.object(recorder.sys, "stdin", self.stdin), self.assertRaises(recorder.StateError):
                recorder.read_command(self.client)
            self.assertEqual(recorder.termios.tcgetattr(self.slave), self.original)

    def test_teach_saves_and_exits_after_one_key_without_enter(self):
        self.client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "initial.json"
            with (
                patch.object(recorder.sys, "stdin", self.stdin),
                patch.object(recorder.select, "select", side_effect=self.send_key(b"S")) as read,
                patch.object(recorder.time, "sleep"),
                redirect_stdout(io.StringIO()),
            ):
                self.assertTrue(recorder.teach(self.client, "idle", path, {}))
            self.assertTrue(path.exists())
            self.assertEqual(read.call_count, 1)
        self.assertEqual(self.client.calls, ["acquire", "gravity", "idle"])
        self.assertEqual(recorder.termios.tcgetattr(self.slave), self.original)


if __name__ == "__main__":
    unittest.main()
