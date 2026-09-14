"""Deterministic fake-hardware tests; never create an AIRBOT or serial connection."""

import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.real_training.airbot_exploration as exp
from scripts.shared.paths import ROOT


class Clock:
    def __init__(self):
        self.t = 10.0

    def now(self):
        return self.t

    def sleep(self, duration):
        self.t += duration


class Sensor:
    def __init__(self, clock):
        self.clock = clock
        self.wrench = [0.0] * 6
        self.age = 0

    def latest(self, net=False):
        assert net is False
        return self.clock.now() - self.age, self.wrench


class FakeRobot:
    def __init__(self, clock, pose):
        self.clock = clock
        self.state = copy.deepcopy(pose)
        self.state["joint_velocity_rad_s"] = [0.0] * 6
        self.controller = "idle"
        self.calls = []
        self.targets = []
        self.lease = False
        self.on_send = lambda: None

    def read(self, controller):
        if controller != self.controller:
            raise exp.StateError("Unexpected controller")
        return copy.deepcopy(self.state)

    def acquire(self):
        self.calls.append("acquire")
        self.lease = True

    def owned(self):
        if not self.lease:
            raise exp.StateError("Lost lease")

    def enter(self):
        self.calls.append("servo")
        self.controller = "servo"

    def send(self, position, quaternion):
        self.owned()
        self.targets.append(list(position))
        self.state["sdk_end_position_m"] = list(position)
        self.state["sdk_end_orientation_xyzw"] = list(quaternion)
        self.on_send()

    def abort(self):
        self.calls.append("stop")
        self.owned()

    def idle(self):
        self.calls.append("idle")
        self.controller = "idle"


def setup_config():
    cfg = exp.read_json(exp.ROOT / "configs/real_training/airbot_exploration.json")
    # Fake gripper cases must not inherit the installed hardware's NULL type.
    cfg.pop("expected_eef_type", None)
    cfg.pop("measured_joint_speed_stop_rad_s", None)
    cfg.pop("orientation_error_limit_rad", None)
    cfg.update(
        calibration_confirmed=True,
        calibration_id="test-only",
        robot_sn="fake",
        sdk_frame_and_tool_note="fake",
        verified_compression_allowance_m=0.02,
        joint_speed_limit_rad_s=0.2,
        lateral_tracking_policy="stop",
        tracking_error_policy="stop",
        joint_min_rad=[-2.0] * 6,
        joint_max_rad=[2.0] * 6,
        joint_current_limits=[1.0] * 6,
        eef_current_limit=1.0,
        max_force_n=10.0,
        max_initial_force_n=3.0,
        max_torque_nm=1.0,
    )
    pose = {
        "sdk_end_position_m": [0.25, 0.0, 0.2],
        "sdk_end_orientation_xyzw": [0, 0, 0, 1],
        "joint_position_rad": [0.1] * 6,
    }
    record = {
        "kind": "airbot_manual_initial_pose",
        "schema_version": 1,
        "sdk_version": "5.2.2",
        "metadata": {"firmware_cached": {"arm_sn": "fake"}},
        "initial_pose": pose,
    }
    return cfg, record


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cfg, self.record = setup_config()
        self.robot = FakeRobot(self.clock, self.record["initial_pose"])
        self.sensor = Sensor(self.clock)
        self.log = io.StringIO()
        self.output = io.StringIO()
        for context in (
            patch.object(exp.time, "perf_counter", self.clock.now),
            patch.object(exp.time, "sleep", self.clock.sleep),
            redirect_stdout(self.output),
            redirect_stderr(self.output),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def run_motion(self, scale=1, finish=lambda check: check()):
        exp.execute(
            self.robot,
            self.sensor,
            self.cfg,
            self.record,
            self.log,
            time_scale=scale,
            finish=finish,
        )

    def events(self):
        return [json.loads(line) for line in self.log.getvalue().splitlines()]

    def configure_no_ft(self):
        self.cfg.update(
            mode="contact-no-ft",
            no_force_contact_confirmed=True,
            expected_eef_type="NULL",
            sponge_thickness_m=0.028,
            position_error_limit_m=0.0005,
            contact_geometry_uncertainty_m=0.0002,
            compression_reserve_m=0.00025,
        )
        self.record["metadata"]["firmware_cached"]["eef_type"] = "NULL"
        for key in (
            "sensor_port",
            "sensor_bias_si",
            "max_force_n",
            "max_torque_nm",
            "max_initial_force_n",
            "eef_current_limit",
        ):
            self.cfg.pop(key, None)

    def configure_air(self):
        self.cfg.update(
            mode="air",
            air_path_clear_confirmed=True,
            air_start_clearance_m=0.050,
            expected_eef_type="NULL",
        )
        self.record["metadata"].update(label="air_motion_start")
        self.record["metadata"]["firmware_cached"]["eef_type"] = "NULL"
        for key in (
            "initial_gap_m",
            "verified_compression_allowance_m",
            "sensor_port",
            "sensor_bias_si",
            "max_force_n",
            "max_torque_nm",
            "max_initial_force_n",
            "eef_current_limit",
        ):
            self.cfg.pop(key, None)

    def test_air_keeps_full_trajectory_without_force_or_compression(self):
        self.configure_air()
        self.sensor = None
        self.run_motion()
        samples = [
            e for e in self.events() if e["event"] == "sample" and e["phase"] == "exploration"
        ]
        self.assertEqual(len(samples), 400)
        for i, event in enumerate(samples, 1):
            expected = exp.target_position(self.record["initial_pose"], self.cfg, i / 100)
            self.assertEqual(event["target_position_m"], expected)
            self.assertEqual(event["target_quaternion_xyzw"], [0, 0, 0, 1])
            self.assertIsNone(event["force"])
            self.assertIsNone(event["pre_send_force"])
            self.assertIsNone(event["estimated_compression_upper_bound_m"])
            self.assertGreaterEqual(event["estimated_air_clearance_m"], 0.029999)
        self.assertEqual(self.robot.targets[-1], self.record["initial_pose"]["sdk_end_position_m"])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "idle"])
        self.assertFalse(self.events()[-2]["contact_expected"])
        self.assertFalse(self.events()[-2]["force_monitoring"])
        self.assertFalse(self.events()[-2]["encoder_ready"])
        self.assertTrue(self.events()[-2]["nominal_protocol_match"])

    def test_air_rejects_unconfirmed_or_insufficient_clearance_before_motion(self):
        self.configure_air()
        valid = copy.deepcopy(self.cfg)
        for key, value in (
            ("air_path_clear_confirmed", False),
            ("air_start_clearance_m", None),
            ("air_start_clearance_m", math.nan),
            ("air_start_clearance_m", 0.0499),
            ("air_start_clearance_m", 0.001),
            ("initial_gap_m", 0.001),
        ):
            self.cfg = copy.deepcopy(valid)
            self.cfg[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.run_motion()
            self.assertEqual(self.robot.calls, [])

    def test_air_refuses_contact_start_record(self):
        self.configure_air()
        self.record["metadata"]["label"] = "contact_motion_start"
        with self.assertRaisesRegex(ValueError, "NEW elevated pose"):
            self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_air_clearance_uses_normal_and_running_boundary(self):
        self.configure_air()
        self.cfg["table_normal_sdk"] = [1, 0, 0]
        state = copy.deepcopy(self.robot.state)
        state["sdk_end_position_m"][0] -= 0.020
        self.assertAlmostEqual(
            exp.check_air_clearance(state, self.record["initial_pose"], self.cfg), 0.030
        )
        state["sdk_end_position_m"][0] -= 0.011
        with self.assertRaisesRegex(exp.StateError, "air clearance"):
            exp.check_air_clearance(state, self.record["initial_pose"], self.cfg)

    def test_air_config_needs_no_contact_fields_and_cannot_be_used_in_contact_mode(self):
        self.configure_air()
        with tempfile.TemporaryDirectory() as folder:
            config, record = Path(folder) / "cfg.json", Path(folder) / "pose.json"
            self.cfg["initial_pose_file"] = str(record)
            record.write_text(json.dumps(self.record))
            config.write_text(json.dumps(self.cfg))
            loaded, _ = exp.load_setup(config, "air")
            self.assertNotIn("initial_gap_m", loaded)
            self.assertNotIn("sensor_port", loaded)
            self.assertEqual(exp.tracking_limit(loaded), exp.TRACKING_LIMIT_M)
            for mode in ("force-guarded", "contact-no-ft"):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mode"):
                    exp.load_setup(config, mode)

    def test_air_stops_on_lost_lease_without_retract(self):
        self.configure_air()

        def lose_control():
            if len(self.robot.targets) == 5:
                self.robot.lease = False

        self.robot.on_send = lose_control
        with self.assertRaises(exp.StateError):
            self.run_motion()
        self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])
        self.assertEqual(len(self.robot.targets), 5)

    def test_air_preview_does_not_connect_or_imply_contact(self):
        with patch.object(exp, "open_client") as connect:
            self.assertEqual(exp.main(["preview", "--mode", "air"]), 0)
            connect.assert_not_called()
        result = json.loads(self.output.getvalue())
        self.assertFalse(result["contact_expected"])
        self.assertIsNone(result["software_tare_duration_s"])
        self.assertIsNone(result["initial_gap_m"])
        self.assertEqual(result["minimum_air_start_clearance_m"], 0.05)
        self.assertEqual(result["table_frame_keyframes_m"]["3"], [0, 0.05, -0.01])

    def test_contact_no_ft_keeps_full_trajectory_and_never_reads_force(self):
        self.configure_no_ft()
        with patch.object(self.sensor, "latest", side_effect=AssertionError("Must not read force")):
            self.run_motion()
        samples = [
            e for e in self.events() if e["event"] == "sample" and e["phase"] == "exploration"
        ]
        self.assertEqual(len(samples), 400)
        start = self.record["initial_pose"]["sdk_end_position_m"]
        for i, event in enumerate(samples, 1):
            self.assertEqual(
                event["target_position_m"],
                exp.target_position(self.record["initial_pose"], self.cfg, i / 100),
            )
            self.assertIsNone(event["force"])
            self.assertIsNone(event["pre_send_force"])
        self.assertAlmostEqual(samples[199]["target_position_m"][2], start[2] - 0.010)
        self.assertAlmostEqual(samples[299]["target_position_m"][1], start[1] + 0.050)
        self.assertEqual(self.robot.targets[-1], start)
        self.assertFalse(self.events()[-2]["force_monitoring"])
        self.assertFalse(self.events()[-2]["encoder_ready"])
        self.assertTrue(self.events()[-2]["nominal_protocol_match"])

    def test_force_guarded_missing_sensor_never_bypasses(self):
        self.sensor = None
        with self.assertRaisesRegex(exp.StateError, "requires a live force sensor"):
            self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_contact_no_ft_invalid_budgets_rejected_before_motion(self):
        self.configure_no_ft()
        valid = copy.deepcopy(self.cfg)
        for key, bad in (
            ("no_force_contact_confirmed", False),
            ("position_error_limit_m", 0.006),
            ("contact_geometry_uncertainty_m", None),
            ("compression_reserve_m", 0),
            ("contact_geometry_uncertainty_m", 0.011),
            ("initial_gap_m", 0),
            ("sponge_thickness_m", 0.019),
            ("position_error_limit_m", math.nan),
        ):
            self.cfg = copy.deepcopy(valid)
            self.cfg[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.run_motion()
            self.assertEqual(self.robot.calls, [])

    def test_contact_compression_estimate_uses_table_normal_and_reserve(self):
        self.configure_no_ft()
        self.cfg["table_normal_sdk"] = [1, 0, 0]
        pose = self.record["initial_pose"]
        state = copy.deepcopy(self.robot.state)
        state["sdk_end_position_m"][0] -= 0.020
        self.assertAlmostEqual(exp.check_compression(state, pose, self.cfg), 0.0192)
        state["sdk_end_position_m"][0] -= 0.0006
        with self.assertRaisesRegex(exp.StateError, "compression"):
            exp.check_compression(state, pose, self.cfg)

    def test_contact_no_ft_still_stops_on_tracking_and_interrupt(self):
        self.configure_no_ft()
        for failure in ("tracking", "interrupt", "lease", "overrun"):
            self.robot = FakeRobot(self.clock, self.record["initial_pose"])

            def fail():
                if len(self.robot.targets) != 5:
                    return
                if failure == "tracking":
                    self.robot.state["sdk_end_position_m"][0] += 0.0006
                elif failure == "lease":
                    self.robot.lease = False
                elif failure == "overrun":
                    self.clock.sleep(0.02)
                else:
                    raise KeyboardInterrupt()

            self.robot.on_send = fail
            with (
                self.subTest(failure=failure),
                self.assertRaises((exp.StateError, KeyboardInterrupt)),
            ):
                self.run_motion()
            self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])
            self.assertEqual(len(self.robot.targets), 5)
            self.assertFalse(self.events()[-1]["force_monitoring"])

    def test_contact_configuration_requires_explicit_matching_mode(self):
        self.configure_no_ft()
        with tempfile.TemporaryDirectory() as folder:
            path, record = Path(folder) / "cfg.json", Path(folder) / "pose.json"
            self.cfg["initial_pose_file"] = str(record)
            record.write_text(json.dumps(self.record))
            path.write_text(json.dumps(self.cfg))
            with self.assertRaisesRegex(ValueError, "mode"):
                exp.load_setup(path)
            loaded, _ = exp.load_setup(path, "contact-no-ft")
            self.assertNotIn("sensor_port", loaded)
            self.assertNotIn("max_force_n", loaded)
            self.assertNotIn("eef_current_limit", loaded)

    def test_contact_cli_uses_distinct_confirmation_and_no_sensor_import(self):
        self.configure_no_ft()
        self.check_cli_without_sensor("contact-no-ft", "CONTACT-NO-FT")

    def test_air_cli_uses_distinct_confirmation_and_no_sensor_import(self):
        self.configure_air()
        self.check_cli_without_sensor("air", "AIR-MOTION")

    def check_cli_without_sensor(self, mode, token):
        from unittest.mock import Mock

        real_execute = exp.execute
        client = Mock()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(exp, "load_setup", return_value=(self.cfg, self.record)),
            patch.object(exp, "open_client", return_value=client) as connect,
            patch.object(exp, "Robot", return_value=self.robot),
            patch.object(exp, "asdict", return_value={"arm_sn": "fake", "eef_type": "NULL"}),
            patch.object(exp.sys.stdin, "isatty", return_value=True),
            patch.dict(
                "sys.modules", {"scripts.force_sensor.kwr75_reader": None, "serial": None}
            ),
            patch.object(
                exp,
                "execute",
                side_effect=lambda *a: real_execute(*a, finish=lambda check: check()),
            ),
        ):
            path = Path(folder) / "motion.jsonl"
            argv = ["run", "--mode", mode, "--execute", "--output", str(path)]
            with patch("builtins.input", return_value="EXPLORE"):
                self.assertEqual(exp.main(argv), 0)
            connect.assert_not_called()
            self.assertFalse(path.exists())
            with patch("builtins.input", return_value=token):
                self.assertEqual(exp.main(argv), 0)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertFalse(events[0]["force_monitoring"])
            self.assertEqual(events[0]["contact_expected"], mode != "air")
            self.assertIsNone(events[0]["ft_frame"])
            self.assertEqual(events[-1]["event"], "session_complete")
            client.close.assert_called_once()

    @unittest.skipUnless(
        importlib.util.find_spec("arm_sdk"), "Requires installed SDK types, no hardware"
    )
    def test_null_gripper_uses_six_axis_servo_request_without_eef_fields(self):
        from unittest.mock import Mock

        from arm_sdk import Controller

        self.configure_no_ft()
        client = Mock()
        client.get_firmware_info.return_value = SimpleNamespace(arm_sn="fake", eef_type="NULL")
        client.has_control.return_value = True
        client._current_controller = Controller.servo_control
        client._stub.move_end_pose.return_value = SimpleNamespace(message="OK")
        robot = exp.Robot(client, self.cfg)
        with (
            patch("scripts.robot_control.adapter.health"),
            patch("scripts.robot_control.adapter.wait_controller"),
        ):
            robot.enter()
        robot.send([0.25, 0, 0.18], [0, 0, 0, 1])
        request = client._stub.move_end_pose.call_args.args[0]
        self.assertEqual(request.arm_dof, 6)
        self.assertEqual(list(request.vel), [0.2] * 6)
        self.assertEqual(list(request.eff), self.cfg["joint_current_limits"])
        self.assertEqual(list(request.target.position), [0.25, 0, 0.18])
        self.assertEqual(request.timeout_ms, 20)
        self.assertFalse(request.blocking)
        self.assertFalse(
            {"eef_pos", "eef_vel", "eef_eff"} & {field.name for field, _ in request.ListFields()}
        )
        self.assertEqual(client._stub.move_end_pose.call_args.kwargs["timeout"], 0.05)
        client.get_eef_joint_state.assert_not_called()
        client.move_eef.assert_not_called()
        client.move_end_pose.assert_not_called()
        client._stub.move_end_pose.return_value.message = "IK failed"
        with self.assertRaisesRegex(exp.StateError, "rejected"):
            robot.send([0.25, 0, 0.18], [0, 0, 0, 1])
        client._current_controller = Controller.direct_control
        with self.assertRaisesRegex(exp.StateError, "require servo"):
            robot.send([0.25, 0, 0.18], [0, 0, 0, 1])
        client.get_firmware_info.return_value.eef_type = "G2"
        with self.assertRaisesRegex(exp.StateError, "type differs"):
            exp.Robot(client, self.cfg)

    def test_exact_400_samples_and_separate_retraction(self):
        self.run_motion()
        samples = [
            e for e in self.events() if e["event"] == "sample" and e["phase"] == "exploration"
        ]
        self.assertEqual(len(samples), 400)
        for i, event in enumerate(samples, 1):
            self.assertEqual(event["protocol_time_s"], i / 100)
            self.assertEqual(
                event["target_position_m"],
                exp.target_position(self.record["initial_pose"], self.cfg, i / 100),
            )
            self.assertAlmostEqual(event["due_perf_s"] - samples[0]["due_perf_s"], (i - 1) / 100)
            self.assertEqual(event["target_quaternion_xyzw"], [0, 0, 0, 1])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "idle"])
        self.assertEqual(len(self.robot.targets), 601)
        for previous, target in zip(self.robot.targets[400:], self.robot.targets[401:]):
            self.assertAlmostEqual(target[2] - previous[2], 0.00005)
        self.assertEqual(self.events()[-2]["press_speed_m_s"], 0.005)
        self.assertEqual(self.events()[-2]["press_duration_s"], 2.0)
        self.assertEqual(self.robot.targets[-1], self.record["initial_pose"]["sdk_end_position_m"])
        self.assertEqual(self.events()[-1]["event"], "session_complete")
        self.assertFalse(self.events()[-2]["encoder_ready"])

    def test_protocol_matches_original_vector_formula_all_ticks(self):
        for i in range(401):
            t = i / 100
            expected = [0, 0.05 * (max(t - 2, 0) if t <= 3 else 4 - t), -0.005 * min(t, 2)]
            self.assertEqual(exp.exploration_offset(t), expected)
        for t in (-1, 4.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                exp.exploration_offset(t)

    def test_rotated_table_axes_not_tool_axes(self):
        self.cfg["table_normal_sdk"] = [1, 0, 0]
        self.cfg["slide_direction_sdk"] = [0, 0, 1]
        self.assertEqual(
            exp.target_position(self.record["initial_pose"], self.cfg, 3), [0.24, 0, 0.25]
        )

    def test_initial_mismatch_sends_no_command(self):
        for key, values in (
            ("sdk_end_position_m", [0.25, 0.0, 0.205]),
            ("joint_position_rad", [0.15] * 6),
            ("sdk_end_orientation_xyzw", [0, 0, 1, 0]),
        ):
            with self.subTest(key=key):
                self.robot = FakeRobot(self.clock, self.record["initial_pose"])
                self.robot.state[key] = values
                with self.assertRaises(exp.StateError):
                    self.run_motion()
                self.assertEqual(self.robot.calls, [])
                self.assertEqual(self.robot.targets, [])

    def test_other_active_controller_not_taken_over(self):
        self.robot.controller = "gravity_comp"
        with self.assertRaises(exp.StateError):
            self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_nonstationary_start_rejected(self):
        self.robot.state["joint_velocity_rad_s"][0] = 0.051
        with self.assertRaises(ValueError):
            self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_bad_force_before_motion_never_acquires(self):
        for wrench, age in (
            ([11, 0, 0, 0, 0, 0], 0),
            ([0, 0, 0, 2, 0, 0], 0),
            ([math.nan] * 6, 0),
            ([0] * 6, 0.021),
            ([0] * 6, -0.01),
        ):
            self.sensor.wrench, self.sensor.age = wrench, age
            with self.subTest(wrench=wrench, age=age), self.assertRaises(exp.StateError):
                self.run_motion()
            self.assertEqual(self.robot.calls, [])

    def test_faults_stop_without_retract_or_idle(self):
        def force_fault():
            self.sensor.wrench[0] = 11

        def lag():
            self.robot.state["sdk_end_position_m"][2] += 0.006

        def joint_fault():
            self.robot.state["joint_position_rad"][0] = 3

        def overrun():
            self.clock.sleep(0.02)

        def lost_lease():
            self.robot.lease = False

        def interrupted():
            raise KeyboardInterrupt()

        for fault in (force_fault, lag, joint_fault, overrun, lost_lease, interrupted):
            self.robot = FakeRobot(self.clock, self.record["initial_pose"])
            self.sensor.wrench = [0] * 6
            self.robot.on_send = lambda: fault() if len(self.robot.targets) == 5 else None
            with (
                self.subTest(fault=fault.__name__),
                self.assertRaises((exp.StateError, KeyboardInterrupt)),
            ):
                self.run_motion()
            self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])
            self.assertEqual(len(self.robot.targets), 5)
            self.assertEqual(self.events()[-1]["event"], "aborted")

    def test_failed_servo_entry_attempts_stop(self):
        with patch.object(self.robot, "enter", side_effect=exp.StateError("failed")):
            with self.assertRaises(exp.StateError):
                self.run_motion()
        self.assertEqual(self.robot.calls, ["acquire", "stop"])

    def test_failed_acquisition_does_not_stop_another_owner(self):
        with patch.object(self.robot, "acquire", side_effect=exp.StateError("occupied")):
            with self.assertRaises(exp.StateError):
                self.run_motion()
        self.assertEqual(self.robot.calls, [])
        self.assertEqual(self.robot.targets, [])

    def test_start_is_rechecked_after_servo_switch(self):
        enter = self.robot.enter

        def drift():
            enter()
            self.robot.state["sdk_end_position_m"][2] += 0.01

        with patch.object(self.robot, "enter", side_effect=drift):
            with self.assertRaisesRegex(exp.StateError, "Start mismatch"):
                self.run_motion()
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.robot.targets, [])

    def test_velocity_and_orientation_faults_are_guarded(self):
        self.robot.lease = True
        self.robot.controller = "servo"
        for key, bad in (
            ("joint_velocity_rad_s", [0.21] * 6),
            ("sdk_end_orientation_xyzw", [0, 0, 1, 0]),
        ):
            state = copy.deepcopy(self.robot.state)
            self.robot.state[key] = bad
            with self.subTest(key=key), self.assertRaises(exp.StateError):
                exp.observe(
                    self.robot,
                    self.sensor,
                    self.cfg,
                    self.record["initial_pose"],
                    self.record["initial_pose"]["sdk_end_position_m"],
                )
            self.robot.state = state

    def test_record_only_slide_lag_completes_original_protocol(self):
        self.cfg["lateral_tracking_policy"] = "record-only"
        start_y = self.record["initial_pose"]["sdk_end_position_m"][1]

        def lag():
            position = self.robot.state["sdk_end_position_m"]
            position[1] = max(start_y, position[1] - 0.010)

        self.robot.on_send = lag
        self.run_motion()
        samples = [
            e for e in self.events() if e["event"] == "sample" and e["phase"] == "exploration"
        ]
        self.assertEqual(len(samples), 400)
        for sample in samples:
            self.assertEqual(
                sample["target_position_m"],
                exp.target_position(
                    self.record["initial_pose"], self.cfg, sample["protocol_time_s"]
                ),
            )
            self.assertEqual(sample["slide_error_record_only"], sample["protocol_time_s"] > 2)
        summary = next(e for e in self.events() if e["event"] == "motion_complete")
        self.assertAlmostEqual(summary["max_abs_slide_error_m"], 0.01)
        self.assertGreater(summary["slide_error_over_5mm_samples"], 0)
        self.assertFalse(summary["position_rms_within_simulation_1mm"])
        self.assertFalse(summary["encoder_ready"])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "idle"])

    def test_slide_policy_keeps_initial_press_and_retract_strict(self):
        self.cfg["lateral_tracking_policy"] = "record-only"
        self.robot.lease = True
        self.robot.controller = "servo"
        pose = self.record["initial_pose"]
        self.robot.state["sdk_end_position_m"][1] += 0.02
        for kwargs in ({}, {"sliding": True, "initial": True}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(exp.StateError, "Position tracking"),
            ):
                exp.observe(
                    self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"], **kwargs
                )

    def test_unfinished_slide_return_stops_before_retraction(self):
        self.cfg["lateral_tracking_policy"] = "record-only"

        def unfinished_return():
            if len(self.robot.targets) == 401:
                self.robot.state["sdk_end_position_m"][1] += 0.020

        self.robot.on_send = unfinished_return
        with self.assertRaisesRegex(exp.ObservationError, "Position tracking"):
            self.run_motion()
        self.assertEqual(len(self.robot.targets), 401)
        self.assertEqual(sum(e["event"] == "sample" for e in self.events()), 400)
        self.assertEqual(
            self.events()[-1]["context"], {"phase": "retract", "index": 1, "check": "pre_send"}
        )
        self.assertFalse(self.events()[-1]["fault_observation"]["slide_error_record_only"])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])

    def test_sliding_keeps_all_other_motion_guards(self):
        self.cfg["lateral_tracking_policy"] = "record-only"
        pose = self.record["initial_pose"]
        for failure in (
            "normal",
            "cross",
            "position_nonfinite",
            "speed",
            "orientation",
            "force",
            "torque",
            "stale",
        ):
            self.robot = FakeRobot(self.clock, pose)
            self.robot.lease = True
            self.robot.controller = "servo"
            self.robot.state["sdk_end_position_m"][1] += 0.02
            self.sensor = Sensor(self.clock)
            if failure == "normal":
                self.robot.state["sdk_end_position_m"][2] -= 0.006
            elif failure == "cross":
                self.robot.state["sdk_end_position_m"][0] += 0.006
            elif failure == "position_nonfinite":
                self.robot.state["sdk_end_position_m"][1] = math.nan
            elif failure == "speed":
                self.robot.state["joint_velocity_rad_s"][4] = 0.21
            elif failure == "orientation":
                self.robot.state["sdk_end_orientation_xyzw"] = [0, 0, 1, 0]
            elif failure == "force":
                self.sensor.wrench[0] = 11
            elif failure == "torque":
                self.sensor.wrench[3] = 2
            else:
                self.sensor.age = 0.021
            with self.subTest(failure=failure), self.assertRaises(exp.ObservationError):
                exp.observe(
                    self.robot,
                    self.sensor,
                    self.cfg,
                    pose,
                    pose["sdk_end_position_m"],
                    sliding=True,
                )

    def test_tracking_projection_uses_table_axes(self):
        root = math.sqrt(0.5)
        self.cfg.update(table_normal_sdk=[root, 0, root], slide_direction_sdk=[root, 0, -root])
        target = self.record["initial_pose"]["sdk_end_position_m"]
        actual = [target[0] + 0.022 * root, target[1] + 0.003, target[2] - 0.018 * root]
        errors = exp.tracking_errors(actual, target, self.cfg)
        self.assertAlmostEqual(errors["normal_error_m"], 0.002)
        self.assertAlmostEqual(errors["slide_error_m"], 0.020)
        self.assertAlmostEqual(errors["cross_slide_error_m"], 0.003)

    def test_record_only_rejected_without_live_force_mode(self):
        for mode in ("air", "contact-no-ft"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exp.lateral_tracking_policy(
                    {"mode": mode, "lateral_tracking_policy": "record-only"}
                )
        with self.assertRaises(ValueError):
            exp.lateral_tracking_policy({"lateral_tracking_policy": "typo"})

    def test_fault_log_contains_triggering_state_and_axis(self):
        def overspeed():
            if len(self.robot.targets) == 5:
                self.robot.state["joint_velocity_rad_s"][4] = -0.215

        self.robot.on_send = overspeed
        with self.assertRaisesRegex(exp.ObservationError, "J5=-0.215"):
            self.run_motion()
        aborted = self.events()[-1]
        self.assertEqual(
            aborted["context"], {"phase": "exploration", "index": 4, "check": "post_send"}
        )
        fault = aborted["fault_observation"]
        self.assertEqual(fault["state"]["joint_velocity_rad_s"][4], -0.215)
        self.assertEqual(fault["target_position_m"], self.robot.targets[-1])
        self.assertIsNone(fault["force"])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])

    def test_measured_speed_stop_is_independent_and_checks_both_signs(self):
        self.cfg.update(joint_speed_limit_rad_s=0.4, measured_joint_speed_stop_rad_s=1.2)
        self.robot.lease = True
        self.robot.controller = "servo"
        pose = self.record["initial_pose"]
        for velocity in (0.402930409, 1.2, -1.2):
            self.robot.state["joint_velocity_rad_s"][5] = velocity
            exp.observe(self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"])
        for velocity in (1.201, -1.201):
            self.robot.state["joint_velocity_rad_s"][5] = velocity
            with (
                self.subTest(velocity=velocity),
                self.assertRaisesRegex(exp.StateError, "limit 1.2 rad/s: J6="),
            ):
                exp.observe(self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"])

    def test_measured_speed_configuration_validation(self):
        self.assertEqual(exp.measured_joint_speed_limit(self.cfg), 0.2)
        for value in (None, True, 0, -0.1, math.nan, math.inf, 0.1, 1.21):
            cfg = dict(self.cfg, measured_joint_speed_stop_rad_s=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                exp.measured_joint_speed_limit(cfg)
        for mode in ("air", "contact-no-ft"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exp.measured_joint_speed_limit(
                    dict(self.cfg, mode=mode, measured_joint_speed_stop_rad_s=1.2)
                )
        with self.assertRaises(ValueError):
            exp.measured_joint_speed_limit(
                dict(self.cfg, joint_speed_limit_rad_s=1.2, measured_joint_speed_stop_rad_s=1.2)
            )

    @unittest.skipUnless(
        importlib.util.find_spec("arm_sdk"), "Requires installed SDK types, no hardware"
    )
    def test_separate_stop_threshold_never_sent_as_command_speed(self):
        from unittest.mock import Mock

        from arm_sdk import Controller

        self.cfg.update(
            expected_eef_type="NULL",
            joint_speed_limit_rad_s=0.4,
            measured_joint_speed_stop_rad_s=1.2,
        )
        client = Mock()
        client.get_firmware_info.return_value = SimpleNamespace(arm_sn="fake", eef_type="NULL")
        client.has_control.return_value = True
        client._current_controller = Controller.servo_control
        client._stub.move_end_pose.return_value = SimpleNamespace(message="OK")
        robot = exp.Robot(client, self.cfg)
        with (
            patch("scripts.robot_control.adapter.health"),
            patch("scripts.robot_control.adapter.wait_controller"),
        ):
            robot.enter()
        robot.send([0.25, 0, 0.18], [0, 0, 0, 1])
        client.set_arm_speed.assert_called_once_with([0.4] * 6)
        self.assertEqual(list(client._stub.move_end_pose.call_args.args[0].vel), [0.4] * 6)

    def test_running_orientation_limit_preserves_start_tolerance(self):
        self.cfg["orientation_error_limit_rad"] = 0.15
        self.robot.lease = True
        self.robot.controller = "servo"
        pose = self.record["initial_pose"]
        for angle in (0.0202657634, 0.149):
            self.robot.state["sdk_end_orientation_xyzw"] = [
                0,
                0,
                math.sin(angle / 2),
                math.cos(angle / 2),
            ]
            result = exp.observe(
                self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"]
            )
            self.assertAlmostEqual(result["orientation_error_rad"], angle)
            self.assertEqual(result["orientation_error_limit_rad"], 0.15)
            with self.assertRaisesRegex(exp.StateError, "Start mismatch"):
                exp.check_start(self.robot.state, pose, self.cfg)
        angle = 0.151
        self.robot.state["sdk_end_orientation_xyzw"] = [
            0,
            0,
            math.sin(angle / 2),
            math.cos(angle / 2),
        ]
        with self.assertRaisesRegex(exp.ObservationError, "more than 0.15 rad") as caught:
            exp.observe(self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"])
        self.assertAlmostEqual(caught.exception.observation["orientation_error_rad"], angle)

    def test_orientation_limit_validation_and_legacy_default(self):
        self.assertEqual(exp.running_orientation_limit(self.cfg), 0.02)
        for value in (None, True, 0, -0.1, math.nan, math.inf, 0.151):
            with self.subTest(value=value), self.assertRaises(ValueError):
                exp.running_orientation_limit(dict(self.cfg, orientation_error_limit_rad=value))
        for mode in ("air", "contact-no-ft"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exp.running_orientation_limit(
                    dict(self.cfg, mode=mode, orientation_error_limit_rad=0.15)
                )

    def test_all_tracking_errors_recorded_through_retract_and_handoff(self):
        self.cfg["tracking_error_policy"] = "record-only"
        angle = 0.2

        def lag():
            self.robot.state["sdk_end_position_m"][0] += 0.006
            self.robot.state["sdk_end_position_m"][1] += 0.010
            self.robot.state["sdk_end_position_m"][2] += 0.006
            self.robot.state["sdk_end_orientation_xyzw"] = [
                0,
                0,
                math.sin(angle / 2),
                math.cos(angle / 2),
            ]

        self.robot.on_send = lag
        self.run_motion()
        samples = [e for e in self.events() if e["event"] == "sample"]
        self.assertEqual(len(samples), 600)
        self.assertTrue(all(e["tracking_error_record_only"] for e in samples))
        self.assertTrue(all(e["position_tracking_exceeds_limit"] for e in samples))
        self.assertTrue(all(e["orientation_tracking_exceeds_limit"] for e in samples))
        complete = next(e for e in self.events() if e["event"] == "motion_complete")
        self.assertFalse(complete["return_within_tracking_limits"])
        self.assertGreater(complete["return_position_error_m"], 0.01)
        self.assertFalse(complete["encoder_ready"])
        self.assertEqual(self.robot.calls, ["acquire", "servo", "idle"])

    def test_all_tracking_policy_preserves_start_and_absolute_guards(self):
        self.cfg["tracking_error_policy"] = "record-only"
        pose = self.record["initial_pose"]
        for failure in ("start", "position_nonfinite", "joint", "speed", "force", "torque", "stale"):
            self.robot = FakeRobot(self.clock, pose)
            self.robot.lease = True
            self.robot.controller = "servo"
            self.sensor = Sensor(self.clock)
            if failure == "start":
                self.robot.state["sdk_end_position_m"][1] += 0.01
                with self.assertRaises(exp.StateError):
                    exp.check_start(self.robot.state, pose, self.cfg)
                with self.assertRaises(exp.StateError):
                    exp.observe(
                        self.robot,
                        self.sensor,
                        self.cfg,
                        pose,
                        pose["sdk_end_position_m"],
                        initial=True,
                    )
                continue
            if failure == "position_nonfinite":
                self.robot.state["sdk_end_position_m"][1] = math.nan
            elif failure == "joint":
                self.robot.state["joint_position_rad"][0] = 3
            elif failure == "speed":
                self.robot.state["joint_velocity_rad_s"][5] = 0.21
            elif failure == "force":
                self.sensor.wrench[0] = 11
            elif failure == "torque":
                self.sensor.wrench[3] = 2
            else:
                self.sensor.age = 0.021
            with self.subTest(failure=failure), self.assertRaises(exp.StateError):
                exp.observe(self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"])

    def test_all_tracking_policy_is_explicit_and_force_guarded_only(self):
        self.assertEqual(exp.tracking_error_policy({}), "stop")
        for cfg in (
            {"tracking_error_policy": "typo"},
            {"mode": "air", "tracking_error_policy": "record-only"},
            {"mode": "contact-no-ft", "tracking_error_policy": "record-only"},
        ):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                exp.tracking_error_policy(cfg)

    def test_gravity_not_tared_from_force_log(self):
        self.sensor.wrench = [0, 0, 2.5, 0, 0, 0]
        self.cfg["sensor_bias_si"] = [0, 0, 0.5, 0, 0, 0]
        result = exp.read_force(self.sensor, self.cfg)
        self.assertEqual(result["bias_corrected_sensor_wrench_si"], [0, 0, 2, 0, 0, 0])
        self.assertEqual(result["raw_sensor_wrench_si"], self.sensor.wrench)

    def test_tare_records_all_axes_and_preserves_raw_and_safety_bias(self):
        baseline = [0.1, 0.2, 2.5, 0.01, 0.02, 0.03]
        self.sensor.wrench = baseline.copy()
        self.cfg["sensor_bias_si"] = [0, 0, 0.5, 0, 0, 0]
        original_cfg = copy.deepcopy(self.cfg)
        self.robot.on_send = lambda: setattr(self.sensor, "wrench", [0.1, 0.2, 3.5, 0.01, 0.02, 0.03])
        self.run_motion()
        tare = next(e for e in self.events() if e["event"] == "tare_complete")
        self.assertGreaterEqual(tare["end_perf_s"] - tare["start_perf_s"], 1)
        self.assertGreaterEqual(tare["distinct_samples"], exp.MIN_TARE_SAMPLES)
        for got, expected in zip(tare["tare_bias_si"], baseline):
            self.assertAlmostEqual(got, expected)
        initial = next(e for e in self.events() if e["event"] == "initial")["force"]
        for value in initial["tared_sensor_wrench_si"]:
            self.assertAlmostEqual(value, 0)
        for row in (e for e in self.events() if e["event"] == "sample"):
            force = row["force"]
            self.assertEqual(force["raw_sensor_wrench_si"][2], 3.5)
            self.assertEqual(force["bias_corrected_sensor_wrench_si"][2], 3.0)
            self.assertAlmostEqual(force["tared_sensor_wrench_si"][2], 1.0)
        self.assertTrue(self.events()[-2]["software_tare_applied"])
        self.assertEqual(self.cfg, original_cfg)

    def test_tare_deduplicates_sensor_frames(self):
        def latest(net=False):
            stamp = min(self.clock.now(), math.floor(self.clock.now() / 0.018) * 0.018)
            return stamp, [0, 0, stamp - 10, 0, 0, 0]
        with patch.object(self.sensor, "latest", side_effect=latest):
            bias = exp.collect_tare(self.robot, self.sensor, self.cfg,
                                    self.record["initial_pose"], self.log)
        event = self.events()[-1]
        stamps = [f["sensor_receive_perf_s"] for f in event["samples"]]
        self.assertEqual(len(stamps), len(set(stamps)))
        self.assertLess(len(stamps), 70)
        self.assertAlmostEqual(bias[2], sum(t - 10 for t in stamps) / len(stamps))
        self.assertEqual(self.robot.calls, [])

    def test_tare_rejects_insufficient_samples_and_unverified_gap(self):
        self.cfg["initial_gap_m"] = 0
        with self.assertRaisesRegex(ValueError, "non-contact"):
            self.run_motion()
        self.assertEqual(self.robot.calls, [])
        self.cfg["initial_gap_m"] = 0.001
        original_read = self.robot.read
        def slow_read(controller):
            self.clock.sleep(0.06)
            return original_read(controller)
        with patch.object(self.robot, "read", side_effect=slow_read):
            with self.assertRaisesRegex(exp.StateError, "Insufficient fresh"):
                self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_tare_failure_never_starts_motion(self):
        for fault in ("stale", "nonfinite", "force", "torque", "moving", "interrupt"):
            self.configure_tare_fault(fault)
            with self.subTest(fault=fault), self.assertRaises((exp.StateError, ValueError, KeyboardInterrupt)):
                self.run_motion()
            self.assertEqual(self.robot.calls, [])
            self.assertEqual(self.robot.targets, [])

    def configure_tare_fault(self, fault):
        self.clock.t = 10
        self.log = io.StringIO()
        self.robot = FakeRobot(self.clock, self.record["initial_pose"])
        self.sensor = Sensor(self.clock)
        original_read = self.robot.read
        def read(controller):
            if self.clock.now() > 10.8:
                if fault == "stale":
                    self.sensor.age = 0.021
                elif fault == "nonfinite":
                    self.sensor.wrench[0] = math.nan
                elif fault == "force":
                    self.sensor.wrench[0] = 4
                elif fault == "torque":
                    self.sensor.wrench[3] = 2
                elif fault == "moving":
                    self.robot.state["joint_velocity_rad_s"][0] = 0.051
                else:
                    raise KeyboardInterrupt()
            return original_read(controller)
        self.robot.read = read

    def test_tare_does_not_mask_running_force_or_torque_limit(self):
        for axis, baseline, overload in ((2, 2.5, 11), (3, 0.8, 1.5)):
            self.robot = FakeRobot(self.clock, self.record["initial_pose"])
            self.sensor.wrench = [0] * 6
            self.sensor.wrench[axis] = baseline
            self.robot.on_send = lambda: self.sensor.wrench.__setitem__(axis, overload)
            with self.subTest(axis=axis), self.assertRaisesRegex(exp.StateError, "Force/torque limit"):
                self.run_motion()
            self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])

    def test_no_ft_modes_do_not_tare(self):
        for configure in (self.configure_air, self.configure_no_ft):
            self.cfg, self.record = setup_config()
            configure()
            with patch.object(self.sensor, "latest", side_effect=AssertionError("No force read")):
                result = exp.collect_tare(self.robot, self.sensor, self.cfg,
                                          self.record["initial_pose"], self.log)
            self.assertIsNone(result)
        self.assertEqual(self.events(), [])

    def test_slow_run_is_not_labeled_simulation_equivalent(self):
        self.run_motion(scale=2)
        result = self.events()[-2]
        self.assertEqual(result["exploration_samples"], 800)
        self.assertFalse(result["nominal_protocol_match"])

    def test_invalid_time_scale(self):
        for value in (0.5, math.nan, math.inf, 11):
            with self.assertRaises(ValueError):
                self.run_motion(scale=value)
        self.assertEqual(self.robot.calls, [])

    def test_logging_failure_still_stops(self):
        original_write = exp.write_event
        def write(stream, event):
            if event["event"] == "initial":
                raise OSError("disk full")
            original_write(stream, event)
        with patch.object(exp, "write_event", side_effect=write):
            with self.assertRaises(OSError):
                self.run_motion()
        self.assertEqual(self.robot.calls, ["acquire", "servo", "stop"])

    def test_tare_logging_failure_never_acquires(self):
        with patch.object(exp, "write_event", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_motion()
        self.assertEqual(self.robot.calls, [])

    def test_eof_during_handoff_is_abort_not_success(self):
        with self.assertRaises(EOFError):
            self.run_motion(finish=lambda _: (_ for _ in ()).throw(EOFError()))
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.events()[-1]["event"], "aborted")

    def test_deadlines_are_rpc_not_just_request_fields(self):
        calls = []
        fake = SimpleNamespace(
            move_end_pose=lambda *a, **k: calls.append(k),
            SwitchController=lambda *a, **k: calls.append(k),
        )
        stub = exp.DeadlineStub(fake)
        stub.move_end_pose("request", metadata=("id",))
        stub.SwitchController("request")
        self.assertEqual(calls[0]["timeout"], 0.05)
        self.assertEqual(calls[1]["timeout"], 1.5)

    def test_preview_and_cli_gates_never_connect(self):
        with patch.object(exp, "open_client") as connect:
            self.assertEqual(exp.main([]), 0)
            self.assertEqual(json.loads(self.output.getvalue())["software_tare_duration_s"], 1.0)
            for argv in (["run"], ["run", "--execute"], ["preview", "--time-scale", "nan"]):
                with self.assertRaises(SystemExit):
                    exp.main(argv)
            connect.assert_not_called()

    def test_cancel_and_existing_log_do_not_connect(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "log.jsonl"
            argv = ["run", "--execute", "--output", str(target)]
            with (
                patch.object(exp, "open_client") as connect,
                patch.object(exp.sys.stdin, "isatty", return_value=True),
                patch.object(exp, "load_setup", return_value=(self.cfg, self.record)),
            ):
                # Stub only the sensor module import; no serial instance is created.
                fake_sensor_module = SimpleNamespace(Kwr75Reader=Sensor)
                with (
                    patch.dict(
                        "sys.modules",
                        {"scripts.force_sensor.kwr75_reader": fake_sensor_module},
                    ),
                    patch("builtins.input", return_value="no"),
                ):
                    self.assertEqual(exp.main(argv), 0)
                self.assertFalse(target.exists())
                target.write_text("keep")
                with self.assertRaises(SystemExit):
                    exp.main(argv)
                self.assertEqual(target.read_text(), "keep")
                connect.assert_not_called()

    def test_plot_preview_and_invalid_modes_never_connect_or_open_window(self):
        with patch.object(exp, "open_client") as connect, patch(
            "scripts.real_training.exploration_live_plot.ExplorationLivePlot.start"
        ) as start:
            self.assertEqual(exp.main(["preview", "--plot"]), 0)
            result = json.loads(self.output.getvalue())
            self.assertTrue(result["live_plot_requested"])
            self.assertFalse(result["live_plot_opened"])
            for argv in (
                ["preview", "--plot", "--mode", "air"],
                ["preview", "--plot", "--mode", "contact-no-ft"],
                ["preview", "--plot-hz", "nan"],
                ["preview", "--plot-hz", "31"],
                ["preview", "--plot-window", "0"],
            ):
                with self.assertRaises(SystemExit):
                    exp.main(argv)
            start.assert_not_called()
            connect.assert_not_called()

    def test_plot_startup_failure_precedes_hardware_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "log.jsonl"
            argv = ["run", "--execute", "--plot", "--output", str(target)]
            with (
                patch.object(exp, "open_client") as connect,
                patch.object(exp.sys.stdin, "isatty", return_value=True),
                patch.object(exp, "load_setup", return_value=(self.cfg, self.record)),
                patch("builtins.input", return_value="EXPLORE"),
                patch.dict("sys.modules", {
                    "scripts.force_sensor.kwr75_reader": SimpleNamespace(Kwr75Reader=Sensor),
                }),
                patch("scripts.real_training.exploration_live_plot.ExplorationLivePlot") as plot,
            ):
                plot.return_value.start.side_effect = RuntimeError("No desktop")
                self.assertEqual(exp.main(argv), 1)
                connect.assert_not_called()
                plot.return_value.close.assert_called_once()
                self.assertFalse(target.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("arm_sdk"), "Requires installed SDK types, no hardware"
    )
    def test_sdk_adapter_preserves_gripper_and_nonblocking_cartesian_api(self):
        from unittest.mock import Mock

        client = Mock()
        client.get_firmware_info.return_value = SimpleNamespace(arm_sn="fake", eef_type="G2")
        client.get_eef_joint_state.return_value = SimpleNamespace(eef_pos=0.03)
        client.has_control.return_value = True
        client.move_end_pose.return_value = True
        robot = exp.Robot(client, self.cfg)
        robot.send([0.25, 0, 0.2], [0, 0, 0, 1])
        args, kwargs = client.move_end_pose.call_args
        self.assertEqual(list(args[0].position), [0.25, 0, 0.2])
        self.assertEqual(list(args[0].orientation), [0, 0, 0, 1])
        self.assertEqual(args[1].eef_pos, 0.03)
        self.assertEqual(args[1].eff, self.cfg["joint_current_limits"])
        self.assertFalse(args[1].blocking)
        self.assertEqual(kwargs, {"timeout_ms": 20})
        client.get_eef_joint_state.return_value = SimpleNamespace(eef_pos=0.08)
        with self.assertRaisesRegex(exp.StateError, "outside SDK bounds"):
            exp.Robot(client, self.cfg)
        client.get_eef_joint_state.return_value = None
        with self.assertRaisesRegex(exp.StateError, "Cannot preserve"):
            exp.Robot(client, self.cfg)
        client.get_firmware_info.return_value.arm_sn = "other"
        with self.assertRaisesRegex(exp.StateError, "serial number"):
            exp.Robot(client, self.cfg)
        client.acquire_control.assert_not_called()
        client.switch_controller.assert_not_called()

    def test_configuration_validates_entire_path_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path, record_path = Path(folder) / "config.json", Path(folder) / "pose.json"
            self.cfg["initial_pose_file"] = str(record_path)
            record_path.write_text(json.dumps(self.record))
            config_path.write_text(json.dumps(self.cfg))
            cfg, record = exp.load_setup(config_path)
            self.assertEqual(record, self.record)
            self.assertEqual(len(cfg["initial_pose_sha256"]), 64)
            for key, value in (
                ("calibration_confirmed", False),
                ("robot_sn", "different"),
                ("initial_gap_m", 0.01),
                ("slide_direction_sdk", [0, 0, 1]),
                ("max_force_n", None),
                ("verified_compression_allowance_m", 0.008),
                ("sensor_bias_si", [0, 0, math.nan, 0, 0, 0]),
                ("joint_speed_limit_rad_s", 1),
            ):
                invalid = copy.deepcopy(self.cfg)
                invalid[key] = value
                config_path.write_text(json.dumps(invalid))
                with self.subTest(key=key), self.assertRaises((ValueError, exp.StateError)):
                    exp.load_setup(config_path)


    def test_exploration_has_no_xyz_bounds_and_ignores_legacy_fields(self):
        for legacy in (False, True):
            self.cfg, self.record = setup_config()
            self.assertNotIn("workspace_min_m", self.cfg)
            self.assertNotIn("workspace_max_m", self.cfg)
            if legacy:
                self.cfg.update(workspace_min_m=[0, 0, 0], workspace_max_m=[0.01] * 3)
            self.record["initial_pose"]["sdk_end_position_m"] = [0.8, -0.3, 0.9]
            self.robot = FakeRobot(self.clock, self.record["initial_pose"])
            self.log = io.StringIO()
            with tempfile.TemporaryDirectory() as folder:
                cfg_path, pose_path = Path(folder) / "config.json", Path(folder) / "pose.json"
                self.cfg["initial_pose_file"] = str(pose_path)
                pose_path.write_text(json.dumps(self.record))
                cfg_path.write_text(json.dumps(self.cfg))
                exp.load_setup(cfg_path)
            self.run_motion()
            self.assertEqual(len(self.robot.targets), 601)
            self.assertEqual(self.robot.targets[-1], [0.8, -0.3, 0.9])

    def test_record_only_feedback_outside_old_xyz_bounds_is_recorded(self):
        self.cfg.update(
            tracking_error_policy="record-only",
            workspace_min_m=[0.1, -0.1, 0.1], workspace_max_m=[0.4, 0.2, 0.4],
        )
        self.robot.lease = True
        self.robot.controller = "servo"
        self.robot.state["sdk_end_position_m"][1] = 0.3
        pose = self.record["initial_pose"]
        observation = exp.observe(self.robot, self.sensor, self.cfg, pose, pose["sdk_end_position_m"])
        self.assertTrue(observation["position_tracking_exceeds_limit"])
        self.assertEqual(observation["state"]["sdk_end_position_m"][1], 0.3)


if __name__ == "__main__":
    unittest.main()
