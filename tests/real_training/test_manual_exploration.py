"""Manual-start state machine and data-link tests; fake hardware only."""

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.real_training import airbot_exploration as legacy
from scripts.real_training import manual_exploration as manual
from scripts.real_training import manual_exploration_contract as contract
from scripts.shared.paths import ROOT
from tests.real_training.test_airbot_exploration import Clock, FakeRobot, Sensor, setup_config


class Robot(FakeRobot):
    def switch(self, mode):
        self.owned()
        self.calls.append(mode)
        self.controller = mode

    def send_joint(self, joints):
        self.owned()
        if self.controller != "servo":
            raise manual.StateError("Joint hold outside servo")
        self.calls.append("hold_joint")
        self.state["joint_position_rad"] = list(joints)


class Keyboard:
    def __init__(self, hook=lambda key: None):
        self.keys = []
        self.hook = hook

    def wait(self, key, monitor, prompt):
        self.keys.append(key)
        self.hook(key)
        monitor()


class Fixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cfg = manual.load_setup(ROOT / "configs/real_training/airbot_exploration_manual.json")
        self.pose = setup_config()[1]["initial_pose"]
        self.pose["joint_position_rad"] = [0] * 6
        self.clock = Clock()
        self.robot = Robot(self.clock, self.pose)
        self.sensor = Sensor(self.clock)
        self.keyboard = Keyboard()
        self.stream = io.StringIO()
        self.output = io.StringIO()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(manual.time, "perf_counter", self.clock.now))
        self.stack.enter_context(patch.object(manual.time, "sleep", self.clock.sleep))
        self.stack.enter_context(redirect_stdout(self.output))
        self.stack.enter_context(redirect_stderr(self.output))

    def run_motion(self, scale=1):
        manual.write_event(self.stream, {"event": "session_start", "mode": "manual-start",
                           "source_kind": "real", "acquisition_protocol": contract.PROTOCOL,
                           "force_monitoring": True, "force_limits_enforced": False,
                           "stationary_validated": False, "config": self.cfg})
        manual.execute(self.robot, self.sensor, self.cfg, self.stream, scale, self.keyboard)
        return self.events()

    def events(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def log_file(self):
        self.run_motion()
        path = self.folder / "exploration.jsonl"
        path.write_text(self.stream.getvalue())
        return path


class ManualTests(Fixture, unittest.TestCase):
    def test_full_flow_and_command_geometry_without_start_file(self):
        rows = self.run_motion()
        self.assertEqual(self.keyboard.keys, ["h", "s", "IDLE"])
        self.assertEqual(self.robot.calls[:3], ["acquire", "gravity_comp", "servo"])
        self.assertEqual(self.robot.calls[-1], "idle")
        samples = [r for r in rows if r["event"] == "sample"]
        self.assertEqual(len(samples), 600)
        for index, expected in ((199, [0.25, 0, 0.19]), (299, [0.25, 0.05, 0.19]),
                                (399, [0.25, 0, 0.19]), (599, [0.25, 0, 0.2])):
            for value, wanted in zip(samples[index]["target_position_m"], expected):
                self.assertAlmostEqual(value, wanted)
        self.assertEqual(rows[-1], {"event": "session_complete", "idle_confirmed": True})
        self.assertNotIn("initial_pose_file", self.cfg)
        self.assertEqual(list(self.folder.iterdir()), [])
        contract.validate(rows)

    def test_no_stationarity_or_load_magnitude_acceptance(self):
        self.robot.state["joint_velocity_rad_s"] = [0.08] * 6
        self.sensor.wrench = [100, -100, 60, 2, -2, 3]
        with patch("scripts.robot_control.airbot_initial_pose.validate_stationary",
                   side_effect=AssertionError("Stationarity must not be checked")), patch.object(
                       legacy, "check_start", side_effect=AssertionError("No recorded start check")):
            rows = self.run_motion()
        self.assertNotIn("stop", self.robot.calls)
        tare = next(r for r in rows if r["event"] == "tare_complete")
        self.assertEqual(tare["tare_bias_si"], self.sensor.wrench)
        initial = next(r for r in rows if r["event"] == "initial")
        self.assertEqual(initial["force"]["tared_sensor_wrench_si"], [0] * 6)
        contract.validate(rows)

    def test_hold_before_tare_and_return_does_not_release_early(self):
        def hook(key):
            if key == "s":
                self.assertIn("hold_joint", self.robot.calls)
                self.assertNotIn("tare_complete", [r["event"] for r in self.events()])
                self.assertEqual(len(self.robot.targets), 0)
            elif key == "IDLE":
                self.assertEqual(len(self.robot.targets), 600)
                self.assertEqual(self.robot.controller, "servo")
                self.assertNotIn("idle", self.robot.calls)
        self.keyboard.hook = hook
        self.run_motion()
        self.assertEqual(len(self.robot.targets), 601)

    def test_return_deviation_only_recorded(self):
        def drift():
            self.robot.state["sdk_end_position_m"][0] += 0.02
            self.robot.state["sdk_end_orientation_xyzw"] = [0, 0, math.sin(0.1), math.cos(0.1)]
        self.robot.on_send = drift
        rows = self.run_motion()
        done = next(r for r in rows if r["event"] == "motion_complete")
        self.assertFalse(done["return_within_tracking_limits"])
        self.assertGreater(done["return_position_error_m"], 0.005)
        self.assertNotIn("stop", self.robot.calls)
        contract.validate(rows)

    def test_sensor_invalid_missing_stale_future_nonfinite(self):
        for item in (None, (9, [0] * 6), (11, [0] * 6), (10, [float("nan")] * 6), (10, [0] * 5)):
            with self.subTest(item=item), patch.object(self.sensor, "latest", return_value=item):
                with self.assertRaises((ValueError, manual.StateError)):
                    manual.ForceReader(self.sensor, self.cfg).read()

    def test_sensor_conflicting_or_backwards_timestamps(self):
        reader = manual.ForceReader(self.sensor, self.cfg)
        reader.read()
        self.sensor.wrench = [1] * 6
        with self.assertRaisesRegex(manual.StateError, "Conflicting"):
            reader.read()
        self.sensor.wrench = [0] * 6
        self.sensor.age = 0.001
        with self.assertRaisesRegex(manual.StateError, "backwards"):
            reader.read()

    def test_tare_insufficient_distinct_samples_never_explores(self):
        self.sensor.latest = lambda net=False: (math.floor(self.clock.now() * 20) / 20, [0] * 6)
        # Exercise the coverage gate independently of the stricter live freshness gate.
        def hold():
            stamp, raw = self.sensor.latest()
            return {"force": {"sensor_receive_perf_s": stamp, "raw_sensor_wrench_si": raw}}
        with self.assertRaisesRegex(manual.StateError, "Insufficient"):
            manual.collect_tare(hold, self.stream)
        self.assertFalse(self.robot.targets)
        self.assertNotIn("tare_complete", [r["event"] for r in self.events()])

    def test_tare_failure_aborts_held_robot_without_motion(self):
        with patch.object(manual, "collect_tare", side_effect=manual.StateError("Insufficient baseline")):
            with self.assertRaises(manual.StateError):
                self.run_motion()
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.events()[-1]["context"]["phase"], "tare")
        self.assertFalse(self.robot.targets)

    def test_motion_faults_stop_without_return_or_idle(self):
        for fault in ("speed", "stale", "lease", "deadline", "rejected"):
            with self.subTest(fault=fault):
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock)
                self.stream = io.StringIO()
                def trigger():
                    if len(self.robot.targets) != 3:
                        return
                    if fault == "speed":
                        self.robot.state["joint_velocity_rad_s"] = [1.21] * 6
                    elif fault == "stale":
                        self.sensor.age = 1
                    elif fault == "lease":
                        self.robot.lease = False
                    elif fault == "deadline":
                        self.clock.sleep(0.1)
                    else:
                        raise manual.StateError("Rejected target")
                self.robot.on_send = trigger
                with self.assertRaises(manual.StateError):
                    self.run_motion()
                self.assertEqual(self.robot.calls[-1], "stop")
                self.assertNotIn("idle", self.robot.calls)
                self.assertEqual(len(self.robot.targets), 3)
                self.assertEqual(self.events()[-1]["event"], "aborted")

    def test_joint_data_validity_and_speed_still_block_initial_state(self):
        for field, value in (("joint_position_rad", [float("nan")] * 6),
                             ("joint_position_rad", [0] * 5),
                             ("sdk_end_position_m", [float("inf")] * 3),
                             ("joint_velocity_rad_s", [1.21] * 6)):
            state = copy.deepcopy(self.robot.state)
            state[field] = value
            with self.subTest(field=field), self.assertRaises(manual.StateError):
                manual.check_state(state, self.cfg)

    def test_drag_speed_above_old_stop_including_reference_capture_completes(self):
        original = self.robot.read
        def read(mode):
            self.robot.state["joint_velocity_rad_s"] = (
                [0, 0, 0.5582418, -1.2673993, 0.4322344, 0]
                if mode == "gravity_comp" else [0] * 6
            )
            return original(mode)
        with patch.object(self.robot, "read", side_effect=read):
            rows = self.run_motion()
        reference = next(row for row in rows if row["event"] == "reference_captured")
        self.assertEqual(reference["runtime_reference_pose"]["joint_velocity_rad_s"][3], -1.2673993)
        phases = {row["phase"]: row for row in rows if row["event"] == "phase_start"}
        self.assertFalse(phases["drag"]["measured_joint_speed_stop_enforced"])
        self.assertTrue(phases["hold"]["measured_joint_speed_stop_enforced"])
        self.assertEqual(rows[-1]["event"], "session_complete")
        self.assertEqual(len([row for row in rows if row["event"] == "sample"]), 600)
        self.assertNotIn("stop", self.robot.calls)

    def test_drag_speed_exemption_ends_at_servo_hold(self):
        def trigger(key):
            if key == "h":
                self.robot.state["joint_velocity_rad_s"] = [1.3] * 6
        self.keyboard = Keyboard(trigger)
        with self.assertRaisesRegex(manual.StateError, "Measured joint speed"):
            self.run_motion()
        self.assertEqual(self.events()[-1]["context"]["phase"], "hold")
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertFalse(self.robot.targets)

    def test_drag_still_rejects_invalid_speed_data_stale_force_and_lost_control(self):
        for fault in ("nan", "shape", "stale", "lease"):
            with self.subTest(fault=fault):
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock)
                self.stream = io.StringIO()
                def trigger(key):
                    if key != "h":
                        return
                    if fault == "nan":
                        self.robot.state["joint_velocity_rad_s"] = [float("nan")] * 6
                    elif fault == "shape":
                        self.robot.state["joint_velocity_rad_s"] = [0] * 5
                    elif fault == "stale":
                        self.sensor.age = 1
                    else:
                        self.robot.lease = False
                self.keyboard = Keyboard(trigger)
                with self.assertRaises(manual.StateError):
                    self.run_motion()
                if fault != "nan":
                    self.assertEqual(self.events()[-1]["context"]["phase"], "drag")
                self.assertEqual(self.robot.calls[-1], "stop")
                self.assertNotIn("servo", self.robot.calls)
                self.assertFalse(self.robot.targets)

    def test_j3_below_old_limit_completes_manual_but_legacy_still_rejects(self):
        self.robot.state["joint_position_rad"][2] = -0.08869306743144989
        old_cfg = legacy.read_json(ROOT / "configs/real_training/airbot_exploration.json")
        with self.assertRaisesRegex(manual.StateError, "Joint position outside"):
            legacy.check_bounds(self.robot.state, old_cfg)
        self.assertNotIn("joint_min_rad", self.cfg)
        self.assertNotIn("joint_max_rad", self.cfg)
        # Old user-supplied fields also must not re-enable the removed check.
        self.cfg.update(joint_min_rad=[0] * 6, joint_max_rad=[0] * 6)
        def drift():
            self.robot.state["joint_position_rad"][2] = -0.2
        self.robot.on_send = drift
        rows = self.run_motion()
        self.assertNotIn("stop", self.robot.calls)
        done = next(r for r in rows if r["event"] == "motion_complete")
        self.assertFalse(done["joint_position_limits_enforced"])
        self.assertEqual(rows[-1]["event"], "session_complete")
        contract.validate(rows)

    def test_switch_failure_and_eof_stop(self):
        for mode in ("gravity_comp", "servo"):
            self.robot = Robot(self.clock, self.pose)
            self.sensor = Sensor(self.clock)
            self.stream = io.StringIO()
            switch = self.robot.switch
            def fail(value):
                if value == mode:
                    raise manual.StateError("Mode switch failed")
                switch(value)
            self.robot.switch = fail
            with self.assertRaises(manual.StateError):
                self.run_motion()
            self.assertEqual(self.robot.calls[-1], "stop")
        self.robot = Robot(self.clock, self.pose)
        self.stream = io.StringIO()
        self.keyboard.hook = lambda key: (_ for _ in ()).throw(EOFError()) if key == "IDLE" else None
        with self.assertRaises(EOFError):
            self.run_motion()
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertNotIn("session_complete", [r["event"] for r in self.events()])

    def test_pre_send_overrun_stops(self):
        original = manual.observe
        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            if any(r.get("phase") == "exploration" for r in self.events()):
                self.clock.sleep(0.02)
            return result
        with patch.object(manual, "observe", side_effect=delayed):
            with self.assertRaisesRegex(manual.StateError, "before send"):
                self.run_motion()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")

    def test_logging_failure_stops_before_further_motion(self):
        write = manual.write_event
        def fail(stream, event):
            if event["event"] == "initial":
                raise OSError("disk full")
            write(stream, event)
        with patch.object(manual, "write_event", side_effect=fail):
            with self.assertRaises(OSError):
                self.run_motion()
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertFalse(self.robot.targets)

    def test_stop_failure_is_reported_without_recovery(self):
        self.keyboard.hook = lambda key: (_ for _ in ()).throw(KeyboardInterrupt())
        self.robot.abort = Mock(side_effect=manual.StateError("stop failed"))
        with self.assertRaises(KeyboardInterrupt):
            self.run_motion()
        self.assertIn("physical emergency stop NOW", self.output.getvalue())
        self.assertEqual(self.events()[-1]["stop_error"], "stop failed")
        self.assertNotIn("idle", self.robot.calls)

    def test_default_and_legacy_cli_route_without_connecting(self):
        with patch.object(manual, "open_client") as connect:
            self.assertEqual(legacy.main(["preview"]), 0)
            self.assertEqual(json.loads(self.output.getvalue())["mode"], "manual-start")
            self.assertFalse(json.loads(self.output.getvalue())["joint_position_limits_enforced"])
            connect.assert_not_called()
        with patch.object(manual, "main", return_value=0) as entry:
            self.assertEqual(legacy.main(["check"]), 0)
            self.assertEqual(entry.call_args.args[0].config, ROOT / "configs/real_training/airbot_exploration_manual.json")
        with patch.object(legacy, "load_setup", side_effect=ValueError("offline path probe")) as load:
            for mode, path in (("force-guarded", "real_training/airbot_exploration.json"),
                               ("air", "robot_control/airbot_air_motion.json"),
                               ("contact-no-ft", "real_training/airbot_contact_motion.json")):
                self.assertEqual(legacy.main(["check", "--mode", mode]), 1)
                self.assertEqual(load.call_args.args[0], ROOT / "configs" / path)

    def test_manual_cli_authorization_and_existing_output_never_connect(self):
        with patch.object(manual, "open_client") as connect:
            for argv in (["run"], ["run", "--execute"]):
                with self.assertRaises(SystemExit):
                    legacy.main(argv)
            path = self.folder / "existing.jsonl"
            path.write_text("keep")
            with patch.object(legacy.sys.stdin, "isatty", return_value=True):
                with self.assertRaises(SystemExit):
                    legacy.main(["run", "--execute", "--output", str(path)])
            self.assertEqual(path.read_text(), "keep")
            connect.assert_not_called()

    def test_manual_plot_failure_precedes_connection(self):
        from scripts.real_training.exploration_live_plot import ExplorationLivePlot

        args = SimpleNamespace(config=ROOT / "configs/real_training/airbot_exploration_manual.json",
                               action="run", plot=True, output=self.folder / "new.jsonl",
                               plot_window=10, plot_hz=10)
        with patch.object(manual, "open_client") as connect, patch.object(
                ExplorationLivePlot, "start", side_effect=RuntimeError("No desktop")), patch.object(
                ExplorationLivePlot, "close"):
            self.assertEqual(manual.main(args), 1)
            connect.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_read_only_check_has_no_start_match_or_sensor(self):
        args = SimpleNamespace(config=ROOT / "configs/real_training/airbot_exploration_manual.json",
                               action="check", host="localhost", port=50051)
        with patch.object(manual, "open_client", return_value=Mock()) as connect, patch.object(
                manual, "HardwareRobot", return_value=self.robot):
            self.assertEqual(manual.main(args), 0)
        result = json.loads(self.output.getvalue())
        self.assertNotIn("start_matches", result)
        self.assertFalse(result["force_checked"])
        self.assertFalse(result["joint_position_limits_enforced"])
        self.assertEqual(self.robot.calls, [])
        connect.return_value.close.assert_called_once()

    def test_keyboard_flushes_each_stage_and_ignores_wrong_keys(self):
        monitor = Mock()
        with patch.object(manual.sys, "stdin") as stdin, patch.object(manual.termios, "tcgetattr", return_value=[]), \
                patch.object(manual.termios, "tcsetattr") as restore, patch.object(manual.termios, "tcflush") as flush, \
                patch.object(manual.tty, "setcbreak"), patch.object(manual.select, "select", return_value=([9], [], [])), \
                patch.object(manual.os, "read", side_effect=[b"s", b"h", b"h", b"s", b"s", b"I", b"D", b"L", b"E", b"\n", b"I", b"D", b"L", b"E", b"\n"]):
            stdin.fileno.return_value = 9
            keys = manual.Keyboard()
            keys.wait("h", monitor, "hold")
            keys.wait("s", monitor, "start")
            keys.wait("IDLE", monitor, "end")
            self.assertEqual(flush.call_count, 3)
            self.assertEqual(restore.call_count, 3)


class ContractTests(Fixture, unittest.TestCase):
    def test_tared_plot_accepts_manual_log_and_never_overwrites(self):
        from scripts.real_training.exploration_live_plot import save_exploration_plot

        path = self.log_file()
        plot = save_exploration_plot(path)
        self.assertGreater(plot.stat().st_size, 10000)
        self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(FileExistsError):
            save_exploration_plot(path)

    def test_new_log_imports_raw_once_with_tare_provenance(self):
        from scripts.real_training.import_airbot import exploration_episode

        self.sensor.wrench = [0, 0, 25, 0, 0, 0]
        path = self.log_file()
        episode, report, header = exploration_episode(path)
        self.assertEqual(header["mode"], "manual-start")
        self.assertTrue(all(row[2] == 25 for row in episode["ft"]))
        self.assertEqual(report["completion"]["exploration_samples"], 400)

    def test_rejects_missing_duplicate_tare_corruption_and_incomplete_motion(self):
        rows = self.run_motion()
        mutations = {
            "missing baseline": lambda r: r.pop(next(i for i, e in enumerate(r) if e["event"] == "tare_complete")),
            "duplicate baseline": lambda r: r.insert(4, copy.deepcopy(next(e for e in r if e["event"] == "tare_complete"))),
            "missing frame": lambda r: r.pop(next(i for i, e in enumerate(r) if e["event"] == "sample")),
            "missing idle": lambda r: r.pop(),
            "aborted": lambda r: r.insert(-1, {"event": "aborted"}),
            "double subtraction": lambda r: next(e for e in r if e["event"] == "sample")["force"]["tared_sensor_wrench_si"].__setitem__(0, 1),
            "baseline changed": lambda r: next(e for e in r if e["event"] == "tare_complete")["tare_bias_si"].__setitem__(0, 1),
            "stale": lambda r: next(e for e in r if e["event"] == "sample")["force"].__setitem__("sensor_age_s", 1),
        }
        for name, change in mutations.items():
            modified = copy.deepcopy(rows)
            change(modified)
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.validate(modified)

    def test_slow_motion_is_not_training_protocol(self):
        rows = self.run_motion(2)
        self.assertEqual(sum(r["event"] == "sample" for r in rows), 1200)
        with self.assertRaisesRegex(ValueError, "400-frame"):
            contract.validate(rows)

    def test_new_binding_is_required_and_old_attempts_cannot_be_relinked(self):
        from scripts.real_training import airbot_programmed_demonstrations as program

        path = self.log_file()
        spec = json.loads((ROOT / "configs/real_training/airbot_programmed_manual_demonstrations.json").read_text())
        spec.update(exploration_log=str(path), exploration_setup_confirmed=True)
        config_path = self.folder / "program.json"
        config_path.write_text(json.dumps(spec))
        cfg, _, frozen = program.load_configuration(config_path)
        self.assertEqual(cfg["exploration_binding"]["sha256"], contract.digest(path))
        header, _ = contract.load_completed(path)
        for key, value in (("robot_sn", "other"), ("expected_eef_type", "G2"),
                           ("sensor_port", "other"), ("slide_direction_sdk", [1, 0, 0]),
                           ("sponge_id", "other")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "setup mismatch"):
                contract.verify_setup(header, {**cfg, key: value})
        session = self.folder / "new_session"
        session.mkdir()
        program.freeze_session(session, frozen)
        program.freeze_session(session, frozen)
        changed = copy.deepcopy(frozen)
        changed["exploration_binding"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "new program session"):
            program.freeze_session(session, changed)
        orphan = self.folder / "orphan"
        orphan.mkdir()
        (orphan / "attempt_old.jsonl").write_text("old")
        with self.assertRaisesRegex(ValueError, "retroactively"):
            program.freeze_session(orphan, frozen)
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "bound before collection"):
            program.accepted_records(session)

    def test_eight_new_demos_import_tared_and_reject_mixed_or_modified_sources(self):
        import h5py
        import numpy as np
        from scripts.real_training import airbot_programmed_demonstrations as program
        from scripts.real_training.config import load_config
        from scripts.real_training.data import _load_raw, write_raw_log
        from scripts.real_training.import_airbot import assemble
        from tests.real_training.test_airbot_programmed_demonstrations import (
            Clock as ProgramClock, Robot as ProgramRobot, Sensor as ProgramSensor, Terminal,
        )

        self.sensor.wrench = [1, 2, 25, 0.1, 0.2, 0.3]
        exploration = self.log_file()
        spec = json.loads((ROOT / "configs/real_training/airbot_programmed_manual_demonstrations.json").read_text())
        spec.update(exploration_log=str(exploration), exploration_setup_confirmed=True, setup_confirmed=True)
        spec["startup_approach"]["direct_path_confirmed"] = True
        config_path = self.folder / "program.json"
        config_path.write_text(json.dumps(spec))
        cfg, _, frozen = program.load_configuration(config_path)
        folder = self.folder / "new_demos"
        folder.mkdir()
        program.freeze_session(folder, frozen)
        clock = ProgramClock()
        robot = ProgramRobot(clock, self.pose)
        sensor = ProgramSensor(clock, robot)
        with patch.object(program.time, "perf_counter", lambda: clock.now), patch.object(program.time, "sleep", clock.sleep):
            count = program.session(robot, sensor, cfg, self.pose, folder, Terminal(["s", "a"] * 8 + ["IDLE"]))
        self.assertEqual(count, 8)
        self.assertEqual(len(program.accepted_records(folder)), 8)
        meta, exp, demos = assemble(exploration, folder, programmed_hold_last=True, subtract_recorded_baseline=True)
        self.assertEqual(meta["compensation"], "recorded_unloaded_baseline_subtracted")
        self.assertFalse(meta["exploration_stationary_validated"])
        self.assertNotIn("initial_pose_sha256", self.cfg)
        self.assertEqual(meta["exploration_binding"]["sha256"], contract.digest(exploration))
        for episode in [*exp.values(), *demos.values()]:
            np.testing.assert_allclose(episode["ft"], episode["ft_raw_before_baseline"] - episode["recorded_unloaded_baseline"])
        np.testing.assert_allclose(exp["normal_exp"]["ft"], 0, atol=1e-12)
        raw = self.folder / "raw.h5"
        write_raw_log(raw, meta, exp, demos, source_kind="real")
        training = load_config(ROOT / "configs/real_training/real_training_manual_start.yaml")
        arrays, _ = _load_raw({**training, "raw_data": str(raw)})
        self.assertEqual(arrays["exploration"].shape, (1, 400, 6))
        self.assertEqual(arrays["ft"].shape, (8, 25, 6))
        with h5py.File(raw, "r+") as h5:
            h5["explorations/normal_exp/ft"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "baseline|raw"):
            _load_raw({**training, "raw_data": str(raw)})
        with self.assertRaisesRegex(ValueError, "baseline subtraction"):
            assemble(exploration, folder, programmed_hold_last=True)

        # A legacy session cannot acquire a new association merely by editing its manifest.
        entry_path = folder / "demo_01.json"
        original = entry_path.read_text()
        entry = json.loads(original)
        entry.pop("exploration_sha256")
        entry_path.write_text(json.dumps(entry))
        with self.assertRaisesRegex(ValueError, "old attempts cannot be relinked"):
            assemble(exploration, folder, programmed_hold_last=True, subtract_recorded_baseline=True)
        entry_path.write_text(original)
        raw_path = folder / entry["raw_file"]
        raw_original = raw_path.read_text()
        rows = [json.loads(line) for line in raw_original.splitlines()]
        rows[0].pop("exploration_sha256")
        raw_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        entry = json.loads(original)
        entry["sha256"] = contract.digest(raw_path)
        entry_path.write_text(json.dumps(entry))
        with self.assertRaisesRegex(ValueError, "old attempts cannot be relinked"):
            program.accepted_records(folder)
        raw_path.write_text(raw_original)
        entry_path.write_text(original)
        session_path = folder / "session.json"
        frozen_original = session_path.read_text()
        old = json.loads(frozen_original)
        old.pop("exploration_binding")
        old["config"].pop("exploration_binding")
        session_path.write_text(json.dumps(old))
        with self.assertRaises(ValueError):
            assemble(exploration, folder, programmed_hold_last=True, subtract_recorded_baseline=True)
        session_path.write_text(frozen_original)
        with exploration.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "bound before collection"):
            assemble(exploration, folder, programmed_hold_last=True, subtract_recorded_baseline=True)


if __name__ == "__main__":
    unittest.main()
