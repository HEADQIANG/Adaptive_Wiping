"""Deterministic no-hardware coverage of program collection and fault handling."""

import copy
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.real_training import airbot_programmed_demonstrations as program
from scripts.real_training.airbot_demonstrations import main as dispatch
from scripts.shared.paths import ROOT


class Clock:
    def __init__(self):
        self.now = 100.0

    def sleep(self, delay):
        self.now += delay


class Robot:
    def __init__(self, clock, pose):
        self.clock, self.pose = clock, pose
        self.position = list(pose["sdk_end_position_m"])
        self.orientation = list(pose["sdk_end_orientation_xyzw"])
        self.calls, self.targets = [], []
        self.lease, self.read_cost, self.return_error = True, 0.0001, 0
        self.speed, self.angle = 0, None

    def owned(self):
        if not self.lease:
            raise program.StateError("Control lost")

    def read(self, controller):
        self.clock.now += self.read_cost
        return {"sdk_end_position_m": list(self.position),
                "sdk_end_orientation_xyzw": self.angle or list(self.orientation),
                "joint_position_rad": [0] * 6, "joint_velocity_rad_s": [self.speed] * 6,
                "host_monotonic_s": self.clock.now, "read_duration_s": self.read_cost}

    def acquire(self):
        self.calls.append("acquire")

    def enter(self):
        self.calls.append("enter")

    def send(self, position, quaternion):
        self.clock.now += 0.0001
        self.position = list(position)
        self.orientation = list(quaternion)
        self.targets.append((list(position), list(quaternion)))
        if math.dist(position, self.pose["sdk_end_position_m"]) < 1e-10:
            self.position[2] -= self.return_error

    def idle(self):
        self.calls.append("idle")

    def abort(self):
        self.calls.append("abort")


class Sensor:
    def __init__(self, clock, robot):
        self.clock, self.robot = clock, robot
        self.cache, self.age, self.sign, self.bias = {}, 0, 1, 0
        self.stiffness = 1000

    def latest(self, net=False):
        stamp = math.floor(self.clock.now * 100 + 1e-7) / 100 - self.age
        if stamp not in self.cache:
            depth = self.robot.pose["sdk_end_position_m"][2] - self.robot.position[2]
            self.cache[stamp] = [0, 0, 20 - self.sign * self.stiffness * depth + self.bias, 0, 0, 0]
        return stamp, self.cache[stamp]

    def flush_csv(self):
        pass


class Terminal:
    def __init__(self, commands=()):
        self.commands = iter(commands)
        self.prompts, self.discards = [], 0

    def discard(self):
        self.discards += 1

    def ask(self, prompt, monitor):
        self.prompts.append(prompt)
        monitor()
        return next(self.commands)


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cfg, _, self.frozen = program.load_configuration(
            ROOT / "configs/real_training/airbot_programmed_demonstrations.json")
        self.cfg.update(setup_confirmed=True)
        self.cfg.update(joint_min_rad=[-4] * 6, joint_max_rad=[4] * 6)
        self.pose = {"sdk_end_position_m": [0.25, 0, 0.15],
                     "sdk_end_orientation_xyzw": [0, 0, 0, 1], "joint_position_rad": [0] * 6}
        self.clock = Clock()
        self.robot = Robot(self.clock, self.pose)
        self.sensor = Sensor(self.clock, self.robot)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(program.time, "perf_counter", side_effect=lambda: self.clock.now))
        self.stack.enter_context(patch.object(program.time, "sleep", side_effect=self.clock.sleep))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def record(self, condition="nominal", name=None):
        path = self.folder / f"{name or condition}.jsonl"
        result = program.record_episode(self.robot, self.sensor, self.cfg, self.pose, condition, path, Terminal())
        events = [json.loads(line) for line in path.read_text().splitlines()]
        return result, events

    def test_startup_direct_move_then_wait_without_recording(self):
        self.robot.position[0] -= 0.03
        self.robot.orientation = [0, 0, math.sin(0.05), math.cos(0.05)]
        initial = self.robot.read("idle")
        terminal = Terminal(["q", "IDLE"])
        recorder = Mock(side_effect=AssertionError("No fresh start command"))
        self.assertEqual(program.session(self.robot, self.sensor, self.cfg, self.pose,
                                         self.folder, terminal, recorder), 0)
        self.assertEqual(self.robot.targets[0], (initial["sdk_end_position_m"], initial["sdk_end_orientation_xyzw"]))
        self.assertLess(math.dist(self.robot.position, self.pose["sdk_end_position_m"]), 1e-10)
        for previous, current in zip(self.robot.targets, self.robot.targets[1:]):
            self.assertLessEqual(math.dist(previous[0], current[0]) / program.DT, 0.01 + 1e-9)
            self.assertLessEqual(program.quaternion_angle(previous[1], current[1]) / program.DT, 0.1 + 1e-7)
        self.assertGreaterEqual(terminal.discards, 1)
        self.assertEqual(len(terminal.prompts), 2)
        events = [json.loads(line) for line in next(self.folder.glob("approach_*.jsonl")).read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "approach_complete")
        self.assertFalse(list(self.folder.glob("attempt_*.jsonl")))
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])

    def test_startup_excessive_distance_rejected_before_servo(self):
        self.robot.position[0] += 1
        with self.assertRaisesRegex(program.StateError, "maximum duration"):
            program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, Terminal())
        self.assertEqual(self.robot.calls, ["acquire"])
        self.assertFalse(self.robot.targets)

    def test_startup_mode_switch_drift_stops_before_command(self):
        def enter():
            self.robot.calls.append("enter")
            self.robot.position[0] += 0.01
        self.robot.enter = enter
        with self.assertRaisesRegex(program.StateError, "tracking"):
            program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, Terminal())
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertFalse(self.robot.targets)

    def test_startup_faults_abort_without_retract(self):
        for fault in ("force", "stream", "lease", "deadline", "rejected", "return"):
            with self.subTest(fault=fault):
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock, self.robot)
                self.robot.position[0] -= 0.01
                original_send = self.robot.send
                def send(position, quaternion):
                    original_send(position, quaternion)
                    if len(self.robot.targets) == 3:
                        if fault == "force":
                            self.sensor.bias = 100
                        elif fault == "stream":
                            self.sensor.age = 1
                        elif fault == "lease":
                            self.robot.lease = False
                        elif fault == "deadline":
                            self.clock.now += 0.03
                        elif fault == "rejected":
                            raise program.StateError("Command rejected")
                        else:
                            self.robot.return_error = 0.003
                self.robot.send = send
                with self.assertRaises((program.StateError, ValueError)):
                    program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, Terminal())
                self.assertEqual(self.robot.calls[-1], "abort")
                self.assertNotIn("idle", self.robot.calls)

    def test_startup_confirmation_and_parameter_validation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["startup_approach"]["direct_path_confirmed"] = False
        self.assertTrue(program.readiness(cfg))
        spec = json.loads((ROOT / "configs/real_training/airbot_programmed_demonstrations.json").read_text())
        for value in (0, -1, True, float("nan"), 0.051):
            spec["startup_approach"]["speed_m_s"] = value
            path = self.folder / "bad.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "speed_m_s"):
                program.load_configuration(path)

    def test_program_stationarity_point_one_without_changing_default(self):
        self.robot.speed = 0.08
        states = [self.robot.read("idle") for _ in range(11)]
        with self.assertRaises(program.NotStationary):
            program.validate_stationary(states)
        program.verify_return([{"pose": s} for s in states], self.cfg, self.pose)
        self.robot.speed = 0.1
        with io.StringIO() as stream:
            self.assertAlmostEqual(program.collect_baseline(self.robot, self.sensor, self.cfg,
                                                           self.pose, stream)[2], 20)

    def test_baseline_retry_discards_all_old_samples(self):
        original_write = program.write_event
        def write(stream, event):
            original_write(stream, event)
            if event["event"] == "tare_start" and event["baseline_attempt"] == 1:
                self.robot.speed, self.sensor.bias = 0.11, 7
            elif event["event"] == "tare_discarded":
                self.robot.speed, self.sensor.bias = 0, 0
        with io.StringIO() as stream, patch.object(program, "write_event", side_effect=write):
            baseline = program.collect_baseline(self.robot, self.sensor, self.cfg, self.pose, stream)
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertAlmostEqual(baseline[2], 20)
        discarded = [r for r in events if r["event"] == "tare_discarded"]
        self.assertEqual(len(discarded), 1)
        self.assertEqual(events[-1]["baseline_attempt"], 2)
        samples = [r for r in events if r.get("phase") == "tare" and r["baseline_attempt"] == 2]
        self.assertGreaterEqual(samples[-1]["observed_perf_s"] - samples[0]["observed_perf_s"], 0.98)
        self.assertFalse(self.robot.targets)

    def test_baseline_persistent_instability_has_total_timeout(self):
        original_write = program.write_event
        def write(stream, event):
            original_write(stream, event)
            if event["event"] == "tare_start":
                self.robot.speed = 0.11
            elif event["event"] == "tare_discarded":
                self.robot.speed = 0
        start = self.clock.now
        with (io.StringIO() as stream, patch.object(program, "write_event", side_effect=write),
              self.assertRaises(program.StateError)):
            program.collect_baseline(self.robot, self.sensor, self.cfg, self.pose, stream)
        self.assertLess(self.clock.now - start, 10.1)
        self.assertGreaterEqual(self.clock.now - start, 10)
        self.assertFalse(self.robot.targets)

    def test_baseline_safety_faults_are_not_retried(self):
        for fault in ("force", "stream", "pose", "lease", "speed"):
            with self.subTest(fault=fault):
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock, self.robot)
                original_write = program.write_event
                def write(stream, event):
                    original_write(stream, event)
                    if event["event"] == "tare_start":
                        if fault == "force":
                            self.sensor.bias = 100
                        elif fault == "stream":
                            self.sensor.age = 1
                        elif fault == "pose":
                            self.robot.position[0] += 0.01
                        elif fault == "lease":
                            self.robot.lease = False
                        else:
                            self.robot.speed = 2
                stream = io.StringIO()
                with patch.object(program, "write_event", side_effect=write), self.assertRaises(program.StateError):
                    program.collect_baseline(self.robot, self.sensor, self.cfg, self.pose, stream)
                self.assertNotIn("tare_discarded", stream.getvalue())
                self.assertNotIn("tare_complete", stream.getvalue())

    def test_retraction_brief_lateness_rebases_without_catchup(self):
        original_write, original_read = program.write_event, self.robot.read
        pending = False
        def write(stream, event):
            nonlocal pending
            original_write(stream, event)
            if event["event"] == "episode_end":
                pending = True
        def read(controller):
            nonlocal pending
            if pending:
                pending = False
                self.clock.now += 0.015
            return original_read(controller)
        self.robot.read = read
        with patch.object(program, "write_event", side_effect=write):
            report, events = self.record()
        self.assertTrue(report["return_confirmed"])
        self.assertTrue(report["quality"]["passed"])
        rows = [e for e in events if e.get("phase") == "retract"]
        self.assertGreater(rows[0]["lateness_s"], 0.005)
        self.assertLess(rows[0]["lateness_s"], 0.01)
        self.assertGreater(rows[1]["schedule_shift_s"], 0.005)
        self.assertTrue(all(b["send_perf_s"] - a["send_perf_s"] >= program.DT - 1e-9
                            for a, b in zip(rows, rows[1:])))

    def test_retraction_large_or_sustained_delay_still_stops(self):
        for sustained in (False, True):
            with self.subTest(sustained=sustained):
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock, self.robot)
                original_write, original_read = program.write_event, self.robot.read
                pending = False
                def write(stream, event):
                    nonlocal pending
                    original_write(stream, event)
                    if event["event"] == "episode_end":
                        pending = True
                    if sustained and event.get("phase") == "retract":
                        self.clock.now += 0.006
                def read(controller):
                    nonlocal pending
                    if pending:
                        pending = False
                        if not sustained:
                            self.clock.now += 0.025
                    return original_read(controller)
                self.robot.read = read
                with patch.object(program, "write_event", side_effect=write), self.assertRaises(program.StateError) as caught:
                    self.record(name=f"retraction_{sustained}")
                timing = caught.exception.retraction_timing
                if sustained:
                    self.assertGreater(timing["elapsed_s"], 3.0)
                else:
                    self.assertEqual(timing["phase"], "before_send")
                events = [json.loads(line) for line in (self.folder / f"retraction_{sustained}.jsonl").read_text().splitlines()]
                self.assertFalse(any(e["event"] == "finished" for e in events))

    def test_startup_six_ms_command_and_logging_fit_ten_ms_cycle(self):
        self.robot.position[0] -= 0.001
        original_send = self.robot.send
        def send(position, quaternion):
            original_send(position, quaternion)
            self.clock.now += 0.006
        self.robot.send = send
        self.sensor.flush_csv = lambda: self.clock.sleep(0.001)
        self.assertEqual(program.session(self.robot, self.sensor, self.cfg, self.pose,
                                         self.folder, Terminal(["q", "IDLE"])), 0)
        events = [json.loads(line) for line in next(self.folder.glob("approach_*.jsonl")).read_text().splitlines()]
        samples = [e for e in events if e.get("phase") == "approach"]
        self.assertTrue(samples)
        self.assertGreater(samples[0]["timing"]["command_s"], 0.005)
        self.assertLess(samples[0]["timing"]["elapsed_s"], program.DT)
        self.assertIn("json_write_s", samples[1]["timing"]["previous_logging_s"])
        self.assertEqual(events[-1]["event"], "approach_complete")

    def test_startup_cycle_overruns_identify_stage_and_stop(self):
        for stage in ("observation_or_target", "command", "csv_logging", "json_logging", "scheduling"):
            with self.subTest(stage=stage):
                folder = self.folder / stage
                folder.mkdir()
                self.robot = Robot(self.clock, self.pose)
                self.sensor = Sensor(self.clock, self.robot)
                self.robot.position[0] -= 0.001
                original_send, original_write = self.robot.send, program.write_event
                def send(position, quaternion):
                    original_send(position, quaternion)
                    if stage == "observation_or_target":
                        self.robot.read_cost = 0.011
                    elif stage == "command" and len(self.robot.targets) == 2:
                        self.clock.now += 0.011
                def flush():
                    if stage == "csv_logging" and len(self.robot.targets) >= 2:
                        self.clock.now += 0.011
                def write(stream, event):
                    original_write(stream, event)
                    if stage == "json_logging" and event.get("phase") == "approach":
                        self.clock.now += 0.011
                def sleep(delay):
                    self.clock.sleep(delay)
                    if stage == "scheduling" and self.robot.targets:
                        self.clock.now += 0.006
                self.robot.send, self.sensor.flush_csv = send, flush
                with (patch.object(program, "write_event", side_effect=write),
                      patch.object(program.time, "sleep", side_effect=sleep),
                      self.assertRaises(program.ApproachTimingError) as caught):
                    program.session(self.robot, self.sensor, self.cfg, self.pose, folder, Terminal())
                self.assertEqual(caught.exception.approach_timing["failed_phase"], stage)
                self.assertEqual(self.robot.calls[-1], "abort")
                self.assertEqual(len(self.robot.targets), 1 if stage in ("observation_or_target", "scheduling") else 2)
                fault = json.loads(next(folder.glob("fault_*.json")).read_text())
                self.assertEqual(fault["approach_timing"]["failed_phase"], stage)
                self.assertFalse(list(folder.glob("demo_*.json")))

    def test_default_preview_and_no_hardware_until_ready(self):
        pending = {**self.cfg, "setup_confirmed": False}
        with (patch.object(program, "load_configuration", return_value=(pending, self.pose, self.frozen)),
              patch.object(program, "open_client", side_effect=AssertionError("hardware"))):
            self.assertEqual(dispatch(["preview", "--mode", "programmed"]), 2)
            self.assertEqual(dispatch(["check", "--mode", "programmed"]), 1)
            self.assertEqual(dispatch(["run", "--mode", "programmed", "--execute"]), 1)
        self.assertFalse(program.readiness(self.cfg))
        self.assertNotIn("control", self.cfg)
        self.assertNotIn("target_force_n", self.cfg)
        self.assertNotIn("force_direction_confirmed", self.cfg)
        self.assertEqual(len(program.readiness(pending)), 1)
        output = io.StringIO()
        with (patch.object(program, "load_configuration", return_value=(self.cfg, self.pose, self.frozen)),
              patch.object(program, "open_client", side_effect=AssertionError("hardware")),
              redirect_stdout(output)):
            self.assertEqual(dispatch(["preview", "--mode", "programmed"]), 0)
        preview = json.loads(output.getvalue())
        self.assertEqual(preview["setup_errors"], [])
        self.assertFalse(preview["force_feedback_enabled"])
        self.assertNotIn("target_force_n", preview)

    def test_closed_loop_configuration_cannot_be_reinterpreted(self):
        spec = json.loads((ROOT / "configs/real_training/airbot_programmed_demonstrations.json").read_text())
        for update in ({"protocol": "airbot_programmed_v1"}, {"target_force_n": 10}, {"control": {}}):
            path = self.folder / "config.json"
            path.write_text(json.dumps({**spec, **update}))
            with self.assertRaises(ValueError):
                program.load_configuration(path)

    def test_geometry_duration_and_counts(self):
        self.assertEqual([program.SEQUENCE.count(c) for c in ("nominal", "under", "over")], [2, 3, 3])
        self.assertEqual([program.slide_offset(t) for t in (0, 1, 3, 4)], [0, 0.05, -0.05, 0])
        self.assertEqual([program.duration(c) for c in ("under", "nominal", "over")], [5.6, 6.0, 6.4])
        for t in (0.3, 1.3, 3.3):
            self.assertAlmostEqual(abs(program.slide_offset(t + 0.01) - program.slide_offset(t)) / 0.01, program.SLIDE_SPEED)
        with self.assertRaises(program.StateError):
            program.target(self.pose, self.cfg, 0, 0.02001)

    def test_force_changes_and_sign_do_not_change_motion(self):
        self.sensor.stiffness = 500
        low_report, _ = self.record(name="low_force")
        low_targets = list(self.robot.targets)
        self.robot.targets.clear()
        self.sensor.cache.clear()
        self.sensor.stiffness, self.sensor.sign = 1500, -1
        high_report, events = self.record(name="high_force")
        self.assertEqual(low_targets, self.robot.targets)
        self.assertLess(low_report["force_report"]["tared"]["mean_si"][2], 0)
        self.assertGreater(high_report["force_report"]["tared"]["mean_si"][2], 0)
        self.assertFalse(high_report["force_report"]["force_feedback_enabled"])
        self.assertNotIn("target_force_n", events[0])

    def test_all_three_real_time_episodes_and_retract_exclusion(self):
        for condition in program.DEPTHS:
            self.robot.position = list(self.pose["sdk_end_position_m"])
            self.sensor.cache.clear()
            report, events = self.record(condition)
            self.assertTrue(report["quality"]["passed"], report)
            self.assertTrue(report["return_confirmed"])
            self.assertIs(events[0]["training_ready"], False)
            self.assertEqual(events[0]["demonstration_source"], "programmed")
            self.assertFalse(report["force_report"]["force_feedback_enabled"])
            self.assertNotIn("condition_matches", report["force_report"])
            self.assertAlmostEqual(report["quality"]["duration_s"], program.duration(condition))
            slides = [r for r in events if r.get("phase") == "slide"]
            self.assertEqual(len(slides), 400)
            self.assertEqual({r["commanded_depth_m"] for r in slides}, {program.DEPTHS[condition]})
            self.assertTrue(all("control" not in r for r in slides))
            self.assertAlmostEqual(slides[99]["lateral_offset_m"], 0.05)
            self.assertAlmostEqual(slides[299]["lateral_offset_m"], -0.05)
            self.assertAlmostEqual(slides[-1]["lateral_offset_m"], 0)
            self.assertEqual(events[-1]["event"], "finished")
            self.assertLessEqual(max(r.get("commanded_depth_m", 0) for r in events), 0.02)
            self.assertEqual(self.robot.position, self.pose["sdk_end_position_m"])

    def test_sensor_and_measured_state_guards(self):
        position = self.pose["sdk_end_position_m"]
        self.sensor.age = 0.1
        with self.assertRaisesRegex(program.StateError, "stale"):
            program.observe(self.robot, self.sensor, self.cfg, self.pose, position)
        self.sensor.age = 0
        self.robot.lease = False
        with self.assertRaisesRegex(program.StateError, "Control"):
            program.observe(self.robot, self.sensor, self.cfg, self.pose, position)
        self.robot.lease = True
        self.robot.position[2] -= 0.021
        with self.assertRaisesRegex(program.StateError, "20 mm"):
            program.observe(self.robot, self.sensor, self.cfg, self.pose, self.robot.position)
        self.robot.position = list(position)
        self.robot.angle = [0, 0, 1, 0]
        with self.assertRaisesRegex(program.StateError, "Orientation"):
            program.observe(self.robot, self.sensor, self.cfg, self.pose, position, initial=True)

    def test_exploration_force_limits_not_disabled_by_tare(self):
        self.sensor.bias = 100
        with self.assertRaisesRegex(program.StateError, "Force/torque"):
            program.observe(self.robot, self.sensor, self.cfg, self.pose, self.robot.position)
        self.sensor.bias = 0
        self.sensor.cache.clear()
        self.robot.position[2] -= 0.01
        self.sensor.sign = -1
        self.sensor.cache.clear()
        row = program.observe(self.robot, self.sensor, self.cfg, self.pose, self.robot.position,
                              [0, 0, 20, 0, 0, 0], sliding=True)
        self.assertAlmostEqual(row["ft"]["tared_sensor_wrench_si"][2], 10)
        with patch.object(self.sensor, "latest", return_value=(self.clock.now, [0, 0, 0, 1, 0, 0])):
            with self.assertRaisesRegex(program.StateError, "Force/torque"):
                program.observe(self.robot, self.sensor, self.cfg, self.pose, self.robot.position,
                                [0, 0, 0, 1, 0, 0])

    def test_running_tracking_policies_are_inherited_from_exploration(self):
        self.robot.position[0] += 0.008
        row = program.observe(self.robot, self.sensor, self.cfg, self.pose, self.pose["sdk_end_position_m"])
        self.assertTrue(row["position_tracking_exceeds_limit"])
        self.assertTrue(row["tracking_error_record_only"])
        strict = {**self.cfg, "tracking_error_policy": "stop", "lateral_tracking_policy": "stop"}
        with self.assertRaisesRegex(program.StateError, "tracking error"):
            program.observe(self.robot, self.sensor, strict, self.pose, self.pose["sdk_end_position_m"])
        self.robot.position = [0.25, 0.008, 0.15]
        strict["lateral_tracking_policy"] = "record-only"
        row = program.observe(self.robot, self.sensor, strict, self.pose, self.pose["sdk_end_position_m"], sliding=True)
        self.assertTrue(row["slide_error_record_only"])
        with self.assertRaisesRegex(program.StateError, "tracking error"):
            program.observe(self.robot, self.sensor, strict, self.pose, self.pose["sdk_end_position_m"])

    def test_return_failure_not_finished(self):
        self.robot.return_error = 0.003
        with self.assertRaisesRegex(program.StateError, "not confirmed"):
            self.record()
        self.assertNotIn('"event": "finished"', (self.folder / "nominal.jsonl").read_text())

    def test_deadline_overrun_does_not_send_or_finish(self):
        with patch.object(program, "collect_baseline", return_value=[0, 0, 20, 0, 0, 0]):
            self.robot.read_cost = 0.012
            with self.assertRaisesRegex(program.StateError, "overrun"):
                self.record()
        self.assertFalse(self.robot.targets)

    def test_quality_checks_gaps_and_coverage(self):
        rows = [{"pose": {"host_monotonic_s": i * 0.01, "read_duration_s": 0.001},
                 "ft": {"sensor_receive_perf_s": i * 0.01}, "observed_perf_s": i * 0.01}
                for i in range(563)]
        self.assertTrue(program.timing_quality(rows, 0, 5.6)["passed"])
        self.assertFalse(program.timing_quality(rows[:-10], 0, 5.6)["passed"])
        self.assertFalse(program.timing_quality(rows[:10] + rows[15:], 0, 5.6)["passed"])
        for row in rows:
            row["ft"]["sensor_receive_perf_s"] = 0
        self.assertFalse(program.timing_quality(rows, 0, 5.6)["passed"])

    def test_session_eight_separate_starts_and_integrity(self):
        terminal = Terminal(["s", "a"] * 8 + ["IDLE"])
        count = program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, terminal)
        self.assertEqual(count, 8)
        self.assertEqual(self.robot.calls, ["acquire", "enter", "idle"])
        records = program.accepted_records(self.folder)
        self.assertEqual([r["condition"] for r in records], list(program.SEQUENCE))
        self.assertEqual(len(terminal.prompts), 17)
        self.assertGreater(terminal.discards, 4000)
        program.write_json(self.folder / "session.json", self.frozen)
        with patch.object(program, "open_client", side_effect=AssertionError("hardware")):
            self.assertEqual(dispatch(["status", "--mode", "programmed", "--output", str(self.folder)]), 0)
        raw = self.folder / records[0]["raw_file"]
        raw.write_text(raw.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            program.accepted_records(self.folder)

    def test_readonly_check_never_acquires_or_moves(self):
        self.robot.position[0] -= 0.02
        client = SimpleNamespace(_stub=object(), close=Mock())
        with (patch.object(program, "load_configuration", return_value=(self.cfg, self.pose, self.frozen)),
              patch.object(program, "open_client", return_value=client),
              patch.object(program, "Robot", return_value=self.robot),
              patch.object(program, "snapshot", side_effect=lambda *_: self.robot.read("idle"))):
            self.assertEqual(dispatch(["check", "--mode", "programmed"]), 0)
        self.assertFalse(self.robot.calls)
        self.assertFalse(self.robot.targets)
        client.close.assert_called_once()

    def test_session_lock_blocks_before_hardware(self):
        import fcntl

        with (self.folder / ".lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (patch.object(program, "load_configuration", return_value=(self.cfg, self.pose, self.frozen)),
                  patch.object(program.sys.stdin, "isatty", return_value=True),
                  patch.object(program, "open_client", side_effect=AssertionError("hardware"))):
                self.assertEqual(dispatch(["run", "--mode", "programmed", "--execute",
                                           "--output", str(self.folder)]), 1)

    def test_execute_reaches_connection_without_consuming_start_key(self):
        with (
            patch.object(program, "load_configuration", return_value=(self.cfg, self.pose, self.frozen)),
            patch.object(program.sys.stdin, "isatty", return_value=True),
            patch.object(program.Terminal, "ask", side_effect=AssertionError("Unexpected startup prompt")) as prompt,
            patch.object(program, "open_client", side_effect=RuntimeError("Test connection boundary")) as connect,
        ):
            self.assertEqual(dispatch(["run", "--mode", "programmed", "--execute",
                                       "--output", str(self.folder)]), 1)
        connect.assert_called_once()
        prompt.assert_not_called()

    def test_waiting_drift_blocks_acceptance(self):
        robot = self.robot
        class DriftingTerminal(Terminal):
            def ask(self, prompt, monitor):
                if prompt.startswith("Returned"):
                    robot.position[1] += 0.003
                return super().ask(prompt, monitor)

        with self.assertRaisesRegex(ValueError, "Return/start"):
            program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder,
                            DriftingTerminal(["s", "a"]))
        self.assertEqual(self.robot.calls[-1], "abort")
        self.assertFalse(list(self.folder.glob("demo_*.json")))

    def test_acquisition_and_servo_entry_failure_boundaries(self):
        with patch.object(self.robot, "acquire", side_effect=program.StateError("refused")):
            with self.assertRaisesRegex(program.StateError, "refused"):
                program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, Terminal())
        self.assertFalse(self.robot.calls)
        with patch.object(self.robot, "enter", side_effect=program.StateError("entry failed")):
            with self.assertRaisesRegex(program.StateError, "entry failed"):
                program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, Terminal())
        self.assertEqual(self.robot.calls, ["acquire", "abort"])

    def test_reject_resume_and_no_automatic_next_attempt(self):
        terminal = Terminal(["s", "r", "s", "a", "q", "IDLE"])
        self.assertEqual(program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, terminal), 1)
        first = (self.folder / "demo_01.json").read_bytes()
        self.assertEqual(len(list(self.folder.glob("attempt_*.jsonl"))), 2)
        terminal = Terminal(["s", "a", "q", "IDLE"])
        self.assertEqual(program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder, terminal), 2)
        self.assertEqual((self.folder / "demo_01.json").read_bytes(), first)

    def test_faults_abort_without_retract_or_idle(self):
        for error in (KeyboardInterrupt(), EOFError(), program.StateError("sensor stale"), OSError("disk full")):
            self.robot.calls.clear()
            def fail(*args):
                raise error
            with self.assertRaises(type(error)):
                program.session(self.robot, self.sensor, self.cfg, self.pose, self.folder,
                                Terminal(["s"]), recorder=fail)
            self.assertEqual(self.robot.calls, ["acquire", "enter", "abort"])

    def test_config_snapshot_and_start_hash_freeze(self):
        program.freeze_session(self.folder, self.frozen)
        program.freeze_session(self.folder, self.frozen)
        changed = copy.deepcopy(self.frozen)
        changed["initial_pose_record"]["initial_pose"]["sdk_end_position_m"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "changed"):
            program.freeze_session(self.folder, changed)
        changed = copy.deepcopy(self.frozen)
        changed["source_hashes"]["extra"] = "new"
        with self.assertRaisesRegex(ValueError, "changed"):
            program.freeze_session(self.folder, changed)

    def test_terminal_discards_batched_and_partial_commands(self):
        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        terminal = program.Terminal()
        with patch.object(program.sys, "stdin", SimpleNamespace(fileno=lambda: slave)):
            os.write(master, b"old\ns\npartial")
            def send_batch():
                os.write(master, b"s\na\ns\n")
            self.assertEqual(terminal.ask("start", send_batch), "s")
            def send_fresh():
                os.write(master, b"q\n")
            self.assertEqual(terminal.ask("next", send_fresh), "q")

    def test_legacy_importer_rejects_before_exploration_read(self):
        try:
            from scripts.real_training.import_airbot import assemble
        except ImportError as exc:
            self.skipTest(str(exc))
        program.write_json(self.folder / "session.json", self.frozen)
        with self.assertRaisesRegex(ValueError, "Variable-length programmed"):
            assemble(self.folder / "nonexistent-exploration.jsonl", self.folder)


if __name__ == "__main__":
    unittest.main()
