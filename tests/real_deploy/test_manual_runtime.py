"""Runtime-origin deployment tests; no hardware."""

import unittest
import csv
import io
from contextlib import redirect_stdout
from unittest.mock import Mock, patch
import numpy as np
from scripts.force_sensor.kwr75_reader import Kwr75Reader
from scripts.real_deploy import manual_runtime as runtime
from tests.real_training import test_manual_exploration as fixtures
from tests.real_deploy import test_fixed_setup as policies


class RuntimeTests(fixtures.Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.sensor.flush_csv = Mock()

    def run_policy(self, *, on_policy_complete=None):
        policy = policies.FakePolicy(self.pose)
        policy.predict_xy = lambda embedding: np.column_stack((np.linspace(1, 1.046, 25), np.linspace(2, 2.183, 25)))[None]
        policy.delta = 0.004
        self.sensor.latest_before = lambda due, net=False: (due, self.sensor.wrench)
        runtime.execute(self.robot, self.sensor, policy, np.zeros((1, 5)), self.cfg, self.stream, self.keyboard,
                        on_policy_complete=on_policy_complete)
        return self.events()

    def test_plot_callback_starts_after_policy_complete_before_gravity_handoff(self):
        def start_plots():
            self.assertEqual(self.events()[-1]["event"], "policy_complete")
            self.assertEqual(self.robot.controller, "servo")
            self.assertEqual(self.robot.calls[-1], "hold_joint")
            self.assertNotIn("g", self.keyboard.keys)

        callback = Mock(side_effect=start_plots)
        self.run_policy(on_policy_complete=callback)
        callback.assert_called_once_with()

    def test_inference_failure_does_not_start_plots(self):
        callback = Mock()
        with patch.object(runtime, "infer", side_effect=runtime.StateError("stale FT")):
            with self.assertRaisesRegex(runtime.StateError, "stale FT"):
                self.run_policy(on_policy_complete=callback)
        callback.assert_not_called()

    def test_order_geometry_and_hold(self):
        def hook(key):
            events = [r["event"] for r in self.events()]
            if key in ("h", "z", "s"):
                self.assertFalse(self.robot.targets)
            if key == "h":
                self.assertNotIn("servo", self.robot.calls)
            if key == "z":
                self.assertNotIn("tare_complete", events)
            if key == "s":
                self.assertIn("tare_complete", events)
            if key == "g":
                self.assertIn("policy_complete", events)
                self.assertEqual(self.robot.controller, "servo")
        self.keyboard.hook = hook
        rows = self.run_policy()
        self.assertEqual(self.keyboard.keys, ["h", "z", "s", "g"])
        self.assertEqual(self.robot.calls[-1], "gravity_comp")
        self.assertNotIn("idle", self.robot.calls)
        path = next(r for r in rows if r["event"] == "reference_captured")
        original, shifted = np.array(path["original_xy_sdk_m"]), np.array(path["translated_xy_sdk_m"])
        np.testing.assert_allclose(np.diff(original, axis=0), np.diff(shifted, axis=0), atol=1e-15)
        np.testing.assert_array_equal(shifted[0], self.pose["sdk_end_position_m"][:2])
        samples = [r for r in rows if r["event"] == "sample"]
        self.assertEqual(len(samples), 1201)
        self.assertEqual(len(self.robot.targets), 1001)
        np.testing.assert_allclose([r["target_sdk_m"] for r in samples[:201]],
                                   np.tile(self.pose["sdk_end_position_m"], (201, 1)))
        self.assertTrue(all(r["control_phase"] == "history_hold" and r["motion_tick"] is None
                            for r in samples[:200]))
        self.assertEqual([r["motion_tick"] for r in samples[200:]], list(range(1001)))
        self.assertEqual([r["tick"] for r in samples if r["delta_h_m"] is not None],
                         list(range(200, 1200, 40)))
        np.testing.assert_allclose([samples[200 + i * 40]["target_sdk_m"][:2]
                                   for i in range(1, 26)], shifted)
        self.assertAlmostEqual(samples[-1]["due_perf_s"] - samples[0]["due_perf_s"], 12)
        self.assertAlmostEqual(samples[240]["target_sdk_m"][2], self.pose["sdk_end_position_m"][2] + 0.004)

    def test_stationary_history_is_used_without_reset_or_skipping_xy(self):
        policy = policies.FakePolicy(self.pose)
        policy.delta = 0.001
        policy.predict_xy = lambda embedding: np.array([[[i * 0.001, i * 0.002] for i in range(25)]])
        loop, _ = runtime.make_loop(policy, np.zeros((1, 5)), self.pose["sdk_end_position_m"])
        raw = np.arange(1201 * 6).reshape(1201, 6) * 0.001
        expected = policy.make_ft_filter(100).process(raw)
        measured = np.array(self.pose["sdk_end_position_m"])
        for tick in range(1201):
            filtered = loop.push(tick, raw[tick], measured)
            np.testing.assert_allclose(filtered, expected[tick], atol=1e-12)
            if tick < 200:
                np.testing.assert_array_equal(loop.target(tick), loop.initial)
                self.assertFalse(policy.histories)
            if loop.last_delta is not None:
                np.testing.assert_allclose(policy.histories[-1][0], expected[np.arange(tick - 160, tick + 1, 40)])
                self.assertAlmostEqual(loop.segment_end[2], measured[2] + policy.delta)
            measured = loop.target(tick)
        self.assertEqual(len(policy.histories), 25)
        np.testing.assert_allclose(measured[:2], loop.xy[-1])
        with self.assertRaises(ValueError):
            loop.push(1201, raw[0], measured)

    def test_history_hold_uses_captured_joints_and_never_sends_early_motion(self):
        held = []
        sent = []
        begun = []
        original = self.robot.send_joint
        original_send = self.robot.send
        original_infer = runtime.infer

        def hold(joints):
            held.append((self.clock.now(), list(joints)))
            original(joints)

        def send(*args):
            sent.append((self.clock.now(), len(held)))
            original_send(*args)

        def infer(*args):
            begun.append((self.clock.now(), len(held)))
            return original_infer(*args)

        self.robot.send_joint = hold
        self.robot.send = send
        with patch.object(runtime, "infer", side_effect=infer):
            self.run_policy()
        history_holds = held[begun[0][1]:sent[0][1]]
        self.assertEqual(len(history_holds), 200)
        self.assertTrue(all(joints == self.pose["joint_position_rad"] for _, joints in history_holds))
        self.assertAlmostEqual(sent[0][0] - begun[0][0], 2)

    def test_sensor_stale_during_history_hold_never_starts_motion(self):
        original = runtime.infer

        def infer(robot, sensor, *args):
            start = self.clock.now()
            sensor.latest_before = lambda due, net=False: (min(due, start + 1), sensor.wrench)
            return original(robot, sensor, *args)

        with patch.object(runtime, "infer", side_effect=infer):
            with self.assertRaisesRegex(runtime.StateError, "fresh causal"):
                self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.events()[-1]["control_phase"], "history_hold")

    def test_completed_runtime_log_loads_for_offline_comparison(self):
        from scripts.real_deploy.plot_run_comparison import load_run, metrics

        original = self.robot.read

        def read(mode):
            row = original(mode)
            row["host_monotonic_s"] = self.clock.now()
            return row

        self.robot.read = read
        runtime.write_event(self.stream, {"event": "session_start", "shadow": False,
                                         "spec": {"workflow": runtime.WORKFLOW}})
        self.run_policy()
        path = self.folder / "events.jsonl"
        path.write_text(self.stream.getvalue(), encoding="utf-8")
        report = metrics(load_run(path))
        self.assertEqual(report["duration_s"], 12)
        self.assertEqual(report["prediction_count"], 25)
        self.assertEqual(report["phases"]["initialization"]["target_xyz_mm"]["X"]["span"], 0)

    def test_tare_fault_never_infers(self):
        with patch.object(runtime, "collect_tare", side_effect=ValueError("bad tare")):
            with self.assertRaisesRegex(ValueError, "bad tare"):
                self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.keyboard.keys, ["h", "z"])

    def test_policy_fault_stops_without_return(self):
        with patch.object(runtime, "infer", side_effect=runtime.StateError("stale FT")):
            with self.assertRaisesRegex(runtime.StateError, "stale FT"):
                self.run_policy()
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.robot.calls.count("gravity_comp"), 1)
        self.assertNotIn("g", self.keyboard.keys)

    def test_stale_causal_sensor_stops(self):
        original = runtime.infer
        def stale(robot, sensor, *args):
            sensor.latest_before = lambda due, net=False: (due - 0.03, [0] * 6)
            return original(robot, sensor, *args)
        with patch.object(runtime, "infer", side_effect=stale):
            with self.assertRaisesRegex(runtime.StateError, "fresh causal"):
                self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")

    def test_long_waits_drain_real_csv_queue_without_dropping_frames(self):
        reader = Kwr75Reader()
        path = self.folder / "sensor.csv"
        recorder = reader.start_csv(path)
        self.addCleanup(reader.stop_csv)
        self.sensor.flush_csv = reader.flush_csv
        produced = 0

        def produce(duration):
            nonlocal produced
            self.clock.sleep(duration)
            if duration > 0:
                recorder.enqueue(self.clock.now(), int(self.clock.now() * 1e9),
                                 produced + 1, [(0,) * 6] * 9)
                produced += 9

        def wait(key, monitor, prompt):
            self.keyboard.keys.append(key)
            # Each wait produces more batches than the entire queue can hold.
            for _ in range(1200):
                produce(runtime.DT)
                monitor()
                self.assertTrue(recorder._queue.empty())

        with patch.object(self.keyboard, "wait", side_effect=wait), patch.object(
                runtime.time, "sleep", side_effect=produce), patch(
                    "scripts.force_sensor.kwr75_reader.serial.Serial",
                    side_effect=AssertionError("No hardware")):
            rows = self.run_policy()
        reader.stop_csv()
        with path.open(newline="") as stream:
            recorded = list(csv.DictReader(stream))
        self.assertGreater(produced, 4 * 1024 * 9)
        self.assertEqual([int(row["frame_index"]) for row in recorded], list(range(1, produced + 1)))
        self.assertIsNone(recorder.error)
        self.assertEqual(rows[-1]["event"], "session_complete")

    def assert_wait_csv_fault(self, key, phase):
        sent = []

        def hook(current):
            if current == key:
                sent.append(len(self.robot.targets))
                self.sensor.flush_csv.side_effect = RuntimeError("CSV queue full")

        self.keyboard.hook = hook
        with self.assertRaisesRegex(RuntimeError, "CSV queue full"):
            self.run_policy()
        self.assertEqual(len(self.robot.targets), sent[0])
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.events()[-1]["phase"], phase)

    def test_drag_csv_fault_stops_before_servo(self):
        self.assert_wait_csv_fault("h", "drag")
        self.assertNotIn("servo", self.robot.calls)

    def test_wait_tare_csv_fault_stops(self):
        self.assert_wait_csv_fault("z", "wait_tare")

    def test_wait_inference_csv_fault_never_sends_policy_target(self):
        self.assert_wait_csv_fault("s", "wait_inference")
        self.assertFalse(self.robot.targets)

    def test_final_hold_csv_fault_prevents_gravity_handoff(self):
        self.assert_wait_csv_fault("g", "final_hold")
        self.assertEqual(self.robot.calls.count("gravity_comp"), 1)

    def test_startup_csv_fault_never_acquires(self):
        self.sensor.flush_csv.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.run_policy()
        self.assertFalse(self.robot.calls)
        self.assertEqual(self.events()[-1]["phase"], "startup")

    def test_tare_csv_fault_never_infers(self):
        original = runtime.collect_tare

        def fail(hold, stream):
            self.sensor.flush_csv.side_effect = OSError("disk full")
            return original(hold, stream)

        with patch.object(runtime, "collect_tare", side_effect=fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertEqual(self.events()[-1]["phase"], "tare")

    def test_policy_csv_fault_prevents_next_target(self):
        def fail():
            self.sensor.flush_csv.side_effect = RuntimeError("CSV queue full")

        self.robot.on_send = fail
        with self.assertRaisesRegex(RuntimeError, "CSV queue full"):
            self.run_policy()
        self.assertEqual(len(self.robot.targets), 1)
        self.assertEqual(self.robot.calls[-1], "stop")

    def test_slow_policy_csv_drain_keeps_deadline_before_send(self):
        def slow():
            self.sensor.flush_csv.side_effect = lambda: self.clock.sleep(0.006)

        self.robot.on_send = slow
        with self.assertRaisesRegex(runtime.StateError, "scheduling deadline"):
            self.run_policy()
        self.assertEqual(len(self.robot.targets), 1)
        self.assertEqual(self.robot.calls[-1], "stop")
        timing = self.events()[-1]
        self.assertEqual(timing["tick"], 201)
        self.assertAlmostEqual(timing["timing_ms"]["recording_check_ms"], 6)

    def test_slow_feedback_records_failed_tick_before_stop(self):
        original = policies.FakePolicy.predict_delta_h

        def slow(policy, *args):
            self.clock.sleep(0.010)
            return original(policy, *args)

        with patch.object(policies.FakePolicy, "predict_delta_h", slow):
            with self.assertRaisesRegex(runtime.StateError, "deadline exceeded before send"):
                self.run_policy()
        self.assertFalse(self.robot.targets)
        self.assertEqual(self.robot.calls[-1], "stop")
        fault = self.events()[-1]
        self.assertEqual(fault["tick"], 200)
        self.assertAlmostEqual(fault["timing_ms"]["policy_ms"], 10)
        self.assertGreaterEqual(fault["timing_ms"]["before_send_ms"], 9)
        self.assertIn("Control timing:", self.output.getvalue())

    def test_slow_observation_is_identified(self):
        original = self.robot.read

        def slow(mode):
            if self.robot.targets:
                self.clock.sleep(0.010)
            return original(mode)

        self.robot.read = slow
        with self.assertRaisesRegex(runtime.StateError, "deadline exceeded before send"):
            self.run_policy()
        self.assertEqual(len(self.robot.targets), 1)
        self.assertAlmostEqual(self.events()[-1]["timing_ms"]["observation_ms"], 10)

    def test_event_writer_fault_stops_without_masking_original_error(self):
        original = self.stream.flush

        def check():
            if self.robot.targets:
                raise OSError("event disk failed")
            return original()

        with patch.object(self.stream, "flush", side_effect=check):
            with self.assertRaisesRegex(OSError, "event disk failed"):
                self.run_policy()
        self.assertEqual(len(self.robot.targets), 1)
        self.assertEqual(self.robot.calls[-1], "stop")
        self.assertIn("Fault log unavailable", self.output.getvalue())

    def test_warmup_does_not_advance_live_history(self):
        policy = policies.FakePolicy(self.pose)
        embedding = np.zeros((1, 5))
        runtime.warm_policy(policy, embedding)
        loop, _ = runtime.make_loop(policy, embedding, self.pose["sdk_end_position_m"])
        self.assertEqual(loop.next_tick, 0)
        self.assertEqual(len(loop.history), 0)
        self.assertIsNone(loop.last_delta)
        self.assertFalse(self.robot.calls)

    def test_warmup_failure_prevents_hardware_connection(self):
        from scripts.real_deploy import manual_setup
        from scripts.real_deploy.airbot_deploy import main

        policy = Mock(warnings=[])
        loaded = ({"workflow": runtime.WORKFLOW}, {}, {}, policy, None, None, {}, {}, {})
        with patch.object(manual_setup, "load_setup", return_value=loaded), patch.object(
                manual_setup.sys.stdin, "isatty", return_value=True), patch.object(
                    runtime, "warm_policy", side_effect=ValueError("warmup failed")), patch.object(
                        manual_setup, "open_client") as connect:
            self.assertEqual(main(["run", "--mode", "manual-tared", "--execute",
                                   "--output", str(self.folder / "events.jsonl")]), 2)
        connect.assert_not_called()

    def test_preflight_ignores_legacy_gates_without_connecting(self):
        from scripts.real_deploy import manual_setup
        from scripts.real_deploy.airbot_deploy import main
        policy = type("Policy", (), {"warnings": []})()
        loaded = ({"workflow": runtime.WORKFLOW}, {}, {}, policy, None, None, {}, {}, {})
        with patch.object(manual_setup, "load_setup", return_value=loaded), patch.object(
                manual_setup, "blockers", side_effect=AssertionError("legacy gate")), patch.object(
                    manual_setup, "open_client") as connect, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["preflight", "--mode", "manual-tared"]), 0)
            connect.assert_not_called()
