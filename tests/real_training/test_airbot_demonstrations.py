"""Software-only manual collection tests; never opens hardware."""

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scripts.real_training.airbot_demonstrations as demo
from scripts.shared.paths import ROOT
from tests.robot_control.test_airbot_initial_pose import FakeClient


class Sensor:
    def flush_csv(self):
        pass


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            (demo.ROOT / "configs/real_training/airbot_demonstrations.json").read_text()
        )
        self.cfg.update(
            workspace_force_policy="stop",
            joint_speed_policy="stop",
            joint_speed_limit_rad_s=0.4,
            setup_confirmed=True,
            setup_note="test-only measured setup",
            max_force_n=10,
            max_torque_nm=0.5,
            joint_min_rad=[-2] * 6,
            joint_max_rad=[2] * 6,
        )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.client = FakeClient()
        self.sensor = Sensor()

    def test_default_record_only_and_guarded_configuration(self):
        cfg = json.loads(
            (demo.ROOT / "configs/real_training/airbot_demonstrations.json").read_text()
        )
        demo.validate_config(cfg)
        cfg["workspace_force_policy"] = "stop"
        with self.assertRaises(ValueError):
            demo.validate_config(cfg)
        demo.validate_config(self.cfg)
        for key, value in (
            ("duration_s", 8),
            ("demonstrations", 9),
            ("max_force_n", float("nan")),
            ("joint_speed_limit_rad_s", 2),
        ):
            cfg = {**self.cfg, key: value}
            with self.assertRaises(ValueError):
                demo.validate_config(cfg)

    def test_record_only_keeps_sensor_validity_and_joint_guards(self):
        cfg = json.loads(
            (demo.ROOT / "configs/real_training/airbot_demonstrations.json").read_text()
        )
        cfg.update(joint_min_rad=[-2] * 6, joint_max_rad=[2] * 6)
        self.client.lease = True
        self.client.state.controller_state = "gravity_comp"
        self.client.pose.position = [10, 10, 10]
        from types import SimpleNamespace

        sensor = SimpleNamespace(latest=lambda **kwargs: (1, [10000] * 6))
        with patch.object(demo.time, "perf_counter", return_value=1.001):
            row = demo.observe(self.client, sensor, cfg)
            self.assertEqual(row["ft"]["raw_sensor_wrench_si"], [10000] * 6)
            self.client.joints.angles[0] = 3
            with self.assertRaises(demo.StateError):
                demo.observe(self.client, sensor, cfg)
        for item in (None, (0, [1] * 6), (1, [float("nan")] * 6)):
            sensor.latest = lambda **kwargs: item
            with patch.object(demo.time, "perf_counter", return_value=1.001):
                with self.assertRaises(demo.StateError):
                    demo.read_demo_force(sensor, cfg)
        with self.assertRaises(ValueError):
            demo.validate_config({**cfg, "workspace_force_policy": "typo"})

    def rows(self):
        return [
            {
                "observed_perf_s": i / 100,
                "pose": {"host_monotonic_s": i / 100, "read_duration_s": 0.001},
                "ft": {"sensor_receive_perf_s": i / 100 - 0.001},
            }
            for i in range(1003)
        ]

    def test_speed_record_only_preserves_values_and_stationary_check(self):
        cfg = {**self.cfg, "joint_speed_policy": "record-only", "joint_speed_limit_rad_s": None}
        demo.validate_config(cfg)
        self.client.lease = True
        self.client.state.controller_state = "gravity_comp"
        velocities = [0.8, -0.9, 1.1, -1.2, 0.5, -0.6]
        self.client.joints.velocities = velocities
        with (
            patch.object(demo, "read_demo_force", return_value={}),
            patch.object(demo.time, "sleep"),
        ):
            row = demo.observe(self.client, self.sensor, cfg)
            self.assertEqual(row["pose"]["joint_velocity_rad_s"], velocities)
            with self.assertRaises(ValueError):
                demo.stationary_start(self.client, self.sensor, cfg)
            self.client.joints.velocities[0] = float("nan")
            with self.assertRaises(demo.StateError):
                demo.observe(self.client, self.sensor, cfg)
        with self.assertRaises(ValueError):
            demo.validate_config({**cfg, "joint_speed_policy": "typo"})
        legacy = {**self.cfg}
        del legacy["joint_speed_policy"]
        self.assertFalse(demo.speed_record_only(legacy))

    def test_quality_rejects_gaps_short_and_slow_reads(self):
        rows = self.rows()
        self.assertTrue(demo.quality(rows, 0)["passed"])
        self.assertFalse(demo.quality(rows[:-3], 0)["passed"])
        self.assertFalse(demo.quality(rows[:5] + rows[9:], 0)["passed"])
        rows[2]["pose"]["read_duration_s"] = 0.03
        self.assertFalse(demo.quality(rows, 0)["passed"])
        rows = self.rows()
        for row in rows:
            row["ft"]["sensor_receive_perf_s"] = 0
        self.assertFalse(demo.quality(rows, 0)["passed"])

    def test_record_loop_and_exclusive_output(self):
        rows = iter(self.rows())
        raw = self.folder / "attempt.jsonl"
        with (
            patch.object(demo, "observe", side_effect=lambda *args: next(rows)),
            patch.object(demo.time, "sleep"),
            patch.object(demo.time, "perf_counter", return_value=0),
        ):
            report = demo.record_episode(self.client, self.sensor, self.cfg, raw)
        self.assertTrue(report["passed"])
        events = [json.loads(line) for line in raw.read_text().splitlines()]
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[-1]["event"], "finished")
        self.assertEqual(len(events), 1005)
        with self.assertRaises(FileExistsError):
            demo.record_episode(self.client, self.sensor, self.cfg, raw)

    def run_session(self, commands, recorder=None):
        commands = iter(commands)

        def record(client, sensor, cfg, path):
            path.write_text("test-only raw log")
            return {"passed": True}

        with (
            redirect_stdout(io.StringIO()),
            patch.object(
                demo,
                "stationary_start",
                return_value={
                    "sdk_end_position_m": [0.25, 0.02, 0.15],
                    "sdk_end_orientation_xyzw": [0, 0, 0, 1],
                },
            ),
            patch.object(demo, "observe"),
        ):
            return demo.session(
                self.client,
                self.sensor,
                "idle",
                self.cfg,
                self.folder,
                reader=lambda *args: next(commands),
                recorder=recorder or record,
            )

    def test_eight_and_no_trajectory_commands(self):
        self.assertEqual(self.run_session(["s", "a"] * 8), 8)
        self.assertEqual(len(demo.accepted_records(self.folder)), 8)
        self.assertEqual(self.client.calls, ["acquire", "gravity", "idle"])
        self.assertFalse(json.loads((self.folder / "demo_01.json").read_text())["training_ready"])

    def test_reject_resume_and_integrity(self):
        self.assertEqual(self.run_session(["s", "r", "s", "a", "q"]), 1)
        self.assertEqual(self.run_session(["s", "a", "q"]), 2)
        record = json.loads((self.folder / "demo_01.json").read_text())
        (self.folder / record["raw_file"]).write_text("corrupt")
        with self.assertRaises(ValueError):
            demo.accepted_records(self.folder)

    def test_failed_quality_not_accepted(self):
        self.assertEqual(self.run_session(["s", "q"], lambda *args: {"passed": False}), 0)
        self.assertEqual(demo.accepted_records(self.folder), {})

    def test_record_fault_and_interrupt_cleanup(self):
        for error in (RuntimeError("sensor stale"), KeyboardInterrupt(), EOFError()):
            self.client = FakeClient()

            def fail(*args):
                raise error

            with self.assertRaises(type(error)):
                self.run_session(["s"], fail)
            self.assertEqual(self.client.calls[-1], "idle")

    def test_refusal_does_not_switch(self):
        self.client.acquire_ok = False
        with self.assertRaises(demo.StateError):
            self.run_session([])
        self.assertEqual(self.client.calls, ["acquire"])

    def test_reference_and_nonoverwrite(self):
        state = {"sdk_end_position_m": [0, 0, 0], "sdk_end_orientation_xyzw": [0, 0, 0, 1]}
        other = copy.deepcopy(state)
        other["sdk_end_position_m"][0] = 0.003
        deviation = demo.check_reference(state, other)
        self.assertEqual(deviation["position_policy"], "record-only")
        self.assertEqual(deviation["position_delta_sdk_m"], [-0.003, 0, 0])
        self.assertAlmostEqual(deviation["position_distance_m"], 0.003)
        other["sdk_end_orientation_xyzw"] = [0, 0, 1, 0]
        deviation = demo.check_reference(state, other)
        self.assertEqual(deviation["orientation_policy"], "record-only")
        self.assertAlmostEqual(deviation["orientation_angle_rad"], math.pi)
        other["sdk_end_orientation_xyzw"] = [0, 0, 0, -1]
        self.assertAlmostEqual(demo.check_reference(state, other)["orientation_angle_rad"], 0)
        demo.write_json(self.folder / "record.json", state)
        with self.assertRaises(FileExistsError):
            demo.write_json(self.folder / "record.json", state)

    def test_offset_resume_preserves_first_demo_and_records_deviation(self):
        self.assertEqual(self.run_session(["s", "a", "q"]), 1)
        first = (self.folder / "demo_01.json").read_bytes()
        reference = (self.folder / "reference.json").read_bytes()
        initial = {
            "sdk_end_position_m": [0.30, 0.02, 0.15],
            "sdk_end_orientation_xyzw": [0, 0, 1, 0],
        }
        commands = iter(["s", "a", "q"])

        def record(client, sensor, cfg, path):
            path.write_text("test-only offset raw")
            return {"passed": True}

        with (
            redirect_stdout(io.StringIO()),
            patch.object(demo, "stationary_start", return_value=initial),
            patch.object(demo, "observe"),
        ):
            count = demo.session(
                self.client,
                self.sensor,
                "idle",
                self.cfg,
                self.folder,
                reader=lambda *args: next(commands),
                recorder=record,
            )
        self.assertEqual(count, 2)
        self.assertEqual((self.folder / "demo_01.json").read_bytes(), first)
        self.assertEqual((self.folder / "reference.json").read_bytes(), reference)
        second = json.loads((self.folder / "demo_02.json").read_text())
        self.assertAlmostEqual(second["reference_start_deviation"]["position_distance_m"], 0.05)
        self.assertAlmostEqual(
            second["reference_start_deviation"]["orientation_angle_rad"], math.pi
        )
        self.assertEqual(second["reference_start_deviation"]["orientation_policy"], "record-only")
        sidecar = (self.folder / second["raw_file"]).with_suffix(".start.json")
        self.assertEqual(
            json.loads(sidecar.read_text())["reference_start_deviation"],
            second["reference_start_deviation"],
        )
        self.assertEqual(len(demo.accepted_records(self.folder)), 2)

    def test_observe_checks_lease_bounds_speed_and_sensor(self):
        self.client.lease = True
        self.client.state.controller_state = "gravity_comp"
        with patch.object(demo, "read_force", return_value={}) as ft:
            demo.observe(self.client, self.sensor, self.cfg)
            ft.assert_called_once()
            self.client.joints.velocities[0] = 0.6
            with self.assertRaises(demo.StateError):
                demo.observe(self.client, self.sensor, self.cfg)
            self.client.joints.velocities[0] = 0
            self.client.pose.position[0] = 2
            demo.observe(self.client, self.sensor, self.cfg)
            self.client.joints.angles[0] = 3
            with self.assertRaises(demo.StateError):
                demo.observe(self.client, self.sensor, self.cfg)
            self.client.lease = False
            with self.assertRaises(demo.StateError):
                demo.observe(self.client, self.sensor, self.cfg)

    def test_guarded_demo_ignores_legacy_workspace_limits(self):
        self.cfg.update(workspace_min_m=[0, 0, 0], workspace_max_m=[0.01] * 3)
        demo.validate_config(self.cfg)
        self.client.lease = True
        self.client.state.controller_state = "gravity_comp"
        with patch.object(demo, "read_force", return_value={}) as force:
            result = demo.observe(self.client, self.sensor, self.cfg)
        self.assertEqual(result["pose"]["sdk_end_position_m"], [0.25, 0.02, 0.15])
        force.assert_called_once()


if __name__ == "__main__":
    unittest.main()
