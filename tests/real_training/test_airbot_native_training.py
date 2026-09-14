"""Native received-data semantics and calibrated-profile isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.real_training.config import load_config, validate_config
from scripts.real_training.data import PROTOCOL, _load_raw, write_raw_log
from scripts.real_training.import_airbot import (
    assemble,
    events,
    exploration_episode,
    received_stream,
)
from scripts.shared.common import ROOT
from scripts.shared.real_preprocessing import CausalFTFilter
from scripts.shared.sampling import previous_samples


class NativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/real_training/real_training_airbot_native.yaml")
        cls.exp_path = ROOT / "archive/real_training/real_robot/exploration_ft_007.jsonl"
        cls.session = (
            ROOT / "archive/real_training/raw_data/manual_demonstrations/session_record_only_002"
        )
        if not cls.exp_path.is_file() or not cls.session.is_dir():
            raise unittest.SkipTest("Real collection audit fixtures are not installed")
        # These read-only archive audits retain their recorded protocol. Current
        # imports reject that protocol, covered separately below.
        with patch.dict(PROTOCOL, press_speed_m_s=0.01):
            cls.meta, cls.exp, cls.demos = assemble(cls.exp_path, cls.session)
            cls.meta = copy.deepcopy(cls.meta)

    def setUp(self):
        protocol = patch.dict(PROTOCOL, press_speed_m_s=0.01)
        protocol.start()
        self.addCleanup(protocol.stop)

    def raw_fixture(self):
        temp = tempfile.TemporaryDirectory(prefix="native-data-test-")
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "raw.h5"
        write_raw_log(path, self.meta, self.exp, self.demos, source_kind="real")
        return path, {**self.cfg, "raw_data": str(path)}

    def test_real_sources_preserved_and_not_claimed_calibrated(self):
        self.assertEqual(len(self.demos), 8)
        self.assertEqual(len(self.exp), 1)
        self.assertEqual(len(self.meta["source_hashes"]), 18)
        self.assertEqual(self.meta["calibration_status"], "unverified")
        self.assertNotIn("sensor_to_ft_frame", self.meta)
        self.assertTrue(all(50 < e["attrs"]["ft_hz"] < 60 for e in self.demos.values()))

    def test_complete_import_shapes_and_causal_filter_parity(self):
        _, cfg = self.raw_fixture()
        arrays, info = _load_raw(cfg)
        self.assertEqual(arrays["ft"].shape, (8, 25, 6))
        self.assertEqual(arrays["exploration"].shape, (1, 400, 6))
        self.assertEqual(info["source_kind"], "real")
        demo = self.demos["demo_01"]
        t = demo["ft_time"] - demo["attrs"]["start_time"]
        held = previous_samples(t, demo["ft"], np.arange(1001) / 100)
        causal = CausalFTFilter()
        streamed = np.array([causal.push(x) for x in held])
        np.testing.assert_allclose(arrays["ft"][0], streamed[np.arange(40, 1001, 40)], atol=1e-12)

    def test_native_data_cannot_enter_calibrated_profile(self):
        _, cfg = self.raw_fixture()
        paper = load_config(ROOT / "configs/real_training/real_training_paper.yaml")
        paper["raw_data"] = cfg["raw_data"]
        with self.assertRaisesRegex(ValueError, "metadata.position_frame"):
            _load_raw(paper)

    def test_profile_preserves_epoch_and_compensation_gates(self):
        for key, value in (("xy_epochs", 1), ("ft_epochs", 1)):
            cfg = copy.deepcopy(self.cfg)
            cfg["training"][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["wrench"]["compensation"] = "sensor_bias_only_gravity_retained"
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_deduplication_and_conflicting_timestamp_rejected(self):
        a = {"sensor_receive_perf_s": 1, "raw_sensor_wrench_si": [1] * 6}
        b = {"sensor_receive_perf_s": 1.018, "raw_sensor_wrench_si": [2] * 6}
        t, f = received_stream([a, a, b, b])
        self.assertEqual(len(t), 2)
        np.testing.assert_array_equal(f[:, 0], [1, 2])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            received_stream([a, {**a, "raw_sensor_wrench_si": [2] * 6}, b])

    def test_aborted_exploration_rejected(self):
        rows = events(self.exp_path)
        rows[-1] = {"event": "aborted"}
        with patch("scripts.real_training.import_airbot.events", return_value=rows):
            with self.assertRaisesRegex(ValueError, "completed"):
                exploration_episode(self.exp_path)

    def test_stale_data_not_padded_or_timestamp_rewritten(self):
        path, cfg = self.raw_fixture()
        with h5py.File(path, "r+") as h5:
            t = h5["demonstrations/demo_01/ft_time"]
            t[20] = t[21] - 0.0001
        with self.assertRaisesRegex(ValueError, "gap exceeding"):
            _load_raw(cfg)

    def test_synthetic_provenance_rejected(self):
        path, cfg = self.raw_fixture()
        with h5py.File(path, "r+") as h5:
            h5.attrs["source_kind"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "real data"):
            _load_raw(cfg)

    def test_source_hash_mismatch_rejected(self):
        with patch(
            "scripts.real_training.import_airbot.file_digest", return_value="changed"
        ):
            with self.assertRaisesRegex(ValueError, "changed demonstration"):
                assemble(self.exp_path, self.session)


class CurrentExplorationProtocolTests(unittest.TestCase):
    def rows(self, speed=0.005):
        samples = [
            {"event": "sample", "phase": "exploration", "index": i, "due_perf_s": 10 + i / 100}
            for i in range(1, 401)
        ]
        completion = {
            "event": "motion_complete", "nominal_protocol_match": True, "time_scale": 1,
            "press_speed_m_s": speed, "press_duration_s": 2.0,
        }
        return [
            {"event": "header", "source_kind": "real", "mode": "force-guarded"},
            *samples, completion, {"event": "session_complete", "idle_confirmed": True},
        ]

    def test_current_press_import(self):
        times = 10 + np.arange(402) / 100
        with patch("scripts.real_training.import_airbot.events", return_value=self.rows()), patch(
            "scripts.real_training.import_airbot.received_stream",
            return_value=(times, np.zeros((402, 6))),
        ):
            episode, _, _ = exploration_episode("synthetic-test-only.jsonl")
        self.assertEqual(episode["attrs"]["start_time"], 10)
        self.assertEqual(PROTOCOL["press_speed_m_s"], 0.005)

    def test_old_or_unmarked_press_rejected(self):
        for speed in (0.01, None):
            rows = self.rows(speed)
            if speed is None:
                del rows[-2]["press_speed_m_s"]
                del rows[-2]["press_duration_s"]
            with self.subTest(speed=speed), patch(
                "scripts.real_training.import_airbot.events", return_value=rows
            ), self.assertRaisesRegex(ValueError, "press protocol"):
                exploration_episode("synthetic-test-only.jsonl")

    def test_incomplete_or_changed_press_duration_rejected(self):
        for duration in (None, 1.0, 4.0):
            rows = self.rows()
            if duration is None:
                del rows[-2]["press_duration_s"]
            else:
                rows[-2]["press_duration_s"] = duration
            with self.subTest(duration=duration), patch(
                "scripts.real_training.import_airbot.events", return_value=rows
            ), self.assertRaisesRegex(ValueError, "press protocol"):
                exploration_episode("synthetic-test-only.jsonl")


if __name__ == "__main__":
    unittest.main()
