"""Hold-last conversion is explicit, bounded to the tail, and auditable."""

import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.real_training.airbot_programmed_demonstrations import DEPTHS, duration
from scripts.real_training.config import load_config
from scripts.real_training.data import _load_raw, encoder_source, load_prepared, prepare, write_raw_log
from scripts.real_training.import_airbot import assemble
from scripts.real_training.programmed_padding import padded_episode
from scripts.shared.common import ROOT


class PaddingTests(unittest.TestCase):
    def fixture(self, condition):
        cfg = {"sponge_id": "normal", "surface_id": "surface", "exploration_id": "exp"}
        record = {"condition": condition, "initial_depth_m": DEPTHS[condition], "quality": {"passed": True}}
        rows = [{"event": "episode_start", "start_perf_s": 100.0}]
        for i in range(round(duration(condition) * 100) + 1):
            rows.append({"event": "sample", "phase": "press" if i < 160 else "slide", "index": i,
                         "pose": {"host_monotonic_s": 100 + i / 100,
                                  "sdk_end_position_m": [i / 1000, 0, 0.15],
                                  "sdk_end_orientation_xyzw": [0, 0, 0, 1]},
                         "ft": {"sensor_receive_perf_s": 100 + i / 100,
                                "raw_sensor_wrench_si": [i / 100, 0, 20, 0, 0, 0]}})
        return rows, record, cfg

    def test_three_durations_constant_tail_and_exact_10s_grid(self):
        for condition, padded in (("under", 440), ("nominal", 400), ("over", 360)):
            with self.subTest(condition=condition):
                rows, record, cfg = self.fixture(condition)
                end = rows[-1]
                excluded = copy.deepcopy(end)
                excluded.update(phase="retract")
                excluded["pose"]["sdk_end_position_m"] = [999] * 3
                episode, report = padded_episode(rows + [excluded], record, cfg)
                self.assertEqual(len(episode["ft_time"]), 1001)
                self.assertEqual(episode["ft_time"][-1], 10)
                self.assertEqual(report["padded_samples"], padded)
                mask = episode["is_padding"].astype(bool)
                self.assertEqual(mask.sum(), padded)
                for key, expected in (("ft", end["ft"]["raw_sensor_wrench_si"]),
                                      ("sdk_end_position", end["pose"]["sdk_end_position_m"]),
                                      ("sdk_end_quaternion", end["pose"]["sdk_end_orientation_xyzw"])):
                    np.testing.assert_array_equal(episode[key][mask], np.tile(expected, (padded, 1)))
                np.testing.assert_allclose(episode["sdk_end_position"][~mask, 0], np.arange(1001 - padded) / 1000)
                self.assertTrue(np.all(episode["pose_source_time"][mask] < episode["pose_time"][mask]))

    def test_missing_internal_frames_cannot_be_padded(self):
        rows, record, cfg = self.fixture("nominal")
        del rows[30]
        with self.assertRaisesRegex(ValueError, "Missing"):
            padded_episode(rows, record, cfg)

    def test_internal_time_gap_rejected(self):
        rows, record, cfg = self.fixture("nominal")
        rows[40]["pose"]["host_monotonic_s"] += 0.03
        with self.assertRaisesRegex(ValueError, "Gaps"):
            padded_episode(rows, record, cfg)


class RealPaddingAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = ROOT / "runs/real_demonstrations/programmed/direct_start_session_004"
        cls.exploration = ROOT / "runs/real_exploration/exploration_tared_plot_003.jsonl"
        if not (cls.session / "demo_08.json").exists() or not cls.exploration.exists():
            raise unittest.SkipTest("Completed real program audit fixture not installed")
        cls.meta, cls.exp, cls.demos = assemble(cls.exploration, cls.session, programmed_hold_last=True)

    def test_old_importer_still_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "Variable-length"):
            assemble(self.exploration, self.session)

    def test_training_shapes_and_padding_provenance(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_programmed_hold_last.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.h5"
            write_raw_log(path, self.meta, self.exp, self.demos, source_kind="real")
            arrays, info = _load_raw({**cfg, "raw_data": str(path)})
            self.assertEqual(arrays["ft"].shape, (8, 25, 6))
            self.assertEqual(arrays["is_padding"].shape, (8, 25))
            self.assertTrue(info["metadata"]["contains_derived_samples"])
            self.assertFalse(info["metadata"]["paper_equivalent_collection"])
            for h, mask in zip(arrays["height"], arrays["is_padding"].astype(bool)):
                self.assertTrue(np.all(h[mask] == h[mask][0]))
            with h5py.File(path, "r+") as h5:
                h5["demonstrations/demo_01/is_padding"][-1] = 0
            with self.assertRaisesRegex(ValueError, "padding mask"):
                _load_raw({**cfg, "raw_data": str(path)})

    def test_modified_tail_rejected(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_programmed_hold_last.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.h5"
            write_raw_log(path, self.meta, self.exp, self.demos, source_kind="real")
            with h5py.File(path, "r+") as h5:
                h5["demonstrations/demo_01/ft"][-1, 0] += 1
            with self.assertRaisesRegex(ValueError, "constant"):
                _load_raw({**cfg, "raw_data": str(path)})

    def test_prepared_data_loads_for_training_with_mask(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_programmed_hold_last.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.h5"
            write_raw_log(path, self.meta, self.exp, self.demos, source_kind="real")
            cfg = {**cfg, "raw_data": str(path), "output_dir": str(Path(temp) / "prepared")}
            prepare(cfg)
            arrays, info = load_prepared(cfg)
            self.assertEqual(arrays["is_padding"].shape, (8, 25))
            self.assertIn("PROGRAMMED_HOLD_LAST_TAIL_IS_DERIVED_NOT_MEASURED", info["warnings"])

    def test_new_tared_profile_uses_recorded_baselines_without_axis_flip(self):
        meta, exp, demos = assemble(self.exploration, self.session, programmed_hold_last=True,
                                    subtract_recorded_baseline=True)
        for name, episode in demos.items():
            np.testing.assert_allclose(episode["ft"], self.demos[name]["ft"] - meta["recorded_baselines_si"][name])
            np.testing.assert_array_equal(episode["ft_raw_before_baseline"], self.demos[name]["ft"])
            np.testing.assert_array_equal(episode["sdk_end_position"], self.demos[name]["sdk_end_position"])
        cfg = load_config(ROOT / "configs/real_training/real_training_programmed_wide1200.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tared.h5"
            write_raw_log(path, meta, exp, demos, source_kind="real")
            arrays, _ = _load_raw({**cfg, "raw_data": str(path)})
            self.assertEqual(arrays["ft"].shape, (8, 25, 6))
            old = load_config(ROOT / "configs/real_training/real_training_programmed_hold_last.yaml")
            with self.assertRaisesRegex(ValueError, "compensation"):
                _load_raw({**old, "raw_data": str(path)})

    def test_encoder_frames_remain_profile_specific(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_programmed_wide1200.yaml")
        if not (ROOT / cfg["encoder"]).exists():
            self.skipTest("New exported encoder not installed")
        _, payload, _ = encoder_source(cfg)
        self.assertEqual(payload["frame"], "ft_frame local with output Y/Z reversed")
        old = load_config(ROOT / "configs/real_training/real_training_programmed_hold_last.yaml")
        with self.assertRaisesRegex(ValueError, "frame"):
            encoder_source({**old, "encoder": cfg["encoder"]})
        with self.assertRaisesRegex(ValueError, "frame"):
            encoder_source({**cfg, "encoder": old["encoder"]})
