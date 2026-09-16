"""Temporary numeric fixtures only; no physical calibration is generated or claimed."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.real_deploy.airbot_deploy import (
    load_exploration,
    replay,
    validate_setup,
)
from scripts.real_training.airbot_calibrated_data import (
    convert_episode,
    convert_exploration,
    convert_training,
)
from scripts.real_training.config import load_config
from scripts.real_training.data import PROTOCOL, _load_raw, inspect
from scripts.shared.airbot_calibration import (
    FRAME_KEYS,
    Calibration,
    checked_received,
)
from scripts.shared.common import ROOT, file_digest, write_json
from scripts.shared.real_preprocessing import CausalFTFilter


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="UNIT_TEST_ONLY_calibration-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "UNIT_TEST_ONLY_calibration.json"
        self.evidence = self.root / "UNIT_TEST_ONLY_evidence.json"
        write_json(self.evidence, {"fixture": "synthetic numeric test; NOT physical evidence"})
        self.record = json.loads(
            (ROOT / "configs/robot_control/airbot_calibration.json").read_text()
        )
        self.record.update(
            calibration_id="UNIT_TEST_ONLY_not_physical_calibration",
            method_notes="Synthetic test fixture",
            evidence=[{"path": self.evidence.name, "sha256": file_digest(self.evidence)}],
        )
        self.record["verification"] = {key: True for key in self.record["verification"]}
        self.record["sensor_to_ft_frame"] = np.eye(4).tolist()
        self.record["sensor_to_ft_frame"][0][3] = 0.01
        self.record["sensor_bias_si"] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        write_json(self.path, self.record)

    def calibration(self):
        return Calibration(self.path)

    def test_sensor_contract_does_not_require_or_transform_robot_pose(self):
        cal = self.calibration()
        self.assertEqual(set(FRAME_KEYS), {"sensor_to_ft_frame", "sensor_bias_si"})
        self.assertNotIn("tcp_definition", cal.metadata())
        self.assertEqual(cal.metadata()["position_frame"], "SDK configured reference")
        self.assertFalse(hasattr(cal.frames, "to_tcp"))
        self.assertFalse(hasattr(cal.frames, "to_end"))

    def test_old_pose_calibration_schema_is_not_reinterpreted_as_sensor_only(self):
        self.record.update(schema_version=1, artifact_kind="airbot_physical_calibration")
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            self.calibration()

    def require_collection(self):
        # Archive integration fixtures retain their original 10 mm/s protocol.
        protocol = patch.dict(PROTOCOL, press_speed_m_s=0.01)
        protocol.start()
        self.addCleanup(protocol.stop)
        exp = ROOT / "archive/real_training/real_robot/exploration_ft_007.jsonl"
        session = (
            ROOT / "archive/real_training/raw_data/manual_demonstrations/session_record_only_002"
        )
        if not exp.exists() or not session.exists():
            self.skipTest("Original real collection is not installed")
        return exp, session

    def test_unfilled_template_is_rejected_before_output(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_airbot_calibrated.yaml")
        cfg["raw_data"] = str(self.root / "not_created" / "raw.h5")
        with self.assertRaisesRegex(ValueError, "Calibration incomplete"):
            convert_training(
                cfg, ROOT / "configs/robot_control/airbot_calibration.json", "missing", "missing"
            )
        self.assertFalse((self.root / "not_created").exists())

    def test_synthetic_calibration_marker_is_rejected(self):
        self.record["source_kind"] = "synthetic"
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "source_kind"):
            self.calibration()

    def test_verification_flags_require_boolean_true(self):
        self.record["verification"]["electronic_bias_verified"] = 1
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "electronic_bias_verified"):
            self.calibration()

    def test_changed_evidence_or_record_is_rejected(self):
        cal = self.calibration()
        write_json(self.evidence, {"fixture": "changed"})
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            cal.assert_unchanged()
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.calibration()

    def test_missing_empty_evidence_and_reflection_are_rejected(self):
        for bad in ([], [{"path": "missing", "sha256": "0" * 64}]):
            record = copy.deepcopy(self.record)
            record["evidence"] = bad
            write_json(self.path, record)
            with self.assertRaises(ValueError):
                self.calibration()
        self.record["sensor_to_ft_frame"][0][0] = -1
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "right-handed"):
            self.calibration()

    def test_sensor_rotation_and_wrench_shift_without_pose_calibration(self):
        self.record["sensor_to_ft_frame"][:3] = np.c_[
            Rotation.from_euler("z", 90, degrees=True).as_matrix(), [0.01, 0, 0]
        ].tolist()
        write_json(self.path, self.record)
        cal = self.calibration()
        np.testing.assert_allclose(
            cal.frames.wrench([3, 2, 3, 0.1, 0.2, 0.3]), [0, 2, 0, 0, 0, 0.02], atol=1e-12
        )

    def test_received_hold_is_causal_and_rejects_gaps(self):
        times = np.arange(-1, 227) * 0.018
        values = np.tile(np.arange(len(times))[:, None], (1, 6))
        grid, held = checked_received(times, values, 4)
        altered = values.copy()
        altered[times > 2] += 999
        _, after = checked_received(times, altered, 4)
        np.testing.assert_array_equal(held[grid <= 2], after[grid <= 2])
        self.assertEqual(held.shape, (401, 6))
        with self.assertRaisesRegex(ValueError, "20 ms"):
            checked_received(np.delete(times, 80), np.delete(values, 80, axis=0), 4)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            checked_received(times[::-1], values[::-1], 4)

    def test_conversion_preserves_original_samples_and_applies_bias_once(self):
        time = np.arange(-1, 558) * 0.018 + 10
        raw = np.tile([1, 4, 3, 0.1, 0.2, 0.3], (len(time), 1))
        episode = {
            "attrs": {"start_time": 10, "ft_hz": 1 / 0.018},
            "ft_time": time,
            "ft": raw,
            "sdk_end_position": [[0.1, 0.2, 0.3]] * 1003,
            "sdk_end_quaternion": [[0, 0, 0, 1]] * 1003,
        }
        out = convert_episode(episode, self.calibration(), demo=True)
        self.assertEqual(out["ft"].shape, (1001, 6))
        self.assertEqual(out["attrs"]["received_ft_hz"], 1 / 0.018)
        np.testing.assert_array_equal(out["received_sensor_ft"], raw)
        np.testing.assert_array_equal(out["received_ft_time"], time)
        np.testing.assert_array_equal(episode["ft"], raw)
        np.testing.assert_allclose(out["ft"][0], [0, 2, 0, 0, 0, 0.02], atol=1e-12)
        np.testing.assert_array_equal(out["sdk_end_position"], episode["sdk_end_position"])
        np.testing.assert_array_equal(out["sdk_end_quaternion"], episode["sdk_end_quaternion"])
        self.assertNotIn("tcp_position", out)

    def test_deployment_rejects_all_frame_and_bias_overrides(self):
        cal = self.calibration()
        meta = cal.metadata()
        cfg = {key: None for key in FRAME_KEYS}
        cal.bind_deployment(cfg, meta)
        self.assertEqual(cfg["sensor_bias_si"], self.record["sensor_bias_si"])
        for key in FRAME_KEYS:
            changed = copy.deepcopy(cfg)
            if key == "sensor_bias_si":
                changed[key][0] += 0.1
            else:
                changed[key][0][3] += 0.001
            with self.assertRaisesRegex(ValueError, "override differs"):
                cal.bind_deployment(changed, meta)
        meta["calibration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "calibration_sha256"):
            cal.bind_deployment({}, meta)

    def test_training_converter_in_temporary_directory(self):
        exp, session = self.require_collection()
        before = file_digest(exp)
        cfg = load_config(ROOT / "configs/real_training/real_training_airbot_calibrated.yaml")
        cfg.update(raw_data=str(self.root / "calibrated.h5"), output_dir=str(self.root / "models"))
        report = convert_training(cfg, self.path, exp, session)
        arrays, info = _load_raw(cfg)
        self.assertEqual(arrays["exploration"].shape, (1, 400, 6))
        self.assertEqual(arrays["xy"].shape, (8, 25, 2))
        self.assertEqual(report["valid_windows_total"], 160)
        self.assertFalse(info["metadata"]["sampling"]["independent_100hz_acquisition"])
        self.assertEqual(info["metadata"]["calibration_sha256"], file_digest(self.path))
        with h5py.File(cfg["raw_data"], "r") as h5:
            demo = h5["demonstrations/demo_01"]
            self.assertNotIn("tcp_position", demo)
            self.assertIn("sdk_end_position", demo)
            self.assertEqual(demo["ft"].shape[0], 1001)
            self.assertEqual(demo["received_sensor_ft"].shape[0], 570)
            self.assertTrue(50 < demo.attrs["received_ft_hz"] < 60)
        self.assertEqual(file_digest(exp), before)
        with self.assertRaises(FileExistsError):
            convert_training(cfg, self.path, exp, session)

    def test_exploration_converter_and_deployment_reconstruction(self):
        exp, _ = self.require_collection()
        target = self.root / "task.npz"
        convert_exploration(self.path, exp, target, audit_only=True)
        self.assertFalse(target.exists())
        convert_exploration(self.path, exp, target)
        policy = SimpleNamespace(
            metadata={"data": {"metadata": self.calibration().metadata()}},
            encoder=SimpleNamespace(
                preprocessor=SimpleNamespace(transform=lambda x: np.zeros_like(x))
            ),
            encode_exploration=lambda x: np.zeros((len(x), 5)),
        )
        np.testing.assert_array_equal(load_exploration(target, policy), np.zeros((1, 5)))
        with self.assertRaises(FileExistsError):
            convert_exploration(self.path, exp, target)
        with np.load(target, allow_pickle=False) as data:
            contents = {key: data[key].copy() for key in data.files}
        contents["ft"][30, 0] += 1
        corrupt = self.root / "corrupt.npz"
        np.savez_compressed(corrupt, **contents)
        with self.assertRaisesRegex(ValueError, "original received observations"):
            load_exploration(corrupt, policy)

    def test_wrong_robot_identity_does_not_publish(self):
        exp, _ = self.require_collection()
        self.record["robot_id"] = "WRONG_ROBOT"
        write_json(self.path, self.record)
        target = self.root / "wrong" / "task.npz"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            convert_exploration(self.path, exp, target)
        self.assertFalse(target.parent.exists())

    def test_calibrated_replay_has_no_second_frame_conversion(self):
        exp, session = self.require_collection()
        cfg = load_config(ROOT / "configs/real_training/real_training_airbot_calibrated.yaml")
        cfg.update(raw_data=str(self.root / "calibrated.h5"), output_dir=str(self.root / "models"))
        convert_training(cfg, self.path, exp, session)
        arrays, info = inspect(cfg)
        info["sha256"] = "UNIT_TEST_ONLY_prepared_binding"
        write_json(
            Path(cfg["output_dir"]) / "policy.pt",
            {"fixture": "Not a real model; loader mocked below"},
        )
        policy = SimpleNamespace(
            metadata={"bindings": {"prepared_sha256": info["sha256"]}},
            warnings=[],
            predict_xy=lambda z: np.broadcast_to(arrays["xy"].mean(axis=0), (len(z), 25, 2)),
            make_ft_filter=lambda hz: CausalFTFilter(hz),
            predict_delta_h=lambda z, ft: np.sum(ft, axis=(1, 2))[:, None] * 0.001,
        )
        with (
            patch(
                "scripts.real_deploy.airbot_deploy.load_prepared",
                return_value=(arrays, info),
            ),
            patch("scripts.real_deploy.airbot_deploy.OfflinePolicy", return_value=policy),
        ):
            report = replay(cfg, self.root / "replay")
        self.assertEqual(report["max_stream_vs_batch_error_m"], 0)
        self.assertEqual(report["position_frame"], "SDK configured reference")
        self.assertEqual(report["calibration_sha256"], file_digest(self.path))

    def test_calibrated_deployment_setup_positive_and_rate_mismatch(self):
        cal = self.calibration()
        cfg = json.loads((ROOT / "configs/real_deploy/airbot_deployment.json").read_text())
        policy_file = self.root / "TEST_ONLY_policy.json"
        write_json(policy_file, {"fixture": "not an actual policy"})
        prepared = self.root / "TEST_ONLY_prepared.h5"
        with h5py.File(prepared, "w") as h5:
            h5.create_dataset("ft_hz", data=np.full(8, 100.0))
        cfg.update(
            policy=str(policy_file),
            policy_sha256=file_digest(policy_file),
            prepared_data=str(prepared),
            calibration_id=cal.record["calibration_id"],
            calibration_record=str(self.path),
            joint_current_limits=[1] * 6,
            joint_speed_limit_rad_s=0.1,
            measured_joint_speed_stop_rad_s=0.2,
            initial_sdk_position_m=[0.1, 0.2, 0.3],
            sdk_end_orientation_xyzw=[0, 0, 0, 1],
        )
        for key in (
            "commissioning_verified",
            "physical_estop_verified",
            "server_no_return_verified",
            "tool_and_swept_path_verified",
            "training_frames_verified",
        ):
            cfg[key] = True
        for key in (
            "max_force_n",
            "max_initial_force_n",
            "max_torque_nm",
            "max_cartesian_speed_m_s",
            "max_delta_h_m",
            "max_tracking_error_m",
            "max_start_error_m",
            "orientation_error_limit_rad",
        ):
            cfg[key] = 0.01
        policy = SimpleNamespace(
            source_kind="real",
            warnings=[],
            metadata={
                "data": {"metadata": cal.metadata()},
                "bindings": {"prepared_sha256": file_digest(prepared)},
                "training_contract": {"profile": "airbot_sensor_calibrated_offline"},
            },
        )
        frames = validate_setup(policy, cfg)
        np.testing.assert_array_equal(frames.bias, self.record["sensor_bias_si"])
        with h5py.File(prepared, "r+") as h5:
            h5["ft_hz"][0] = 200
        policy.metadata["bindings"]["prepared_sha256"] = file_digest(prepared)
        with self.assertRaisesRegex(ValueError, "100 Hz adapter"):
            validate_setup(policy, cfg)


if __name__ == "__main__":
    unittest.main()
