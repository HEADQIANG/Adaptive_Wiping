"""Offline-only fixtures; no physical data or robot commands."""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts.real_training.__main__ import main
from scripts.real_training.config import load_config, validate_config
from scripts.real_training.data import (
    inspect,
    load_prepared,
    prepare,
    write_raw_log,
)
from scripts.real_training.testing import synthetic_fixture
from scripts.real_training.training import (
    cross_validate,
    evaluate,
    export,
    fit_scalers,
    train,
)
from scripts.shared.common import ROOT, file_digest
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_models import FTEncoder, HeightFeedback, XYDecoder
from scripts.shared.real_preprocessing import (
    CausalFTFilter,
    ChannelScaler,
    make_windows,
)
from scripts.shared.sampling import previous_samples


class RealTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wiping-real-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.base = load_config(ROOT / "configs/real_training/real_training_paper.yaml")
        self.cfg = synthetic_fixture(self.folder, self.base)
        torch.set_num_threads(1)

    def quiet(self, fn, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def test_paper_configuration_and_real_missing_data_gate(self):
        self.assertEqual(self.base["training"]["xy_epochs"], 10000)
        self.assertEqual(self.base["training"]["ft_epochs"], 2000)
        cfg = copy.deepcopy(self.base)
        cfg["raw_data"] = str(self.folder / "not_present.h5")
        cfg["output_dir"] = str(self.folder / "never_created")
        with self.assertRaisesRegex(FileNotFoundError, "Missing training inputs"):
            inspect(cfg)
        with self.assertRaisesRegex(FileNotFoundError, "Missing inputs"):
            train(cfg)
        self.assertFalse(Path(cfg["output_dir"]).exists())
        cfg["training"]["xy_epochs"] = 1
        with self.assertRaisesRegex(ValueError, "10000"):
            validate_config(cfg)

    def test_raw_contract_shapes_times_and_next_step_targets(self):
        arrays, info = inspect(self.cfg, testing=True)
        self.assertEqual(arrays["exploration"].shape, (1, 400, 6))
        self.assertEqual(arrays["xy"].shape, (8, 25, 2))
        self.assertEqual(info["valid_windows_total"], 160)
        np.testing.assert_allclose(info["policy_time_s"], np.arange(1, 26) * 0.4)
        np.testing.assert_allclose(info["exploration_time_s"], np.arange(1, 401) / 100)
        with h5py.File(self.cfg["raw_data"]) as h5:
            np.testing.assert_array_equal(
                arrays["exploration"][0], h5["explorations/normal_exp/ft"][1:]
            )
            position = h5["demonstrations/demo_01/tcp_position"][:]
            np.testing.assert_array_equal(arrays["xy"][0], position[40::40, :2])
            self.assertEqual(arrays["initial_height"][0], position[0, 2])
        np.testing.assert_allclose(
            arrays["delta_h"],
            np.diff(np.column_stack((arrays["initial_height"], arrays["height"]))),
        )
        windows, targets = make_windows(arrays["ft"], arrays["height"])
        self.assertEqual(windows.shape, (8, 20, 5, 6))
        np.testing.assert_array_equal(windows[:, 0], arrays["ft"][:, :5])
        np.testing.assert_array_equal(windows[:, -1], arrays["ft"][:, 19:24])
        np.testing.assert_array_equal(
            targets[:, 0, 0], arrays["height"][:, 5] - arrays["height"][:, 4]
        )
        np.testing.assert_array_equal(
            targets[:, -1, 0], arrays["height"][:, 24] - arrays["height"][:, 23]
        )

    def test_previous_sample_is_causal_and_rejects_staleness(self):
        value = previous_samples(
            np.array([0, 0.009, 0.021]), np.array([1, 2, 3]), np.array([0.01, 0.02])
        )
        np.testing.assert_array_equal(value, [2, 2])
        with self.assertRaisesRegex(ValueError, "stale"):
            previous_samples(np.array([0, 0.1]), [1, 2], [0.05])
        with self.assertRaisesRegex(ValueError, "No past"):
            previous_samples(np.array([0.001, 0.1]), [1, 2], [0])

    def test_causal_filter_stream_batch_future_and_reset(self):
        rng = np.random.default_rng(1)
        ft = rng.normal(size=(1001, 6))
        batch = CausalFTFilter().process(ft)
        stream_filter = CausalFTFilter()
        stream = np.array([stream_filter.push(row) for row in ft])
        np.testing.assert_array_equal(batch, stream)
        changed = ft.copy()
        changed[501:] += 1000
        np.testing.assert_array_equal(batch[:501], CausalFTFilter().process(changed)[:501])
        stream_filter.reset()
        np.testing.assert_array_equal(stream_filter.process(ft), batch)
        steady = np.tile(np.arange(6), (100, 1))
        np.testing.assert_allclose(CausalFTFilter().process(steady), steady, atol=1e-12)
        chunked = CausalFTFilter()
        np.testing.assert_array_equal(
            np.concatenate((chunked.process(ft[:317]), chunked.process(ft[317:]))), batch
        )

    def test_preparation_filter_state_does_not_cross_episodes(self):
        before, _ = inspect(self.cfg, testing=True)
        with h5py.File(self.cfg["raw_data"], "r+") as h5:
            h5["demonstrations/demo_01/ft"][:] += 500
        after, _ = inspect(self.cfg, testing=True)
        np.testing.assert_array_equal(before["ft"][1:], after["ft"][1:])

    def test_asynchronous_200hz_ft_filters_before_policy_sampling(self):
        time = np.arange(2002) / 200 - 0.0025
        ft = np.column_stack([np.sin(time * (i + 1)) for i in range(6)])
        with h5py.File(self.cfg["raw_data"], "r+") as h5:
            group = h5["demonstrations/demo_01"]
            group.attrs["ft_hz"] = 200.0
            for key, values in (("ft_time", time + 100), ("ft", ft)):
                del group[key]
                group.create_dataset(key, data=values)
        arrays, _ = inspect(self.cfg, testing=True)
        expected = previous_samples(time, CausalFTFilter(200).process(ft), np.arange(1, 26) * 0.4)
        np.testing.assert_array_equal(arrays["ft"][0], expected)

    def test_encoder_preparation_uses_fixed_thread_count(self):
        torch.set_num_threads(2)
        first, _ = inspect(self.cfg, testing=True)
        self.assertEqual(torch.get_num_threads(), 1)
        second, _ = inspect(self.cfg, testing=True)
        np.testing.assert_array_equal(first["sponge"], second["sponge"])

    def test_raw_validation_rejects_invalid_inputs(self):
        path = Path(self.cfg["raw_data"])
        original = path.read_bytes()

        def metadata_change(h5, key, value):
            meta = json.loads(h5.attrs["metadata"])
            meta[key] = value
            h5.attrs["metadata"] = json.dumps(meta)

        def gap(h5):
            group = h5["demonstrations/demo_01"]
            for key in ("ft_time", "ft"):
                data = group[key][:]
                del group[key]
                group.create_dataset(key, data=np.delete(data, np.arange(101, 105), axis=0))

        cases = [
            ("complete", lambda h: h.attrs.__setitem__("complete", False)),
            ("string_complete", lambda h: h.attrs.__setitem__("complete", "false")),
            (
                "episode_string_complete",
                lambda h: h["demonstrations/demo_01"].attrs.__setitem__("complete", "false"),
            ),
            ("units", lambda h: metadata_change(h, "position_units", "mm")),
            ("frame", lambda h: metadata_change(h, "ft_frame", "world")),
            ("calibration", lambda h: metadata_change(h, "calibration_id", "")),
            ("protocol", lambda h: metadata_change(h, "exploration_protocol", {})),
            (
                "reflection",
                lambda h: metadata_change(h, "sensor_to_ft_frame", np.diag([-1, 1, 1, 1]).tolist()),
            ),
            (
                "association",
                lambda h: h["demonstrations/demo_01"].attrs.__setitem__(
                    "exploration_id", "missing"
                ),
            ),
            (
                "surface",
                lambda h: h["demonstrations/demo_01"].attrs.__setitem__("surface_id", "other"),
            ),
            ("nominal_rate", lambda h: h["demonstrations/demo_01"].attrs.__setitem__("ft_hz", 500)),
            ("duplicate_time", lambda h: h["demonstrations/demo_01/ft_time"].__setitem__(1, 100.0)),
            ("nan", lambda h: h["demonstrations/demo_01/ft"].__setitem__((5, 0), np.nan)),
            (
                "orientation",
                lambda h: h["demonstrations/demo_01/tcp_quaternion"].__setitem__((5, 3), 2),
            ),
            ("gap", gap),
            (
                "incomplete_duration",
                lambda h: h["demonstrations/demo_01"].attrs.__setitem__("start_time", 101.0),
            ),
        ]
        for name, change in cases:
            with self.subTest(name=name):
                path.write_bytes(original)
                with h5py.File(path, "r+") as h5:
                    change(h5)
                with self.assertRaises((ValueError, KeyError)):
                    inspect(self.cfg, testing=True)

    def test_synthetic_data_encoder_and_config_cannot_be_used_formally(self):
        formal = copy.deepcopy(self.base)
        formal.update(
            raw_data=self.cfg["raw_data"],
            encoder=self.cfg["encoder"],
            output_dir=self.cfg["output_dir"],
        )
        with self.assertRaisesRegex(ValueError, "synthetic"):
            inspect(formal)
        with self.assertRaisesRegex(ValueError, "profile"):
            inspect(self.cfg)
        with h5py.File(self.cfg["raw_data"], "r+") as h5:
            h5.attrs["source_kind"] = "real"
            for category in ("explorations", "demonstrations"):
                for group in h5[category].values():
                    group.attrs["source_kind"] = "real"
        with self.assertRaisesRegex(ValueError, "Synthetic encoder"):
            inspect(formal)

    def test_preparation_integrity_no_overwrite_and_no_renormalization(self):
        fingerprint = file_digest(self.cfg["encoder"])
        prepare(self.cfg, testing=True)
        arrays, info = load_prepared(self.cfg, testing=True)
        self.assertEqual(fingerprint, file_digest(self.cfg["encoder"]))
        self.assertEqual(info["encoder_sha256"], fingerprint)
        with self.assertRaises(FileExistsError):
            prepare(self.cfg, testing=True)
        with self.assertRaises(FileExistsError):
            write_raw_log(self.cfg["raw_data"], {}, {}, {}, source_kind="synthetic")
        with h5py.File(Path(self.cfg["output_dir"]) / "prepared.h5", "r+") as h5:
            h5["xy"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "content hash"):
            load_prepared(self.cfg, testing=True)

    def test_scalers_are_fold_local_invertible_and_unclipped(self):
        arrays, _ = inspect(self.cfg, testing=True)
        train_indices = list(range(7))
        before = fit_scalers(arrays, train_indices)
        changed = copy.deepcopy(arrays)
        for key in ("xy", "ft", "delta_h"):
            changed[key][7] += 100
        after = fit_scalers(changed, train_indices)
        for key in before:
            self.assertEqual(before[key].state(), after[key].state())
        self.assertTrue(np.any(after["xy"].transform(changed["xy"][[7]]) > 0.9))
        x = np.array([[3.0, 4.0], [3.0, 8.0]])
        scaler = ChannelScaler().fit(x)
        self.assertEqual(scaler.constant.tolist(), [True, False])
        np.testing.assert_allclose(scaler.inverse(scaler.transform(x)), x, atol=1e-6)
        self.assertGreater(scaler.transform(x + 10).max(), 0.9)
        restored = ChannelScaler.from_state(scaler.state())
        np.testing.assert_array_equal(scaler.transform(x), restored.transform(x))

    def test_network_shapes_gradients_and_determinism(self):
        z, ft = torch.randn(3, 5), torch.randn(3, 5, 6)
        xy, feedback = XYDecoder(), HeightFeedback()
        self.assertEqual(xy(z).shape, (3, 25, 2))
        self.assertEqual(FTEncoder()(ft).shape, (3, 6))
        self.assertEqual(feedback(z, ft).shape, (3, 1))
        loss = xy(z).square().mean() + feedback(z, ft).square().mean()
        loss.backward()
        for model in (xy, feedback):
            self.assertTrue(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            )
            model.eval()
        torch.testing.assert_close(xy(z), xy(z), rtol=0, atol=0)
        torch.testing.assert_close(feedback(z, ft), feedback(z, ft), rtol=0, atol=0)

    def test_full_pipeline_cv_isolation_export_and_standalone_inference(self):
        fingerprint = file_digest(self.cfg["encoder"])
        prepare(self.cfg, testing=True)
        arrays, _ = load_prepared(self.cfg, testing=True)
        cv = self.quiet(cross_validate, self.cfg, testing=True)
        self.assertEqual(len(cv["folds"]), 8)
        for i, fold in enumerate(cv["folds"]):
            self.assertNotIn(fold["held_out_demo_id"], fold["train_demo_ids"])
            self.assertEqual(len(fold["train_demo_ids"]), 7)
            expected = fit_scalers(arrays, [j for j in range(8) if j != i])
            self.assertEqual(
                fold["scalers"], {key: value.state() for key, value in expected.items()}
            )
        with (
            patch.object(
                torch.optim.Adam,
                "step",
                side_effect=AssertionError("Completed folds must not retrain"),
            ),
            patch("scripts.real_training.training._plot"),
        ):
            resumed_cv = self.quiet(cross_validate, self.cfg, resume=True, testing=True)
        self.assertEqual(cv["metrics"], resumed_cv["metrics"])
        result = self.quiet(train, self.cfg, testing=True)
        self.assertTrue(result["completed"])
        with self.assertRaises(FileExistsError):
            train(self.cfg, testing=True)
        evaluation = evaluate(self.cfg, testing=True)
        self.assertIn("training_set", evaluation["scope"])
        self.assertIn("leave_one", cv["scope"])
        self.assertTrue(
            all(np.isfinite(value) for m in evaluation["metrics"].values() for value in m.values())
        )
        exported = export(self.cfg, testing=True)
        self.assertTrue(exported["roundtrip_verified"])
        self.assertEqual(file_digest(self.cfg["encoder"]), fingerprint)
        policy = OfflinePolicy(exported["policy"])
        z = policy.encode_exploration(arrays["exploration"])
        self.assertIsNone(policy.predict_delta_h(z, np.zeros((1, 4, 6))))
        xy = policy.predict_xy(z)
        dh = policy.predict_delta_h(z, arrays["ft"][:1, :5])
        Path(self.cfg["encoder"]).rename(self.folder / "moved_source_encoder.pt")
        standalone = OfflinePolicy(exported["policy"])
        np.testing.assert_array_equal(z, standalone.encode_exploration(arrays["exploration"]))
        np.testing.assert_array_equal(xy, standalone.predict_xy(z))
        np.testing.assert_array_equal(dh, standalone.predict_delta_h(z, arrays["ft"][:1, :5]))
        self.assertEqual(standalone.source_kind, "synthetic")
        self.assertFalse(standalone.hardware_ready)
        self.assertTrue(
            all(
                not p.requires_grad and p.grad is None
                for p in standalone.encoder.model.parameters()
            )
        )
        np.testing.assert_array_equal(
            policy.make_ft_filter().process(arrays["exploration"][0]),
            CausalFTFilter().process(arrays["exploration"][0]),
        )
        for invalid in (np.zeros((1, 6, 6)), np.full((1, 5, 6), np.nan)):
            with self.assertRaises(ValueError):
                standalone.predict_delta_h(z, invalid)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_resume_exactly_matches_uninterrupted_across_both_branches(self):
        self.cfg["training"].update(xy_epochs=3, ft_epochs=3)
        prepare(self.cfg, testing=True)
        self.quiet(train, self.cfg, testing=True)
        baseline = torch.load(Path(self.cfg["output_dir"]) / "final/training.pt", weights_only=True)
        for stop in (1, 4):
            with self.subTest(stop_after=stop):
                cfg = copy.deepcopy(self.cfg)
                cfg["output_dir"] = str(self.folder / f"resumed_{stop}")
                prepare(cfg, testing=True)
                interrupted = self.quiet(train, cfg, testing=True, _stop_after=stop)
                self.assertFalse(interrupted["completed"])
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    evaluate(cfg, testing=True)
                resumed = self.quiet(train, cfg, testing=True, resume=True)
                self.assertTrue(resumed["completed"])
                saved = torch.load(Path(cfg["output_dir"]) / "final/training.pt", weights_only=True)
                for key in (
                    "xy",
                    "feedback",
                    "optimizers",
                    "rng",
                    "epochs",
                    "history",
                    "scalers",
                    "encoder",
                ):
                    self.assert_nested_equal(baseline[key], saved[key])

    def test_resume_rejects_changed_configuration_and_source(self):
        prepare(self.cfg, testing=True)
        self.quiet(train, self.cfg, testing=True, _stop_after=1)
        changed = copy.deepcopy(self.cfg)
        changed["training"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            train(changed, testing=True, resume=True)
        with h5py.File(self.cfg["raw_data"], "r+") as h5:
            h5["demonstrations/demo_01/ft"][0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "Raw data changed"):
            train(self.cfg, testing=True, resume=True)

    def test_nonfinite_update_does_not_replace_valid_checkpoint(self):
        prepare(self.cfg, testing=True)
        original_step = torch.optim.Adam.step

        def bad_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
            return result

        with patch.object(torch.optim.Adam, "step", bad_step):
            with self.assertRaisesRegex(RuntimeError, "Non-finite updated"):
                train(self.cfg, testing=True)
        saved = torch.load(Path(self.cfg["output_dir"]) / "final/training.pt", weights_only=True)
        self.assertEqual(saved["epochs"], {"xy": 0, "feedback": 0})
        self.assertTrue(all(torch.isfinite(value).all() for value in saved["xy"].values()))

    def test_stale_evaluation_prevents_export(self):
        prepare(self.cfg, testing=True)
        self.quiet(train, self.cfg, testing=True)
        with self.assertRaisesRegex(RuntimeError, "evaluate"):
            export(self.cfg, testing=True)
        evaluate(self.cfg, testing=True)
        path = Path(self.cfg["output_dir"]) / "final/training.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["xy"]["linear.bias"][0] += 0.01
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "stale"):
            export(self.cfg, testing=True)

    def test_resume_rejects_nonfinite_optimizer_and_invalid_history(self):
        prepare(self.cfg, testing=True)
        self.quiet(train, self.cfg, testing=True, _stop_after=1)
        path = Path(self.cfg["output_dir"]) / "final/training.pt"
        saved = torch.load(path, weights_only=True)
        damaged = copy.deepcopy(saved)
        next(iter(damaged["optimizers"]["xy"]["state"].values()))["exp_avg"].fill_(float("inf"))
        torch.save(damaged, path)
        with self.assertRaisesRegex(ValueError, "non-finite tensors"):
            train(self.cfg, testing=True, resume=True)
        damaged = copy.deepcopy(saved)
        damaged["history"] = []
        torch.save(damaged, path)
        with self.assertRaisesRegex(ValueError, "history"):
            train(self.cfg, testing=True, resume=True)

    def test_exploration_out_of_range_is_reported_without_refitting(self):
        source = torch.load(self.cfg["encoder"], weights_only=True)
        fingerprint = file_digest(self.cfg["encoder"])
        with h5py.File(self.cfg["raw_data"], "r+") as h5:
            h5["explorations/normal_exp/ft"][:] += 100
        arrays, info = inspect(self.cfg, testing=True)
        self.assertEqual(info["exploration_out_of_range_per_channel"], [1.0] * 6)
        self.assertIn(
            "exploration_outside_encoder_normalization_range_not_clipped", info["warnings"]
        )
        self.assertGreater(arrays["exploration"].min(), 90)
        self.assertEqual(fingerprint, file_digest(self.cfg["encoder"]))
        self.assertEqual(
            source["preprocessing"],
            torch.load(self.cfg["encoder"], weights_only=True)["preprocessing"],
        )

    def test_import_and_help_do_not_need_robot_or_simulator_modules(self):
        code = """
import sys
class NoHardware:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('mujoco', 'robosuite', 'rospy', 'rclpy', 'airbot_py', 'airbot_hardware_py'):
            raise RuntimeError('Forbidden hardware import: ' + fullname)
sys.meta_path.insert(0, NoHardware())
from scripts.real_training.__main__ import main
main(['--help'])
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("offline training", result.stdout)


if __name__ == "__main__":
    unittest.main()
