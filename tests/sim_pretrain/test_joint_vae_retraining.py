"""Initialization, schedule, trained checkpoint selection, and FT export contract."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import h5py
import torch
import yaml

from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.common import file_digest
from scripts.shared.model import SpongeVAE
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.retrain_wide_vae import (
    export_encoder, initialize_original, load_model, passed, run, schedule, train_joint, validate_config,
)


class JointRetrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = yaml.safe_load((ROOT / "configs/sim_pretrain/retrain_wide_1200.yaml").read_text())
        rng = np.random.default_rng(15)
        codes = rng.normal(size=(24, 5))
        basis = rng.normal(size=(5, 400, 5))
        projection = rng.normal(size=(5, 6))
        self.raw = (np.einsum("ni,itc,cd->ntd", codes, basis, projection) * .03).astype(np.float32)

    def test_original_architecture_initialization_and_rank_rejection(self):
        model, info = initialize_original(self.raw)
        self.assertIsInstance(model, SpongeVAE)
        self.assertEqual(model.decode_hidden[2].p, 0)
        self.assertEqual(info["fit_split"], "train")
        self.assertLess(info["original_architecture_initialized_mse"], 1e-10)
        self.assertEqual(set(model.state_dict()), set(SpongeVAE().state_dict()))
        with torch.no_grad():
            result, mu, _ = model(torch.from_numpy(self.raw), sample=False)
            shuffled = model.decode(mu.flip(0))
        self.assertGreater(float((shuffled - torch.from_numpy(self.raw)).square().mean()), .001)
        with self.assertRaises(ValueError):
            initialize_original(np.zeros_like(self.raw))

    def test_schedule_reaches_final_beta_and_lr(self):
        t = self.cfg["training"]
        self.assertEqual(schedule(50, 1e-5, t), (0.0, False, 1e-4))
        beta, sample, _ = schedule(51, 1e-5, t)
        self.assertTrue(sample)
        self.assertAlmostEqual(beta, 1e-7)
        self.assertEqual(schedule(150, 1e-5, t)[0], 1e-5)
        self.assertEqual(schedule(400, 1e-5, t), (1e-5, True, 1e-5))
        with self.assertRaises(ValueError):
            schedule(0, 1e-5, t)

    def test_joint_training_selects_only_after_warmup_and_preserves_input(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"].update(epochs=6, deterministic_warmup_epochs=1, kl_ramp_epochs=2,
                               batch_size=8, checkpoint_every=3)
        prep = Preprocessor().fit(self.raw[:18])
        arrays = {"train": prep.transform(self.raw[:18]), "validation": prep.transform(self.raw[18:])}
        model, _ = initialize_original(arrays["train"])
        initial = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "training"
            with patch("builtins.print"):
                result = train_joint(model, arrays, prep, out, cfg, 1e-5, 42, {})
            loaded, saved = load_model(out / "best.pt")
            self.assertGreaterEqual(saved["epoch"], 3)
            self.assertTrue(all(v > 0 for v in result["parameter_update_l2"].values()))
            self.assertTrue((out / "last.pt").is_file())
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, initial[k]))
            self.assertEqual(loaded.decode_hidden[2].p, 0)

    def test_export_keeps_tare_frame_and_exact_frozen_predictions(self):
        prep = Preprocessor().fit(self.raw)
        model, _ = initialize_original(prep.transform(self.raw))
        saved = dict(preprocessing=prep.state(), channels=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            frame="ft_frame local with output Y/Z reversed", units=["N"]*3+["N*m"]*3,
            ft_processing="initial_noncontact_tare_yz_flip_v2", epoch=150,
            data_provenance_hash="fixture", retraining_config={}, source_run="synthetic")
        metric = dict(passed=True, mse=.001, kl=2., baseline_mse=.01,
                      shuffled_latent_mse_mean=.02, latent_variance=[1.]*5)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "encoder.pt"
            with self.assertRaises(ValueError):
                export_encoder(model, saved, {"passed": False}, metric, out)
            self.assertFalse(out.exists())
            export_encoder(model, saved, metric, metric, out)
            encoder = FrozenSpongeEncoder(out)
            self.assertEqual(encoder.metadata["frame"], saved["frame"])
            self.assertEqual(encoder.metadata["ft_processing"], saved["ft_processing"])
            self.assertFalse(encoder.metadata["evaluation"]["collapse_warning"])
            with torch.no_grad():
                expected = model.encoder(torch.from_numpy(prep.transform(self.raw)))[0].numpy()
            np.testing.assert_array_equal(expected, encoder.encode(self.raw))

    def test_bad_config_and_collapsed_export_criteria(self):
        validate_config(self.cfg)
        bad = copy.deepcopy(self.cfg)
        bad["training"]["beta_candidates"] = [0]
        with self.assertRaises(ValueError):
            validate_config(bad)
        metric = dict(mse=.001, baseline_mse=.01, fixed_latent_mse=.001,
                      shuffled_latent_mse_mean=.001)
        self.assertFalse(passed(metric, self.cfg["acceptance"]))

    def test_reorganized_dataset_link_loads_and_still_rejects_hash_mismatch(self):
        class InitializationReached(Exception):
            pass

        prep = Preprocessor().fit(self.raw[:18])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sim_training/fixture"
            source.mkdir(parents=True)
            data = root / "sim_data/fixture/dataset.h5"
            data.parent.mkdir(parents=True)
            with h5py.File(data, "w") as f:
                f.attrs.update(complete=True, config_hash="fixture", ft_processing="fixture")
                for split, samples in (("train", self.raw[:18]),
                                       ("validation", self.raw[18:21]), ("test", self.raw[21:])):
                    group = f.create_group(split)
                    group.create_dataset("ft", data=samples)
                    group.create_dataset("valid", data=np.ones(len(samples), dtype=bool))
            (source / "dataset.h5").symlink_to("../../sim_data/fixture/dataset.h5")
            original_hash = file_digest(data)
            torch.save(dict(data_provenance_hash=original_hash, config_hash="fixture",
                            preprocessing=prep.state(),
                            config={"dataset": {"train": 18, "validation": 3, "test": 3}}),
                       source / "vae_last.pt")
            for name in ("evaluation.json", "preprocessing.json", "history.json"):
                (source / name).write_text("{}")
            cfg = copy.deepcopy(self.cfg)
            cfg["source_run"] = str(source)
            config_path = root / "retrain.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            with patch("scripts.sim_pretrain.experiments.retrain_wide_vae.initialize_original",
                       side_effect=InitializationReached) as initialize:
                with self.assertRaises(InitializationReached):
                    run(cfg, root / "output", config_path)
                np.testing.assert_array_equal(initialize.call_args.args[0], prep.transform(self.raw[:18]))
            self.assertEqual(file_digest(data), original_hash)
            with h5py.File(data, "r+") as f:
                f["train/ft"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "Source checkpoint/data mismatch"):
                run(cfg, root / "rejected", config_path)


if __name__ == "__main__":
    unittest.main()
