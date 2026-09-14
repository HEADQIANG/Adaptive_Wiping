"""Synthetic software checks, not evidence of real-robot model quality."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.repair_sponge_vae import (
    RepairedVAE,
    export_encoder,
    initialize,
    mse,
    restore,
    run,
    train_seed,
)


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(5)
        codes = rng.normal(size=(20, 5))
        basis = rng.normal(size=(5, 400, 5))
        projection = rng.normal(size=(5, 6))
        self.x = (0.05 * np.einsum("ni,itc,cd->ntd", codes, basis, projection)).astype(np.float32)

    def test_initialization_reconstruction_and_latent_use(self):
        model = RepairedVAE().eval()
        info = initialize(model, self.x)
        self.assertEqual(info["fitting_split"], "train")
        self.assertLess(mse(model, self.x), 1e-10)
        x = torch.from_numpy(self.x)
        a, mu, _ = model(x, sample=False)
        b = model(x, sample=True)[0]
        self.assertFalse(torch.equal(a, b))
        self.assertGreater(float((model.decode(mu.flip(0)) - x).square().mean().detach()), 0.001)

    def test_export_compatibility_and_failure_gate(self):
        prep = Preprocessor().fit(self.x)
        model = RepairedVAE().eval()
        initialize(model, prep.transform(self.x))
        metric = {
            "passed": True,
            "mse": 0.001,
            "kl": 1.0,
            "latent_variance": [1.0] * 5,
            "shuffled_latent_mse_mean": 0.01,
            "reconstruction_pass": True,
            "latent_utilization_pass": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "encoder.pt"
            with self.assertRaises(ValueError):
                export_encoder(model, prep, {}, "hash", {"passed": False}, metric, 50, path)
            self.assertFalse(path.exists())
            export_encoder(model, prep, {}, "hash", metric, metric, 50, path)
            frozen = FrozenSpongeEncoder(path)
            expected = model.encoder(torch.from_numpy(prep.transform(self.x)))[0].detach().numpy()
            np.testing.assert_array_equal(frozen.encode(self.x), expected)
            self.assertFalse(any(p.requires_grad for p in frozen.model.parameters()))

    def test_rank_and_finite_validation(self):
        for x in (np.zeros((10, 400, 6)), np.full((10, 400, 6), np.nan), self.x[:3]):
            with self.assertRaises(ValueError):
                initialize(RepairedVAE(), x)

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "scripts.sim_pretrain.experiments.repair_sponge_vae.load_config",
                return_value={},
            ):
                with self.assertRaises(FileExistsError):
                    run("unused", folder)

    def test_training_checkpoint_selection_starts_at_epoch_50(self):
        prep = Preprocessor().fit(self.x)
        x = prep.transform(self.x)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            with patch("builtins.print"):
                train_seed(x[:16], x[16:], prep, out, 42, epochs=50)
            model, saved = restore(out / "best.pt")
            self.assertEqual(saved["epoch"], 50)
            self.assertTrue(np.isfinite(mse(model, x)))
            self.assertTrue((out / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
