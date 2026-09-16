"""Checks for information-preserving probes and posterior-noise attribution."""

import unittest

import numpy as np
import torch

from scripts.sim_pretrain.experiments.diagnose_latent_collapse import (
    SharedDecoder, noise_optimal_decoder, readout,
)
from scripts.sim_pretrain.experiments.diagnose_original_vae_beta import informative_original
from scripts.shared.model import SpongeVAE


class LatentDiagnosisTests(unittest.TestCase):
    def test_original_decoder_transfer_preserves_predictions_and_dropout(self):
        torch.manual_seed(9)
        rng = np.random.default_rng(9)
        codes = rng.normal(size=(40, 5))
        converted = codes @ rng.normal(size=(5, 5)) + rng.normal(size=5)
        decoder = SharedDecoder().eval()
        source_encoder = SpongeVAE().encoder
        model, error = informative_original(source_encoder.state_dict(), decoder.state_dict(),
                                            codes, converted)
        model.eval()
        self.assertLess(error, 1e-20)
        self.assertEqual(model.decode_hidden[2].p, .1)
        with torch.no_grad():
            expected = decoder(torch.tensor(converted, dtype=torch.float32))
            actual = model.decode(torch.tensor(codes, dtype=torch.float32))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
        for k, v in source_encoder.state_dict().items():
            self.assertTrue(torch.equal(v, model.encoder.state_dict()[k]))

    def test_tiny_codes_retain_information_and_test_does_not_select(self):
        rng = np.random.default_rng(8)
        features = {s: rng.normal(size=(n, 5)) * 1e-5
                    for s, n in [("train", 120), ("validation", 30), ("test", 30)]}
        weight = rng.normal(size=(5, 24)) * 1e4
        targets = {s: z @ weight + 2.0 for s, z in features.items()}
        sc, center, fitted, selection = readout(features, targets)
        np.testing.assert_allclose(sc.transform(features["validation"]) @ fitted + center,
                                   targets["validation"], atol=1e-10)
        features["test"][:] = np.nan
        targets["test"][:] = np.nan
        sc2, center2, fitted2, selection2 = readout(features, targets)
        self.assertEqual(selection, selection2)
        np.testing.assert_array_equal(fitted, fitted2)
        np.testing.assert_array_equal(sc.mean, sc2.mean)
        np.testing.assert_array_equal(center, center2)

    def test_noise_optimum_matches_monte_carlo_and_rejects_template_collapse(self):
        rng = np.random.default_rng(4)
        z = rng.normal(size=(150, 5))
        target = z @ rng.normal(size=(5, 8))
        lv = np.zeros_like(z)
        zm, ym, weight = noise_optimal_decoder(z, lv, target)
        expected = np.mean(((z - zm) @ weight + ym - target) ** 2) + np.sum(weight ** 2) / 8
        samples = z[None] + rng.normal(size=(3000, *z.shape))
        measured = np.mean(((samples - zm) @ weight + ym - target) ** 2)
        self.assertAlmostEqual(expected, measured, delta=0.025)
        _, _, suppressed = noise_optimal_decoder(z * 1e-4, lv, target)
        self.assertLess(np.linalg.norm(suppressed), np.linalg.norm(weight) * .001)


if __name__ == "__main__":
    unittest.main()
