"""Compare current numerical models with immutable pre-refactor source evidence."""

import importlib.util
import unittest

import torch

from scripts.shared import model, real_models
from scripts.shared.paths import read_path


def historical_module(name, path):
    spec = importlib.util.spec_from_file_location(name, read_path(path, historical=True))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelParityTests(unittest.TestCase):
    def test_vae_outputs_and_loss_match_snapshot(self):
        old = historical_module("historical_vae", "adaptive_wiping/model.py")
        torch.manual_seed(173)
        current, previous = model.SpongeVAE(), old.SpongeVAE()
        previous.load_state_dict(current.state_dict(), strict=True)
        inputs = torch.linspace(-1, 1, 4800).reshape(2, 400, 6)
        for training in (False, True):
            with self.subTest(training=training), torch.no_grad():
                current.train(training)
                previous.train(training)
                torch.manual_seed(174)
                expected = previous(inputs, sample=training)
                torch.manual_seed(174)
                actual = current(inputs, sample=training)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                actual_loss = model.vae_loss(actual[0], inputs, *actual[1:])
                expected_loss = old.vae_loss(expected[0], inputs, *expected[1:])
                for a, b in zip(actual_loss, expected_loss):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_real_model_outputs_match_snapshot(self):
        old = historical_module("historical_real", "adaptive_wiping/real_training/models.py")
        sponge = torch.linspace(-1, 1, 10).reshape(2, 5)
        history = torch.linspace(-2, 2, 60).reshape(2, 5, 6)
        for name, args in (
            ("XYDecoder", (sponge,)),
            ("FTEncoder", (history,)),
            ("HeightFeedback", (sponge, history)),
        ):
            torch.manual_seed(175)
            current, previous = getattr(real_models, name)(), getattr(old, name)()
            previous.load_state_dict(current.state_dict(), strict=True)
            for training in (False, True):
                with self.subTest(model=name, training=training), torch.no_grad():
                    current.train(training)
                    previous.train(training)
                    torch.manual_seed(176)
                    expected = previous(*args)
                    torch.manual_seed(176)
                    actual = current(*args)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
