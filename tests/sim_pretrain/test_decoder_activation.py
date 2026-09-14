"""Synthetic paired-decoder checks; never train on or overwrite research artifacts."""

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import ROOT, write_json
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.decoder_activation_experiment import (
    PROTOCOL,
    ActivationVAE,
    activation_diagnostic,
    compare_reproduction,
    continue_after_pair,
    execute,
    initial_state,
    load_activation_model,
    next_stage,
    payload,
    plot_pair,
    state_hash,
    train_activation,
)
from scripts.sim_pretrain.experiments.vae_ablation import (
    CANDIDATES,
    FrozenAblationEncoder,
    export_encoder,
    fit_run,
    run_spec,
    seed_all,
    verify_snapshot,
)


class DecoderActivationTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.raw = np.random.default_rng(2).normal(size=(12, 400, 6))
        self.prep = Preprocessor().fit(self.raw[:8])
        self.x = self.prep.transform(self.raw)
        self.initial = initial_state()

    def test_only_activation_changes_and_initialization_shared(self):
        models = [ActivationVAE(a) for a in ("relu", "leaky_relu")]
        for model in models:
            model.load_state_dict(self.initial)
            self.assertEqual(state_hash(model.state_dict()), state_hash(self.initial))
        self.assertIsInstance(models[0].decode_hidden[1], torch.nn.ReLU)
        self.assertIsInstance(models[1].decode_hidden[1], torch.nn.LeakyReLU)
        self.assertEqual(models[1].decode_hidden[1].negative_slope, 0.01)
        for a, b in zip(models[0].encoder.parameters(), models[1].encoder.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(models[1].decode_hidden[0].weight.shape, (2000, 5))
        self.assertEqual(models[1].decode_frame.weight.shape, (6, 5))
        self.assertEqual(PROTOCOL["small_steps"], 2000)
        self.assertEqual(PROTOCOL["vae_candidates"], CANDIDATES)
        self.assertEqual(state_hash(self.initial), state_hash(initial_state()))
        self.assertNotEqual(state_hash(self.initial), state_hash(initial_state(43)))
        with self.assertRaises(ValueError):
            ActivationVAE("gelu")

    def test_leaky_negative_gradient_and_deterministic_random_paths(self):
        for activation, negative_gradient in (("relu", 0.0), ("leaky_relu", 0.01)):
            model = ActivationVAE(activation)
            x = torch.tensor([-2.0, 2.0], requires_grad=True)
            model.decode_hidden[1](x).sum().backward()
            torch.testing.assert_close(x.grad, torch.tensor([negative_gradient, 1.0]))
            trajectory = torch.from_numpy(self.x[:3])
            model.eval()
            torch.testing.assert_close(
                model(trajectory, sample=False)[0], model(trajectory, sample=False)[0]
            )
            self.assertFalse(
                torch.equal(model(trajectory, sample=True)[0], model(trajectory, sample=True)[0])
            )

    def test_relu_reproduces_previous_trainer_and_detects_corruption(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("scripts.sim_pretrain.experiments.vae_ablation.plot_history"),
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.plot_history"
            ),
        ):
            out = Path(tmp)
            fit_run(
                out / "old",
                "original_linear",
                run_spec(),
                self.x[:8],
                None,
                self.prep,
                self.x[:8].mean(0),
                {},
                small=True,
                small_steps=5,
            )
            train_activation(
                out / "new",
                "relu",
                run_spec(),
                self.x[:8],
                None,
                self.prep,
                self.x[:8].mean(0),
                {},
                small=True,
                small_steps=5,
                initial=self.initial,
            )
            result = compare_reproduction(out / "new", out / "old")
            self.assertTrue(result["passed"])
            self.assertTrue(result["weights_bitwise_equal"])
            self.assertTrue(result["history_bitwise_equal"])
            history = json.loads((out / "new/history.json").read_text())
            history[1]["train"]["kl"] += 1.0
            write_json(out / "new/history.json", history)
            self.assertFalse(compare_reproduction(out / "new", out / "old")["passed"])

    def test_activation_checkpoint_roundtrip_and_encoder_freezing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for activation in ("relu", "leaky_relu"):
                model = ActivationVAE(activation).eval()
                path = out / f"{activation}.pt"
                save_checkpoint(
                    path, payload(model, self.prep, run_spec(), 1, self.x[:8].mean(0), {})
                )
                loaded, metadata = load_activation_model(path)
                self.assertEqual(metadata["decoder_activation"], activation)
                x = torch.from_numpy(self.x)
                torch.testing.assert_close(
                    model(x, sample=False)[0], loaded(x, sample=False)[0], rtol=0, atol=0
                )
                export_encoder(
                    model, self.prep, out / "encoder.pt", {"decoder_activation": activation}
                )
                encoder = FrozenAblationEncoder(out / "encoder.pt")
                np.testing.assert_allclose(
                    encoder.encode(self.raw), model(x, sample=False)[1].detach().numpy(), atol=1e-7
                )
                self.assertTrue(
                    all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
                )

    def test_gates_are_bounded_and_stop_failed_groups(self):
        completed = {"state": "completed", "passed": True}
        failed = {"state": "completed", "passed": False}
        self.assertEqual(next_stage({"passed": False}, completed), "stop_reproduction_mismatch")
        self.assertEqual(next_stage({"passed": True}), "leaky_small")
        self.assertEqual(next_stage({"passed": True}, failed), "stop_small_ae_failed")
        self.assertEqual(next_stage({"passed": True}, completed), "full_ae")
        self.assertEqual(next_stage({"passed": True}, completed, failed), "stop_full_ae_failed")
        self.assertEqual(next_stage({"passed": True}, completed, completed), "vae_sweep")
        with patch(
            "scripts.sim_pretrain.experiments.decoder_activation_experiment.train_activation"
        ) as train:
            result = continue_after_pair(
                Path("unused"), failed, {}, None, None, {}, {"reproduction": {"passed": True}}
            )
            self.assertIsNone(result)
            train.assert_not_called()
        self.assertNotIn("test", inspect.signature(train_activation).parameters)
        self.assertNotIn("labels", inspect.signature(train_activation).parameters)

    def test_validation_only_continuation_uses_six_fresh_specs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.train_activation"
            ) as train,
        ):
            train.return_value = {
                "state": "completed",
                "passed": True,
                "validation": {"mse": 0.001},
            }
            report = {"reproduction": {"passed": True}, "runs": []}
            arrays = {"train": self.x[:8], "validation": self.x[8:]}
            result = continue_after_pair(
                Path(tmp),
                {"state": "completed", "passed": True},
                arrays,
                self.prep,
                self.x[:8].mean(0),
                {},
                report,
            )
            self.assertEqual(train.call_count, 7)
            self.assertEqual(result["order"], 0)
            for call, candidate in zip(train.call_args_list[1:], CANDIDATES):
                spec = call.args[2]
                self.assertEqual(spec, run_spec(candidate))
                self.assertFalse(spec["deterministic"])
                self.assertIs(call.args[3], arrays["train"])
                self.assertIs(call.args[4], arrays["validation"])

    def test_full_vae_selection_epochs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.plot_history"
            ),
        ):
            out = Path(tmp) / "vae"
            result = train_activation(
                out,
                "leaky_relu",
                run_spec(CANDIDATES[2]),
                self.x[:8],
                self.x[8:],
                self.prep,
                self.x[:8].mean(0),
                {},
                epochs=51,
            )
            self.assertEqual(result["state"], "completed")
            self.assertIn(result["best_epoch"], (50, 51))
            _, saved = load_activation_model(out / "last.pt")
            self.assertEqual(saved["epoch"], 51)
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(history[0]["beta"], 0.0)
            self.assertEqual(history[49]["beta"], 0.001)
            self.assertEqual(history[50]["beta"], 0.001)

    def test_nonfinite_failure_and_overwrite_protection(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.plot_history"
            ),
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.finite_step",
                side_effect=FloatingPointError("fixture loss"),
            ),
        ):
            out = Path(tmp) / "bad"
            status = train_activation(
                out,
                "leaky_relu",
                run_spec(),
                self.x[:8],
                None,
                self.prep,
                self.x[:8].mean(0),
                {},
                small=True,
                small_steps=2,
            )
            self.assertEqual(status["state"], "failed")
            self.assertIn("fixture", status["failure_reason"])
            with self.assertRaises(FileExistsError):
                train_activation(
                    out,
                    "leaky_relu",
                    run_spec(),
                    self.x[:8],
                    None,
                    self.prep,
                    self.x[:8].mean(0),
                    {},
                    small=True,
                    small_steps=2,
                )

    def test_diagnostic_and_pair_plot_are_training_only(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "scripts.sim_pretrain.experiments.decoder_activation_experiment.plot_history"
            ),
        ):
            out = Path(tmp)
            for activation, name in (("relu", "relu_small_ae"), ("leaky_relu", "leaky_small_ae")):
                train_activation(
                    out / name,
                    activation,
                    run_spec(),
                    self.x[:8],
                    None,
                    self.prep,
                    self.x[:8].mean(0),
                    {},
                    initial=self.initial,
                    small=True,
                    small_steps=2,
                )
            plot_pair(out, self.x[:8], self.prep, np.arange(8))
            self.assertTrue((out / "activation_comparison.png").is_file())
            self.assertTrue((out / "train_reconstruction_000.png").is_file())
            self.assertFalse(list(out.rglob("test*.json")))
            diagnostic = activation_diagnostic(ActivationVAE("leaky_relu"), self.x[:8], 0)
            self.assertEqual(diagnostic["exact_zero_activation_fraction"], 0.0)

    def test_previous_sources_preserved_and_output_cannot_be_nested(self):
        previous = ROOT / "archive/sim_pretrain/normal_vae_ablation_v1"
        if not previous.exists():
            self.skipTest("Previous research experiment is not installed")
        for filename in ("source_hashes.json", "original_hashes.json"):
            before = json.loads((previous / filename).read_text())
            self.assertTrue(verify_snapshot(before, historical=True)["unchanged"])
        cfg = ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
        with self.assertRaisesRegex(ValueError, "outside"):
            execute(cfg, previous, previous / "forbidden_child")
        with self.assertRaisesRegex(ValueError, "outside"):
            execute(cfg, previous, cfg.parent / "forbidden_child")


if __name__ == "__main__":
    unittest.main()
