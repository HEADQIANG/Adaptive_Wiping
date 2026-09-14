"""Software checks only; tiny synthetic fixtures never replace the real experiment."""

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.shared.common import ROOT, file_digest, load_config
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.vae_ablation import (
    ARCHITECTURES,
    CANDIDATES,
    PHASES,
    PROTOCOL,
    SHUFFLE_SEEDS,
    AblationVAE,
    FrozenAblationEncoder,
    beta_at,
    choose_candidate,
    diagnostic_arrays,
    export_encoder,
    finite_step,
    fit_run,
    infer,
    load_experiment_model,
    reserve_output,
    run_spec,
    seed_all,
    snapshot,
    split_audit,
    verify_snapshot,
)
from scripts.sim_pretrain.experiments.vae_property_probe import (
    FrozenPropertyPredictor,
    Standardizer,
    fit_mlp,
    fit_pca5,
    property_metrics,
    ridge_coefficients,
    run_probes,
    transform_pca,
)
from scripts.sim_pretrain.learning import load_checkpoint


class AblationTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.raw = np.random.default_rng(8).normal(size=(12, 400, 6))
        self.prep = Preprocessor().fit(self.raw)
        self.x = self.prep.transform(self.raw)

    def test_beta_boundaries_and_protocol(self):
        for target in (0.001, 0.01, 0.06):
            self.assertEqual(beta_at(1, target, True), 0)
            self.assertAlmostEqual(beta_at(2, target, True), target / 49)
            self.assertAlmostEqual(beta_at(25, target, True), target * 24 / 49)
            self.assertEqual(beta_at(50, target, True), target)
            self.assertEqual(beta_at(51, target, True), target)
            self.assertEqual(beta_at(200, target, True), target)
            self.assertEqual(beta_at(1, target, False), target)
        with self.assertRaises(ValueError):
            beta_at(0, 0.001, True)
        self.assertEqual(PROTOCOL["selection_epochs"], [50, 200])
        self.assertEqual(len(CANDIDATES), 6)
        self.assertEqual(len(SHUFFLE_SEEDS), 20)
        self.assertEqual(PHASES, dict(press=(0, 200), forward=(200, 300), reverse=(300, 400)))

    def test_original_structure_and_loss_unchanged(self):
        seed_all(42)
        original = SpongeVAE()
        seed_all(42)
        experiment = AblationVAE()
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, experiment.state_dict()[key], rtol=0, atol=0)
        x = torch.from_numpy(self.x)
        original.eval()
        experiment.eval()
        for a, b in zip(original(x, sample=False), experiment(x, sample=False)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        prediction, mu, logvar = experiment(x, sample=False)
        loss, mse, kl = vae_loss(prediction, x, mu, logvar, 0.001)
        torch.testing.assert_close(mse, ((prediction - x) ** 2).mean())
        torch.testing.assert_close(kl, 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(-1).mean())
        torch.testing.assert_close(loss, mse + 0.001 * kl)

    def test_deterministic_and_random_branches(self):
        x = torch.from_numpy(self.x)
        for architecture in ARCHITECTURES:
            model = AblationVAE(architecture, dropout=0)
            model.train()
            a, mu, _ = model(x, sample=False)
            torch.testing.assert_close(a, model.decode(mu))
            torch.testing.assert_close(a, model(x, sample=False)[0], rtol=0, atol=0)
            self.assertFalse(torch.equal(model(x, sample=True)[0], model(x, sample=True)[0]))
            self.assertEqual(mu.shape, (12, 5))
            self.assertEqual(model.decode_frame.weight.shape, (6, 5))
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 399, 6))

    def test_export_roundtrip_and_frozen_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ARCHITECTURES:
                model = AblationVAE(architecture, 0).eval()
                path = Path(tmp) / f"{architecture}.pt"
                export_encoder(model, self.prep, path, {"fixture": True})
                loaded = FrozenAblationEncoder(path)
                expected = infer(model, self.x)[1].numpy()
                np.testing.assert_allclose(loaded.encode(self.raw), expected, atol=1e-7)
                np.testing.assert_array_equal(loaded.encode(self.raw), loaded.encode(self.raw))
                self.assertTrue(
                    all(not p.requires_grad and p.grad is None for p in loaded.model.parameters())
                )
                features = torch.from_numpy(loaded.encode(self.raw))
                predictor = torch.nn.Linear(5, 3)
                predictor(features).square().mean().backward()
                self.assertTrue(all(p.grad is None for p in loaded.model.parameters()))
                self.assertIsNotNone(predictor.weight.grad)

    def test_no_split_or_preprocessing_leakage(self):
        raw = dict(train=self.raw[:6], validation=self.raw[6:9], test=self.raw[9:])
        labels = {
            s: np.arange(len(x) * 3).reshape(len(x), 3) + 100 * i
            for i, (s, x) in enumerate(raw.items())
        }
        audit = split_audit(raw, labels)
        self.assertEqual(audit["removed_samples"], 0)
        duplicate = {**raw, "test": raw["train"][:3]}
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            split_audit(duplicate, labels)
        duplicate_labels = {**labels, "test": labels["train"][:3]}
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            split_audit(raw, duplicate_labels)
        prep = Preprocessor().fit(raw["train"])
        state = prep.state()
        prep.transform(raw["test"] + 1000)
        self.assertEqual(state, prep.state())
        signature = inspect.signature(fit_run)
        self.assertNotIn("test", signature.parameters)
        self.assertNotIn("labels", signature.parameters)

    def test_synthetic_informative_and_collapsed_diagnostics(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(40, 1, 1))
        target = np.broadcast_to(values, (40, 400, 6)).copy()
        template = target.mean(0)
        mu = np.repeat(values[:, 0, 0, None], 5, axis=1)
        prediction = target + 0.01
        fixed = np.broadcast_to(template + 0.01, target.shape)
        shuffled = [
            prediction[np.random.default_rng(seed).permutation(40)] for seed in SHUFFLE_SEEDS
        ]
        result = diagnostic_arrays(
            target, prediction, mu, np.zeros_like(mu), fixed, shuffled, template, self.prep
        )
        self.assertTrue(result["passed"])
        self.assertGreater(result["fixed_relative_increase"], 0.1)
        collapsed = diagnostic_arrays(
            target,
            fixed,
            np.zeros_like(mu),
            np.zeros_like(mu),
            fixed,
            [fixed] * 20,
            template,
            self.prep,
        )
        self.assertFalse(collapsed["latent_utilization_pass"])
        self.assertFalse(collapsed["passed"])
        self.assertEqual(collapsed["fixed_relative_increase"], 0.0)

    def test_small_run_and_full_epoch_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch("scripts.sim_pretrain.experiments.vae_ablation.plot_history"):
                small = fit_run(
                    base / "small",
                    ARCHITECTURES[0],
                    run_spec(),
                    self.x[:8],
                    None,
                    self.prep,
                    self.x.mean(0),
                    {},
                    small=True,
                    small_steps=2,
                )
                result = fit_run(
                    base / "vae",
                    ARCHITECTURES[0],
                    run_spec(CANDIDATES[2]),
                    self.x[:8],
                    self.x[8:],
                    self.prep,
                    self.x[:8].mean(0),
                    {},
                    epochs=51,
                )
            self.assertEqual(small["state"], "completed")
            self.assertFalse(small["passed"])
            self.assertEqual(result["epoch"], 51)
            self.assertIn(result["best_epoch"], (50, 51))
            _, last = load_experiment_model(base / "vae" / "last.pt")
            self.assertEqual(last["epoch"], 51)
            history = json.loads((base / "vae" / "history.json").read_text())
            self.assertEqual(history[0]["beta"], 0)
            for row in history:
                for split in ("train", "validation"):
                    self.assertAlmostEqual(
                        row[split]["weighted_kl"], row["beta"] * row[split]["kl"]
                    )

    def test_nonfinite_loss_and_gradient_stop(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            finite_step([torch.tensor(float("nan"))], model, opt)
        handle = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            finite_step([model(torch.ones(2, 2)).square().mean()], model, opt)
        handle.remove()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("scripts.sim_pretrain.experiments.vae_ablation.plot_history"),
            patch(
                "scripts.sim_pretrain.experiments.vae_ablation.finite_step",
                side_effect=FloatingPointError("fixture loss"),
            ),
        ):
            result = fit_run(
                Path(tmp) / "failed",
                ARCHITECTURES[0],
                run_spec(),
                self.x[:8],
                None,
                self.prep,
                self.x.mean(0),
                {},
                small=True,
                small_steps=2,
            )
            self.assertEqual(result["state"], "failed")
            self.assertIn("fixture", result["failure_reason"])

    def test_selection_uses_validation_and_table_order(self):
        rows = [
            {
                "state": "completed",
                "passed": True,
                "order": i,
                "validation": {"mse": 0.001},
                "test": {"mse": 0.1 - 0.01 * i},
            }
            for i in range(6)
        ]
        self.assertIs(choose_candidate(rows[::-1]), rows[0])
        rows[0]["passed"] = False
        self.assertIs(choose_candidate(rows), rows[1])
        for row in rows:
            row["passed"] = False
        self.assertIsNone(choose_candidate(rows))

    def test_overwrite_protection_and_hash_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = reserve_output(Path(tmp) / "run")
            with self.assertRaises(FileExistsError):
                reserve_output(out)
            path = out / "fixture.json"
            path.write_text("{}")
            before = snapshot([path])
            self.assertTrue(verify_snapshot(before)["unchanged"])
            path.write_text('{"changed": true}')
            self.assertFalse(verify_snapshot(before)["unchanged"])

    def test_existing_baseline_and_production_sources_preserved(self):
        config = ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
        if not config.exists():
            self.skipTest("Local frozen dataset not installed")
        cfg = load_config(config)
        baseline = config.parent / "vae_last.pt"
        self.assertEqual(
            file_digest(baseline),
            "65812a35f41ae8ce8f176a2e62fe832edeebc6592d2f40c05df08c209655c7ba",
        )
        self.assertEqual(
            file_digest(config.parent / "dataset.h5"),
            "de35cc1d706d113ffdfc5c89942325f0eb3e555ed1d0180fdb9289dafb739e49",
        )
        checkpoint, _, _ = load_checkpoint(cfg)
        for name, sha in checkpoint["training_provenance"]["sources"].items():
            self.assertEqual(file_digest(ROOT / name), sha, name)


class PropertyProbeTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_scaling_is_train_only_and_retains_tiny_latent_variance(self):
        training = np.arange(30).reshape(10, 3) * 1e-8
        scaler = Standardizer().fit(training)
        state = scaler.state()
        np.testing.assert_allclose(scaler.transform(training).std(0), 1)
        scaler.transform(training + 100)
        self.assertEqual(scaler.state(), state)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(training)), training, atol=1e-20)
        constant = Standardizer().fit(np.ones((5, 3)))
        np.testing.assert_array_equal(constant.transform(np.ones((5, 3))), 0)
        constant = Standardizer().fit(np.full((1000, 3), 0.1))
        np.testing.assert_array_equal(constant.scale, 1.0)

    def test_train_pca_and_ridge(self):
        rng = np.random.default_rng(5)
        training = rng.normal(size=(25, 6)) @ rng.normal(size=(6, 40))
        pca = fit_pca5(training)
        np.testing.assert_allclose(pca["mean"], training.mean(0))
        np.testing.assert_allclose(pca["components"] @ pca["components"].T, np.eye(5), atol=1e-10)
        before = pca["mean"].copy()
        self.assertEqual(transform_pca(training + 200, pca).shape, (25, 5))
        np.testing.assert_array_equal(before, pca["mean"])
        for dimensions in (4, 40):
            x = rng.normal(size=(25, dimensions))
            y = rng.normal(size=(25, 3))
            coefficients = ridge_coefficients(x, y)
            for alpha, coefficient in zip((1e-4, 1e-2, 1, 100), coefficients):
                expected = np.linalg.solve(x.T @ x + alpha * np.eye(dimensions), x.T @ y)
                np.testing.assert_allclose(coefficient, expected, atol=1e-8)

    def test_identifiable_and_constant_property_bootstrap(self):
        rng = np.random.default_rng(8)
        target = rng.normal(size=(100, 3))
        prediction = target + 0.05 * rng.normal(size=target.shape)
        result = property_metrics(target, prediction, np.zeros(3))
        repeated = property_metrics(target, prediction, np.zeros(3))
        self.assertEqual(result, repeated)
        self.assertTrue(all(v["predictive_information_detected"] for v in result.values()))
        collapsed = property_metrics(target, np.zeros_like(target), np.zeros(3))
        self.assertFalse(any(v["predictive_information_detected"] for v in collapsed.values()))
        self.assertTrue(
            all(v["paired_rmse_improvement_ci95"] == [0, 0] for v in collapsed.values())
        )

    def test_mlp_export_and_no_feature_gradients(self):
        rng = np.random.default_rng(15)
        features = {"train": rng.normal(size=(32, 5)), "validation": rng.normal(size=(8, 5))}
        labels = {s: x[:, :3] for s, x in features.items()}
        input_scaler, label_scaler = (
            Standardizer().fit(features["train"]),
            Standardizer().fit(labels["train"]),
        )
        x = {s: input_scaler.transform(v) for s, v in features.items()}
        y = {s: label_scaler.transform(v) for s, v in labels.items()}
        metadata = {
            "format": "supervised_property_probe_v1",
            "input_standardizer": input_scaler.state(),
            "label_standardizer": label_scaler.state(),
            "supervised": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model, result = fit_mlp(x, y, out, metadata, epochs=3, patience=2)
            self.assertEqual(result["state"], "completed")
            loaded = FrozenPropertyPredictor(out / "mlp_best.pt")
            with torch.no_grad():
                expected = label_scaler.inverse(
                    model(torch.tensor(x["validation"], dtype=torch.float32)).numpy()
                )
            np.testing.assert_allclose(loaded.predict(features["validation"]), expected)
            self.assertTrue(all(not p.requires_grad for p in loaded.model.parameters()))

    def test_end_to_end_frozen_probes_select_before_test(self):
        rng = np.random.default_rng(55)
        arrays = {
            s: rng.normal(size=(n, 400, 6)).astype(np.float32)
            for s, n in (("train", 32), ("validation", 8), ("test", 10))
        }
        labels = {s: x[:, 0, :3].copy() for s, x in arrays.items()}
        original, new = AblationVAE(), AblationVAE()

        def short_fit(*args, **kwargs):
            return fit_mlp(*args, **kwargs, epochs=2, patience=2)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "scripts.sim_pretrain.experiments.vae_property_probe.fit_mlp",
                side_effect=short_fit,
            ),
        ):
            out = Path(tmp) / "probes"
            report = run_probes(out, arrays, labels, original, new, {"fixture": True})
            self.assertTrue(report["encoders_frozen_without_gradients"])
            self.assertEqual(set(report["inputs"]), {"original_mu", "new_mu", "pca5", "full_ft"})
            self.assertTrue((out / "property_scatter.png").is_file())
            for name, result in report["inputs"].items():
                self.assertEqual(len(result["candidates"]), 5)
                selected_before_test = json.loads((out / name / "selection.json").read_text())
                self.assertNotIn("test", selected_before_test["selected"])
                self.assertEqual(
                    selected_before_test["selected"]["name"], result["selected"]["name"]
                )
                self.assertEqual(
                    result["selected"]["validation_standardized_mse"],
                    min(r["validation_standardized_mse"] for r in result["candidates"]),
                )
                self.assertTrue((out / name / "selected_predictor.pt").is_file())


if __name__ == "__main__":
    unittest.main()
