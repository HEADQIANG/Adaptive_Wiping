"""Legacy replay and native continuation must equal uninterrupted training."""

import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts.shared.common import digest, file_digest, load_config, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.continue_pretraining import (
    continue_training,
    finalize_export,
    initialize_training,
    load_continued_model,
    require_exact_replay,
    reserve_output,
    restore_rng,
    rng_state,
    run_epoch,
)
from scripts.sim_pretrain.learning import configure_torch, train


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wiping-continuation-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper_mu1p2.yaml")
        self.cfg["dataset"] = {"train": 8, "validation": 3, "test": 3}
        self.cfg["training"]["epochs"] = 2
        configure_torch(self.cfg)

    def dataset(self, name, epochs):
        folder = self.root / name
        folder.mkdir()
        cfg = copy.deepcopy(self.cfg)
        cfg["output_dir"] = str(folder)
        cfg["training"]["epochs"] = epochs
        rng = np.random.default_rng(17)
        with h5py.File(folder / "dataset.h5", "w") as h5:
            h5.attrs.update(config_hash=digest(cfg), complete=True)
            for split, count in cfg["dataset"].items():
                g = h5.create_group(split)
                g.create_dataset("ft", data=rng.normal(size=(count, 400, 6)))
                g.create_dataset("valid", data=np.ones(count, dtype=bool))
        write_json(folder / "config.json", cfg)
        write_json(
            folder / "dataset_integrity.json", {"sha256": file_digest(folder / "dataset.h5")}
        )
        return cfg, folder / "config.json"

    def arrays(self):
        rng = np.random.default_rng(42)
        return {
            split: rng.normal(size=(count, 400, 6)).astype(np.float32)
            for split, count in self.cfg["dataset"].items()
        }

    def test_replay_and_saved_state_match_original_uninterrupted_training(self):
        cfg, config = self.dataset("source", 2)
        train(cfg)
        source_files = {p: file_digest(p) for p in config.parent.iterdir() if p.is_file()}
        fake_report = {
            "frozen_export_verified": True,
            "splits": {"test": {"after": {"passed": False}}},
        }
        with patch(
            "scripts.sim_pretrain.experiments.continue_pretraining.evaluate_and_export",
            return_value=fake_report,
        ):
            status = continue_training(config, self.root / "four", 2)
            self.assertEqual(status["state"], "completed")
            status = continue_training(config, self.root / "six", 2, self.root / "four/vae_last.pt")
        self.assertEqual(status["last_completed_epoch"], 6)
        self.assertTrue(status["protected_inputs"]["unchanged"])
        recovery = json.loads((self.root / "four/recovery_verification.json").read_text())
        self.assertTrue(recovery["weights_exact"] and recovery["all_history_exact"])
        recovery = json.loads((self.root / "six/recovery_verification.json").read_text())
        self.assertEqual(recovery["method"], "saved_optimizer_rng")
        reference, _ = self.dataset("reference", 6)
        train(reference)
        _, saved = load_continued_model(self.root / "six/vae_last.pt")
        expected = torch.load(self.root / "reference/vae_last.pt", weights_only=True)
        for key, value in expected["model"].items():
            self.assertTrue(torch.equal(value, saved["model"][key]), key)
        self.assertEqual(
            json.loads((self.root / "six/history.json").read_text()),
            json.loads((self.root / "reference/history.json").read_text()),
        )
        self.assertTrue(saved["optimizer"]["state"])
        self.assertIn("rng_state", saved)
        self.assertEqual(source_files, {p: file_digest(p) for p in source_files})

    def test_history_mismatch_stops_before_added_epochs(self):
        cfg, config = self.dataset("source", 2)
        train(cfg)
        path = config.parent / "history.json"
        history = json.loads(path.read_text())
        history[0]["train"]["mse"] += 0.1
        write_json(path, history)
        with self.assertRaisesRegex(ValueError, "Replay history mismatch"):
            continue_training(config, self.root / "mismatch", 2)
        status = json.loads((self.root / "mismatch/status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertFalse((self.root / "mismatch/vae_last.pt").exists())

    def test_weight_mismatch_is_not_tolerated(self):
        model, _, _ = initialize_training(self.cfg, self.arrays())
        expected = copy.deepcopy(model.state_dict())
        expected[next(iter(expected))].flatten()[0] += 0.001
        with self.assertRaisesRegex(ValueError, "weights differ"):
            require_exact_replay(model, [], {"model": expected}, [])

    def test_rng_roundtrip_covers_all_three_generators(self):
        saved = rng_state()
        expected = (torch.rand(5), np.random.rand(5), random.random())
        restore_rng(saved)
        actual = (torch.rand(5), np.random.rand(5), random.random())
        self.assertTrue(torch.equal(expected[0], actual[0]))
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])

    def test_output_overlap_overwrite_and_invalid_budget_rejected(self):
        source = self.root / "source"
        source.mkdir()
        for path in (source, source / "child", self.root):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                reserve_output(path, [source])
        out = reserve_output(self.root / "new", [source])
        with self.assertRaises(FileExistsError):
            reserve_output(out, [source])
        with self.assertRaisesRegex(ValueError, "positive"):
            continue_training("unused", "unused", 0)

    def test_only_training_uses_stochastic_branch_no_test_loader(self):
        model, optimizer, loaders = initialize_training(self.cfg, self.arrays())
        self.assertEqual(set(loaders), {"train", "validation"})
        seen = []
        original = model.forward

        def record(x, sample=True):
            seen.append((model.training, sample, torch.is_grad_enabled()))
            return original(x, sample=sample)

        with patch.object(model, "forward", side_effect=record):
            run_epoch(model, optimizer, loaders, 0.06, 1)
        self.assertEqual(seen, [(True, True, True), (False, False, False)])

    def test_nonfinite_loss_and_gradient_stop_training(self):
        arrays = self.arrays()
        arrays["train"][0, 0, 0] = np.nan
        model, optimizer, loaders = initialize_training(self.cfg, arrays)
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            run_epoch(model, optimizer, loaders, 0.06, 1)
        model, optimizer, loaders = initialize_training(self.cfg, self.arrays())
        handle = next(model.parameters()).register_hook(lambda gradient: gradient * float("nan"))
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            run_epoch(model, optimizer, loaders, 0.06, 1)

    def test_export_recovery_checks_same_batch_and_never_trains(self):
        self.cfg["dataset"]["test"] = 100
        cfg, config = self.dataset("source", 2)
        train(cfg)
        out = self.root / "export"

        def save_embeddings(model, values, prep, output, label):
            from scripts.sim_pretrain.experiments.vae_ablation import infer

            _, mu, logvar = infer(model, values)
            np.savez_compressed(
                output / "test_embeddings.npz", mu=mu.numpy(), logvar=logvar.numpy()
            )

        with (
            patch("scripts.sim_pretrain.experiments.continue_pretraining.plot_history"),
            patch(
                "scripts.sim_pretrain.experiments.continue_pretraining.plot_diagnostics",
                side_effect=save_embeddings,
            ),
            patch(
                "scripts.sim_pretrain.experiments.continue_pretraining.verify_frozen_export",
                side_effect=AssertionError("synthetic export failure"),
            ),
        ):
            with self.assertRaisesRegex(AssertionError, "synthetic"):
                continue_training(config, out, 2)
        weights_before = {
            name: file_digest(out / name) for name in ("vae_last.pt", "encoder.pt", "history.json")
        }
        with patch(
            "scripts.sim_pretrain.experiments.continue_pretraining.run_epoch",
            side_effect=AssertionError("must not retrain"),
        ):
            self.assertEqual(finalize_export(out)["state"], "completed")
        self.assertEqual(weights_before, {name: file_digest(out / name) for name in weights_before})
        proof = json.loads((out / "evaluation.json").read_text())
        self.assertTrue(proof["frozen_export_verified"])
        self.assertTrue(proof["export_weights_and_preprocessing_exact"])
        with self.assertRaisesRegex(ValueError, "failed postprocessing"):
            finalize_export(out)

    def test_export_recovery_rejects_unfinished_training(self):
        out = self.root / "unfinished"
        out.mkdir()
        write_json(
            out / "status.json", {"state": "failed", "last_completed_epoch": 3, "target_epoch": 4}
        )
        with self.assertRaisesRegex(ValueError, "final epoch"):
            finalize_export(out)


if __name__ == "__main__":
    unittest.main()
