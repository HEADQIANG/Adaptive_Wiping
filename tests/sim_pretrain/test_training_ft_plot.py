"""Synthetic visualization fixtures never write into the frozen training dataset."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import (
    ROOT,
    digest,
    file_digest,
    load_config,
    write_json,
)
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.plot_training_ft import (
    PANEL_CHANNELS,
    checked_indices,
    load_training,
    make_figure,
    quantile_band,
    visualize,
)


class TrainingFTPlotTests(unittest.TestCase):
    def fixture(self, folder):
        folder.mkdir()
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["output_dir"] = str(folder)
        cfg["dataset"] = dict(train=4, validation=2, test=2)
        cfg["research_comparison"] = {"controller": "normal"}
        config = folder / "config.json"
        write_json(config, cfg)
        raw = np.random.default_rng(9).normal(size=(4, 400, 6))
        prep = Preprocessor(**cfg["filter"], sample_hz=100).fit(raw)
        write_json(folder / "preprocessing.json", prep.state())
        with h5py.File(folder / "dataset.h5", "w") as h5:
            h5.attrs.update(complete=True, config_hash=digest(cfg))
            train = h5.create_group("train")
            train.create_dataset("ft", data=raw)
            train.create_dataset("time", data=np.broadcast_to(np.arange(1, 401) / 100.0, (4, 400)))
            train.create_dataset("parameters", data=np.ones((4, 3)))
            train.create_dataset("valid", data=np.ones(4, dtype=bool))
            train.create_dataset("motion_passed", data=np.zeros(4, dtype=bool))
        write_json(
            folder / "dataset_integrity.json", {"sha256": file_digest(folder / "dataset.h5")}
        )
        return config, raw, prep

    def test_indices_and_channel_units(self):
        np.testing.assert_array_equal(checked_indices([0, 3], 4), [0, 3])
        for invalid in ([], [-1], [4], [0, 0], [0.5], [[0, 1]]):
            with self.assertRaises(ValueError):
                checked_indices(invalid, 4)
        self.assertEqual(PANEL_CHANNELS, (0, 3, 1, 4, 2, 5))
        plt, fig, axes = make_figure("Fixture")
        self.assertEqual(axes[0, 0].get_ylabel(), "Fx (N)")
        self.assertEqual(axes[0, 1].get_ylabel(), "Tx (N m)")
        self.assertEqual(axes[2, 0].get_ylabel(), "Fz (N)")
        self.assertEqual(axes[2, 1].get_ylabel(), "Tz (N m)")
        plt.close(fig)

    def test_training_only_and_exact_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, raw, prep = self.fixture(Path(tmp) / "dataset")
            loaded = load_training(config)
            np.testing.assert_array_equal(loaded["raw"], raw)
            np.testing.assert_array_equal(loaded["filtered"], prep.filtered(raw))
            np.testing.assert_array_equal(loaded["normalized"], prep.transform(raw))
            # transform stores float32; inverse divides in float32 before rescaling.
            scale = np.max(np.where(prep.constant, 1.0, prep.maximum - prep.minimum)) / 0.9
            np.testing.assert_allclose(
                prep.inverse(loaded["normalized"]),
                loaded["filtered"],
                rtol=0,
                atol=2 * np.finfo(np.float32).eps * scale,
            )
            self.assertEqual(loaded["motion_passed"].sum(), 0)
            self.assertEqual(len(loaded["raw"]), 4)
            with h5py.File(config.parent / "dataset.h5") as h5:
                self.assertNotIn("test", h5)
                self.assertNotIn("validation", h5)

    def test_integrity_and_stale_preprocessing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, prep = self.fixture(Path(tmp) / "dataset")
            state = prep.state()
            state["minimum"][0] -= 1
            write_json(config.parent / "preprocessing.json", state)
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_training(config)
            write_json(config.parent / "preprocessing.json", prep.state())
            with h5py.File(config.parent / "dataset.h5", "r+") as h5:
                h5["train/ft"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_training(config)

    def test_end_to_end_plots_retained_rows_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config, raw, prep = self.fixture(base / "dataset")
            manifest = base / "prior" / "manifest.json"
            write_json(
                manifest,
                {
                    "dataset_sha256": file_digest(config.parent / "dataset.h5"),
                    "dataset_config_sha256": file_digest(config),
                    "small_train_indices": [0, 2],
                },
            )
            before = file_digest(config.parent / "dataset.h5")
            result = visualize(config, base / "plots", [0, 2], manifest)
            self.assertTrue(result["inputs_unchanged"])
            self.assertEqual(result["training_count"], 4)
            self.assertEqual(result["removed_samples"], 0)
            self.assertEqual(len(result["figures"]), 7)
            self.assertTrue(
                all((base / "plots" / name).stat().st_size > 1000 for name in result["figures"])
            )
            with np.load(base / "plots/plotted_data.npz") as data:
                np.testing.assert_array_equal(data["raw"], raw[[0, 2]])
                np.testing.assert_array_equal(data["normalized"], prep.transform(raw)[[0, 2]])
                np.testing.assert_allclose(data["filtered_band"], quantile_band(prep.filtered(raw)))
            self.assertEqual(file_digest(config.parent / "dataset.h5"), before)
            with self.assertRaises(FileExistsError):
                visualize(config, base / "plots", [0])
            with self.assertRaisesRegex(ValueError, "outside"):
                visualize(config, config.parent / "nested", [0])
            with self.assertRaisesRegex(ValueError, "outside"):
                visualize(config, manifest.parent / "nested", [0], manifest)


if __name__ == "__main__":
    unittest.main()
