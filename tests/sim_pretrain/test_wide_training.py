import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts.shared.common import digest, file_digest, load_config, write_json
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.collect_wide_training import assignments, validate_numerical, simulate
from scripts.sim_pretrain.learning import train


class WideTrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("configs/sim_pretrain/pretrain_wide_2000.yaml")

    def test_assignments_and_numerical_only_policy(self):
        self.assertEqual(sum(self.cfg["dataset"].values()), 2000)
        plan = assignments(self.cfg)
        self.assertEqual(plan, assignments(self.cfg))
        for rows in plan.values():
            a = np.asarray(rows)
            self.assertTrue(np.all(a >= [2000,0,10,.001]))
            self.assertTrue(np.all(a <= [10000,1.2,10000,1.2]))
        data = {key:np.zeros(shape) for key, shape in FIELDS.items()}
        data["time"] = np.arange(1,401)/100
        data["orientation_error"][:] = 1
        data["saturated"][:] = 1
        validate_numerical(data)

        class FakeEnv:
            def __init__(self, *args): pass
            def rollout(self): return data
            def close(self): pass

        with patch("scripts.sim_pretrain.simulation.PretrainingWipe", FakeEnv):
            _, _, result, diagnostic, error = simulate((self.cfg,"train",0,[5000,.5,500,.05]))
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertFalse(diagnostic["motion"]["passed"])
        data["ft"][0,0] = np.nan
        with self.assertRaises(ValueError): validate_numerical(data)

    def test_periodic_results_and_exact_cpu_resume(self):
        from scripts.sim_pretrain.learning import training_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg)
            cfg["output_dir"] = tmp
            cfg["dataset"] = dict(train=4,validation=2,test=2)
            cfg["training"].update(epochs=3, checkpoint_every=1)
            rng = np.random.default_rng(2)
            with h5py.File(Path(tmp)/"dataset.h5", "w") as h5:
                h5.attrs.update(config_hash=digest(cfg), complete=True, frame="test_frame")
                for split,count in cfg["dataset"].items():
                    g=h5.create_group(split)
                    g.create_dataset("ft",data=rng.normal(size=(count,400,6)))
                    g.create_dataset("valid",data=np.ones(count,dtype=bool))
            write_json(Path(tmp)/"dataset_integrity.json",dict(sha256=file_digest(Path(tmp)/"dataset.h5")))
            train(cfg)
            original=torch.load(Path(tmp)/"vae_last.pt",weights_only=True)
            for epoch in (1,2,3):
                for name in ("vae.pt","training.png","reconstruction.png","validation.json"):
                    self.assertTrue((Path(tmp)/f"epoch_{epoch:04d}"/name).exists())
            self.assertEqual(original["frame"],"test_frame")
            # Restore a real intermediate checkpoint, then reproduce remaining epochs.
            from scripts.shared.checkpoints import save_checkpoint
            save_checkpoint(Path(tmp)/"vae_resume.pt",torch.load(Path(tmp)/"epoch_0001/vae.pt",weights_only=True))
            (Path(tmp)/"vae_last.pt").unlink()
            train(cfg,resume=True)
            resumed=torch.load(Path(tmp)/"vae_last.pt",weights_only=True)
            for key in original["model"]:
                torch.testing.assert_close(original["model"][key],resumed["model"][key],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
