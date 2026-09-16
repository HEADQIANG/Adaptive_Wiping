import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.shared.common import digest, file_digest, load_config, write_json
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.train_collected_subset import prepare


class CollectedSubsetTests(unittest.TestCase):
    def test_preserves_source_and_disjoint_mapping_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/"source", Path(tmp)/"runs/sim_training/out"
            storage = patch("scripts.shared.paths.RUNS", Path(tmp)/"runs")
            storage.start()
            self.addCleanup(storage.stop)
            source.mkdir()
            out.mkdir(parents=True)
            cfg = load_config("configs/sim_pretrain/pretrain_wide_1200.yaml")
            cfg["dataset"] = dict(train=4,validation=1,test=1)
            cfg["output_dir"] = str(out)
            original_cfg = copy.deepcopy(cfg)
            original_cfg["dataset"] = dict(train=8)
            manifest = dict(config=original_cfg,provenance={})
            write_json(source/"manifest.json",manifest)
            with h5py.File(source/"dataset.h5","w") as h5:
                h5.attrs.update(config_hash=digest(original_cfg),provenance_json=json.dumps(manifest),complete=False)
                g = h5.create_group("train")
                g.create_dataset("valid",data=np.ones(8,dtype=bool))
                g.create_dataset("parameters",data=np.tile([.5,500,.05],(8,1)))
                g.create_dataset("gain",data=np.full(8,5000))
                for key,shape in FIELDS.items():
                    a=np.zeros((8,*shape))
                    if key == "time": a[:]=np.arange(1,401)/100
                    if key == "ft":
                        for i in range(8): a[i]=i
                    g.create_dataset(key,data=a)
                diagnostic=json.dumps(dict(motion=dict(passed=False),metrics=dict(saturation_fraction=1)))
                g.create_dataset("diagnostics",data=[diagnostic]*8,dtype=h5py.string_dtype())
                g.create_dataset("error",data=[""]*8,dtype=h5py.string_dtype())
            before=file_digest(source/"dataset.h5")
            prepare(cfg,source,out)
            self.assertTrue((out/"dataset.h5").is_symlink())
            self.assertTrue((Path(tmp)/"runs/sim_data/out/dataset.h5").is_file())
            self.assertEqual(file_digest(source/"dataset.h5"),before)
            with h5py.File(out/"dataset.h5","r") as h5:
                self.assertTrue(h5.attrs["complete"])
                indices=[]
                for split,size in cfg["dataset"].items():
                    self.assertEqual(h5[split]["ft"].shape,(size,400,6))
                    for i,index in enumerate(h5[split]["source_index"][:]):
                        indices.append(int(index))
                        np.testing.assert_array_equal(h5[split]["ft"][i],np.full((400,6),index))
                self.assertEqual(sorted(indices),list(range(6)))
            target_hash=file_digest(out/"dataset.h5")
            prepare(cfg,source,out)
            self.assertEqual(target_hash,file_digest(out/"dataset.h5"))
            report=json.loads((out/"collection.json").read_text())
            self.assertEqual(report["motion_passed"],0)
            self.assertEqual(report["any_saturation"],6)
            (source/"vae_best.pt").touch()
            with self.assertRaises(ValueError): prepare(cfg,source,out)


if __name__ == "__main__":
    unittest.main()
