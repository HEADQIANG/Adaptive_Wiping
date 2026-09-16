"""Central asset completeness and compiled numerical parity, without hardware."""

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.shared.assets import ASSETS, MANIFEST, ROBOSUITE_MODELS, asset_files, verify
from scripts.shared.common import load_config, provenance
from scripts.shared.paths import ROOT, ROBOSUITE_PACKAGE
from scripts.sim_pretrain.experiments.test_ur5e_contact import UR5eWipe, comparison_config
from scripts.sim_pretrain.simulation import PretrainingWipe

import robosuite


class ModelAssetTests(unittest.TestCase):
    def test_inventory_and_original_bytes_match(self):
        result = verify(sources=True)
        self.assertTrue(result["passed"], result["errors"])
        records = json.loads(MANIFEST.read_text())["files"]
        recorded = {ROOT / r["path"] for r in records}
        # The source-copy manifest covers runtime models, inertia references and
        # licenses. Mechanical CAD in asserts/connector is a separate resource.
        actual = {
            p
            for folder in (ROBOSUITE_MODELS, ASSETS / "references", ASSETS / "licenses")
            for p in folder.rglob("*")
            if p.is_file()
        }
        self.assertEqual(recorded, actual)
        self.assertTrue(all(not p.is_symlink() for p in actual))

    def test_xml_file_dependencies_stay_inside_central_root(self):
        for path in ROBOSUITE_MODELS.rglob("*.xml"):
            for element in ET.parse(path).iter():
                if element.get("file"):
                    target = (path.parent / element.get("file")).resolve()
                    self.assertTrue(target.is_relative_to(ROBOSUITE_MODELS), target)
                    self.assertTrue(target.is_file(), target)

    def test_new_provenance_tracks_central_resources(self):
        sources = provenance()["sources"]
        for path in asset_files():
            self.assertIn(path.relative_to(ROOT).as_posix(), sources)
        vendor_assets = (ROBOSUITE_PACKAGE / "models/assets").relative_to(ROOT).as_posix() + "/"
        self.assertFalse(any(p.startswith(vendor_assets) for p in sources))

    def test_airbot_and_ur5e_compile_and_match_vendor_dynamics(self):
        base = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        tabletop = copy.deepcopy(base)
        base["simulation"].update(base_mount="stand", tool_mount="legacy", inertia_profile="legacy")
        tabletop["simulation"].update(
            base_mount="tabletop",
            tool_mount="direct_wrist_v3",
            inertia_profile="discoverse_standard",
        )
        cases = (
            (PretrainingWipe, base),
            (PretrainingWipe, tabletop),
            (UR5eWipe, comparison_config(base, "ik_ff")),
        )
        self.assertEqual(Path(robosuite.models.assets_root), ROBOSUITE_MODELS)
        for factory, cfg in cases:
            with self.subTest(robot=factory.__name__, mount=cfg["simulation"].get("base_mount")):
                central = factory(cfg, 300)
                try:
                    with patch.object(
                        robosuite.models,
                        "assets_root",
                        str(ROBOSUITE_PACKAGE / "models/assets"),
                    ):
                        original = factory(cfg, 300)
                    try:
                        for element in central.model.asset:
                            if element.get("file"):
                                target = Path(element.get("file")).resolve()
                                self.assertTrue(target.is_relative_to(ROBOSUITE_MODELS), target)
                                self.assertTrue(target.is_file())
                        for name in (
                            "body_mass",
                            "body_inertia",
                            "body_pos",
                            "body_quat",
                            "geom_pos",
                            "geom_size",
                            "geom_friction",
                            "geom_solref",
                            "actuator_ctrlrange",
                            "jnt_range",
                            "mesh_vert",
                        ):
                            np.testing.assert_array_equal(
                                getattr(central.sim.model, name),
                                getattr(original.sim.model, name),
                                err_msg=name,
                            )
                        for _ in range(10):
                            central.sim.step()
                            original.sim.step()
                            for name in ("qpos", "qvel", "sensordata"):
                                np.testing.assert_array_equal(
                                    getattr(central.sim.data, name),
                                    getattr(original.sim.data, name),
                                    err_msg=name,
                                )
                    finally:
                        original.close()
                finally:
                    central.close()


if __name__ == "__main__":
    unittest.main()
