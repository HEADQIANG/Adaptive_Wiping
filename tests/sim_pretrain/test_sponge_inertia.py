"""Geometry-based sponge inertia without changes to mass, geometry or arm dynamics."""

import copy
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts.shared.assets import ROBOSUITE_MODELS
from scripts.shared.common import ROOT, file_digest, load_config
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.inertia import set_sponge_inertia
from scripts.sim_pretrain.simulation import PretrainingWipe


class SpongeInertiaTests(unittest.TestCase):
    def test_formula_and_only_inertia_changes(self):
        root = ET.parse(ROBOSUITE_MODELS / "grippers/wiping_gripper.xml").getroot()
        before = copy.deepcopy(root)
        set_sponge_inertia(root, "", "uniform_box_v1")
        inertia = root.find(".//body[@name='wiping_gripper']/inertial")
        np.testing.assert_allclose(np.fromstring(inertia.get("diaginertia"), sep=" "),
                                   [8.5e-6, 3.825e-5, 4.225e-5], rtol=1e-12)
        inertia.set("diaginertia", before.find(".//inertial").get("diaginertia"))
        self.assertEqual(ET.tostring(root), ET.tostring(before))

    def test_legacy_and_invalid_geometry(self):
        root = ET.parse(ROBOSUITE_MODELS / "grippers/wiping_gripper.xml").getroot()
        before = ET.tostring(root)
        set_sponge_inertia(root, "")
        self.assertEqual(before, ET.tostring(root))
        with self.assertRaisesRegex(ValueError, "profile"):
            set_sponge_inertia(root, "", "invalid")
        root.find(".//inertial").set("pos", "0 0 0.01")
        before = ET.tostring(root)
        with self.assertRaisesRegex(ValueError, "COM"):
            set_sponge_inertia(root, "", "uniform_box_v1")
        self.assertEqual(before, ET.tostring(root))

    def test_compilation_reset_and_unchanged_physical_parameters(self):
        path = ROBOSUITE_MODELS / "grippers/wiping_gripper.xml"
        source_hash = file_digest(path)
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        self.assertEqual(cfg["simulation"]["tool_inertia_profile"], "uniform_box_v1")
        other = load_config(ROOT / "configs/sim_pretrain/pretrain_paper_mu1p2.yaml")
        self.assertEqual(other["simulation"]["tool_inertia_profile"], "uniform_box_v1")
        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg["simulation"].pop("tool_inertia_profile")
        env = PretrainingWipe(cfg, 300)
        self.addCleanup(env.close)
        legacy = PretrainingWipe(legacy_cfg, 300)
        self.addCleanup(legacy.close)
        m, old = env.sim.model, legacy.sim.model
        body = m.body_name2id(env.robot.gripper["right"].naming_prefix + "wiping_gripper")
        np.testing.assert_allclose(m.body_inertia[body], [8.5e-6, 3.825e-5, 4.225e-5])
        np.testing.assert_allclose(old.body_inertia[body], [0.01] * 3)
        others = np.arange(m.nbody) != body
        np.testing.assert_array_equal(m.body_inertia[others], old.body_inertia[others])
        for field in ("body_mass", "body_ipos", "body_iquat", "body_pos", "body_quat",
                      "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
                      "geom_solimp", "site_pos", "site_quat", "dof_armature", "dof_damping",
                      "dof_frictionloss", "jnt_range", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(m, field), getattr(old, field), err_msg=field)
        self.assertTrue(sensor_load_check(env)["passed"])
        env.reset()
        np.testing.assert_allclose(env.sim.model.body_inertia[body], [8.5e-6, 3.825e-5, 4.225e-5])
        self.assertEqual(file_digest(path), source_hash)


if __name__ == "__main__":
    unittest.main()
