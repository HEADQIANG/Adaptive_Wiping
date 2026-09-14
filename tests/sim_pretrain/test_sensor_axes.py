"""Nominal KWR52 mounting axes, independent of physical sensor calibration."""

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import mock_open, patch

import numpy as np

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.simulation import PretrainingWipe
from scripts.sim_pretrain.tool_mount import orient_ft_frame


class SensorAxesTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")

    def test_profile_validation_and_legacy_default(self):
        for name in ("pretrain_paper.yaml", "pretrain_paper_mu1p2.yaml"):
            cfg = load_config(ROOT / "configs/sim_pretrain" / name)
            self.assertEqual(cfg["simulation"]["ft_frame_profile"], "kwr52_left_v1")
            self.assertTrue(cfg["output_dir"].endswith("kwr52_left_v1"))
        cfg = self.config()
        cfg["simulation"]["ft_frame_profile"] = "invalid"
        with patch("pathlib.Path.open", mock_open(read_data=json.dumps(cfg))):
            with self.assertRaisesRegex(ValueError, "ft_frame_profile"):
                load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        root = ET.Element("worldbody")
        orient_ft_frame(root, "gripper0_")
        self.assertEqual(ET.tostring(root), b"<worldbody />")
        with self.assertRaisesRegex(ValueError, "Unknown FT"):
            orient_ft_frame(root, "gripper0_", "invalid")
        with self.assertRaisesRegex(ValueError, "audited AIRBOT"):
            orient_ft_frame(root, "gripper0_", "kwr52_left_v1")

    def test_zero_pose_moving_axes_wrench_and_unchanged_physics(self):
        cfg = self.config()
        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg["simulation"].pop("ft_frame_profile")
        env = PretrainingWipe(cfg, 300)
        self.addCleanup(env.close)
        legacy = PretrainingWipe(legacy_cfg, 300)
        self.addCleanup(legacy.close)
        m, d = env.sim.model, env.sim.data
        old_m, old_d = legacy.sim.model, legacy.sim.data
        prefix = env.robot.gripper["right"].naming_prefix
        sensor = m.site_name2id(prefix + "ft_frame")
        old_sensor = old_m.site_name2id(prefix + "ft_frame")
        tool = m.body_name2id(prefix + "wiping_gripper")
        base = m.body_name2id("robot0_base")
        c, s = np.sqrt(3) / 2, 0.5
        tool_from_sensor = np.array([[-c, -s, 0], [s, -c, 0], [0, 0, 1]])
        for field in (
            "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
            "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity",
            "geom_friction", "geom_solref", "geom_solimp", "site_pos", "jnt_range",
            "actuator_ctrlrange",
        ):
            np.testing.assert_array_equal(getattr(m, field), getattr(old_m, field), err_msg=field)
        other_sites = np.arange(m.nsite) != sensor
        np.testing.assert_array_equal(m.site_quat[other_sites], old_m.site_quat[other_sites])
        for q in ([0] * 6, [0.2, -1.2, 1.1, 1.3, -1, -1.1]):
            d.qpos[:] = old_d.qpos[:] = q
            d.qvel[:] = old_d.qvel[:] = 0
            d.xfrc_applied[:] = old_d.xfrc_applied[:] = 0
            d.xfrc_applied[tool] = old_d.xfrc_applied[tool] = [.7, -.3, .9, .02, .05, -.04]
            env.sim.forward()
            legacy.sim.forward()
            rotation = d.site_xmat[sensor].reshape(3, 3)
            old_rotation = old_d.site_xmat[old_sensor].reshape(3, 3)
            np.testing.assert_allclose(rotation, old_rotation @ tool_from_sensor, atol=1e-11)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            np.testing.assert_allclose(d.site_xpos[sensor], old_d.site_xpos[old_sensor], atol=1e-12)
            np.testing.assert_allclose(d.site_xpos[env.site_id], old_d.site_xpos[legacy.site_id], atol=1e-12)
            np.testing.assert_allclose(d.geom_xpos, old_d.geom_xpos, atol=1e-12)
            np.testing.assert_allclose(
                env.wrench().reshape(2, 3),
                legacy.wrench().reshape(2, 3) @ tool_from_sensor,
                atol=1e-10,
            )
            if not any(q):
                # Independent physical target: forward=base X, left=base Y, up=base Z.
                base_from_sensor = d.xmat[base].reshape(3, 3).T @ rotation
                expected = np.array([[0, 0, 1], [c, s, 0], [-s, c, 0]])
                np.testing.assert_allclose(base_from_sensor, expected, atol=1e-5)
                np.testing.assert_allclose(c * base_from_sensor[:, 0] + s * base_from_sensor[:, 1],
                                           [0, 1, 0], atol=1e-5)
        for _ in range(10):
            env.sim.step()
            legacy.sim.step()
            np.testing.assert_allclose(d.qpos, old_d.qpos, atol=1e-12, rtol=0)
            np.testing.assert_allclose(d.qvel, old_d.qvel, atol=1e-12, rtol=0)
        self.assertTrue(sensor_load_check(env)["passed"])
        saved_quat = m.site_quat[sensor].copy()
        env.reset()
        np.testing.assert_array_equal(env.sim.model.site_quat[sensor], saved_quat)
        env.sim.forward()
        np.testing.assert_allclose(
            env.sim.data.site_xmat[sensor].reshape(3, 3),
            env.sim.data.site_xmat[env.site_id].reshape(3, 3) @ tool_from_sensor,
            atol=1e-11,
        )

    def test_ur5e_comparison_keeps_original_axes(self):
        from scripts.sim_pretrain.experiments.test_ur5e_contact import comparison_config

        self.assertEqual(comparison_config(self.config(), "ik_ff")["simulation"]["ft_frame_profile"],
                         "legacy")


if __name__ == "__main__":
    unittest.main()
