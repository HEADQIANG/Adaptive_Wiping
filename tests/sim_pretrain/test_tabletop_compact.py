"""Tabletop installation and compact fixed tool regression checks."""

import copy
import unittest

import numpy as np

from scripts.shared.common import ROOT, load_config
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.simulation import PretrainingWipe


class TabletopCompactTests(unittest.TestCase):
    def test_compiled_assembly_and_reset(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["simulation"]["tool_mount"] = "compact_v2"
        env = PretrainingWipe(cfg, 300)
        old_cfg = copy.deepcopy(cfg)
        old_cfg["simulation"]["tool_mount"] = "bridge_v1"
        old = PretrainingWipe(old_cfg, 300)
        try:
            m, d = env.sim.model, env.sim.data
            prefix = env.robot.gripper["right"].naming_prefix
            self.assertEqual((m.nq, m.nv), (6, 6))
            self.assertEqual(type(env.robot.robot_model.base).__name__, "NullMount")
            self.assertFalse(any("mount_adapter" in n for n in m.body_names))
            self.assertFalse(any("rail_" in n or "torso" in n for n in m.geom_names))
            root = m.body_name2id("robot0_base")
            np.testing.assert_allclose(m.body_pos[root], [0.07, 0, 0.8975], atol=1e-8)
            plane = m.site_name2id("robot0_table_mount_plane")
            self.assertAlmostEqual(d.site_xpos[plane, 2], 0.9)
            tool = m.body_name2id(prefix + "wiping_gripper")
            hand = m.body_name2id("robot0_right_hand")
            self.assertEqual(m.body_parentid[tool], hand)
            np.testing.assert_allclose(m.body_pos[tool], [0, 0, -0.0494])
            self.assertAlmostEqual(m.body_mass[tool], 0.0624)
            plate = m.geom_name2id(prefix + "mount_backplate_visual")
            sponge = m.geom_name2id(prefix + "wiping_surface_vis")
            self.assertAlmostEqual(
                m.geom_pos[plate, 2] + m.geom_size[plate, 2],
                m.geom_pos[sponge, 2] - m.geom_size[sponge, 2],
            )
            self.assertAlmostEqual(
                0.015 + m.body_pos[tool, 2] + m.geom_pos[plate, 2] - m.geom_size[plate, 2], -0.0514
            )
            ft = m.site_name2id(prefix + "ft_frame")
            np.testing.assert_allclose(m.site_pos[ft], [0, 0, -0.017])
            for q in ([0, -1, 1.2, 1.5, -1.2, -1.5], [0.2, -1.2, 1.1, 1.3, -1, -1.1]):
                d.qpos[:] = old.sim.data.qpos[:] = q
                env.sim.forward()
                old.sim.forward()
                shift = d.xmat[hand].reshape(3, 3) @ [0, 0, -0.0644]
                np.testing.assert_allclose(env.pose()[0] - old.pose()[0], shift, atol=1e-12)
                for name in [m.geom_id2name(i) for i in env.tool_ids] + [
                    prefix + "wiping_surface_vis"
                ]:
                    i, j = m.geom_name2id(name), old.sim.model.geom_name2id(name)
                    np.testing.assert_allclose(
                        d.geom_xpos[i] - old.sim.data.geom_xpos[j], shift, atol=1e-12
                    )
                np.testing.assert_allclose(
                    d.site_xpos[ft] - old.sim.data.site_xpos[ft], shift, atol=1e-12
                )
            self.assertTrue(sensor_load_check(env)["passed"])
            env.reset()
            self.assertAlmostEqual(env.sim.data.site_xpos[plane, 2], 0.9)
            np.testing.assert_allclose(env.sim.model.body_pos[tool], [0, 0, -0.0494])
        finally:
            env.close()
            old.close()

    def test_outside_table_rejected(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["simulation"]["base_xy"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "footprint"):
            PretrainingWipe(cfg, 300)

    def test_historical_stand_remains_available(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["simulation"].pop("base_mount")
        cfg["simulation"]["tool_mount"] = "legacy"
        env = PretrainingWipe(cfg, 300)
        try:
            self.assertEqual(type(env.robot.robot_model.base).__name__, "RethinkMount")
            self.assertTrue(any("torso" in n for n in env.sim.model.geom_names))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
