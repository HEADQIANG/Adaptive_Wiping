"""Engineering mount geometry, inertia and sensor regression checks."""

import copy
import unittest

import numpy as np

from scripts.shared.common import ROOT, load_config
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.simulation import PretrainingWipe
from scripts.sim_pretrain.tool_mount import BRIDGE_BOXES, box_mass_inertia


class ToolMountTests(unittest.TestCase):
    def test_compiled_mount_and_legacy_kinematics(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["simulation"]["tool_mount"] = "bridge_v1"
        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg["simulation"]["tool_mount"] = "legacy"
        legacy = PretrainingWipe(legacy_cfg, 300)
        env = PretrainingWipe(cfg, 300)
        try:
            m, d = env.sim.model, env.sim.data
            prefix = env.robot.gripper["right"].naming_prefix
            adapter = m.body_name2id(prefix + "mount_adapter")
            tool = m.body_name2id(prefix + "wiping_gripper")
            self.assertEqual(m.body_parentid[tool], adapter)
            self.assertEqual(m.nv, 6)
            self.assertEqual(m.nq, 6)
            self.assertEqual(m.body_jntnum[adapter], 0)
            self.assertAlmostEqual(
                m.body_mass[adapter], sum(box_mass_inertia(s)[0] for _, _, s in BRIDGE_BOXES)
            )
            self.assertAlmostEqual(m.body_mass[tool], 0.0624)
            sensor = m.site_name2id(prefix + "ft_frame")
            np.testing.assert_allclose(m.site_pos[sensor], [0, 0, -0.017])
            for name, pos, size in BRIDGE_BOXES:
                visual = m.geom_name2id(prefix + "mount_" + name + "_visual")
                collision = m.geom_name2id(prefix + "mount_" + name + "_collision")
                np.testing.assert_array_equal(m.geom_pos[visual], m.geom_pos[collision])
                np.testing.assert_array_equal(m.geom_size[visual], m.geom_size[collision])
                self.assertNotIn(collision, env.tool_ids)
                self.assertEqual(m.geom_contype[visual], 0)
                self.assertEqual(m.geom_contype[collision], 1)
            for q in ([0, -1, 1.2, 1.5, -1.2, -1.5], [0.2, -1.2, 1.1, 1.3, -1, -1.1]):
                d.qpos[:] = q
                legacy.sim.data.qpos[:] = q
                env.sim.forward()
                legacy.sim.forward()
                np.testing.assert_allclose(env.pose()[0], legacy.pose()[0], atol=1e-12)
                np.testing.assert_allclose(
                    d.geom_xpos[env.tool_ids],
                    legacy.sim.data.geom_xpos[legacy.tool_ids],
                    atol=1e-12,
                )
            self.assertTrue(sensor_load_check(env)["passed"])
            env.reset()
            self.assertAlmostEqual(env.sim.model.body_mass[adapter], m.body_mass[adapter])
        finally:
            env.close()
            legacy.close()

    def test_plates_and_rails_meet(self):
        _, plate_pos, plate_size = BRIDGE_BOXES[0]
        for _, pos, size in BRIDGE_BOXES[1:]:
            self.assertAlmostEqual(plate_pos[2] + plate_size[2], pos[2] - size[2])
            self.assertAlmostEqual(pos[2] + size[2], -0.002)
        # Gripper is +.015 relative to right_hand; backing interface is -.017 in gripper.
        self.assertAlmostEqual(0.015 - 0.017, -0.002)


if __name__ == "__main__":
    unittest.main()
