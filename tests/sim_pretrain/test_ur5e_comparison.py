"""UR5e comparison preserves native mechanics and leaves AIRBOT config alone."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.experiments.test_ur5e_contact import (
    UR5eWipe,
    comparison_config,
)


class UR5eComparisonTests(unittest.TestCase):
    def test_native_models_and_controller_variants(self):
        source = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        before = copy.deepcopy(source)
        for mode, gain, expected in (
            ("ik_ff", 300, "JointPositionFeedforward"),
            ("osc", 150, "OperationalSpaceController"),
        ):
            cfg = comparison_config(source, mode)
            self.assertEqual(source, before)
            env = UR5eWipe(cfg, gain, mode=mode)
            try:
                audit = env.audit()
                self.assertEqual(audit["robot"], "UR5e")
                self.assertEqual(audit["controller"], expected)
                np.testing.assert_allclose(audit["armature"], 5 / np.arange(1, 7))
                np.testing.assert_allclose(audit["frictionloss"], 0.01)
                np.testing.assert_allclose(audit["damping"], 0.001)
                np.testing.assert_allclose(
                    np.asarray(audit["actuator_ctrlrange"])[:, 1], [150] * 3 + [28] * 3
                )
                self.assertAlmostEqual(audit["tool_mass"], 0.03)
                q = env.sim.data.qpos.copy()
                self.assertTrue(sensor_load_check(env)["passed"])
                np.testing.assert_array_equal(q, env.sim.data.qpos)
                env.reset()
                self.assertEqual(env.audit(), audit)
                position, rotation = env.pose()
                env.command(position, rotation)
                controller = env.robot.part_controllers["right"]
                if mode == "osc":
                    np.testing.assert_allclose(controller.goal_pos, position)
                    np.testing.assert_allclose(
                        controller.goal_ori, rotation.as_matrix(), atol=1e-12
                    )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
