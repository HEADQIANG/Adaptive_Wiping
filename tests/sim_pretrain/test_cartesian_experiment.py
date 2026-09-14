"""Experimental controller semantics and isolated fixture checks."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import opspace_matrices, orientation_error

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.analyze_contact_experiment import (
    quasistatic_budget,
)
from scripts.sim_pretrain.experiments.contact_breakaway import build_rig
from scripts.sim_pretrain.experiments.contact_control_experiment import (
    CANDIDATES,
    ExperimentWipe,
)
from scripts.sim_pretrain.simulation import PretrainingWipe


class CartesianExperimentTests(unittest.TestCase):
    def test_stock_osc_feedforward_equation_and_real_command(self):
        env = ExperimentWipe(
            load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"),
            (0.9, 1000, 0.02),
            CANDIDATES["osc_ff"],
        )
        try:
            start = env.prepare(0.08)
            env.command(start + [0, 0.0005, 0])
            controller = env.robot.part_controllers["right"]
            np.testing.assert_allclose(controller.nominal_twist, [0, 0.05, 0, 0, 0, 0], atol=1e-12)
            stock = OperationalSpaceController.run_controller(controller).copy()
            _, lp, lr, _ = opspace_matrices(
                controller.mass_matrix, controller.J_full, controller.J_pos, controller.J_ori
            )
            expected = (
                stock
                + controller.J_full.T
                @ np.r_[
                    lp @ (controller.kd[:3] * controller.nominal_twist[:3]),
                    lr @ (controller.kd[3:] * controller.nominal_twist[3:]),
                ]
            )
            np.testing.assert_allclose(controller.run_controller(), expected, atol=1e-12)
            np.testing.assert_allclose(controller.nominal_twist, [0, 0.05, 0, 0, 0, 0], atol=1e-12)
        finally:
            env.close()

    def test_quasistatic_wrench_shift_and_balance(self):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body pos="0 0 1"><freejoint/>
        <geom type="box" size=".05 .04 .03" mass="1"/>
        <site name="tcp" pos="0 0 -.03"/><site name="ft" pos="0 0 .03"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        wrench = np.array([0, -3, 1, -0.1, 0, 0])
        direct = np.zeros(6)
        mujoco.mj_applyFT(model, data, wrench[:3], wrench[3:], data.site_xpos[1], 1, direct)
        result = quasistatic_budget(model, data, 0, 1, np.arange(6), wrench, np.full(6, 10))
        np.testing.assert_allclose(
            result["required_peak_abs_nm"], np.abs(data.qfrc_bias - direct), atol=1e-12
        )
        self.assertTrue(result["within_limits"])
        data.qvel[0] = 0.1
        with self.assertRaises(ValueError):
            quasistatic_budget(model, data, 0, 1, np.arange(6), wrench, np.full(6, 10))

    def test_controller_twist_torque_and_reset(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        env = ExperimentWipe(cfg, (0.9, 1000, 0.02), CANDIDATES["cart_2000"])
        try:
            start = env.prepare(0.08)
            controller = env.robot.part_controllers["right"]
            action = np.r_[start + [0, 0.0005, 0], env.goal_rotation.as_rotvec()]
            controller.set_goal(action)
            np.testing.assert_allclose(controller.nominal_twist, [0, 0.05, 0, 0, 0, 0], atol=1e-12)
            torque = controller.run_controller()
            error = np.r_[
                controller.goal_pos - controller.ref_pos,
                orientation_error(controller.goal_ori, controller.ref_ori_mat),
            ]
            velocity = np.r_[controller.ref_pos_vel, controller.ref_ori_vel]
            expected = (
                controller.J_full.T
                @ (controller.kp * error + controller.kd * (controller.nominal_twist - velocity))
                + controller.torque_compensation
            )
            np.testing.assert_allclose(torque, expected)
            q = env.sim.data.qpos.copy()
            env.sim.data.qpos[0] += 0.001
            env.sim.forward()
            action[1] += 0.0005
            controller.set_goal(action)
            np.testing.assert_allclose(controller.nominal_twist, [0, 0.05, 0, 0, 0, 0], atol=1e-12)
            env.sim.data.qpos[:] = q
            env.sim.forward()
            limits = np.asarray(cfg["simulation"]["torque_limits"])
            np.testing.assert_array_equal(
                env.sim.model.actuator_ctrlrange[:6], np.c_[-limits, limits]
            )
            controller.reset_goal()
            np.testing.assert_array_equal(controller.nominal_twist, np.zeros(6))
            start = env.prepare(0.08)
            self.assertEqual(env.robot.composite_controller.name, "BASIC")
            self.assertEqual(env.robot.composite_controller._action_split_indexes["right"], (0, 6))
            self.assertEqual(env.action_dim, 6)
            env.command(start + [0, 0.0005, 0])
            controller = env.robot.part_controllers["right"]
            np.testing.assert_allclose(controller.goal_pos, start + [0, 0.0005, 0])
            np.testing.assert_allclose(controller.nominal_twist, [0, 0.05, 0, 0, 0, 0], atol=1e-12)
            self.assertTrue(np.isfinite(env.sim.data.qpos).all())
        finally:
            env.close()

    def test_fixture_preserves_native_and_mapped_contact_parameters(self):
        env = PretrainingWipe(load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"), 300)
        try:
            original = env.sim.model.geom_friction.copy()
            native, d, ids, table = build_rig(env, "native", (0.9, 1000, 0.02))
            friction, _, fids, _ = build_rig(env, "friction_only", (0.9, 1000, 0.02))
            mapped, _, mids, mt = build_rig(env, "mapped", (0.9, 1000, 0.02))
            self.assertEqual((native.nq, native.nv), (2, 2))
            for name in dir(native.opt):
                if not name.startswith("_"):
                    np.testing.assert_array_equal(
                        getattr(native.opt, name), getattr(env.sim.model._model.opt, name)
                    )
            np.testing.assert_allclose(native.geom_friction[ids, 0], 0.001)
            np.testing.assert_allclose(friction.geom_friction[fids, 0], 0.9)
            np.testing.assert_array_equal(native.geom_solref[ids], friction.geom_solref[fids])
            np.testing.assert_array_equal(native.geom_solimp[ids], friction.geom_solimp[fids])
            np.testing.assert_allclose(
                mapped.geom_solref[mids], np.tile([-1000, -2 * np.sqrt(1000)], (len(mids), 1))
            )
            self.assertTrue(np.all(mapped.geom_priority[mids] > mapped.geom_priority[mt]))
            np.testing.assert_array_equal(env.sim.model.geom_friction, original)
            rotation = d.xmat[1].copy()
            d.qpos[:] = [0.01, -0.001]
            mujoco.mj_forward(native, d)
            np.testing.assert_array_equal(d.xmat[1], rotation)
            np.testing.assert_allclose(d.xpos[1], [0, 0.01, 0.016], atol=1e-12)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
