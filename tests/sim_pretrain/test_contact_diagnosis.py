"""Check diagnostic wrench conventions against native MuJoCo generalized forces."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.sim_pretrain.experiments.diagnose_contact import contact_result


class ContactDiagnosticTests(unittest.TestCase):
    def test_contact_wrench_sign_origin_and_empty_case(self):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option cone="pyramidal"/><worldbody>
        <geom name="table" type="plane" size="1 1 .1"/>
        <body pos="0 0 .049"><freejoint/>
        <geom name="tool" type="box" size=".05 .04 .05" mass="1" friction="3.5 .005 .0001"/>
        </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        data.qvel[:3] = [0.1, -0.2, 0]
        mujoco.mj_forward(model, data)
        origin = np.array([0.013, -0.022, 0.1])
        result = contact_result(model, data, [1], 0, origin, np.arange(6))
        self.assertGreater(result["normal_sum"], 0)
        self.assertGreater(result["wrench"][2], 0)
        np.testing.assert_allclose(result["generalized"], data.qfrc_constraint, atol=1e-9)
        expected = np.zeros(6)
        mujoco.mj_applyFT(
            model, data, result["wrench"][:3], result["wrench"][3:], origin, 1, expected
        )
        np.testing.assert_allclose(expected, data.qfrc_constraint, atol=1e-9)
        opposite = contact_result(model, data, [0], 1, origin, np.arange(6))
        np.testing.assert_allclose(opposite["wrench"], -result["wrench"], atol=1e-9)
        data.qpos[2] = 0.5
        mujoco.mj_forward(model, data)
        empty = contact_result(model, data, [1], 0, origin, np.arange(6))
        self.assertEqual(empty["rows"].shape, (0, 16))
        np.testing.assert_array_equal(empty["wrench"], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
