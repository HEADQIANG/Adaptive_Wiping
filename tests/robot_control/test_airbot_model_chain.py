import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_KDL = importlib.util.find_spec("PyKDL") is not None
if HAS_KDL:
    pass
    try:
        from scripts.robot_control.tools.audit_airbot_model_chain import (
            chain_from_urdf,
            metrics,
            solve,
        )
    finally:
        pass


@unittest.skipUnless(HAS_KDL, "Run with system python3 and python3-pykdl")
class ModelChainTests(unittest.TestCase):
    def test_rotated_joint_axis_and_fixed_tip(self):
        xml = """<robot name="fixture">
        <joint name="hinge" type="revolute">
          <parent link="base_link"/><child link="arm"/>
          <origin xyz="1 2 3" rpy="1.5707963267948966 0 0"/>
          <axis xyz="0 0 1"/>
        </joint>
        <joint name="tip" type="fixed">
          <parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/>
        </joint></robot>"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.urdf"
            path.write_text(xml)
            chain = chain_from_urdf(path, "tip")
            np.testing.assert_allclose(solve(chain, [0])[:3, 3], [2, 2, 3], atol=1e-12)
            np.testing.assert_allclose(solve(chain, [np.pi / 2])[:3, 3], [1, 2, 4], atol=1e-12)
            with self.assertRaises(ValueError):
                solve(chain, [])
            with self.assertRaises(ValueError):
                chain_from_urdf(path, "missing")

    def test_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.urdf"
            path.write_text(
                '<robot><joint name="cycle" type="fixed"><parent link="tip"/><child link="tip"/></joint></robot>'
            )
            with self.assertRaises(ValueError):
                chain_from_urdf(path, "tip")

    def test_known_translation_metrics(self):
        transform = np.eye(4)
        transform[0, 3] = 0.001
        result = metrics([transform, transform], np.eye(4))
        self.assertAlmostEqual(result["position_rms_mm"], 1)
        self.assertAlmostEqual(result["orientation_rms_deg"], 0)


if __name__ == "__main__":
    unittest.main()
