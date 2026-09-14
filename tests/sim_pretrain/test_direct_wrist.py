"""Direct wrist assembly, unchanged arm inertia, and sensor-frame checks."""

import copy
import unittest

import numpy as np

from scripts.shared.common import ROOT, file_digest, load_config
from scripts.shared.assets import ROBOSUITE_MODELS
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.simulation import PretrainingWipe
from scripts.sim_pretrain.wrist import WRIST_FACE_Z, WRIST_RADIUS, WRIST_REAR_Z


class DirectWristTests(unittest.TestCase):
    def test_compiled_geometry_dynamics_and_reset(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        cfg["simulation"]["tool_mount"] = "direct_wrist_v3"
        old_cfg = copy.deepcopy(cfg)
        old_cfg["simulation"]["tool_mount"] = "compact_v2"
        path = ROBOSUITE_MODELS / "robots/airbot_play/meshes/link6.obj"
        original_hash = file_digest(path)
        env, old = PretrainingWipe(cfg, 300), PretrainingWipe(old_cfg, 300)
        try:
            m, d = env.sim.model, env.sim.data
            prefix = env.robot.gripper["right"].naming_prefix
            link = m.body_name2id("robot0_link6")
            tool = m.body_name2id(prefix + "wiping_gripper")
            hand = m.body_name2id("robot0_right_hand")
            ft = m.site_name2id(prefix + "ft_frame")
            self.assertEqual((m.nq, m.nv), (6, 6))
            self.assertEqual(m.body_parentid[tool], hand)
            self.assertAlmostEqual(m.body_mass[tool], 0.03)
            self.assertFalse(
                any("mount_backplate" in name or "mount_adapter" in name for name in m.geom_names)
            )
            np.testing.assert_allclose(m.body_pos[tool], [0, 0, WRIST_FACE_Z])
            np.testing.assert_allclose(m.site_pos[ft], [0, 0, -0.015])
            np.testing.assert_array_equal(m.actuator_ctrlrange, old.sim.model.actuator_ctrlrange)
            for i in range(1, 7):
                a, b = (
                    m.body_name2id(f"robot0_link{i}"),
                    old.sim.model.body_name2id(f"robot0_link{i}"),
                )
                for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
                    np.testing.assert_array_equal(
                        getattr(m, field)[a], getattr(old.sim.model, field)[b]
                    )
            source_link = env.model.worldbody.find(".//body[@name='robot0_link6']")
            self.assertIsNone(source_link.find("geom[@mesh='robot0_airbot_camera_stand']"))
            cylinder = source_link.find("geom[@group='0']")
            self.assertEqual(cylinder.get("type"), "cylinder")
            np.testing.assert_allclose(
                np.fromstring(cylinder.get("size"), sep=" "),
                [WRIST_RADIUS, (WRIST_FACE_Z - WRIST_REAR_Z) / 2],
            )
            wrist = env.model.asset.find("mesh[@name='robot0_airbot_link6']")
            vertices = np.fromstring(wrist.get("vertex"), sep=" ").reshape(-1, 3)
            np.testing.assert_allclose(
                vertices.min(axis=0), [-WRIST_RADIUS, -WRIST_RADIUS, WRIST_REAR_Z], atol=1e-7
            )
            np.testing.assert_allclose(
                vertices.max(axis=0), [WRIST_RADIUS, WRIST_RADIUS, WRIST_FACE_Z], atol=1e-7
            )
            self.assertIsNone(wrist.get("file"))
            for q in ([0, -1, 1.2, 1.5, -1.2, -1.5], [0.2, -1.2, 1.1, 1.3, -1, -1.1]):
                d.qpos[:] = old.sim.data.qpos[:] = q
                env.sim.forward()
                old.sim.forward()
                support = d.xpos[link] + d.xmat[link].reshape(3, 3) @ [0, 0, WRIST_FACE_Z]
                np.testing.assert_allclose(d.site_xpos[ft], support, atol=1e-12)
                shift = d.xmat[hand].reshape(3, 3) @ [0, 0, WRIST_FACE_Z + 0.0494]
                np.testing.assert_allclose(env.pose()[0] - old.pose()[0], shift, atol=1e-12)
                for gid in env.tool_ids:
                    old_gid = old.sim.model.geom_name2id(m.geom_id2name(gid))
                    np.testing.assert_allclose(
                        d.geom_xpos[gid] - old.sim.data.geom_xpos[old_gid], shift, atol=1e-12
                    )
            self.assertTrue(sensor_load_check(env)["passed"])
            env.reset()
            np.testing.assert_allclose(env.sim.model.body_pos[tool], [0, 0, WRIST_FACE_Z])
            self.assertEqual(file_digest(path), original_hash)
        finally:
            env.close()
            old.close()


if __name__ == "__main__":
    unittest.main()
