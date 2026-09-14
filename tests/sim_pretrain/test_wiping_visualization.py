"""Visualization records the real rollout without changing its dynamics."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.shared.common import ROOT, load_config
from scripts.sim_pretrain.experiments.visualize_wiping import (
    RecordedWipe,
    configure_view,
    decorate,
    restore_frame,
)
from scripts.sim_pretrain.simulation import PretrainingWipe


class VisualizationTests(unittest.TestCase):
    def test_recording_and_replay_preserve_real_rollout(self):
        cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        baseline = PretrainingWipe(cfg, 300, (3.5, 1000, 0.02))
        try:
            expected = baseline.rollout()
        finally:
            baseline.close()
        env = RecordedWipe(cfg, 300, (3.5, 1000, 0.02))
        try:
            actual = env.rollout()
            np.testing.assert_array_equal(actual["position"], expected["position"])
            np.testing.assert_array_equal(actual["ft"], expected["ft"])
            self.assertEqual(len(env.frames), 401)
            data = mujoco.MjData(env.sim.model._model)
            live_qpos = env.sim.data.qpos.copy()
            for i in (0, 100, 200, 300, 400):
                restore_frame(env.sim.model._model, data, env.frames[i])
                np.testing.assert_allclose(
                    data.site_xpos[env.site_id], env.frames[i]["position"], atol=1e-12
                )
            np.testing.assert_array_equal(env.sim.data.qpos, live_qpos)
            scene = mujoco.MjvScene(env.sim.model._model, maxgeom=1000)
            decorate(scene, env.frames, 400)
            self.assertGreater(scene.ngeom, 2)
            camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
            configure_view(camera, option, False)
            self.assertEqual(option.geomgroup[0], 0)
            self.assertEqual(option.geomgroup[1], 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
