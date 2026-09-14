"""Engineering gate boundaries and isolated nominal-reference invariants."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.shared.common import ROOT, load_config
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import FIELDS, collect, sanity
from scripts.sim_pretrain.reference import NominalJointReference
from scripts.sim_pretrain.simulation import PretrainingWipe, exploration_offsets


def motion_fixture():
    times = np.arange(1, 401) / 100
    target = exploration_offsets(times)
    return {
        "time": times,
        "position": target.copy(),
        "target_position": target,
        "orientation_error": np.zeros(400),
        "contact_count": np.ones(400),
        "saturated": np.zeros((400, 6), dtype=bool),
    }


class ContactGateTests(unittest.TestCase):
    def test_sanity_and_collector_reject_static_contact(self):
        class FakeEnv:
            def __init__(self, *args):
                pass

            def set_parameters(self, parameters):
                pass

            def rollout(self, extra_height=0):
                data = {key: np.zeros(shape) for key, shape in FIELDS.items()}
                data.update(motion_fixture())
                if extra_height:
                    data["contact_count"][:] = 0
                else:
                    data["position"][:, 1] = 0
                return data

            def unload(self, data):
                return [0.0] * 6

            def close(self):
                pass

        with tempfile.TemporaryDirectory(prefix="contact-gate-test-") as tmp:
            cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
            cfg["output_dir"] = tmp
            cfg["dataset"] = {"train": 1}
            with (
                patch("scripts.sim_pretrain.simulation.PretrainingWipe", FakeEnv),
                patch(
                    "scripts.sim_pretrain.diagnostics.sensor_load_check",
                    return_value={"passed": True},
                ),
            ):
                report = sanity(cfg)
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["grid"]), 27)
                self.assertTrue(all(not row["contact_motion"]["passed"] for row in report["grid"]))
                with self.assertRaises(RuntimeError):
                    collect(cfg)
                self.assertFalse((Path(tmp) / "dataset.h5").exists())
                with patch(
                    "scripts.sim_pretrain.collection.require_sanity",
                    return_value={"selected_gain": 300},
                ):
                    result = collect(cfg)
                self.assertFalse(result["complete"])
                self.assertEqual(result["valid_counts"]["train"], 0)

    def test_correct_motion_and_normal_compliance(self):
        data = motion_fixture()
        data["position"][:, 2] += 0.02
        result = contact_motion_acceptance(data)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["metrics"]["forward_travel_m"], 0.05)
        self.assertAlmostEqual(result["metrics"]["reverse_travel_m"], 0.05)

    def test_static_and_missing_return_fail(self):
        data = motion_fixture()
        data["position"][:, 1] = 0
        self.assertFalse(contact_motion_acceptance(data)["passed"])
        data = motion_fixture()
        data["position"][300:, 1] = 0.05
        result = contact_motion_acceptance(data)
        self.assertIn("reverse_travel", result["failed_checks"])
        self.assertIn("return_error", result["failed_checks"])

    def test_contact_fraction_gap_pose_and_saturation(self):
        data = motion_fixture()
        data["contact_count"][200:205] = 0
        self.assertTrue(contact_motion_acceptance(data)["passed"])
        data["contact_count"][205] = 0
        result = contact_motion_acceptance(data)
        self.assertIn("forward_contact", result["failed_checks"])
        self.assertIn("forward_contact_gap", result["failed_checks"])
        data = motion_fixture()
        data["orientation_error"][0] = np.deg2rad(2.001)
        self.assertIn("orientation", contact_motion_acceptance(data)["failed_checks"])
        data = motion_fixture()
        data["saturated"][:4, 0] = True
        self.assertTrue(contact_motion_acceptance(data)["passed"])
        data["saturated"][4, 0] = True
        self.assertIn("saturation", contact_motion_acceptance(data)["failed_checks"])

    def test_lateral_error_and_invalid_data(self):
        for axis, name, offset in ((0, "x_error_max", 0.0031), (1, "y_error_max", 0.0051)):
            data = motion_fixture()
            data["position"][:, axis] += offset
            self.assertIn(name, contact_motion_acceptance(data)["failed_checks"])
        data = motion_fixture()
        data["position"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            contact_motion_acceptance(data)


class NominalReferenceTests(unittest.TestCase):
    def test_live_state_independence_velocity_and_guards(self):
        env = PretrainingWipe(load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"), 300)
        try:
            live = env.robot.composite_controller.joint_action_policy
            a, b = NominalJointReference(live), NominalJointReference(live)
            pos, rot = env.pose()
            q_before = env.sim.data.qpos.copy()
            v_before = env.sim.data.qvel.copy()
            previous = a.last_goal.copy()
            for index in range(1, 5):
                target = np.r_[pos + [0, 0.0001 * index, 0], rot.as_rotvec()]
                goal_a = a.solve(target)
                np.testing.assert_array_equal(env.sim.data.qpos, q_before)
                np.testing.assert_array_equal(env.sim.data.qvel, v_before)
                env.sim.data.qpos[live.dof_ids] += 0.01
                env.sim.forward()
                goal_b = b.solve(target)
                np.testing.assert_array_equal(goal_a, goal_b)
                np.testing.assert_array_equal(a.velocity, b.velocity)
                np.testing.assert_allclose(a.velocity, (goal_a - previous) / 0.01)
                self.assertLessEqual(np.abs(a.velocity).max(), 2 + 1e-10)
                env.sim.data.qpos[:] = q_before
                env.sim.forward()
                previous = goal_a.copy()
            guard = NominalJointReference(live, tracking_limit=1e-8)
            with self.assertRaisesRegex(RuntimeError, "tracking guard"):
                guard.solve(np.r_[pos + [0, 0.001, 0], rot.as_rotvec()])
            unreachable = NominalJointReference(live)
            with self.assertRaisesRegex(RuntimeError, "infeasible"):
                unreachable.solve(np.r_[pos + [0, 1, 0], rot.as_rotvec()])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
