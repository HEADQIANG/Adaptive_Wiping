"""Small synthetic fixtures test software only; they are never paper datasets."""

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch
from robosuite.controllers.parts.generic.joint_pos import JointPositionController

from scripts.shared.common import (
    DISCOVERSE_INERTIA_SOURCE,
    ROOT,
    digest,
    file_digest,
    load_config,
    provenance,
    sample_parameters,
    write_json,
)
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.assets import ROBOSUITE_MODELS
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.collection import (
    FIELDS,
    collect,
    require_sanity,
    tracking_passes,
    validate_trajectory,
)
from scripts.sim_pretrain.control import JointPositionFeedforward
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.inertia import restore_discoverse_inertia
from scripts.sim_pretrain.learning import evaluate, export, load_data, train
from scripts.sim_pretrain.simulation import (
    PretrainingWipe,
    contact_parameters,
    exploration_offsets,
    parameter_grid,
)


def fixture_trajectory():
    data = {key: np.zeros(shape) for key, shape in FIELDS.items()}
    data["time"] = np.arange(1, 401) / 100
    data["contact_count"][20:] = 1
    data["position"] = exploration_offsets(data["time"])
    data["target_position"] = data["position"].copy()
    return data


class PretrainingTests(unittest.TestCase):
    def test_rollout_tares_all_channels_and_preserves_raw(self):
        from scipy.spatial.transform import Rotation

        env = MagicMock()
        env.prepare.return_value = np.zeros(3)
        env.contacts.return_value = 0
        env.pretrain_cfg = {"simulation": {"press_speed": 0.005, "slide_speed": 0.05, "exploration_ramp_s": 0.2}}
        env.sim.data.time = 0.0
        env.sim.data.site_xmat = np.eye(3).reshape(1, 9)
        env.sim.model.site_name2id.return_value = 0
        env.pose.return_value = (np.zeros(3), Rotation.identity())
        env.goal_rotation = Rotation.identity()
        env.parameters = (1.2, 100, 0.16)
        env.robot._joint_positions = np.zeros(6)
        env._interval_saturated = np.zeros(6, dtype=bool)
        bias = np.array([1, -2, 3, -4, 5, -6.0])
        delta = np.arange(1, 7) * 0.1
        env.wrench.side_effect = [bias, bias] + [bias + delta] * 399

        def advance(*args, **kwargs):
            env.sim.data.time += 0.01

        env.command.side_effect = advance
        samples = []
        data = PretrainingWipe.rollout(env, sample_callback=lambda t, ft: samples.append((t, ft)))
        self.assertEqual(len(samples), 401)
        np.testing.assert_array_equal(data["target_position"], exploration_offsets(data["time"], ramp_s=0.2))
        self.assertEqual(samples[0][0], 0.0)
        np.testing.assert_array_equal(samples[0][1], np.zeros(6))
        np.testing.assert_array_equal([row[0] for row in samples[1:]], data["time"])
        np.testing.assert_array_equal([row[1] for row in samples[1:]], data["ft"])
        np.testing.assert_array_equal(data["ft"][0], np.zeros(6))
        signs = np.array([1, -1, -1, 1, -1, -1])
        np.testing.assert_allclose(data["ft"][1:], np.tile(delta * signs, (399, 1)))
        np.testing.assert_allclose((data["ft_raw"] - data["initial_bias"]) * signs, data["ft"])
        np.testing.assert_array_equal(data["initial_bias"], bias)
        np.testing.assert_allclose(data["ft_raw"][1:], np.tile(bias + delta, (399, 1)))
        env.contacts.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "non-contact"):
            PretrainingWipe.rollout(env)

    def setUp(self):
        self.cfg = load_config(ROOT / "configs/sim_pretrain/pretrain_paper.yaml")
        torch.set_num_threads(1)

    def test_exploration_boundaries(self):
        np.testing.assert_allclose(
            exploration_offsets([0, 2, 3, 4]),
            [[0, 0, 0], [0, 0, -0.01], [0, 0.05, -0.01], [0, 0, -0.01]],
        )
        offsets = exploration_offsets(np.arange(401) / 100)
        np.testing.assert_allclose(np.diff(offsets[:201, 2]), -0.00005, atol=1e-12)
        np.testing.assert_allclose(np.diff(offsets[200:301, 1]), 0.0005, atol=1e-12)

    def test_real_exploration_matches_legacy_simulation_every_tick(self):
        from scripts.shared.exploration_protocol import exploration_offset

        times = np.arange(401) / 100
        np.testing.assert_array_equal(
            exploration_offsets(times), [exploration_offset(float(t)) for t in times]
        )
        for name in ("pretrain_paper.yaml", "pretrain_paper_mu1p2.yaml"):
            cfg = load_config(ROOT / "configs/sim_pretrain" / name)
            self.assertEqual(cfg["simulation"]["press_speed"], 0.005)
            self.assertEqual(cfg["simulation"]["exploration_ramp_s"], 0.2)
            self.assertFalse(np.allclose(exploration_offsets(times, ramp_s=0.2), exploration_offsets(times)))

    def test_smooth_trajectory_distances_monotonicity_and_peak_speeds(self):
        t = np.linspace(0, 4, 40001)
        for ramp in (0.1, 0.2, 0.5):
            p = exploration_offsets(t, ramp_s=ramp)
            np.testing.assert_allclose(p[[0, 20000, 30000, 40000]],
                [[0, 0, 0], [0, 0, -0.01], [0, 0.05, -0.01], [0, 0, -0.01]], atol=1e-14)
            self.assertTrue(np.all(np.diff(p[:20001, 2]) <= 1e-15))
            self.assertTrue(np.all(np.diff(p[20000:30001, 1]) >= -1e-15))
            self.assertTrue(np.all(np.diff(p[30000:, 1]) <= 1e-15))
            np.testing.assert_allclose(p[20000:, 2], -0.01, atol=1e-14)
            np.testing.assert_array_equal(p[:20001, 1], np.zeros(20001))
            speed = np.abs(np.diff(p, axis=0) / (t[1] - t[0])).max(axis=0)
            np.testing.assert_allclose(speed, [0, 0.05 / (1-ramp), 0.01 / (2-ramp)], rtol=1e-6)
        custom = exploration_offsets([2, 3, 4], press_speed=0.004, slide_speed=0.03, ramp_s=0.2)
        np.testing.assert_allclose(custom, [[0, 0, -0.008], [0, 0.03, -0.008], [0, 0, -0.008]])

    def test_smooth_trajectory_has_continuous_velocity_and_acceleration(self):
        h = 1e-4
        for t in (0, 0.2, 1.8, 2, 2.2, 2.8, 3, 3.2, 3.8, 4):
            p = exploration_offsets([t-h, t, t+h], ramp_s=0.2)
            left, right = (p[1]-p[0])/h, (p[2]-p[1])/h
            np.testing.assert_allclose(left, right, atol=1e-6)
            np.testing.assert_allclose((p[2]-2*p[1]+p[0])/h**2, 0, atol=0.002)
            if t in (0, 2, 3, 4):
                np.testing.assert_allclose((left+right)/2, 0, atol=1e-7)
        for bad in (-0.1, 0.51, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                exploration_offsets([0, 2, 3, 4], ramp_s=bad)

    def test_contact_mapping(self):
        friction, solref, solimp = contact_parameters(1.2, 100, 0.16)
        np.testing.assert_allclose(friction, [1.2, 0.005, 0.0001])
        np.testing.assert_allclose(solref, [-100, -20])
        np.testing.assert_allclose(solimp, [0.2, 0.9, 0.16, 0.5, 2])
        self.assertEqual(len(list(parameter_grid(self.cfg))), 27)
        with self.assertRaises(ValueError):
            contact_parameters(-1, 100, 0.1)

    def test_expanded_gain_reaches_controller_without_changing_actuator_limits(self):
        env = PretrainingWipe(self.cfg, 5000)
        try:
            controller = env.robot.part_controllers["right"]
            np.testing.assert_array_equal(controller.kp, np.full(6, 5000))
            np.testing.assert_array_equal(controller.kp_max, np.full(6, 5000))
            np.testing.assert_allclose(controller.kd, np.full(6, 2*np.sqrt(5000)))
            limits = np.asarray(self.cfg["simulation"]["torque_limits"])
            np.testing.assert_array_equal(env.sim.model.actuator_ctrlrange[:6], np.column_stack([-limits, limits]))
            env.reset()
            np.testing.assert_array_equal(env.robot.part_controllers["right"].kp, np.full(6, 5000))
        finally:
            env.close()
        for gain in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                PretrainingWipe(self.cfg, gain)

    def test_sampling_reproducibility_and_split_isolation(self):
        a = sample_parameters(self.cfg, 0, 1000)
        np.testing.assert_array_equal(a, sample_parameters(self.cfg, 0, 1000))
        self.assertFalse(np.array_equal(a, sample_parameters(self.cfg, 1, 1000)))
        for i, key in enumerate(("friction", "stiffness_direct", "width")):
            low, high = self.cfg["randomization"][key]
            self.assertTrue(np.all((a[:, i] >= low) & (a[:, i] <= high)))

    def test_filter_axis_and_training_only_statistics(self):
        raw = np.ones((2, 400, 6)) * np.arange(6)
        raw[1] += 1
        prep = Preprocessor().fit(raw)
        np.testing.assert_allclose(prep.filtered(raw), raw, atol=1e-12)
        scaled = prep.transform(raw)
        np.testing.assert_allclose(scaled[0], 0, atol=1e-7)
        np.testing.assert_allclose(scaled[1], 0.9, atol=1e-7)
        state = copy.deepcopy(prep.state())
        self.assertGreater(float(prep.transform(raw + 10).max()), 0.9)
        self.assertEqual(state, prep.state())
        np.testing.assert_allclose(prep.inverse(scaled), raw, atol=1e-6)
        np.testing.assert_array_equal(Preprocessor.from_state(state).transform(raw), scaled)

    def test_constant_channel_and_bad_inputs(self):
        raw = np.zeros((2, 400, 6))
        prep = Preprocessor().fit(raw)
        self.assertTrue(np.isfinite(prep.transform(raw)).all())
        self.assertTrue(prep.constant.all())
        with self.assertRaises(ValueError):
            prep.transform(np.zeros((400, 6)))
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prep.transform(raw)

    def test_vae_shape_loss_and_gradient(self):
        model = SpongeVAE()
        x = torch.randn(3, 400, 6)
        reconstruction, mu, logvar = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (3, 5))
        loss, mse, kl = vae_loss(reconstruction, x, mu, logvar)
        self.assertTrue(torch.allclose(loss, mse + 0.06 * kl))
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        self.assertEqual(vae_loss(x, x, torch.zeros(3, 5), torch.zeros(3, 5))[0].item(), 0)
        self.assertEqual(vae_loss(x, x, torch.ones(3, 5), torch.zeros(3, 5))[2].item(), 2.5)
        model.eval()
        torch.testing.assert_close(model(x, sample=False)[0], model(x, sample=False)[0])

    def test_trajectory_validation(self):
        data = fixture_trajectory()
        validate_trajectory(data)
        data["time"][0] = 0
        with self.assertRaises(ValueError):
            validate_trajectory(data)
        data = fixture_trajectory()
        data["contact_count"][:] = 0
        with self.assertRaises(ValueError):
            validate_trajectory(data)

    def test_tracking_threshold_is_not_relaxed(self):
        metrics = {"position_rms_m": 0.00219, "orientation_max_deg": 0.084, "contact_fraction": 0}
        self.assertFalse(tracking_passes(metrics, self.cfg))
        metrics["position_rms_m"] = 0.0009
        self.assertTrue(tracking_passes(metrics, self.cfg))
        metrics["contact_fraction"] = 0.1
        self.assertFalse(tracking_passes(metrics, self.cfg))

    def test_gate_blocks_failed_and_stale_reports(self):
        with tempfile.TemporaryDirectory(prefix="wiping-gate-") as tmp:
            self.cfg["output_dir"] = tmp
            with self.assertRaisesRegex(RuntimeError, "Run sanity"):
                require_sanity(self.cfg)
            report = {
                "config_hash": digest(self.cfg),
                "provenance": provenance(),
                "passed": False,
                "selected_gain": None,
            }
            write_json(Path(tmp) / "sanity.json", report)
            with self.assertRaisesRegex(RuntimeError, "not passed"):
                collect(self.cfg)
            report["config_hash"] = "stale"
            write_json(Path(tmp) / "sanity.json", report)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                require_sanity(self.cfg)
            self.assertFalse((Path(tmp) / "dataset.h5").exists())

    def test_collector_preserves_failed_assignment_and_resume(self):
        class FakeEnv:
            calls = 0

            def __init__(self, *args):
                pass

            def set_parameters(self, parameters):
                pass

            def unload(self, data):
                return [0.0] * 6

            def rollout(self):
                FakeEnv.calls += 1
                if FakeEnv.calls == 1:
                    raise RuntimeError("intentional test failure")
                return fixture_trajectory()

            def close(self):
                pass

        with tempfile.TemporaryDirectory(prefix="wiping-collector-test-") as tmp:
            self.cfg["output_dir"] = tmp
            self.cfg["dataset"] = dict(train=2, validation=1, test=1)
            report = {"selected_gain": 300, "test_fixture": True}
            with (
                patch(
                    "scripts.sim_pretrain.collection.require_sanity", return_value=report
                ),
                patch("scripts.sim_pretrain.simulation.PretrainingWipe", FakeEnv),
            ):
                self.assertFalse(collect(self.cfg)["complete"])
                with h5py.File(Path(tmp) / "dataset.h5") as h5:
                    assignment = h5["train/parameters"][0]
                    self.assertTrue(np.isnan(h5["train/ft"][0]).all())
                self.assertFalse(collect(self.cfg)["complete"])
                self.assertEqual(FakeEnv.calls, 4)
                self.assertTrue(collect(self.cfg, retry_failed=True)["complete"])
                with h5py.File(Path(tmp) / "dataset.h5") as h5:
                    np.testing.assert_array_equal(assignment, h5["train/parameters"][0])
                self.assertEqual(load_data(self.cfg)[0]["train"].shape, (2, 400, 6))

    def test_training_evaluation_export_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="wiping-learning-test-") as tmp:
            self.cfg["output_dir"] = tmp
            self.cfg["dataset"] = dict(train=8, validation=3, test=3)
            self.cfg["training"]["epochs"] = 2
            rng = np.random.default_rng(8)
            with h5py.File(Path(tmp) / "dataset.h5", "w") as h5:
                h5.attrs.update(
                    config_hash=digest(self.cfg),
                    complete=True,
                    provenance_hash="synthetic-test-only",
                )
                for split, count in self.cfg["dataset"].items():
                    group = h5.create_group(split)
                    group.create_dataset("ft", data=rng.normal(size=(count, 400, 6)))
                    group.create_dataset("valid", data=np.ones(count, dtype=bool))
            write_json(
                Path(tmp) / "dataset_integrity.json",
                {"sha256": file_digest(Path(tmp) / "dataset.h5")},
            )
            self.assertEqual(train(self.cfg)["epoch"], 2)
            self.assertEqual(evaluate(self.cfg)["epoch"], 2)
            export(self.cfg)
            encoder = FrozenSpongeEncoder(Path(tmp) / "encoder.pt")
            raw = rng.normal(size=(2, 400, 6))
            embedding = encoder.encode(raw)
            self.assertEqual(embedding.shape, (2, 5))
            self.assertTrue(np.isfinite(embedding).all())
            np.testing.assert_array_equal(
                embedding, FrozenSpongeEncoder(Path(tmp) / "encoder.pt").encode(raw)
            )
            self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))
            with self.assertRaises(FileExistsError):
                train(self.cfg)
            with h5py.File(Path(tmp) / "dataset.h5", "a") as h5:
                h5["train/ft"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_data(self.cfg)

    def test_actual_robot_actuators_contact_priority_and_sensor(self):
        env = PretrainingWipe(self.cfg, 300)
        try:
            env.set_parameters((0.7, 100, 0.12))
            m = env.sim.model
            np.testing.assert_allclose(
                m.geom_solref[env.tool_ids], np.tile([-100, -20], (len(env.tool_ids), 1))
            )
            self.assertTrue(np.all(m.geom_priority[env.tool_ids] > m.geom_priority[env.table_id]))
            np.testing.assert_allclose(m.actuator_ctrlrange[:6, 1], [10, 10, 10, 5, 5, 5])
            state = env.sim.data.qpos.copy()
            self.assertTrue(sensor_load_check(env)["passed"])
            np.testing.assert_array_equal(env.sim.data.qpos, state)
        finally:
            env.close()

    def test_feedforward_velocity_limit_reset_and_invalid_input(self):
        env = PretrainingWipe(self.cfg, 300)
        try:
            controller = env.robot.part_controllers["right"]
            self.assertIsInstance(controller, JointPositionFeedforward)
            goal = controller.joint_pos.copy()
            controller.set_goal(goal)
            np.testing.assert_array_equal(controller.goal_velocity, np.zeros(6))
            controller.set_goal(goal + 0.001)
            np.testing.assert_allclose(controller.goal_velocity, 0.1)
            controller.set_goal(goal + 0.001)
            np.testing.assert_array_equal(controller.goal_velocity, np.zeros(6))
            jump = np.array([1, -0.5, 0.25, 0, 0, 0])
            controller.set_goal(goal + 0.001 + jump)
            np.testing.assert_allclose(controller.goal_velocity, jump * controller.velocity_limit)
            controller.reset_goal()
            np.testing.assert_array_equal(controller.goal_velocity, np.zeros(6))
            controller.set_goal(goal + 1)
            np.testing.assert_array_equal(controller.goal_velocity, np.zeros(6))
            with self.assertRaises(ValueError):
                controller.set_goal(np.full(6, np.nan))
            with self.assertRaises(ValueError):
                controller.set_goal(np.zeros(5))
            env.reset()
            refreshed = env.robot.part_controllers["right"]
            self.assertIsNot(refreshed, controller)
            self.assertIsInstance(refreshed, JointPositionFeedforward)
            np.testing.assert_array_equal(refreshed.goal_velocity, np.zeros(6))
        finally:
            env.close()

    def test_feedforward_torque_equation_and_substep_hold(self):
        env = PretrainingWipe(self.cfg, 300)
        try:
            controller = env.robot.part_controllers["right"]
            goal = controller.joint_pos.copy()
            controller.set_goal(goal)
            controller.set_goal(goal + np.arange(6) * 0.0001)
            velocity = controller.goal_velocity.copy()
            for compensation in (True, False):
                controller.use_torque_compensation = compensation
                base = JointPositionController.run_controller(controller).copy()
                extra = controller.kd * velocity
                if compensation:
                    extra = controller.mass_matrix @ extra
                for _ in range(5):
                    np.testing.assert_allclose(controller.run_controller(), base + extra)
                    np.testing.assert_array_equal(controller.goal_velocity, velocity)
            controller.reset_goal()
            np.testing.assert_allclose(
                controller.run_controller(), JointPositionController.run_controller(controller)
            )
        finally:
            env.close()

    def test_feedforward_disabled_preserves_original_controller(self):
        self.cfg["simulation"]["target_velocity_feedforward"] = False
        env = PretrainingWipe(self.cfg, 300)
        try:
            self.assertIs(type(env.robot.part_controllers["right"]), JointPositionController)
            env.reset()
            self.assertIs(type(env.robot.part_controllers["right"]), JointPositionController)
        finally:
            env.close()

    def test_restored_inertials_match_reference_after_compile_and_reset(self):
        reference = ET.parse(DISCOVERSE_INERTIA_SOURCE).getroot()
        robot_xml = ROBOSUITE_MODELS / "robots/airbot_play/robot.xml"
        source_hashes = [file_digest(path) for path in (robot_xml, DISCOVERSE_INERTIA_SOURCE)]
        env = PretrainingWipe(self.cfg, 300)
        try:
            for reset in (False, True):
                if reset:
                    env.reset()
                model = env.sim.model
                for i in range(1, 7):
                    expected = reference.find(f".//body[@name='link{i}']/inertial")
                    bid = model.body_name2id(f"robot0_link{i}")
                    self.assertEqual(model.body_mass[bid], float(expected.get("mass")))
                    np.testing.assert_allclose(
                        model.body_ipos[bid], np.fromstring(expected.get("pos"), sep=" ")
                    )
                    np.testing.assert_allclose(
                        model.body_inertia[bid], np.fromstring(expected.get("diaginertia"), sep=" ")
                    )
                    q = np.fromstring(expected.get("quat", "1 0 0 0"), sep=" ")
                    q /= np.linalg.norm(q)
                    np.testing.assert_allclose(
                        np.outer(model.body_iquat[bid], model.body_iquat[bid]),
                        np.outer(q, q),
                        atol=1e-12,
                    )
                np.testing.assert_array_equal(model.dof_armature[:6], np.zeros(6))
        finally:
            env.close()
        self.assertEqual(
            source_hashes, [file_digest(path) for path in (robot_xml, DISCOVERSE_INERTIA_SOURCE)]
        )

    def test_inertia_override_does_not_change_geometry_or_control_limits(self):
        legacy_cfg = copy.deepcopy(self.cfg)
        legacy_cfg["simulation"]["inertia_profile"] = "legacy"
        old = PretrainingWipe(legacy_cfg, 300)
        new = PretrainingWipe(self.cfg, 300)
        try:
            for name in (
                "geom_pos",
                "geom_quat",
                "geom_size",
                "body_pos",
                "body_quat",
                "jnt_range",
                "dof_damping",
                "dof_frictionloss",
                "actuator_ctrlrange",
            ):
                np.testing.assert_array_equal(
                    getattr(old.sim.model, name), getattr(new.sim.model, name)
                )
            np.testing.assert_allclose(old.sim.model.dof_armature[:6], [5 / i for i in range(1, 7)])
            for i in range(new.sim.model.nbody):
                if new.sim.model.body_id2name(i) not in [f"robot0_link{k}" for k in range(1, 7)]:
                    for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
                        np.testing.assert_array_equal(
                            getattr(old.sim.model, name)[i], getattr(new.sim.model, name)[i]
                        )
        finally:
            old.close()
            new.close()

    def test_missing_inertial_rejects_without_partial_override(self):
        tree = ET.parse(ROBOSUITE_MODELS / "robots/airbot_play/robot.xml")
        world = tree.getroot().find("worldbody")
        body = world.find(".//body[@name='link6']")
        body.remove(body.find("inertial"))
        before = ET.tostring(world)
        with self.assertRaises(ValueError):
            restore_discoverse_inertia(world, "")
        self.assertEqual(before, ET.tostring(world))


if __name__ == "__main__":
    unittest.main()
