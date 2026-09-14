"""Scoped robosuite environment: no mutation of the vendored robot XML."""

import copy
import itertools
import sys
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG, Wipe
from scipy.spatial.transform import Rotation, Slerp

# Use the project fork containing AirbotPlay, not an unrelated installed robosuite.
import robosuite
from scripts.shared.common import ROOT
from scripts.shared.exploration_protocol import PRESS_DURATION_S, PRESS_SPEED_M_S
from scripts.shared.paths import ROBOSUITE_PACKAGE
from scripts.shared.assets import ROBOSUITE_MODELS
from scripts.sim_pretrain.control import JointPositionFeedforward
from scripts.sim_pretrain.inertia import restore_discoverse_inertia, set_sponge_inertia

# Set the vendor's supported resource root without modifying its source files.
# "robosuite_models" avoids its legacy XML rewriter's special "robosuite" path token.
robosuite.models.assets_root = str(ROBOSUITE_MODELS)

# User-requested output convention only; native sensor axes and physics are unchanged.
FT_OUTPUT_SIGNS = (1, -1, -1, 1, -1, -1)
FT_PROCESSING = "initial_noncontact_tare_yz_flip_v2"
FT_OUTPUT_FRAME = "ft_frame local with output Y/Z reversed"


def contact_parameters(mu, stiffness_direct, width):
    if mu < 0 or stiffness_direct <= 0 or width <= 0:
        raise ValueError("Invalid contact parameters")
    return (
        np.array([mu, 0.005, 0.0001]),
        np.array([-stiffness_direct, -2 * np.sqrt(stiffness_direct)]),
        np.array([0.2, 0.9, width, 0.5, 2.0]),
    )


def exploration_offsets(times, press_speed=PRESS_SPEED_M_S, slide_speed=0.05, ramp_s=0.0):
    """Fixed phase durations/distances; optional smoothstep velocity ramps."""
    if not np.isfinite(ramp_s) or not 0 <= ramp_s <= 0.5:
        raise ValueError("exploration ramp_s must be finite and in [0, 0.5]")
    times = np.asarray(times)
    result = np.zeros((len(times), 3))
    if ramp_s > 0:
        def distance(local_time, duration, length):
            t = np.clip(local_time, 0, duration)
            peak_speed = length / (duration - ramp_s)

            def ramp_integral(elapsed):
                u = np.clip(elapsed / ramp_s, 0, 1)
                return peak_speed * ramp_s * (u**3 - 0.5 * u**4)

            # Integrate v/v_peak = 3u^2 - 2u^3; mirror the ramp at the phase end.
            return np.where(
                t < ramp_s, ramp_integral(t),
                np.where(t > duration - ramp_s, length - ramp_integral(duration - t),
                         peak_speed * (t - 0.5 * ramp_s)),
            )

        result[:, 2] = -distance(times, PRESS_DURATION_S, press_speed * PRESS_DURATION_S)
        result[:, 1] = distance(times - 2, 1.0, slide_speed) - distance(times - 3, 1.0, slide_speed)
        return result
    result[:, 2] = -press_speed * np.minimum(times, PRESS_DURATION_S)
    result[:, 1] = slide_speed * np.where(times <= 3, np.maximum(times - 2, 0), 4 - times)
    return result


class PretrainingWipe(Wipe):
    def __init__(self, cfg, gain, parameters=(1.75, 500.25, 0.16)):
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("Joint position gain must be finite and positive")
        self.pretrain_cfg = cfg
        self.parameters = parameters
        controller = load_composite_controller_config(
            controller=str(ROBOSUITE_PACKAGE / "controllers/config/robots/default_airbot_play.json")
        )
        arm = controller["body_parts"]["right"]
        arm.pop("kd", None)
        arm.pop("kv", None)
        arm.update(kp=gain, damping_ratio=1, kp_limits=[0, max(1000, gain)])
        controller["composite_controller_specific_configs"]["ik_integration_dt"] = 0.01
        task = copy.deepcopy(DEFAULT_WIPE_CONFIG)
        task.update(
            num_markers=0, early_terminations=False, table_offset=cfg["simulation"]["table_offset"]
        )
        super().__init__(
            robots="AirbotPlay",
            controller_configs=controller,
            task_config=task,
            use_camera_obs=False,
            use_object_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            initialization_noise=None,
            control_freq=cfg["simulation"]["sample_hz"],
            hard_reset=False,
            horizon=100000,
            ignore_done=True,
            seed=cfg["simulation"]["seed"],
        )
        self.robot = self.robots[0]
        if not np.isclose(self.sim.model.opt.timestep, cfg["simulation"]["timestep"]):
            self.close()
            raise RuntimeError("Compiled physics timestep differs from configuration")
        self.site_id = self.robot.eef_site_id["right"]
        self.tool_ids = [
            self.sim.model.geom_name2id(n) for n in self.robot.gripper["right"].contact_geoms
        ]
        self.table_id = self.sim.model.geom_name2id("table_collision")
        self.set_parameters(parameters)
        self.goal_rotation = Rotation.from_euler("z", -np.pi / 2) * Rotation.from_rotvec(
            [np.pi, 0, 0]
        )

    def _load_robots(self):
        # Wipe hardcodes base_types="default"; override its per-robot config before loading.
        if self.pretrain_cfg["simulation"].get("base_mount", "stand") == "tabletop":
            for config in self.robot_configs:
                config["base_type"] = "NullMount"
        super()._load_robots()

    def _load_model(self):
        super()._load_model()
        prefix = self.robots[0].robot_model.naming_prefix
        motors = [x for x in self.model.actuator if x.get("name", "").startswith(prefix)]
        if len(motors) != 6:
            raise RuntimeError("Expected exactly six AIRBOT actuators")
        for old, limit in zip(motors, self.pretrain_cfg["simulation"]["torque_limits"]):
            attrs = {k: old.get(k) for k in ("name", "joint")}
            attrs.update(gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}")
            self.model.actuator.remove(old)
            ET.SubElement(self.model.actuator, "motor", attrs)
        for joint in self.model.worldbody.iter("joint"):
            if joint.get("name", "").startswith(prefix):
                joint.set("frictionloss", "0")
        if (
            self.pretrain_cfg["simulation"].get("inertia_profile", "legacy")
            == "discoverse_standard"
        ):
            restore_discoverse_inertia(self.model.worldbody, prefix)
        set_sponge_inertia(
            self.model.worldbody, self.robots[0].gripper["right"].naming_prefix,
            self.pretrain_cfg["simulation"].get("tool_inertia_profile", "legacy"),
        )
        mount_profile = self.pretrain_cfg["simulation"].get("tool_mount", "legacy")
        if mount_profile == "direct_wrist_v3":
            from scripts.sim_pretrain.wrist import install_direct_wrist

            install_direct_wrist(self.model.worldbody, self.model.asset, prefix)
        if mount_profile in ("bridge_v1", "compact_v2", "direct_wrist_v3"):
            from scripts.sim_pretrain.tool_mount import install_tool_mount

            install_tool_mount(
                self.model.worldbody,
                prefix,
                self.robots[0].gripper["right"].naming_prefix,
                profile=mount_profile,
            )
        from scripts.sim_pretrain.tool_mount import orient_ft_frame

        orient_ft_frame(
            self.model.worldbody,
            self.robots[0].gripper["right"].naming_prefix,
            self.pretrain_cfg["simulation"].get("ft_frame_profile", "legacy"),
        )
        if self.pretrain_cfg["simulation"].get("base_mount", "stand") == "tabletop":
            from scripts.sim_pretrain.tool_mount import place_base_on_table

            place_base_on_table(
                self.model.worldbody,
                prefix,
                self.pretrain_cfg["simulation"]["base_xy"],
                self.table_offset,
                self.table_full_size,
            )

    def _reset_internal(self):
        super()._reset_internal()
        if self.pretrain_cfg["simulation"].get("target_velocity_feedforward", False):
            robot = self.robots[0]
            composite = robot.composite_controller
            controller = JointPositionFeedforward(
                velocity_limit=composite.joint_action_policy.max_dq,
                **robot.part_controller_config["right"],
            )
            controller.reset_goal()
            composite.part_controllers["right"] = controller

    def reward(self, action=None):
        return 0.0

    def set_parameters(self, parameters):
        self.parameters = tuple(float(x) for x in parameters)
        friction, solref, solimp = contact_parameters(*self.parameters)
        m = self.sim.model
        m.geom_friction[self.tool_ids] = friction
        m.geom_solref[self.tool_ids] = solref
        m.geom_solimp[self.tool_ids] = solimp
        m.geom_priority[self.tool_ids] = 1
        m.geom_priority[self.table_id] = 0
        self.sim.forward()

    def pose(self):
        return (
            self.sim.data.site_xpos[self.site_id].copy(),
            Rotation.from_matrix(self.sim.data.site_xmat[self.site_id].reshape(3, 3).copy()),
        )

    def start_position(self, extra_height=0.0):
        pos, rot = self.pose()
        minimum = np.inf
        desired = self.goal_rotation.as_matrix()
        for gid in self.tool_ids:
            center = desired @ rot.inv().apply(self.sim.data.geom_xpos[gid] - pos)
            axes = desired @ rot.as_matrix().T @ self.sim.data.geom_xmat[gid].reshape(3, 3)
            kind = int(self.sim.model.geom_type[gid])
            if kind == 6:  # box
                extent = np.abs(axes[2]) @ self.sim.model.geom_size[gid]
            elif kind == 2:  # sphere
                extent = self.sim.model.geom_size[gid, 0]
            else:
                raise ValueError(f"Unsupported tool collision geometry type: {kind}")
            minimum = min(minimum, center[2] - extent)
        return np.array(
            [
                *self.pretrain_cfg["simulation"]["start_xy"],
                self.table_offset[2]
                - minimum
                + self.pretrain_cfg["simulation"]["initial_gap"]
                + extra_height,
            ]
        )

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, "_interval_saturated"):
            limits = np.asarray(self.pretrain_cfg["simulation"]["torque_limits"])
            self._interval_saturated |= np.abs(self.sim.data.ctrl[:6]) >= limits - 1e-6
        if getattr(self, "_check_contacts", False):
            self.contacts()

    def command(self, position, rotation=None, check_contacts=False):
        rotation = self.goal_rotation if rotation is None else rotation
        self._check_contacts = check_contacts
        self._interval_saturated = np.zeros(6, dtype=bool)
        action = self.robot.composite_controller.create_action_vector(
            {"right": np.r_[position, rotation.as_rotvec()]}
        )
        self.step(action)
        self.sim.forward()
        if check_contacts:
            self.contacts()
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            raise RuntimeError("Non-finite robot state")

    def move(self, target, duration, rotation=None):
        p0, r0 = self.pose()
        from scripts.sim_pretrain.reference import NominalJointReference

        policy = self.robot.composite_controller.joint_action_policy
        if isinstance(policy, NominalJointReference):
            # Continue the commanded path, not a jump back to the compliant actual pose.
            nominal = policy.full_model_data
            p0 = nominal.site_xpos[self.site_id].copy()
            r0 = Rotation.from_matrix(nominal.site_xmat[self.site_id].reshape(3, 3).copy())
        rotation = self.goal_rotation if rotation is None else rotation
        slerp = Slerp([0, 1], Rotation.from_quat([r0.as_quat(), rotation.as_quat()]))
        for fraction in np.linspace(0, 1, int(duration * 100) + 1)[1:]:
            blend = fraction * fraction * (3 - 2 * fraction)
            self.command(p0 + blend * (target - p0), slerp(blend))

    def prepare(self, extra_height=0.0):
        self.reset()
        self.set_parameters(self.parameters)
        start = self.start_position(extra_height)
        self.move(start + [0, 0, 0.06], 3.0)
        self.move(start, 2.0)
        for _ in range(100):
            self.command(start)
        pos, rot = self.pose()
        err = np.linalg.norm(pos - start)
        angle = (self.goal_rotation.inv() * rot).magnitude()
        if err > 0.003 or angle > np.deg2rad(2):
            raise RuntimeError(
                f"Unable to settle at start: position={err:.6f} m, angle={np.rad2deg(angle):.3f} deg"
            )
        if self.pretrain_cfg["simulation"].get("reference_mode", "actual") == "nominal":
            from scripts.sim_pretrain.reference import NominalJointReference

            composite = self.robot.composite_controller
            composite.joint_action_policy = NominalJointReference(composite.joint_action_policy)
            controller = composite.part_controllers["right"]
            if isinstance(controller, JointPositionFeedforward):
                controller.seed_reference(composite.joint_action_policy.last_goal)
        return start

    def unload(self, data):
        target = data["target_position"][0] + [0, 0, 0.08]
        self.move(target, 2.0)
        for _ in range(100):
            self.command(target)
        if self.contacts():
            raise RuntimeError("Tool remains in contact after unloading")
        return self.wrench().tolist()

    def wrench(self):
        return np.r_[self.robot.ee_force["right"], self.robot.ee_torque["right"]].copy()

    def contacts(self):
        count = 0
        unexpected = []
        tool = set(self.tool_ids)
        friction, solref, solimp = contact_parameters(*self.parameters)
        for c in self.sim.data.contact[: self.sim.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self.table_id in pair and pair.intersection(tool):
                count += 1
                if not np.allclose(c.solref, solref) or not np.allclose(c.solimp, solimp):
                    raise RuntimeError(
                        "Effective contact solver parameters differ from requested parameters"
                    )
                if not np.isclose(c.friction[0], max(friction[0], 1e-5)):
                    raise RuntimeError("Effective sliding friction differs from requested friction")
            elif c.dist < -1e-5:
                unexpected.append(
                    [self.sim.model.geom_id2name(c.geom1), self.sim.model.geom_id2name(c.geom2)]
                )
        if unexpected:
            raise RuntimeError(f"Non-tool contact: {unexpected[:3]}")
        return count

    def rollout(self, extra_height=0.0, sample_callback=None):
        start = self.prepare(extra_height)
        if self.contacts():
            raise RuntimeError("Initial FT tare requires a non-contact start")
        bias = self.wrench()
        times = np.arange(1, 401) / 100.0
        s = self.pretrain_cfg["simulation"]
        targets = start + exploration_offsets(
            times, s["press_speed"], s["slide_speed"], s.get("exploration_ramp_s", 0.0),
        )
        rows = {
            key: []
            for key in (
                "ft",
                "ft_raw",
                "position",
                "quat_xyzw",
                "sensor_quat_xyzw",
                "joints",
                "contact_count",
                "saturated",
                "orientation_error",
            )
        }
        time0 = self.sim.data.time
        if sample_callback is not None:
            sample_callback(0.0, np.zeros(6))
        for i, target in enumerate(targets):
            self.command(target, check_contacts=True)
            if not np.isclose(self.sim.data.time - time0, times[i], atol=1e-8):
                raise RuntimeError("Simulation sampling clock drift")
            pos, rot = self.pose()
            wrench = self.wrench()
            if not np.isfinite(wrench).all():
                raise RuntimeError("Non-finite force/torque")
            ft_site = self.sim.model.site_name2id(
                self.robot.gripper["right"].naming_prefix + "ft_frame"
            )
            rows["ft_raw"].append(wrench)
            rows["ft"].append((wrench - bias) * FT_OUTPUT_SIGNS)
            rows["position"].append(pos)
            rows["quat_xyzw"].append(rot.as_quat())
            rows["sensor_quat_xyzw"].append(
                Rotation.from_matrix(self.sim.data.site_xmat[ft_site].reshape(3, 3)).as_quat()
            )
            rows["joints"].append(self.robot._joint_positions.copy())
            rows["contact_count"].append(self.contacts())
            rows["saturated"].append(self._interval_saturated.copy())
            rows["orientation_error"].append((self.goal_rotation.inv() * rot).magnitude())
            if sample_callback is not None:
                sample_callback(float(times[i]), rows["ft"][-1].copy())
        result = {key: np.asarray(value) for key, value in rows.items()}
        result.update(
            time=times,
            target_position=targets,
            initial_bias=bias,
            parameters=np.asarray(self.parameters),
        )
        return result


def rollout_metrics(data):
    error = np.linalg.norm(data["position"] - data["target_position"], axis=1)
    return {
        "position_rms_m": float(np.sqrt(np.mean(error**2))),
        "position_max_m": float(error.max()),
        "orientation_max_deg": float(np.rad2deg(data["orientation_error"]).max()),
        "contact_fraction": float(np.mean(data["contact_count"] > 0)),
        "saturation_fraction": float(np.mean(np.any(data["saturated"], axis=1))),
        "force_peak_n": float(np.linalg.norm(data["ft"][:, :3], axis=1).max()),
    }


def parameter_grid(cfg):
    return itertools.product(
        *[
            [bounds[0], sum(bounds) / 2, bounds[1]]
            for bounds in (
                cfg["randomization"][k] for k in ("friction", "stiffness_direct", "width")
            )
        ]
    )
