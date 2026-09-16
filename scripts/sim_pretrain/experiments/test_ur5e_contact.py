"""Isolated UR5e comparison; no AIRBOT production config or model changes."""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation, Slerp

from scripts.shared.common import (
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT, ROBOSUITE_PACKAGE
from scripts.shared.assets import ROBOSUITE_MODELS
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import tracking_passes, validate_trajectory
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.simulation import (
    DEFAULT_WIPE_CONFIG,
    PretrainingWipe,
    Wipe,
    parameter_grid,
    rollout_metrics,
)


def comparison_config(source, mode):
    cfg = copy.deepcopy(source)
    cfg["simulation"].update(
        robot="UR5e",
        base_mount="stand",
        tool_mount="legacy",
        ft_frame_profile="legacy",
        inertia_profile="robosuite_native",
        start_xy=[0.15, 0.0],
        reference_mode="actual",
        target_velocity_feedforward=mode == "ik_ff",
        torque_limits=[150, 150, 150, 28, 28, 28],
    )
    cfg["simulation"].pop("base_xy", None)
    return cfg


class UR5eWipe(PretrainingWipe):
    def __init__(self, cfg, gain, parameters=(1.75, 500.25, 0.16), mode="ik_ff"):
        if mode not in ("ik_ff", "osc"):
            raise ValueError("Expected ik_ff or osc")
        self.pretrain_cfg = cfg
        self.parameters = parameters
        self.mode = mode
        filename = "default_airbot_play.json" if mode == "ik_ff" else "default_ur5e.json"
        controller = load_composite_controller_config(
            controller=str(ROBOSUITE_PACKAGE / "controllers/config/robots" / filename)
        )
        arm = controller["body_parts"]["right"]
        if mode == "ik_ff":
            arm.pop("kd", None)
            arm.pop("kv", None)
            arm.update(kp=gain, damping_ratio=1, kp_limits=[0, 1000])
            controller["composite_controller_specific_configs"]["ik_integration_dt"] = 0.01
        else:
            # Preserve stock OSC gains/damping; express the same trajectory in world absolute poses.
            if gain != 150:
                raise ValueError("The stock OSC comparison uses its native gain150")
            arm.update(input_type="absolute", input_ref_frame="world")
        task = copy.deepcopy(DEFAULT_WIPE_CONFIG)
        task.update(
            num_markers=0, early_terminations=False, table_offset=cfg["simulation"]["table_offset"]
        )
        Wipe.__init__(
            self,
            robots="UR5e",
            controller_configs=controller,
            task_config=task,
            use_camera_obs=False,
            use_object_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            initialization_noise=None,
            control_freq=100,
            hard_reset=False,
            horizon=100000,
            ignore_done=True,
            seed=cfg["simulation"]["seed"],
        )
        self.robot = self.robots[0]
        self.site_id = self.robot.eef_site_id["right"]
        self.tool_ids = [
            self.sim.model.geom_name2id(name) for name in self.robot.gripper["right"].contact_geoms
        ]
        self.table_id = self.sim.model.geom_name2id("table_collision")
        np.testing.assert_allclose(
            self.sim.model.actuator_ctrlrange[:6],
            np.c_[
                -np.asarray(cfg["simulation"]["torque_limits"]), cfg["simulation"]["torque_limits"]
            ],
        )
        if not np.isclose(self.sim.model.opt.timestep, 0.002):
            raise ValueError("Unexpected physics timestep")
        self.set_parameters(parameters)
        self.goal_rotation = Rotation.from_euler("z", -np.pi / 2) * Rotation.from_rotvec(
            [np.pi, 0, 0]
        )

    def _load_model(self):
        # Explicitly skip AIRBOT-only inertia, actuator and engineering mount overrides.
        Wipe._load_model(self)

    def move(self, target, duration, rotation=None):
        p0, r0 = self.pose()
        rotation = self.goal_rotation if rotation is None else rotation
        slerp = Slerp([0, 1], Rotation.from_quat([r0.as_quat(), rotation.as_quat()]))
        for fraction in np.linspace(0, 1, int(duration * 100) + 1)[1:]:
            blend = fraction * fraction * (3 - 2 * fraction)
            self.command(p0 + blend * (target - p0), slerp(blend))

    def audit(self):
        controller = self.robot.part_controllers["right"]
        model = self.sim.model
        dofs = controller.qvel_index
        base = model.body_name2id(self.robot.robot_model.naming_prefix + "base")
        return {
            "robot": type(self.robot.robot_model).__name__,
            "controller": type(controller).__name__,
            "kp": controller.kp.tolist(),
            "joint_names": controller.joint_names,
            "base_world_position": self.sim.data.body_xpos[base].tolist(),
            "armature": model.dof_armature[dofs].tolist(),
            "damping": model.dof_damping[dofs].tolist(),
            "frictionloss": model.dof_frictionloss[dofs].tolist(),
            "actuator_ctrlrange": model.actuator_ctrlrange[:6].tolist(),
            "tool_mass": float(model.body_mass[model.geom_bodyid[self.tool_ids[0]]]),
        }


def experiment_sources():
    extra = [
        Path(__file__),
        ROBOSUITE_PACKAGE / "controllers/config/robots/default_ur5e.json",
        ROBOSUITE_PACKAGE / "models/robots/manipulators/ur5e_robot.py",
        ROBOSUITE_PACKAGE / "controllers/parts/arm/osc.py",
        ROBOSUITE_PACKAGE / "controllers/composite/composite_controller.py",
    ]
    extra.extend(sorted((ROBOSUITE_MODELS / "robots/ur5e").rglob("*")))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in extra if path.is_file()}


def run_case(cfg, mode, gain, parameters, height, path):
    row = {"parameters": list(parameters), "gain": gain, "passed": False}
    env = None
    try:
        env = UR5eWipe(cfg, gain, parameters, mode)
        row["compiled_model"] = env.audit()
        data = env.rollout(extra_height=height)
        np.savez_compressed(path, **data)
        row["metrics"] = rollout_metrics(data)
        if height:
            row["passed"] = bool(tracking_passes(row["metrics"], cfg))
        else:
            validate_trajectory(data)
            row["contact_motion"] = contact_motion_acceptance(data)
            row["unloaded_wrench"] = env.unload(data)
            row["unloaded"] = True
            row["passed"] = row["contact_motion"]["passed"]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("ik_ff", "osc", "both"), default="both")
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = load_config(args.config)
    report = {
        "scope": "UR5e simulation comparison only; never authorizes AIRBOT collection",
        "source_config": source,
        "provenance": provenance(),
        "experiment_sources": experiment_sources(),
        "runtime": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "variants": {},
    }
    modes = ("ik_ff", "osc") if args.mode == "both" else (args.mode,)
    for mode in modes:
        cfg = comparison_config(source, mode)
        cfg["output_dir"] = str(out / mode)
        folder = out / mode
        folder.mkdir()
        write_json(folder / "config.json", cfg)
        result = {"free_space": [], "contacts": [], "selected_gain": None, "passed": False}
        report["variants"][mode] = result
        for gain in source["simulation"]["gain_candidates"] if mode == "ik_ff" else [150]:
            row = run_case(
                cfg, mode, gain, (1.75, 500.25, 0.16), 0.08, folder / f"free_gain_{gain}.npz"
            )
            result["free_space"].append(row)
            print(
                mode,
                "free",
                gain,
                row.get("metrics", row.get("error")),
                "passed",
                row["passed"],
                flush=True,
            )
            write_json(out / "report.json", report)
            if row["passed"]:
                result["selected_gain"] = gain
                break
        # Failed free-space variants still get three explicitly diagnostic contacts, not a passing gate.
        gain = result["selected_gain"] or (300 if mode == "ik_ff" else 150)
        cases = (
            list(parameter_grid(cfg))
            if result["selected_gain"] is not None
            else [(0, 1000, 0.02), (3.5, 1000, 0.02), (3.5, 0.5, 0.02)]
        )
        result["contact_scope"] = (
            "full_27" if len(cases) == 27 else "diagnostic_only_3_free_space_failed"
        )
        for index, parameters in enumerate(cases):
            row = run_case(cfg, mode, gain, parameters, 0, folder / f"contact_{index:02d}.npz")
            result["contacts"].append(row)
            print(
                mode,
                "contact",
                index,
                parameters,
                "passed",
                row["passed"],
                row.get("contact_motion", {}).get("failed_checks", row.get("error")),
                flush=True,
            )
            write_json(out / "report.json", report)
        env = None
        try:
            env = UR5eWipe(cfg, gain, mode=mode)
            result["sensor_check"] = sensor_load_check(env)
        except Exception as exc:
            result["sensor_check"] = {"passed": False, "error": str(exc)}
        finally:
            if env is not None:
                env.close()
        result["passed"] = (
            result["selected_gain"] is not None
            and all(row["passed"] for row in result["contacts"])
            and result["sensor_check"]["passed"]
        )
        write_json(out / "report.json", report)
    report["source_unchanged"] = (
        report["provenance"] == provenance()
        and report["experiment_sources"] == experiment_sources()
    )
    write_json(out / "report.json", report)
    print(
        "Finished",
        {
            key: {
                "passed": value["passed"],
                "contacts_passed": sum(r["passed"] for r in value["contacts"]),
                "contacts_tested": len(value["contacts"]),
            }
            for key, value in report["variants"].items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
