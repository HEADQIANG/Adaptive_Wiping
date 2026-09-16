"""Bounded Cartesian-control experiment; never modifies the production gate."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
from robosuite.controllers.composite.composite_controller import CompositeController
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import opspace_matrices, orientation_error
from scipy.spatial.transform import Rotation, Slerp

from scripts.shared.common import (
    digest,
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT, ROBOSUITE_PACKAGE
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import tracking_passes, validate_trajectory
from scripts.sim_pretrain.diagnostics import sensor_load_check
from scripts.sim_pretrain.experiments.diagnose_contact import contact_result
from scripts.sim_pretrain.experiments.visualize_wiping import (
    RecordedWipe,
    export_gif,
)
from scripts.sim_pretrain.simulation import parameter_grid, rollout_metrics

CANDIDATES = {
    "osc_ff": {"mode": "osc", "kp": [150] * 6, "kd": [2 * np.sqrt(150)] * 6},
    "cart_2000": {
        "mode": "impedance",
        "kp": [2000, 2000, 150, 80, 80, 80],
        "kd": [45, 45, 20, 2, 2, 2],
    },
    "cart_6000": {
        "mode": "impedance",
        "kp": [6000, 6000, 150, 120, 120, 120],
        "kd": [80, 80, 20, 2.5, 2.5, 2.5],
    },
    "cart_12000": {
        "mode": "impedance",
        "kp": [12000, 12000, 150, 120, 120, 120],
        "kd": [120, 120, 20, 2.5, 2.5, 2.5],
    },
}
REPRESENTATIVES = [(mu, k, 0.02) for mu in (0, 0.9, 3.5) for k in (0.5, 1000)]


class CartesianFeedback(OperationalSpaceController):
    """Stock OSC interface with nominal twist FF or physical Cartesian impedance."""

    def __init__(self, spec, **kwargs):
        self.mode = spec["mode"]
        super().__init__(**kwargs)
        self.kd = np.asarray(spec["kd"], dtype=float)
        self.nominal_twist = np.zeros(6)
        self.previous_position = None
        self.previous_rotation = None
        self.command_wrench = np.zeros(6)

    def seed(self, position, rotation):
        self.previous_position = np.asarray(position).copy()
        self.previous_rotation = rotation.as_matrix().copy()
        self.nominal_twist[:] = 0

    def reset_goal(self, goal_update_mode="achieved"):
        super().reset_goal(goal_update_mode)
        self.nominal_twist = np.zeros(6)
        self.previous_position = None
        self.previous_rotation = None

    def set_goal(self, action):
        super().set_goal(action)
        if self.previous_position is not None:
            self.nominal_twist[:3] = (self.goal_pos - self.previous_position) * self.control_freq
            self.nominal_twist[3:] = (
                Rotation.from_matrix(self.goal_ori @ self.previous_rotation.T).as_rotvec()
                * self.control_freq
            )
        if (
            np.linalg.norm(self.nominal_twist[:3]) > 0.2
            or np.linalg.norm(self.nominal_twist[3:]) > 2
        ):
            raise RuntimeError("Discontinuous nominal pose reference")
        self.previous_position = self.goal_pos.copy()
        self.previous_rotation = self.goal_ori.copy()

    def run_controller(self):
        if self.mode == "osc":
            torque = super().run_controller()
            _, lp, lr, _ = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
            feedforward = np.r_[
                lp @ (self.kd[:3] * self.nominal_twist[:3]),
                lr @ (self.kd[3:] * self.nominal_twist[3:]),
            ]
            self.torques = torque + self.J_full.T @ feedforward
            self.command_wrench = np.linalg.lstsq(
                self.J_full.T, self.torques - self.torque_compensation, rcond=None
            )[0]
        else:
            self.update()
            error = np.r_[
                self.goal_pos - self.ref_pos, orientation_error(self.goal_ori, self.ref_ori_mat)
            ]
            velocity = np.r_[self.ref_pos_vel, self.ref_ori_vel]
            self.command_wrench = self.kp * error + self.kd * (self.nominal_twist - velocity)
            # Physical stiffness/damping, without lambda preconditioning or joint-reference clipping.
            self.torques = self.J_full.T @ self.command_wrench + self.torque_compensation
            self.new_update = True
        return self.torques


class ExperimentWipe(RecordedWipe):
    def __init__(self, cfg, parameters, spec=None):
        self.controller_spec = copy.deepcopy(spec)
        self.cartesian_active = False
        self.load_rows = []
        super().__init__(cfg, 300, parameters)

    def prepare(self, extra_height=0):
        self.cartesian_active = False
        self.load_rows = []
        start = super().prepare(extra_height)
        if self.controller_spec is not None:
            robot = self.robot
            old = robot.composite_controller
            params = dict(robot.part_controller_config["right"])
            params.update(
                input_type="absolute",
                input_ref_frame="world",
                control_ori=True,
                impedance_mode="fixed",
                kp=self.controller_spec["kp"],
                kp_limits=[0, 20000],
                uncouple_pos_ori=True,
                interpolator_pos=None,
                interpolator_ori=None,
            )
            controller = CartesianFeedback(self.controller_spec, **params)
            controller.update_initial_joints(robot._joint_positions)
            controller.seed(start, self.goal_rotation)
            basic = CompositeController(robot.sim, robot.robot_model, old.grippers)
            basic.part_controllers.update(old.part_controllers)
            basic.part_controllers["right"] = controller
            basic.part_controller_config = old.part_controller_config
            basic.setup_action_split_idx()
            robot.composite_controller = basic
            basic.update_state()
            self.cartesian_active = True
        return start

    def move(self, target, duration, rotation=None):
        if not self.cartesian_active:
            return super().move(target, duration, rotation)
        controller = self.robot.part_controllers["right"]
        p0 = controller.goal_pos.copy()
        r0 = Rotation.from_matrix(controller.goal_ori)
        rotation = self.goal_rotation if rotation is None else rotation
        slerp = Slerp([0, 1], Rotation.from_quat([r0.as_quat(), rotation.as_quat()]))
        for fraction in np.linspace(0, 1, int(duration * 100) + 1)[1:]:
            blend = fraction**2 * (3 - 2 * fraction)
            self.command(p0 + blend * (target - p0), slerp(blend))

    def command(self, position, rotation=None, check_contacts=False):
        super().command(position, rotation, check_contacts)
        if not self.recording:
            return
        controller = self.robot.part_controllers["right"]
        result = contact_result(
            self.sim.model._model,
            self.sim.data._data,
            self.tool_ids,
            self.table_id,
            self.pose()[0],
            controller.qvel_index,
        )
        self.load_rows.append(
            {
                "contact_wrench": result["wrench"],
                "normal_sum": result["normal_sum"],
                "requested_torque": controller.torques.copy(),
                "applied_torque": self.sim.data.ctrl[:6].copy(),
                "nominal_twist": getattr(controller, "nominal_twist", np.zeros(6)).copy(),
                "contact_count_loaded": int(np.sum(result["rows"][:, 8] > 1e-8)),
            }
        )
        if self.cartesian_active:
            error = np.linalg.norm(self.pose()[0] - position)
            angle = (self.goal_rotation.inv() * self.pose()[1]).magnitude()
            if check_contacts and (error > 0.1 or angle > np.deg2rad(20)):
                raise RuntimeError("Experimental tracking guard: 100 mm / 20 degrees")

    def rollout(self, extra_height=0):
        data = super().rollout(extra_height)
        data.update(
            {key: np.asarray([row[key] for row in self.load_rows]) for key in self.load_rows[0]}
        )
        return data


def sources():
    paths = [
        Path(__file__),
        ROOT / "scripts/sim_pretrain/experiments/contact_breakaway.py",
        ROOT / "scripts/sim_pretrain/experiments/diagnose_contact.py",
        ROOT / "scripts/sim_pretrain/experiments/visualize_wiping.py",
        ROBOSUITE_PACKAGE / "controllers/parts/arm/osc.py",
        ROBOSUITE_PACKAGE / "controllers/composite/composite_controller.py",
        ROBOSUITE_PACKAGE / "utils/control_utils.py",
    ]
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def run_case(cfg, parameters, spec, height, path, gif=None):
    env = None
    row = {"parameters": list(parameters), "passed": False, "free_space": bool(height)}
    try:
        env = ExperimentWipe(cfg, parameters, spec)
        data = env.rollout(height)
        np.savez_compressed(path, **data)
        row["metrics"] = rollout_metrics(data)
        if height:
            row["passed"] = tracking_passes(row["metrics"], cfg)
        else:
            validate_trajectory(data)
            row["contact_motion"] = contact_motion_acceptance(data)
            row["normal_phase_mean_n"] = [
                float(data["normal_sum"][a:b].mean()) for a, b in ((0, 200), (200, 300), (300, 400))
            ]
            row["normal_peak_n"] = float(data["normal_sum"].max())
            row["normal_press_last_100ms_n"] = float(data["normal_sum"][190:200].mean())
            row["unloaded_wrench"] = env.unload(data)
            row["unloaded"] = True
            row["passed"] = row["contact_motion"]["passed"]
        if gif:
            args = argparse.Namespace(
                gif=str(gif),
                speed=1,
                collisions=False,
                tabletop=True,
                closeup=True,
                mu=parameters[0],
                stiffness=parameters[1],
            )
            export_gif(env.sim.model._model, env.frames[:401], args)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None and env.load_rows:
            np.savez_compressed(
                path.with_name(path.stem + "_partial.npz"),
                **{key: np.asarray([r[key] for r in env.load_rows]) for key in env.load_rows[0]},
            )
    finally:
        if env is not None:
            env.close()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=["all", "baseline", *CANDIDATES], default="all")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    report = {
        "config": cfg,
        "config_hash": digest(cfg),
        "provenance": provenance(),
        "sources": sources(),
        "planned_candidates": CANDIDATES,
        "representatives": REPRESENTATIVES,
        "preparation": "unchanged production IK; candidate starts only after settling",
        "scope": "simulation experiment, no collection authorization",
        "variants": {},
        "runtime": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(out / "report.json", report)
    names = ["baseline", *CANDIDATES] if args.candidate == "all" else [args.candidate]
    for name in names:
        folder = out / name
        folder.mkdir()
        spec = CANDIDATES.get(name)
        result = {"spec": spec, "contacts": [], "passed": False}
        report["variants"][name] = result
        result["free"] = run_case(cfg, (1.75, 500.25, 0.16), spec, 0.08, folder / "free.npz")
        print(name, "free", result["free"].get("metrics", result["free"].get("error")), flush=True)
        write_json(out / "report.json", report)
        cases = REPRESENTATIVES if result["free"]["passed"] else [(0.9, 1000, 0.02)]
        for index, parameters in enumerate(cases):
            row = run_case(
                cfg,
                parameters,
                spec,
                0,
                folder / f"contact_{index:02d}.npz",
                folder / "mu0p9.gif" if args.gif and parameters == (0.9, 1000, 0.02) else None,
            )
            result["contacts"].append(row)
            print(
                name,
                parameters,
                row.get("contact_motion", {}).get("metrics", row.get("error")),
                flush=True,
            )
            write_json(out / "report.json", report)
        result["representative_passed"] = result["free"]["passed"] and all(
            r["passed"] for r in result["contacts"]
        )
        if result["representative_passed"] and name != "baseline":
            result["grid"] = []
            for index, parameters in enumerate(parameter_grid(cfg)):
                row = run_case(cfg, parameters, spec, 0, folder / f"grid_{index:02d}.npz")
                result["grid"].append(row)
                print(name, "grid", index, "passed", row["passed"], flush=True)
                write_json(out / "report.json", report)
            env = ExperimentWipe(cfg, (1.75, 500.25, 0.16), spec)
            try:
                result["sensor_check"] = sensor_load_check(env)
            finally:
                env.close()
            result["passed"] = (
                all(r["passed"] for r in result["grid"]) and result["sensor_check"]["passed"]
            )
        write_json(out / "report.json", report)
        if result["passed"]:
            report["selected_candidate"] = name
            break
    report["source_unchanged"] = (
        provenance() == report["provenance"] and sources() == report["sources"]
    )
    report["passed"] = (
        any(v["passed"] for v in report["variants"].values()) and report["source_unchanged"]
    )
    write_json(out / "report.json", report)
    print(
        "FINISHED",
        {name: value["passed"] for name, value in report["variants"].items()},
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
