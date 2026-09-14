"""Read-only instrumentation of free-space rollouts; never opens the collection gate."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from scripts.shared.common import (
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT, ROBOSUITE_PACKAGE
from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics


class TraceWipe(PretrainingWipe):
    def __init__(self, *args, **kwargs):
        self.recording = False
        self.trace = []
        super().__init__(*args, **kwargs)

    def prepare(self, extra_height=0.0):
        self.recording = False
        start = super().prepare(extra_height)
        self.trace = []
        self.time_zero = self.sim.data.time
        self.previous_goal = self.robot.part_controllers["right"].goal_qpos.copy()
        self.goal_velocity_raw = np.zeros(6)
        self.recording = True
        return start

    def command(self, position, rotation=None, check_contacts=False):
        self.cartesian_target = np.asarray(position).copy()
        super().command(position, rotation, check_contacts)

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if not self.recording:
            return
        controller = self.robot.part_controllers["right"]
        goal = controller.goal_qpos.copy()
        if policy_step:
            self.goal_velocity_raw = (goal - self.previous_goal) * self.control_freq
            self.previous_goal = goal
        velocity = getattr(controller, "goal_velocity", np.zeros(6))
        matrix = controller.mass_matrix if controller.use_torque_compensation else np.eye(6)
        p = matrix @ (controller.kp * (goal - controller.joint_pos))
        d = matrix @ (-controller.kd * controller.joint_vel)
        ff = matrix @ (controller.kd * velocity)
        bias = (
            controller.torque_compensation.copy()
            if controller.use_torque_compensation
            else np.zeros(6)
        )
        armature = self.sim.model.dof_armature[controller.qvel_index].copy()
        acceleration_command = controller.kp * (goal - controller.joint_pos) + controller.kd * (
            velocity - controller.joint_vel
        )
        armature_torque = (
            armature * acceleration_command if controller.use_torque_compensation else np.zeros(6)
        )
        requested = controller.torques.copy()
        indexes = self.robot._ref_actuators_indexes_dict["right"]
        applied = self.sim.data.ctrl[indexes].copy()
        if not np.allclose(requested, p + d + ff + bias, atol=1e-10):
            raise RuntimeError("Torque decomposition does not match the real controller")
        bounds = self.sim.model.actuator_ctrlrange[indexes]
        if not np.allclose(applied, np.clip(requested, bounds[:, 0], bounds[:, 1]), atol=1e-10):
            raise RuntimeError("Recorded applied torque does not match actuator clipping")
        self.trace.append(
            dict(
                time=float(self.sim.data.time - self.time_zero),
                policy_step=policy_step,
                q=controller.joint_pos.copy(),
                dq=controller.joint_vel.copy(),
                q_goal=goal,
                dq_goal_raw=self.goal_velocity_raw.copy(),
                dq_goal=velocity.copy(),
                ik_error_correction_dq=self.robot.composite_controller.joint_action_policy.dq.copy(),
                position=self.sim.data.site_xpos[self.site_id].copy(),
                velocity=controller.J_pos @ controller.joint_vel,
                target=self.cartesian_target.copy(),
                p=p,
                d=d,
                ff=ff,
                bias=bias,
                requested=requested,
                applied=applied,
                armature=armature,
                mass_diagonal=np.diag(controller.mass_matrix).copy(),
                armature_torque=armature_torque,
                jacobian_singular_values=np.linalg.svd(controller.J_full, compute_uv=False),
            )
        )

    def arrays(self):
        return {key: np.asarray([row[key] for row in self.trace]) for key in self.trace[0]}


def intervals(mask, times, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        [float(times[a]), float(times[b - 1] + dt)]
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def summarize(trace, rollout, limits, dt):
    t = trace["time"]
    reversal = (t >= 3 - 1e-8) & (t < 3.5 - 1e-8)
    clipped = np.abs(trace["requested"] - trace["applied"]) > 1e-8
    error = np.linalg.norm(rollout["position"] - rollout["target_position"], axis=1)
    peak = int(np.argmax(error))
    switch = np.flatnonzero((t >= 3 - 1e-8) & (trace["velocity"][:, 1] <= 0))
    recovery = np.flatnonzero((rollout["time"] > 3.1) & (error <= 0.001))
    rows = []
    for j in range(6):
        where = int(np.argmax(np.abs(trace["requested"][:, j])))
        rows.append(
            dict(
                joint=j + 1,
                limit_nm=limits[j],
                peak_requested_nm=float(trace["requested"][where, j]),
                peak_time_s=float(t[where]),
                components_at_peak_nm={
                    key: float(trace[key][where, j]) for key in ("p", "d", "ff", "bias")
                },
                armature_contribution_at_peak_nm=float(trace["armature_torque"][where, j]),
                remaining_torque_at_peak_nm=float(
                    trace["requested"][where, j] - trace["armature_torque"][where, j]
                ),
                clipped_duration_s=float(np.sum(clipped[:, j]) * dt),
                reversal_clipped_duration_s=float(np.sum(clipped[reversal, j]) * dt),
                clipping_intervals_s=intervals(clipped[:, j], t, dt),
                reversal_peak_dq_target_raw=float(
                    np.max(np.abs(trace["dq_goal_raw"][reversal, j]))
                ),
                reversal_peak_dq_target=float(np.max(np.abs(trace["dq_goal"][reversal, j]))),
                reversal_peak_dq_actual=float(np.max(np.abs(trace["dq"][reversal, j]))),
            )
        )
    window = (t >= 2.9) & (t <= 3.5)
    policy = trace["policy_step"] & window
    changes = np.diff(trace["q_goal"][trace["policy_step"]], axis=0)
    return dict(
        metrics=rollout_metrics(rollout),
        peak_error_time_s=float(rollout["time"][peak]),
        actual_y_velocity_reversal_s=float(t[switch[0]]) if len(switch) else None,
        first_below_1mm_after_3p1_s=float(rollout["time"][recovery[0]]) if len(recovery) else None,
        reversal_squared_error_fraction=float(
            np.sum(error[(rollout["time"] > 3) & (rollout["time"] <= 3.5)] ** 2) / np.sum(error**2)
        ),
        velocity_limit_active_policy_steps=int(
            np.sum(
                np.any(
                    np.abs(trace["dq_goal_raw"][policy] - trace["dq_goal"][policy]) > 1e-8, axis=1
                )
            )
        ),
        max_goal_step_rad_per_joint=np.max(np.abs(changes), axis=0).tolist(),
        jacobian_min_singular_value=float(trace["jacobian_singular_values"][window, -1].min()),
        compiled_armature_kg_m2=trace["armature"][0].tolist(),
        initial_mass_diagonal_kg_m2=trace["mass_diagonal"][0].tolist(),
        initial_diagonal_without_armature_kg_m2=(
            trace["mass_diagonal"][0] - trace["armature"][0]
        ).tolist(),
        joints=rows,
    )


def plots(out, traces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, (trace, rollout) in traces.items():
        if not name.startswith("ff_"):
            continue
        t = trace["time"]
        window = (t >= 2.9) & (t <= 3.5)
        j = int(np.argmax(np.sum(np.abs(trace["requested"] - trace["applied"])[window], axis=0)))
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
        axes[0].plot(
            rollout["time"],
            1000 * np.linalg.norm(rollout["position"] - rollout["target_position"], axis=1),
        )
        axes[0].set_ylabel("Position error (mm)")
        axes[1].plot(t, trace["velocity"][:, 1] * 1000, label="Actual world Y")
        axes[1].plot([2.9, 3, 3, 3.5], [50, 50, -50, -50], "k--", label="Nominal target Y")
        axes[1].set_ylabel("TCP speed (mm/s)")
        for key, label in (
            ("dq", "Actual"),
            ("dq_goal", "Target (bounded)"),
            ("dq_goal_raw", "Target (raw)"),
        ):
            axes[2].plot(t, trace[key][:, j], label=label)
        axes[2].set_ylabel(f"Joint {j + 1} speed (rad/s)")
        for key in ("requested", "applied", "p", "d", "ff", "bias"):
            axes[3].plot(
                t,
                trace[key][:, j],
                label=key,
                linewidth=1.5 if key in ("requested", "applied") else 1,
            )
        axes[3].set_ylabel(f"Joint {j + 1} torque (N m)")
        for ax in axes:
            ax.set_xlim(2.9, 3.5)
            ax.axvline(3, color="gray", alpha=0.4)
            ax.grid(alpha=0.2)
        for ax in axes[1:]:
            ax.legend(loc="upper right", ncol=2, fontsize=8)
        axes[-1].set_xlabel("Exploration time (s); physics values sampled before integration")
        fig.suptitle(name + ": unchanged free-space trajectory and torque limits")
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs/sim_pretrain/reversal_diagnosis"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError("Diagnostic report exists; use --output with a new directory")
    report = dict(
        config=cfg,
        provenance=provenance(),
        script_sha256=file_digest(__file__),
        scope="free-space diagnostics only, not training data or a physics-gate override",
        runs={},
    )
    robot_base = ROBOSUITE_PACKAGE / "models/robots/robot_model.py"
    report["robot_model_base_sha256"] = file_digest(robot_base)
    report["armature_interpretation"] = (
        "M = M_rigid + diag(armature). Torque contributions are algebraic attribution at the recorded state; "
        "they are not a simulated outcome with armature removed. No dynamics parameters are changed."
    )
    traces = {}
    for enabled in (False, True):
        for gain in cfg["simulation"]["gain_candidates"]:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["simulation"]["target_velocity_feedforward"] = enabled
            name = f"{'ff' if enabled else 'no_ff'}_gain_{gain}"
            env = None
            try:
                env = TraceWipe(run_cfg, gain)
                rollout = env.rollout(extra_height=0.08)
                trace = env.arrays()
                np.savez_compressed(out / f"{name}_physics.npz", **trace)
                np.savez_compressed(out / f"{name}_rollout.npz", **rollout)
                row = summarize(
                    trace,
                    rollout,
                    cfg["simulation"]["torque_limits"],
                    cfg["simulation"]["timestep"],
                )
                if not enabled:
                    row["velocity_limit_active_policy_steps"] = None
                baseline_dir = "pretrain_paper_velocity_ff" if enabled else "pretrain_paper"
                old = ROOT / "outputs" / baseline_dir / "sanity_traces" / f"free_gain_{gain}.npz"
                if old.exists():
                    with np.load(old) as previous:
                        row["max_position_difference_from_previous_run_m"] = float(
                            np.max(np.abs(previous["position"] - rollout["position"]))
                        )
                report["runs"][name] = row
                traces[name] = (trace, rollout)
                print(name, json.dumps(row), flush=True)
            except Exception as exc:
                report["runs"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(name, report["runs"][name], flush=True)
            finally:
                if env is not None:
                    env.close()
            write_json(out / "report.json", report)
    plots(out, traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
