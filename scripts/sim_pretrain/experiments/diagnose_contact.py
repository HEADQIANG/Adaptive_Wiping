"""Contact control-chain diagnostics; no controller changes or collection-gate writes."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.shared.common import (
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.diagnose_reversal import TraceWipe
from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics


def contact_result(model, data, tool_ids, table_id, origin, dofs):
    """World wrench ON the tool, about origin; MuJoCo force is on geom2."""
    wrench = np.zeros(6)
    generalized = np.zeros(model.nv)
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if not (
            table_id in (contact.geom1, contact.geom2)
            and (contact.geom1 in tool_ids or contact.geom2 in tool_ids)
        ):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        sign = 1 if contact.geom2 in tool_ids else -1
        rotation = contact.frame.reshape(3, 3).T
        force = sign * rotation @ local[:3]
        torque = sign * rotation @ local[3:]
        wrench += np.r_[force, torque + np.cross(contact.pos - origin, force)]
        body = model.geom_bodyid[contact.geom2 if sign == 1 else contact.geom1]
        mujoco.mj_applyFT(model, data, force, torque, contact.pos, body, generalized)
        # This ratio is translational only, not the full elliptic/pyramidal cone utilization.
        ratio = (
            np.linalg.norm(local[1:3]) / (contact.friction[0] * local[0])
            if local[0] > 1e-9
            else 0.0
        )
        rows.append(
            np.r_[
                index,
                contact.geom1,
                contact.geom2,
                contact.dim,
                contact.dist,
                contact.pos,
                local,
                contact.friction[0],
                ratio,
            ]
        )
    rows = np.asarray(rows).reshape(-1, 16)
    return dict(
        wrench=wrench,
        generalized=generalized[dofs],
        rows=rows,
        normal_sum=float(rows[:, 8].sum()),
        tangent_norm_sum=float(np.linalg.norm(rows[:, 9:11], axis=1).sum()),
        max_translation_friction_ratio=float(rows[:, 15].max()) if len(rows) else 0.0,
    )


class ContactTraceWipe(TraceWipe):
    def prepare(self, extra_height=0.0):
        start = super().prepare(extra_height)
        self.diagnostic_data = mujoco.MjData(self.sim.model._model)
        self.fk_data = mujoco.MjData(self.sim.model._model)
        self.contact_rows = []
        return start

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if not self.recording:
            return
        row = self.trace[-1]
        controller = self.robot.part_controllers["right"]
        ik = self.robot.composite_controller.joint_action_policy
        model, live = self.sim.model._model, self.sim.data._data
        dofs = np.asarray(controller.qvel_index)
        jac = controller.J_full.copy()
        if policy_step:
            if len(ik.site_ids) != 1 or not np.array_equal(ik.dof_ids, dofs):
                raise RuntimeError("Diagnostic requires the six-joint single-site AIRBOT IK")
            ik_jac = np.vstack(ik.jac_temps)[:, ik.dof_ids].copy()
            raw = ik_jac.T @ np.linalg.solve(
                ik_jac @ ik_jac.T + ik.damping**2 * np.eye(6), ik.twist
            )
            raw += (np.eye(6) - np.linalg.pinv(ik_jac) @ ik_jac) @ (
                ik.Kn * (ik.q0 - live.qpos[ik.dof_ids])
            )
            scale = min(1.0, ik.max_dq / max(np.max(np.abs(raw)), 1e-30)) if ik.max_dq > 0 else 1.0
            if not np.allclose(raw * scale, ik.dq, atol=1e-10):
                raise RuntimeError("Reconstructed IK velocity differs from actual solver")
            pre_clip = live.qpos[ik.dof_ids] + ik.dq * ik.integration_dt
            expected = np.clip(pre_clip, *model.jnt_range[ik.dof_ids].T)
            if not np.allclose(expected, row["q_goal"], atol=1e-10):
                raise RuntimeError("Reconstructed IK goal differs from actual controller goal")
            self.ik_snapshot = dict(
                ik_dq_raw=raw,
                ik_scale=scale,
                ik_joint_limit_clipped=np.abs(expected - pre_clip) > 1e-10,
                ik_twist=ik.twist.copy(),
                ik_base_q=row["q"].copy(),
            )
        row.update(self.ik_snapshot)

        # step1 has not solved forces for the new ctrl yet. Forward a COPY so wrench,
        # qacc and generalized forces refer to this same state and newly applied ctrl.
        data = self.diagnostic_data
        mujoco.mj_copyData(data, model, live)
        mujoco.mj_forward(model, data)
        origin = data.site_xpos[self.site_id].copy()
        contacts = contact_result(model, data, self.tool_ids, self.table_id, origin, dofs)
        if len(contacts["rows"]):
            self.contact_rows.extend(
                np.c_[np.full(len(contacts["rows"]), row["time"]), contacts["rows"]]
            )
        if not np.allclose(jac.T @ contacts["wrench"], contacts["generalized"], atol=1e-8):
            raise RuntimeError("Contact wrench frame / moment-arm reconstruction failed")
        grip = self.robot.gripper["right"]
        sensor_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, grip.important_sensors[key])
            for key in ("force_ee", "torque_ee")
        ]
        sensor = np.concatenate(
            [data.sensordata[model.sensor_adr[i] : model.sensor_adr[i] + 3] for i in sensor_ids]
        )
        ft_site = self.sim.model.site_name2id(grip.naming_prefix + "ft_frame")
        sensor_rotation = data.site_xmat[ft_site].reshape(3, 3).copy()
        at_sensor = contacts["wrench"].copy()
        at_sensor[3:] += np.cross(origin - data.site_xpos[ft_site], at_sensor[:3])
        contact_local = np.r_[sensor_rotation.T @ at_sensor[:3], sensor_rotation.T @ at_sensor[3:]]
        feedback_wrench = np.linalg.lstsq(jac.T, row["requested"] - row["bias"], rcond=None)[0]
        balance = (
            controller.mass_matrix @ data.qacc[dofs]
            + data.qfrc_bias[dofs]
            - data.qfrc_actuator[dofs]
            - data.qfrc_passive[dofs]
            - data.qfrc_constraint[dofs]
            - data.qfrc_applied[dofs]
        )
        mujoco.mj_copyData(self.fk_data, model, live)
        self.fk_data.qpos[controller.qpos_index] = row["q_goal"]
        mujoco.mj_kinematics(model, self.fk_data)
        row.update(
            jacobian=jac,
            mass_matrix=controller.mass_matrix.copy(),
            q_error=row["q_goal"] - row["q"],
            kp=controller.kp.copy(),
            kd=controller.kd.copy(),
            goal_fk_position=self.fk_data.site_xpos[self.site_id].copy(),
            orientation_error=float(
                (
                    self.goal_rotation.inv()
                    * Rotation.from_matrix(data.site_xmat[self.site_id].reshape(3, 3))
                ).magnitude()
            ),
            contact_wrench_world_tcp=contacts["wrench"],
            contact_qfrc=contacts["generalized"],
            contact_normal_sum=contacts["normal_sum"],
            contact_tangent_norm_sum=contacts["tangent_norm_sum"],
            contact_max_translation_ratio=contacts["max_translation_friction_ratio"],
            contact_count=len(contacts["rows"]),
            sensor_local=sensor,
            sensor_position=data.site_xpos[ft_site].copy(),
            sensor_rotation=sensor_rotation,
            contact_wrench_local_sensor=contact_local,
            feedback_equivalent_wrench_world_tcp=feedback_wrench,
            feedback_wrench_residual=jac.T @ feedback_wrench - (row["requested"] - row["bias"]),
            qfrc_actuator=data.qfrc_actuator[dofs].copy(),
            qfrc_passive=data.qfrc_passive[dofs].copy(),
            qfrc_constraint=data.qfrc_constraint[dofs].copy(),
            qacc=data.qacc[dofs].copy(),
            dynamics_balance_residual=balance,
        )


def summarize_contact(trace, rollout, baseline):
    policy = trace["policy_step"]
    slide = (trace["time"] >= 2 - 1e-8) & (trace["time"] < 4 - 1e-8)
    forward = (trace["time"] >= 2.8 - 1e-8) & (trace["time"] < 3 - 1e-8)
    peak = int(np.argmax(np.linalg.norm(rollout["position"] - rollout["target_position"], axis=1)))
    result = dict(
        metrics=rollout_metrics(rollout),
        baseline_max_position_difference_m=float(
            np.max(np.abs(rollout["position"] - baseline["position"]))
        ),
        baseline_max_ft_difference=float(np.max(np.abs(rollout["ft"] - baseline["ft"]))),
        y_range_mm=float(np.ptp(rollout["position"][:, 1]) * 1000),
        peak_error_time_s=float(rollout["time"][peak]),
        peak_error_xyz_mm=(
            (rollout["position"][peak] - rollout["target_position"][peak]) * 1000
        ).tolist(),
        ik_limited_policy_steps=int(np.sum(trace["ik_scale"][policy] < 1 - 1e-8)),
        ik_limited_slide_policy_steps=int(np.sum(trace["ik_scale"][policy & slide] < 1 - 1e-8)),
        ik_min_scale=float(trace["ik_scale"][policy].min()),
        ik_raw_peak_rad_s=float(np.max(np.abs(trace["ik_dq_raw"][policy]))),
        ik_joint_limit_policy_steps=int(
            np.sum(np.any(trace["ik_joint_limit_clipped"][policy], axis=1))
        ),
        ff_limited_policy_steps=int(
            np.sum(
                np.any(
                    np.abs(trace["dq_goal_raw"][policy] - trace["dq_goal"][policy]) > 1e-8, axis=1
                )
            )
        ),
        max_policy_joint_error_rad=float(np.max(np.abs(trace["q_error"][policy]))),
        max_requested_nm_per_joint=np.max(np.abs(trace["requested"]), axis=0).tolist(),
        clipped_substeps=int(
            np.sum(np.any(np.abs(trace["requested"] - trace["applied"]) > 1e-8, axis=1))
        ),
        constraint_contact_residual_max_nm=float(
            np.max(np.abs(trace["qfrc_constraint"] - trace["contact_qfrc"]))
        ),
        dynamics_balance_residual_max_nm=float(np.max(np.abs(trace["dynamics_balance_residual"]))),
        wrench_reconstruction_residual_max_nm=float(
            np.max(np.abs(trace["feedback_wrench_residual"]))
        ),
        jacobian_min_singular_value=float(trace["jacobian_singular_values"][:, -1].min()),
    )
    result["forward_end_2p8_to_3p0_mean"] = {
        key: np.mean(trace[key][forward], axis=0).tolist()
        for key in (
            "ik_scale",
            "ik_dq_raw",
            "ik_error_correction_dq",
            "q_error",
            "dq_goal",
            "dq",
            "position",
            "target",
            "goal_fk_position",
            "p",
            "d",
            "ff",
            "bias",
            "requested",
            "contact_wrench_world_tcp",
            "contact_normal_sum",
            "contact_tangent_norm_sum",
            "contact_max_translation_ratio",
            "feedback_equivalent_wrench_world_tcp",
            "sensor_local",
            "contact_wrench_local_sensor",
            "orientation_error",
        )
    }
    for name, time in (("press_end", 2), ("forward_end", 3), ("reverse_end", 4)):
        index = int(np.argmin(np.abs(trace["time"] - (time - 0.01))))
        result[name] = {
            key: trace[key][index].tolist()
            for key in (
                "time",
                "position",
                "target",
                "goal_fk_position",
                "ik_scale",
                "ik_dq_raw",
                "ik_error_correction_dq",
                "q_error",
                "dq_goal",
                "p",
                "d",
                "ff",
                "requested",
                "contact_wrench_world_tcp",
                "feedback_equivalent_wrench_world_tcp",
                "orientation_error",
            )
        }
    return result


def plots(out, traces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, len(traces), figsize=(15, 16), sharex=True, constrained_layout=True)
    for col, (name, (trace, rollout)) in enumerate(traces.items()):
        t = trace["time"]
        ax = axes[:, col]
        ax[0].set_title(name)
        ax[0].plot(rollout["time"], rollout["target_position"][:, 1] * 1000, "k--", label="Target")
        ax[0].plot(t, trace["position"][:, 1] * 1000, label="Actual")
        ax[0].plot(t, trace["goal_fk_position"][:, 1] * 1000, label="FK(joint goal)")
        ax[1].plot(t, np.rad2deg(trace["orientation_error"]), label="Orientation error")
        ax[2].plot(t, np.max(np.abs(trace["ik_dq_raw"]), axis=1), label="IK raw")
        ax[2].plot(t, np.max(np.abs(trace["ik_error_correction_dq"]), axis=1), label="IK bounded")
        ax[2].plot(t, np.max(np.abs(trace["dq_goal"]), axis=1), label="Goal difference (FF)")
        ax[3].plot(t, trace["contact_normal_sum"], label="Sum normal")
        ax[3].plot(t, trace["contact_tangent_norm_sum"], label="Sum tangent norm")
        ax[3].plot(
            t, trace["feedback_equivalent_wrench_world_tcp"][:, 1], label="Feedback equiv. Fy"
        )
        ax[4].plot(t, trace["contact_max_translation_ratio"], label="Max |Ft| / (mu Fn)")
        j = 0
        for key in ("p", "d", "ff", "requested", "applied"):
            ax[5].plot(t, trace[key][:, j], label=key)
        for a in ax:
            a.axvline(2, color="gray", alpha=0.3)
            a.axvline(3, color="gray", alpha=0.3)
            a.grid(alpha=0.2)
            a.legend(fontsize=7, loc="upper left")
        ax[-1].set_xlabel("Time before integration (s)")
    for ax, label in zip(
        axes[:, 0],
        (
            "World Y (mm)",
            "Angle (deg)",
            "Joint speed (rad/s)",
            "Force (N)",
            "Translation ratio",
            "Joint 1 torque (N m)",
        ),
    ):
        ax.set_ylabel(label)
    fig.savefig(out / "contact_control_chain.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs/sim_data/contact_diagnosis"))
    parser.add_argument("--gain", type=float, default=300)
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        config=cfg,
        gain=args.gain,
        provenance=provenance(),
        diagnostic_sources={
            str(p.relative_to(ROOT)): file_digest(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/sim_pretrain/experiments/diagnose_reversal.py",
            )
        },
        scope="diagnostics only; unchanged production controller, parameters and gate",
        runs={},
    )
    traces = {}
    for name, parameters in (
        ("mu0_k1000", (0, 1000, 0.02)),
        ("mu3p5_k1000", (3.5, 1000, 0.02)),
        ("mu3p5_k0p5", (3.5, 0.5, 0.02)),
    ):
        print(f"Running {name}: uninstrumented reference then diagnostic", flush=True)
        env = PretrainingWipe(cfg, args.gain, parameters)
        try:
            baseline = env.rollout()
        finally:
            env.close()
        env = ContactTraceWipe(cfg, args.gain, parameters)
        try:
            rollout = env.rollout()
            trace = env.arrays()
            row = summarize_contact(trace, rollout, baseline)
            row["parameters"] = list(parameters)
            row["compiled"] = dict(
                armature=trace["armature"][0].tolist(),
                cone=int(env.sim.model.opt.cone),
                ik_dt=env.robot.composite_controller.joint_action_policy.integration_dt,
                ik_max_dq=env.robot.composite_controller.joint_action_policy.max_dq,
            )
            np.savez_compressed(out / f"{name}_physics.npz", **trace)
            np.savez_compressed(out / f"{name}_rollout.npz", **rollout)
            np.savez_compressed(out / f"{name}_baseline.npz", **baseline)
            np.savez_compressed(
                out / f"{name}_contacts.npz", contacts=np.asarray(env.contact_rows).reshape(-1, 17)
            )
            if (
                row["baseline_max_position_difference_m"] > 1e-12
                or row["baseline_max_ft_difference"] > 1e-10
            ):
                raise RuntimeError("Instrumentation changed rollout; diagnostic rejected")
            report["runs"][name] = row
            traces[name] = (trace, rollout)
            write_json(out / "report.json", report)
            print(name, json.dumps(row), flush=True)
        finally:
            env.close()
    plots(out, traces)
    print(f"Diagnostic complete: {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
