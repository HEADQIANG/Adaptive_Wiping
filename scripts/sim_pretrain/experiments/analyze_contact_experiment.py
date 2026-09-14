"""Re-score isolated experiments and check sampled quasistatic torque budgets."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from scripts.shared.common import digest, file_digest, provenance, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import tracking_passes, validate_trajectory
from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics


def quasistatic_budget(model, data, tcp, ft, dofs, wrenches, limits):
    """World wrenches ON the tool at FT; zero-velocity, zero-acceleration balance."""
    if np.any(data.qvel):
        raise ValueError("Quasistatic budget requires zero velocity")
    wrenches = np.asarray(wrenches).reshape(-1, 6)
    shifted = wrenches.copy()
    shifted[:, 3:] += np.cross(data.site_xpos[ft] - data.site_xpos[tcp], shifted[:, :3])
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, tcp)
    generalized = shifted @ np.vstack([jp, jr])[:, dofs]
    direct = []
    for wrench in wrenches:
        force = np.zeros(model.nv)
        mujoco.mj_applyFT(
            model, data, wrench[:3], wrench[3:], data.site_xpos[ft], model.site_bodyid[ft], force
        )
        direct.append(force[dofs])
    np.testing.assert_allclose(generalized, direct, atol=1e-10)
    required = data.qfrc_bias[dofs] - generalized
    peak = np.max(np.abs(required), axis=0)
    return {
        "gravity_nm": data.qfrc_bias[dofs].tolist(),
        "required_peak_abs_nm": peak.tolist(),
        "limits_nm": np.asarray(limits).tolist(),
        "peak_limit_ratio": float(np.max(peak / limits)),
        "within_limits": bool(np.all(peak <= limits)),
        "wrench_mapping_residual_nm": float(np.max(np.abs(generalized - direct))),
    }


def assay_summary(folder, report):
    rows, wrench_samples = [], []
    for index, case in enumerate(report["cases"]):
        path = folder / f"rig_{index:02d}.npz"
        with np.load(path) as archive:
            trace = archive["trace"]
        assert trace.ndim == 2 and trace.shape[1] == 14 and np.isfinite(trace).all()
        row = {
            "index": index,
            "mode": case["mode"],
            "parameters": case["parameters"],
            "normal_target_n": case["normal_target_n"],
            "completed": case["completed"],
            "trace_sha256": file_digest(path),
            "max_penetration_m": float(max(0, -trace[:, 2].min() - 0.001)),
        }
        settled = trace[(trace[:, 0] >= 1.8) & (trace[:, 0] <= 2)]
        if len(settled):
            row["settled_normal_n"] = float(settled[:, 13].mean())
            row["settled_penetration_m"] = float(max(0, -settled[:, 2].mean() - 0.001))
        if case.get("wiping_speed_reached"):
            # The final 26 samples confirm >=50 mm/s for 50 ms, not steady sliding.
            speed = trace[-26:]
            assert np.all(speed[:, 3] >= 0.05)
            assert np.isclose(speed[-1, 0] - speed[0, 0], 0.05)
            row.update(
                speed_window_normal_minmax_n=[float(speed[:, 13].min()), float(speed[:, 13].max())],
                speed_window_drag_mean_n=float(-speed[:, 8].mean()),
                speed_window_moment_peak_nm=np.max(np.abs(speed[:, 10:13]), axis=0).tolist(),
                speed_window_applied_fy_minmax_n=[
                    float(speed[:, 5].min()),
                    float(speed[:, 5].max()),
                ],
            )
        if case["mode"] == "mapped" and case["parameters"][0] in (0.9, 3.5) and case["completed"]:
            ramp = trace[trace[:, 0] >= 2]
            wrench_samples.append((row, ramp[:, 7:13]))
        rows.append(row)
    return rows, wrench_samples


def robot_summary(folder, report):
    rows, diagnostic_traces = [], {}
    expected_twist = np.zeros((400, 6))
    expected_twist[:200, 2] = -0.01
    expected_twist[200:300, 1] = 0.05
    expected_twist[300:, 1] = -0.05
    for name, variant in report["variants"].items():
        cases = [("free", variant["free"])]
        cases += [(f"contact_{i:02d}", row) for i, row in enumerate(variant["contacts"])]
        cases += [(f"grid_{i:02d}", row) for i, row in enumerate(variant.get("grid", []))]
        for label, case in cases:
            path = folder / name / f"{label}.npz"
            if not path.exists():
                assert "error" in case
                rows.append({"variant": name, "case": label, "error": case["error"]})
                continue
            with np.load(path) as archive:
                data = dict(archive)
            metrics = rollout_metrics(data)
            assert metrics == case["metrics"]
            if label == "free":
                passed = tracking_passes(metrics, report["config"])
            else:
                validate_trajectory(data)
                gate = contact_motion_acceptance(data)
                assert gate == case["contact_motion"]
                passed = gate["passed"] and case.get("unloaded", False)
            assert passed == case["passed"]
            row = {
                "variant": name,
                "case": label,
                "passed": passed,
                "trace_sha256": file_digest(path),
            }
            if name != "baseline":
                np.testing.assert_allclose(data["nominal_twist"], expected_twist, atol=1e-11)
                row["nominal_twist_max_error"] = float(
                    np.max(np.abs(data["nominal_twist"] - expected_twist))
                )
            if label != "free":
                row["normal_slide_mean_n"] = float(data["normal_sum"][200:].mean())
                row["normal_slide_minmax_n"] = [
                    float(data["normal_sum"][200:].min()),
                    float(data["normal_sum"][200:].max()),
                ]
                row["loaded_slide_fraction"] = float(
                    np.mean(data["contact_count_loaded"][200:] > 0)
                )
                row["failed_checks"] = case["contact_motion"]["failed_checks"]
                row["unloaded"] = case.get("unloaded", False)
            if case["parameters"] == [0.9, 1000, 0.02] and label.startswith("contact"):
                diagnostic_traces[name] = data
            rows.append(row)
    return rows, diagnostic_traces


def figures(out, traces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    colors = ["#333333", "#9471ad", "#138276", "#cf4e45", "#3478b2"]
    for (name, data), color in zip(traces.items(), colors):
        axes[0].plot(data["time"], 1000 * data["position"][:, 1], label=name, color=color)
        axes[1].plot(data["time"], data["normal_sum"], color=color)
        axes[2].plot(data["time"], np.rad2deg(data["orientation_error"]), color=color)
        axes[3].plot(
            data["time"],
            np.max(np.abs(data["requested_torque"]) / [10, 10, 10, 5, 5, 5], axis=1),
            color=color,
        )
    data = next(iter(traces.values()))
    axes[0].plot(data["time"], 1000 * data["target_position"][:, 1], "k--", label="nominal target")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].set_title("AIRBOT direct wrist: mu=0.9, k=1000, width=0.02")
    axes[2].axhline(2, color="gray", linestyle=":")
    axes[3].axhline(1, color="gray", linestyle=":")
    for ax, label in zip(
        axes,
        ["World Y (mm)", "Normal load (N)", "Orientation error (deg)", "Requested torque / limit"],
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.axvline(2, color="gray", linewidth=0.6)
        ax.axvline(3, color="gray", linewidth=0.6)
    axes[-1].set_xlabel("Exploration time (s)")
    fig.savefig(out / "mu0p9_comparison.png", dpi=150)
    plt.close(fig)

    checks = {}
    for path in sorted(out.parent.glob("*/mu0p9.gif")):
        with Image.open(path) as gif:
            selected = []
            for index in (0, min(60, gif.n_frames - 1), gif.n_frames - 1):
                gif.seek(index)
                selected.append(gif.convert("RGB"))
            a, b = np.asarray(selected[0])[25:], np.asarray(selected[1])[25:]
            checks[path.parent.name] = {
                "frames": gif.n_frames,
                "image_std": float(a.std()),
                "changed_pixel_fraction_excluding_header": float(np.mean(np.any(a != b, axis=2))),
            }
            assert a.std() > 10 and np.any(a != b)
            sheet = Image.new("RGB", (640 * 3, 480))
            for index, frame in enumerate(selected):
                sheet.paste(frame, (640 * index, 0))
            sheet.save(out / f"{path.parent.name}_frames.png")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True)
    parser.add_argument("--robot", required=True)
    args = parser.parse_args()
    rig, robot = Path(args.rig), Path(args.robot)
    rr = json.loads((rig / "report.json").read_text())
    cr = json.loads((robot / "report.json").read_text())
    assert rr["source_unchanged"] and cr["source_unchanged"]
    assert rr["provenance"] == cr["provenance"] == provenance()
    assert digest(rr["config"]) == cr["config_hash"] == digest(cr["config"])
    assert (
        file_digest(ROOT / "scripts/sim_pretrain/experiments/contact_breakaway.py")
        == rr["script_sha256"]
    )
    assert all(file_digest(ROOT / path) == sha for path, sha in cr["sources"].items())
    out = robot / "analysis"
    out.mkdir(exist_ok=False)
    assays, wrenches = assay_summary(rig, rr)
    rows, traces = robot_summary(robot, cr)
    baseline = json.loads((ROOT / cr["config"]["output_dir"] / "sanity.json").read_text())
    assert (
        baseline["provenance"] == cr["provenance"] and baseline["config_hash"] == cr["config_hash"]
    )
    with np.load(ROOT / cr["config"]["output_dir"] / "mu0p9_contact.npz") as prior:
        for key in ("position", "ft", "joints"):
            np.testing.assert_array_equal(prior[key], traces["baseline"][key])
    budgets = []
    env = PretrainingWipe(cr["config"], 300)
    try:
        model = env.sim.model._model
        snapshot = mujoco.MjData(model)
        controller = env.robot.part_controllers["right"]
        dofs = np.asarray(controller.qvel_index)
        ft = env.sim.model.site_name2id(env.robot.gripper["right"].naming_prefix + "ft_frame")
        limits = np.asarray(cr["config"]["simulation"]["torque_limits"])
        for name, data in traces.items():
            for index in (199, 299, 399):
                snapshot.qpos[:] = env.sim.data.qpos
                snapshot.qpos[controller.qpos_index] = data["joints"][index]
                snapshot.qvel[:] = 0
                mujoco.mj_forward(model, snapshot)
                np.testing.assert_allclose(
                    snapshot.site_xpos[env.site_id], data["position"][index], atol=1e-10
                )
                for assay, samples in wrenches:
                    budget = quasistatic_budget(
                        model, snapshot, env.site_id, ft, dofs, samples, limits
                    )
                    budget.update(
                        variant=name,
                        time_s=float(data["time"][index]),
                        rig_index=assay["index"],
                        joints_rad=data["joints"][index].tolist(),
                    )
                    budgets.append(budget)
    finally:
        env.close()
    report = {
        "verification_passed": True,
        "experiment_passed": cr["passed"],
        "production_unchanged": True,
        "production_baseline_reproduced": True,
        "analysis_sha256": file_digest(__file__),
        "input_hashes": {
            str(path): file_digest(path) for path in (rig / "report.json", robot / "report.json")
        },
        "assays": assays,
        "rescored_traces": rows,
        "budgets": budgets,
        "budget_scope": "Hypothetical SAME WORLD rig loads at recorded mu=.9 robot FT poses at 2/3/4 s; all ramp samples, qvel=qacc=0, tau=gravity-J.T*wrench. Actual tilted contact distribution, inertia, transients, stability and trajectory reachability are NOT established.",
        "speed_scope": "50 ms above 50 mm/s during a rising force ramp; not steady-state drag or constant normal load during sliding.",
        "gif_checks": figures(out, traces),
    }
    write_json(out / "verification.json", report)
    print(
        json.dumps(
            {
                "verification_passed": True,
                "experiment_passed": cr["passed"],
                "rescored": len(rows),
                "budgets": len(budgets),
                "maximum_quasistatic_limit_ratio": max(row["peak_limit_ratio"] for row in budgets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
