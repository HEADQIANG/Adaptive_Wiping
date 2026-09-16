"""Read-only force-domain and demonstration alignment diagnostics; no hardware API."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.real_training.config import load_config as load_real_config
from scripts.real_training.data import load_prepared
from scripts.shared.common import file_digest, load_config, write_json
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.shared.sampling import previous_samples
from scripts.sim_pretrain.learning import load_data


def summary(x):
    x = np.asarray(x, dtype=float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite data")
    return {
        "min": np.min(x, axis=0).tolist(),
        "median": np.median(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
    }


def loo_mean(x):
    """Each prediction excludes the held-out trajectory; RMSE is per coordinate."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite trajectories")
    prediction = (x.sum(0, keepdims=True) - x) / (len(x) - 1)
    error = prediction - x
    return {
        "rmse_mm": float(1000 * np.sqrt(np.mean(error**2))),
        "mae_mm": float(1000 * np.mean(np.abs(error))),
        "per_demo_rmse_mm": (1000 * np.sqrt(np.mean(error**2, axis=(1, 2)))).tolist(),
    }


def onset(time, xy, start, threshold=0.002, hold=5):
    """First sustained displacement, not a force/contact onset detector."""
    active = np.linalg.norm(np.asarray(xy) - start, axis=1) >= threshold
    for i in range(len(active) - hold + 1):
        if np.all(active[i : i + hold]):
            return float(time[i])
    return None


def arc_resample(xy, count=101):
    xy = np.asarray(xy, dtype=float)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-12]
    if distance[-1] < 1e-9:
        raise ValueError("Stationary trajectory has no arc-length parameterization")
    grid = np.linspace(0, distance[-1], count)
    return np.column_stack([np.interp(grid, distance[keep], xy[keep, j]) for j in range(2)])


def force_audit(sim, real, initial, prep, unloaded, quaternions, reference):
    filtered_sim, filtered_real = prep.filtered(sim), prep.filtered(real)[0]
    sim_force_norm = np.linalg.norm(filtered_sim[..., :3], axis=-1)
    real_force_norm = np.linalg.norm(filtered_real[:, :3], axis=-1)
    # Translation affects torque, but neither rotation nor origin shift changes |F|.
    initial = np.asarray(initial)
    relative_rotation = Rotation.from_quat(reference).inv() * Rotation.from_quat(quaternions)
    start_subtracted = filtered_real - initial
    span = np.where(prep.constant, 1.0, prep.maximum - prep.minimum)
    normalized = 0.9 * (filtered_real - prep.minimum) / span
    return {
        "channels": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "units": ["N"] * 3 + ["N m"] * 3,
        "simulation_train_filtered": summary(filtered_sim.reshape(-1, 6)),
        "real_exploration_filtered": summary(filtered_real),
        "simulation_unloaded": summary(unloaded),
        "simulation_force_norm_N": summary(sim_force_norm.flatten()),
        "real_force_norm_N": summary(real_force_norm),
        "initial_raw_wrench": initial.tolist(),
        "initial_force_norm_N": float(np.linalg.norm(initial[:3])),
        "initial_force_mass_equivalent_kg_not_measured_mass": float(
            np.linalg.norm(initial[:3]) / 9.81
        ),
        "fraction_real_force_norm_above_simulation_train_max": float(
            np.mean(real_force_norm > sim_force_norm.max())
        ),
        "out_of_normalization_range_per_channel": np.mean(
            (normalized < 0) | (normalized > 0.9), axis=0
        ).tolist(),
        "counterfactual_start_subtracted_force_norm_N": summary(
            np.linalg.norm(start_subtracted[:, :3], axis=1)
        ),
        "counterfactual_is_not_bias_compensation": True,
        "orientation_change_from_initial_degrees": summary(
            np.rad2deg(relative_rotation.magnitude())
        ),
        "bias_gravity_contact_separately_identified": False,
        "reason": "One initial wrench cannot separate electronic bias, tool gravity and contact; subsequent poses are loaded",
    }


def xy_audit(h5, prepared):
    grid = np.arange(1001) / 100
    trajectories, starts, details, thresholds = [], [], [], []
    for name, group in sorted(h5["demonstrations"].items()):
        time = group["pose_time"][:] - group.attrs["start_time"]
        position = group["sdk_end_position"][:]
        pose = previous_samples(time, position, grid)
        q = previous_samples(time, group["sdk_end_quaternion"][:], grid)
        xy, start = pose[:, :2], pose[0, :2]
        rotations = Rotation.from_quat(q[0]).inv() * Rotation.from_quat(q)
        starts.append(start)
        trajectories.append(xy)
        thresholds.append({str(mm): onset(grid, xy, start, mm / 1000) for mm in (1, 2, 5)})
        ft_time = group["ft_time"][:] - group.attrs["start_time"]
        pose_ages = grid - time[np.searchsorted(time, grid + 1e-9, side="right") - 1]
        ft_ages = grid - ft_time[np.searchsorted(ft_time, grid + 1e-9, side="right") - 1]
        details.append(
            {
                "demo": name,
                "start_xyz_m": pose[0].tolist(),
                "xy_span_mm": (np.ptp(xy, axis=0) * 1000).tolist(),
                "path_length_mm": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() * 1000),
                "end_distance_from_start_mm": float(np.linalg.norm(xy[-1] - start) * 1000),
                "time_of_max_y_s": float(grid[np.argmax(xy[:, 1])]),
                "onset_s_by_threshold_mm": thresholds[-1],
                "max_orientation_change_deg": float(np.rad2deg(rotations.magnitude()).max()),
                "max_pose_age_ms": float(pose_ages.max() * 1000),
                "max_ft_receive_age_ms": float(ft_ages.max() * 1000),
            }
        )
    trajectories, starts = np.array(trajectories), np.array(starts)
    relative = trajectories - starts[:, None]
    policies = prepared["xy"]
    comparisons = {
        "absolute_25_policy_times": loo_mean(policies),
        "start_relative_25_policy_times": loo_mean(policies - starts[:, None]),
        "absolute_100hz": loo_mean(trajectories),
        "start_relative_100hz": loo_mean(relative),
    }
    onset_comparisons = {}
    for mm in (1, 2, 5):
        times = [row[str(mm)] for row in thresholds]
        if any(t is None for t in times):
            onset_comparisons[str(mm)] = {"available": False}
            continue
        # This uses recorded future onset: descriptive alignment, not held-out prediction.
        aligned_time = np.arange(int(np.floor((10 - max(times)) * 100)) + 1) / 100
        aligned = np.array(
            [
                np.column_stack([np.interp(aligned_time + t, grid, x[:, j]) for j in range(2)])
                for t, x in zip(times, relative)
            ]
        )
        same_duration = relative[:, : len(aligned_time)]
        onset_comparisons[str(mm)] = {
            "available": True,
            "uses_held_out_future_onset": True,
            "onset_s": times,
            "common_duration_s": float(aligned_time[-1]),
            "unaligned_same_duration": loo_mean(same_duration),
            "onset_aligned": loo_mean(aligned),
        }
    arc = np.array([arc_resample(x) for x in relative])
    report = {
        "frame": "SDK configured reference; not calibrated sponge TCP",
        "identical_sponge_embeddings": bool(np.all(prepared["sponge"] == prepared["sponge"][0])),
        "start_xy_span_mm": (np.ptp(starts, axis=0) * 1000).tolist(),
        "episodes": details,
        "mean_trajectory_comparisons": comparisons,
        "onset_alignment_diagnostics": onset_comparisons,
        "arc_length_shape_only": {
            **loo_mean(arc),
            "uses_entire_held_out_path": True,
            "not_a_predictive_validation_score": True,
        },
        "centroid_aligned_oracle_25_times": {
            **loo_mean(policies - policies.mean(1, keepdims=True)),
            "uses_held_out_targets": True,
        },
        "training_or_production_data_changed": False,
    }
    return report, trajectories, relative, arc


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new audit output directory")
    cfg = load_real_config(ROOT / "configs/real_training/real_training_airbot_native_repaired.yaml")
    sim_cfg_path = ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"
    sim_cfg = load_config(sim_cfg_path)
    sim, _ = load_data(sim_cfg)
    arrays, info = load_prepared(cfg)
    paths = [
        ROOT / cfg["raw_data"],
        ROOT / cfg["encoder"],
        ROOT / cfg["output_dir"] / "prepared.h5",
        ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/dataset.h5",
        sim_cfg_path,
        ROOT / "configs/robot_control/airbot_calibration.json",
        Path(__file__),
        ROOT / "configs/real_training/real_training_airbot_native_repaired.yaml",
    ]
    paths += [Path(p) for p in info["metadata"]["source_hashes"]]
    hashes = {str(p): file_digest(p) for p in paths}
    for p, digest in info["metadata"]["source_hashes"].items():
        if hashes[p] != digest:
            raise ValueError(f"Recorded source changed: {p}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"inputs_and_script": hashes, "hardware_ready": False})
    write_json(output / "status.json", {"state": "running"})
    try:
        records = [
            json.loads(s)
            for s in (ROOT / "archive/real_training/real_robot/exploration_ft_007.jsonl")
            .read_text()
            .splitlines()
            if s.strip()
        ]
        initial = next(r for r in records if r["event"] == "initial")
        samples = [r for r in records if r["event"] == "sample" and r["phase"] == "exploration"]
        quaternions = np.array([r["state"]["sdk_end_orientation_xyzw"] for r in samples])
        prep = Preprocessor(**sim_cfg["filter"], sample_hz=100).fit(sim["train"])
        with h5py.File(
            ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/dataset.h5", "r"
        ) as h5:
            unloaded = h5["train/unloaded_wrench"][:]
        force = force_audit(
            sim["train"],
            arrays["exploration"],
            initial["force"]["raw_sensor_wrench_si"],
            prep,
            unloaded,
            quaternions,
            initial["state"]["sdk_end_orientation_xyzw"],
        )
        position = np.array([r["state"]["sdk_end_position_m"] for r in samples])
        target = np.array([r["target_position_m"] for r in samples])
        force["exploration_motion"] = {
            "logged_completion": next(r for r in records if r["event"] == "motion_complete"),
            "axis_tracking_rmse_mm": (
                1000 * np.sqrt(np.mean((position - target) ** 2, axis=0))
            ).tolist(),
            "measured_displacements_from_initial_mm_at_2_3_4s": (
                1000 * (position[[199, 299, 399]] - initial["state"]["sdk_end_position_m"])
            ).tolist(),
        }
        with h5py.File(ROOT / cfg["raw_data"], "r") as h5:
            xy, trajectories, relative, arc = xy_audit(h5, arrays)
        report = {
            "scope": "offline_diagnosis_not_calibration_or_training",
            "force": force,
            "xy": xy,
            "hardware_ready": False,
            "sensor_identity_note": "Historical KWR75 labels are retained; operator identified KWR52-TiS S/N 1F06303",
            "inference_limits": [
                "Signed axis mapping is not identified by norm comparisons",
                "Start subtraction removes gravity and any contact as well as bias",
                "Time/arc oracle alignment is not deployable performance evidence",
            ],
        }
        plot(
            output,
            prep.filtered(sim["train"]),
            prep.filtered(arrays["exploration"])[0],
            np.array(initial["force"]["raw_sensor_wrench_si"]),
            trajectories,
            relative,
            arc,
        )
        changed = [p for p, sha in hashes.items() if file_digest(p) != sha]
        if changed:
            raise RuntimeError(f"Inputs changed during audit: {changed}")
        report["inputs_unchanged"] = True
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"state": "completed", "hardware_ready": False})
        return report
    except Exception as exc:
        write_json(output / "status.json", {"state": "failed", "error": str(exc)})
        raise


def plot(output, sim, real, initial, absolute, relative, arc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(1, 401) / 100
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        low, high = np.quantile(sim[:, :, i], [0.01, 0.99], axis=0)
        ax.fill_between(time, low, high, alpha=0.25, color="gray", label="Simulation train 1-99%")
        ax.plot(time, real[:, i], label="Real, original")
        ax.plot(time, real[:, i] - initial[i], label="Start-subtracted diagnostic only")
        ax.set(
            xlabel="Time (s)",
            ylabel=f"{['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'][i]} ({'N' if i < 3 else 'N m'})",
        )
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "force_domain.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, values, title in zip(
        axes,
        (absolute, relative, arc),
        ("Absolute SDK XY", "Initial-position relative XY", "Full-path arc normalization (oracle)"),
    ):
        for i, x in enumerate(values):
            ax.plot(x[:, 0] * 1000, x[:, 1] * 1000, label=f"Demo {i + 1}", linewidth=1)
            ax.scatter(x[0, 0] * 1000, x[0, 1] * 1000, s=12)
        ax.set(title=title, xlabel="X (mm)", ylabel="Y (mm)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7)
    fig.savefig(output / "xy_alignment.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/real_training/real_input_alignment_audit_v1")
    from scripts.shared.run_paths import new_output

    result = run(new_output(parser.parse_args().output))
    print(
        json.dumps(
            {"force": result["force"], "xy": result["xy"], "inputs_unchanged": True}, indent=2
        )
    )
