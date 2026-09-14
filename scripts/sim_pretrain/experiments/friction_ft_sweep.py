"""Isolated normal-mode friction sweep; no writes to training data or models."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.common import (
    CHANNELS,
    file_digest,
    load_config,
    output_dir,
    write_json,
)
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import validate_trajectory
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    EXTRA_FIELDS,
    CollectionWipe,
    experiment_provenance,
)

DEFAULT_MUS = (0.0, 0.5, 0.9, 2.5, 3.5)
PHASES = {"press": (0, 200), "forward": (200, 300), "reverse": (300, 400)}
PANEL_CHANNELS = (0, 3, 1, 4, 2, 5)
COLORS = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00")
STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))


def prepare_output(config_path, out, mus, stiffness, width):
    cfg = load_config(config_path)
    mode = cfg.get("research_comparison", {})
    if mode.get("controller") != "normal" or mode.get("normal_gain") != 300:
        raise ValueError("Require the frozen normal-mode configuration with gain=300")
    values = np.asarray([*mus, stiffness, width], dtype=float)
    if (
        not len(mus)
        or len(set(mus)) != len(mus)
        or not np.isfinite(values).all()
        or min(mus) < 0
        or stiffness <= 0
        or width <= 0
    ):
        raise ValueError("Require distinct finite mu >= 0 and positive stiffness/width")
    out = Path(out).resolve()
    if out.is_relative_to(output_dir(cfg).resolve()) or out.is_relative_to(
        Path(config_path).resolve().parent
    ):
        raise ValueError("Output must be outside the frozen dataset directory")
    out.mkdir(parents=True, exist_ok=False)
    return cfg, out


def protected_hashes(config_path, cfg):
    paths = [Path(config_path).resolve(), Path(__file__).resolve()]
    paths += [
        output_dir(cfg) / name
        for name in (
            "dataset.h5",
            "dataset_integrity.json",
            "preprocessing.json",
            "vae_best.pt",
            "vae_last.pt",
            "encoder.pt",
        )
    ]
    return {str(path): file_digest(path) for path in paths if path.exists()}


def summarize_case(data):
    validate_trajectory(data)
    for key, shape in EXTRA_FIELDS.items():
        if np.shape(data[key]) != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid extra field: {key}")
    ft = data["ft"]
    phases = {}
    for name, (start, stop) in PHASES.items():
        part = ft[start:stop]
        phases[name] = {
            "sample_count": stop - start,
            "raw_mean": part.mean(axis=0).tolist(),
            "raw_rms": np.sqrt(np.mean(part**2, axis=0)).tolist(),
            "raw_abs_peak": np.max(np.abs(part), axis=0).tolist(),
            "normal_mean_n": float(data["normal_sum"][start:stop].mean()),
        }
    return {
        "acceptance": contact_motion_acceptance(data),
        "phases": phases,
        "raw_abs_peak": np.max(np.abs(ft), axis=0).tolist(),
        "force_norm_peak_n": float(np.linalg.norm(ft[:, :3], axis=1).max()),
        "torque_norm_peak_nm": float(np.linalg.norm(ft[:, 3:], axis=1).max()),
    }


def decorate_axis(ax, ylabel):
    ax.set(xlim=(0, 4), xlabel="Time (s)", ylabel=ylabel)
    ax.axvline(2, color="0.5", linewidth=0.8, linestyle=":")
    ax.axvline(3, color="0.5", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.18)


def plot_ft(path, records, cfg, filtered, single=False):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
    p = records[0]["parameters"]
    mode = "Raw + filtered" if single else ("Filtered" if filtered else "Raw")
    extra = f", mu={p[0]:g}" if single else ""
    fig.suptitle(
        f"Normal IK+FF, gain=300, stiffness_direct={p[1]:g}, width={p[2]:g}{extra}\n"
        f"{mode} local sensor FT (includes tool weight)\n"
        "Press: 0-2 s  |  Forward: 2-3 s  |  Reverse: 3-4 s",
        fontsize=13,
    )
    for ax, channel in zip(axes.flat, PANEL_CHANNELS):
        for index, data in enumerate(records):
            if single:
                ax.plot(
                    data["time"],
                    data["ft"][:, channel],
                    color="0.65",
                    linewidth=1,
                    label="Raw (100 Hz)",
                )
            key = "ft_filtered" if filtered else "ft"
            label = f"mu={data['parameters'][0]:g}"
            if single:
                label = f"Filtered ({cfg['filter']['cutoff_hz']:g} Hz)"
            ax.plot(
                data["time"],
                data[key][:, channel],
                color=COLORS[index % len(COLORS)],
                linestyle=STYLES[index % len(STYLES)],
                linewidth=1.5,
                label=label,
            )
        unit = "N" if channel < 3 else "N m"
        decorate_axis(ax, f"{CHANNELS[channel]} ({unit})")
    axes[0, 0].legend(fontsize=9, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_motion(path, records):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    fig.suptitle("Normal-mode friction sweep: motion and contact (unfiltered)", fontsize=13)
    for index, data in enumerate(records):
        target = data["target_position"]
        origin = target[0, 1]
        values = (
            (data["position"][:, 1] - origin) * 1000,
            data["normal_sum"],
            np.rad2deg(data["orientation_error"]),
            (data["position"][:, 2] - target[:, 2]) * 1000,
        )
        for ax, value in zip(axes.flat, values):
            ax.plot(
                data["time"],
                value,
                color=COLORS[index % len(COLORS)],
                linestyle=STYLES[index % len(STYLES)],
                label=f"mu={data['parameters'][0]:g}",
            )
    first = records[0]
    axes[0, 0].plot(
        first["time"],
        (first["target_position"][:, 1] - first["target_position"][0, 1]) * 1000,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Target",
    )
    for ax, label in zip(
        axes.flat,
        (
            "World Y displacement (mm)",
            "Summed contact normal force (N)",
            "Orientation error (deg)",
            "Z tracking error: actual - target (mm)",
        ),
    ):
        decorate_axis(ax, label)
    axes[0, 0].legend(fontsize=9, ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_sweep(
    config_path,
    out,
    mus=DEFAULT_MUS,
    stiffness=1000.0,
    width=0.02,
    factory=CollectionWipe,
    make_plots=True,
):
    cfg, out = prepare_output(config_path, out, mus, stiffness, width)
    before = protected_hashes(config_path, cfg)
    sources = experiment_provenance()
    report = {
        "schema_version": 1,
        "state": "running",
        "scope": "Isolated simulation, not training data",
        "config": cfg,
        "controller": "normal IK+FF, gain=300",
        "seed": cfg["simulation"]["seed"],
        "parameters": [[float(mu), float(stiffness), float(width)] for mu in mus],
        "parameter_names": ["friction", "stiffness_direct", "width"],
        "channels": CHANNELS,
        "units": ["N"] * 3 + ["N m"] * 3,
        "frame": "ft_frame local",
        "notes": [
            "Raw sensor FT includes tool weight; not bias-subtracted pure contact force",
            "contact_wrench is world-frame contact wrench ON tool, moment about TCP",
            "mu=0 is clamped to 1e-5 by MuJoCo for sliding; torsion=.005, rolling=.0001 remain fixed",
            "stiffness_direct is a MuJoCo parameter, not calibrated N/m",
            "One independent reset per mu; no averaging, normalization or training-set insertion",
            "Motion acceptance is recorded, not a rejection gate",
        ],
        "filter": cfg["filter"],
        "phase_slices": PHASES,
        "protected_hashes_before": before,
        "provenance": sources,
        "runs": [],
    }
    write_json(out / "summary.json", report)
    prep = Preprocessor(**cfg["filter"], sample_hz=cfg["simulation"]["sample_hz"])
    records = []
    for index, parameters in enumerate(report["parameters"]):
        row = {"index": index, "parameters": parameters, "state": "running"}
        report["runs"].append(row)
        write_json(out / "summary.json", report)
        folder = out / f"mu_{parameters[0]:g}"
        folder.mkdir()
        env = None
        try:
            print(
                f"[{index + 1}/{len(mus)}] normal, mu={parameters[0]:g}, k={stiffness:g}, width={width:g}",
                flush=True,
            )
            env = factory(cfg, tuple(parameters), None)
            data = env.rollout()
            np.testing.assert_array_equal(data["parameters"], parameters)
            row.update(summarize_case(data))
            if records:
                np.testing.assert_array_equal(data["time"], records[0]["time"])
                np.testing.assert_array_equal(
                    data["target_position"], records[0]["target_position"]
                )
            data["ft_filtered"] = prep.filtered(data["ft"][None])[0]
            np.savez_compressed(folder / "trajectory.npz", **data)
            unloaded = np.asarray(env.unload(data))
            if unloaded.shape != (6,) or not np.isfinite(unloaded).all():
                raise ValueError("Invalid unloaded wrench")
            row.update(
                state="completed",
                unloaded=True,
                unloaded_wrench=unloaded.tolist(),
                trajectory_sha256=file_digest(folder / "trajectory.npz"),
            )
            records.append(data)
            if make_plots:
                plot_ft(folder / "ft.png", [data], cfg, True, single=True)
            metrics = row["acceptance"]["metrics"]
            print(
                f"  forward/reverse: {metrics['forward_travel_m'] * 1000:.3f}/"
                f"{metrics['reverse_travel_m'] * 1000:.3f} mm; "
                f"motion_passed={row['acceptance']['passed']}",
                flush=True,
            )
        except Exception as exc:
            row.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            print(row["error"], flush=True)
        finally:
            if env is not None:
                env.close()
            write_json(folder / "summary.json", row)
            write_json(out / "summary.json", report)
    if all(row["state"] == "completed" for row in report["runs"]):
        arrays = {key: np.stack([data[key] for data in records]) for key in records[0]}
        np.savez_compressed(out / "sweep.npz", **arrays)
        if make_plots:
            plot_ft(out / "ft_raw_comparison.png", records, cfg, False)
            plot_ft(out / "ft_filtered_comparison.png", records, cfg, True)
            plot_motion(out / "motion_contact.png", records)
        report["state"] = "completed"
    else:
        report["state"] = "failed"
    report["protected_hashes_after"] = protected_hashes(config_path, cfg)
    report["sources_and_inputs_unchanged"] = (
        before == report["protected_hashes_after"] and sources == experiment_provenance()
    )
    if not report["sources_and_inputs_unchanged"]:
        report.update(state="failed", error="Protected inputs or source changed")
    write_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mu", nargs="+", type=float, default=list(DEFAULT_MUS))
    parser.add_argument("--stiffness", type=float, default=1000.0)
    parser.add_argument("--width", type=float, default=0.02)
    args = parser.parse_args()
    result = run_sweep(args.dataset_config, args.output, args.mu, args.stiffness, args.width)
    print(f"Sweep {result['state']}: {Path(args.output).resolve()}", flush=True)
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
