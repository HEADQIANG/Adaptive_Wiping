"""Summarize complete paired raw simulation datasets without fitting a network."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import CHANNELS, file_digest, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    MODES,
    POLICY,
    verify,
)


def split_statistics(group):
    ft = group["ft"][:]
    if ft.ndim != 3 or ft.shape[1:] != (400, 6) or not np.isfinite(ft).all():
        raise ValueError("Invalid raw FT dataset")
    if not np.all(group["valid"][:]) or not np.all(group["unloaded"][:]):
        raise ValueError("Incomplete or non-unloaded split")
    acceptance = [json.loads(row) for row in group["acceptance_json"].asstr()[:]]
    metrics = {
        key: np.asarray([row["metrics"][key] for row in acceptance])
        for key in acceptance[0]["metrics"]
    }
    failures = {
        key: sum(key in row["failed_checks"] for row in acceptance)
        for key in acceptance[0]["checks"]
    }
    normal = group["normal_sum"][:, 200:]
    result = {
        "count": len(ft),
        "raw_ft_shape": list(ft.shape),
        "motion_pass_count": int(group["motion_passed"][:].sum()),
        "unloaded_count": int(group["unloaded"][:].sum()),
        "attempt_count": int(group["attempts"][:].sum()),
        "failed_motion_check_counts": failures,
        "raw_channel_min": ft.min(axis=(0, 1)).tolist(),
        "raw_channel_max": ft.max(axis=(0, 1)).tolist(),
        "raw_channel_std": ft.std(axis=(0, 1)).tolist(),
        "slide_normal_mean_n": float(normal.mean()),
        "slide_zero_load_sample_fraction": float(np.mean(normal <= 1e-8)),
        "motion_metrics": {
            key: {
                "min": float(value.min()),
                "median": float(np.median(value)),
                "max": float(value.max()),
                "mean": float(value.mean()),
            }
            for key, value in metrics.items()
        },
    }
    return result


def summarize(out):
    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    configs = {name: json.loads((out / name / "config.json").read_text()) for name in MODES}
    verification = verify(out, manifest, configs)
    if not verification["complete"]:
        raise ValueError("Wait for both collections to finish before summarizing")
    report = {
        "scope": "Raw simulation collection only; no network training or calibrated material estimator",
        "policy": POLICY,
        "channels": CHANNELS,
        "frame": "ft_frame local",
        "units": ["N"] * 3 + ["N*m"] * 3,
        "modes": {},
        "summary_source_sha256": file_digest(__file__),
        "paired_target_max_difference_m": {},
    }
    bands, examples = {}, {}
    assignments = []
    for name in MODES:
        path = out / name / "dataset.h5"
        with h5py.File(path, "r") as h5:
            report["modes"][name] = {
                "dataset_sha256": file_digest(path),
                "splits": {
                    split: split_statistics(h5[split]) for split in configs[name]["dataset"]
                },
            }
            ft = h5["train/ft"][:]
            bands[name] = np.quantile(ft, [0.1, 0.5, 0.9], axis=0)
            examples[name] = {
                key: h5[f"train/{key}"][0]
                for key in ("ft", "position", "target_position", "normal_sum", "time", "parameters")
            }
            assignments.append(
                {split: h5[f"{split}/parameters"][:] for split in configs[name]["dataset"]}
            )
    for split in configs["normal"]["dataset"]:
        np.testing.assert_array_equal(assignments[0][split], assignments[1][split])
    all_assignments = np.concatenate(list(assignments[0].values()))
    report["unique_parameter_assignments"] = len(np.unique(all_assignments, axis=0))
    report["total_parameter_assignments_per_mode"] = len(all_assignments)
    assert report["unique_parameter_assignments"] == len(all_assignments)
    with (
        h5py.File(out / "normal/dataset.h5", "r") as normal,
        h5py.File(out / "impedance/dataset.h5", "r") as impedance,
    ):
        for split in configs["normal"]["dataset"]:
            a, b = normal[f"{split}/target_position"][:], impedance[f"{split}/target_position"][:]
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            report["paired_target_max_difference_m"][split] = float(np.abs(a - b).max())
            np.testing.assert_array_equal(normal[f"{split}/time"][:], impedance[f"{split}/time"][:])
    from scripts.shared.run_paths import new_output

    destination = new_output(out / "data_summary.json").parent
    plot(destination, bands, examples)
    write_json(destination / "data_summary.json", report)
    print(
        json.dumps(
            {
                "modes": {
                    name: {split: row["count"] for split, row in result["splits"].items()}
                    for name, result in report["modes"].items()
                },
                "unique_assignments": report["unique_parameter_assignments"],
            },
            indent=2,
        )
    )
    return report


def plot(out, bands, examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"normal": "#3478b2", "impedance": "#c44e52"}
    times = np.arange(1, 401) / 100
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        for name, quantiles in bands.items():
            ax.fill_between(
                times,
                quantiles[0, :, index],
                quantiles[2, :, index],
                color=colors[name],
                alpha=0.16,
            )
            ax.plot(times, quantiles[1, :, index], color=colors[name], label=name)
        ax.set_ylabel(f"{CHANNELS[index]} ({'N' if index < 3 else 'N m'})")
        ax.set_xlabel("Time (s)")
        ax.axvline(2, color="gray", linewidth=0.5)
        ax.axvline(3, color="gray", linewidth=0.5)
        ax.grid(alpha=0.15)
    axes.flat[0].legend()
    fig.suptitle("Raw local FT: training median and 10-90% interval")
    fig.savefig(out / "raw_ft_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, data in examples.items():
        axes[0].plot(times, 1000 * data["position"][:, 1], label=name, color=colors[name])
        axes[1].plot(times, data["normal_sum"], color=colors[name])
        axes[2].plot(times, data["ft"][:, 2], color=colors[name])
    first = examples["normal"]
    axes[0].plot(times, 1000 * first["target_position"][:, 1], "k--", label="target")
    axes[0].legend()
    for ax, label in zip(axes, ["World Y (mm)", "Contact normal sum (N)", "Local sensor Fz (N)"]):
        ax.set_ylabel(label)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Time (s)")
    mu, stiffness, width = first["parameters"]
    fig.suptitle(f"Paired training sample 0: mu={mu:.3f}, k={stiffness:.3f}, width={width:.3f}")
    fig.savefig(out / "paired_sample.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    summarize(parser.parse_args().output)
