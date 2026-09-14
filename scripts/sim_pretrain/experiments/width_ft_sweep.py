"""Single-factor width comparison reusing the unchanged normal FT collector."""

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.friction_ft_sweep import (
    CHANNELS,
    COLORS,
    PANEL_CHANNELS,
    STYLES,
    CollectionWipe,
    decorate_axis,
    experiment_provenance,
    file_digest,
    plot_ft,
    plt,
    prepare_output,
    protected_hashes,
    run_sweep,
    write_json,
)

DEFAULT_WIDTHS = (0.02, 0.05, 0.1, 0.2, 0.3)


def plot_comparison(path, records, filtered=True, motion=False):
    fig, axes = plt.subplots(
        2 if motion else 3, 2, figsize=(13, 7 if motion else 9), layout="constrained"
    )
    mu, stiffness, _ = records[0]["parameters"]
    kind = (
        "Motion and contact (unfiltered)"
        if motion
        else ("Filtered local sensor FT" if filtered else "Raw local sensor FT")
    )
    fig.suptitle(
        f"Normal IK+FF, gain=300, mu={mu:g}, stiffness_direct={stiffness:g}\n"
        f"Width sweep: {kind}\n"
        "Press: 0-2 s  |  Forward: 2-3 s  |  Reverse: 3-4 s",
        fontsize=13,
    )
    if motion:
        labels = (
            "World Y displacement (mm)",
            "Summed contact normal force (N)",
            "Orientation error (deg)",
            "Z tracking error: actual - target (mm)",
        )
    else:
        labels = [f"{CHANNELS[c]} ({'N' if c < 3 else 'N m'})" for c in PANEL_CHANNELS]
    for i, data in enumerate(records):
        if motion:
            target = data["target_position"]
            values = (
                (data["position"][:, 1] - target[0, 1]) * 1000,
                data["normal_sum"],
                np.rad2deg(data["orientation_error"]),
                (data["position"][:, 2] - target[:, 2]) * 1000,
            )
        else:
            ft = data["ft_filtered" if filtered else "ft"]
            values = [ft[:, c] for c in PANEL_CHANNELS]
        for ax, value in zip(axes.flat, values):
            ax.plot(
                data["time"],
                value,
                color=COLORS[i % len(COLORS)],
                linestyle=STYLES[i % len(STYLES)],
                linewidth=1.5,
                label=f"width={data['parameters'][2]:g}",
            )
    if motion:
        first = records[0]
        target = first["target_position"]
        axes[0, 0].plot(
            first["time"],
            (target[:, 1] - target[0, 1]) * 1000,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Target",
        )
    for ax, label in zip(axes.flat, labels):
        decorate_axis(ax, label)
    axes[0, 0].legend(fontsize=9, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_width_sweep(
    config_path,
    out,
    widths=DEFAULT_WIDTHS,
    mu=0.9,
    stiffness=1000.0,
    factory=CollectionWipe,
    make_plots=True,
):
    values = np.asarray(widths, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Require distinct finite positive widths")
    names = [f"width_{width:g}" for width in values]
    if len(set(names)) != len(names):
        raise ValueError("Width output names collide; use more widely spaced widths")
    cfg, out = prepare_output(config_path, out, [mu], stiffness, values[0])
    before = protected_hashes(config_path, cfg)
    before[str(Path(__file__).resolve())] = file_digest(__file__)
    sources = experiment_provenance()
    report = {
        "schema_version": 1,
        "state": "running",
        "sweep_parameter": "width",
        "parameters": [[float(mu), float(stiffness), float(w)] for w in values],
        "parameter_names": ["friction", "stiffness_direct", "width"],
        "controller": "normal IK+FF, gain=300",
        "config": cfg,
        "channels": CHANNELS,
        "units": ["N"] * 3 + ["N m"] * 3,
        "frame": "ft_frame local",
        "filter": cfg["filter"],
        "notes": [
            "Isolated fresh simulation per width; not added to training data",
            "Width is solimp transition width, not sponge geometry",
            "Raw FT includes tool weight; no bias subtraction or normalization",
            "stiffness_direct is not calibrated N/m; failed motion is retained",
        ],
        "protected_hashes_before": before,
        "provenance": sources,
        "runs": [],
    }
    write_json(out / "summary.json", report)
    records = []
    for index, (width, name) in enumerate(zip(values, names)):
        row = {"index": index, "parameters": report["parameters"][index], "state": "running"}
        report["runs"].append(row)
        write_json(out / "summary.json", report)
        try:
            result = run_sweep(
                config_path,
                out / name,
                [mu],
                stiffness,
                float(width),
                factory=factory,
                make_plots=False,
            )
            row.update(result["runs"][0], index=index)
            if result["state"] != "completed":
                raise RuntimeError(
                    result.get("error", row.get("error", "Condition did not complete"))
                )
            with np.load(out / name / "sweep.npz", allow_pickle=False) as arrays:
                data = {key: arrays[key][0].copy() for key in arrays.files}
            if records:
                for key in ("time", "target_position"):
                    np.testing.assert_array_equal(data[key], records[0][key])
            if make_plots:
                plot_ft(out / name / "ft.png", [data], cfg, True, single=True)
            records.append(data)
            row["state"] = "completed"
        except Exception as exc:
            row.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "summary.json", report)
    if all(row["state"] == "completed" for row in report["runs"]):
        np.savez_compressed(
            out / "sweep.npz", **{key: np.stack([r[key] for r in records]) for key in records[0]}
        )
        if make_plots:
            plot_comparison(out / "ft_filtered_comparison.png", records)
            plot_comparison(out / "ft_raw_comparison.png", records, filtered=False)
            plot_comparison(out / "motion_contact.png", records, motion=True)
        report["state"] = "completed"
    else:
        report["state"] = "failed"
    after = {path: file_digest(path) for path in before}
    report["protected_hashes_after"] = after
    report["sources_and_inputs_unchanged"] = before == after and sources == experiment_provenance()
    if not report["sources_and_inputs_unchanged"]:
        report.update(state="failed", error="Protected inputs or sources changed")
    write_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "archive/sim_pretrain/two_control_pretraining_v2/normal/config.json"),
    )
    parser.add_argument("--width", type=float, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--stiffness", type=float, default=1000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_width_sweep(args.dataset_config, args.output, args.width, args.mu, args.stiffness)
    print(f"Width sweep {result['state']}: {Path(args.output).resolve()}", flush=True)
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
