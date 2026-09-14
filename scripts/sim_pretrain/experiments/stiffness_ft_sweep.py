"""Normal-mode stiffness sweep using the existing isolated FT collection runner."""

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.shared.common import CHANNELS, file_digest, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.friction_ft_sweep import (
    COLORS,
    PANEL_CHANNELS,
    STYLES,
    CollectionWipe,
    decorate_axis,
    experiment_provenance,
    plot_ft,
    plt,
    prepare_output,
    protected_hashes,
    run_sweep,
)
from scripts.sim_pretrain.simulation import contact_parameters

DEFAULT_STIFFNESSES = (0.5, 10.0, 100.0, 250.0, 500.0, 1000.0)
SERIES_COLORS = (*COLORS, "#555555")
SERIES_STYLES = (*STYLES, (0, (3, 1, 1, 1, 1, 1)))


def validate_stiffnesses(stiffnesses):
    values = np.asarray(stiffnesses, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Require distinct finite positive stiffness values")
    names = [f"k_{value:g}" for value in values]
    if len(set(names)) != len(names):
        raise ValueError("Stiffness values produce duplicate output names")
    return values


def plot_comparison(path, records, filtered=True, motion=False):
    shape, size = ((2, 2), (13, 7)) if motion else ((3, 2), (13, 9))
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained")
    mu, _, width = records[0]["parameters"]
    description = (
        "Motion and contact (unfiltered)"
        if motion
        else (("Filtered" if filtered else "Raw") + " local sensor FT (includes tool weight)")
    )
    fig.suptitle(
        f"Normal IK+FF, gain=300, mu={mu:g}, width={width:g}\n"
        f"{description} | k = stiffness_direct\n"
        "Press: 0-2 s  |  Forward: 2-3 s  |  Reverse: 3-4 s",
        fontsize=13,
    )
    for index, data in enumerate(records):
        if motion:
            target = data["target_position"]
            values = (
                (data["position"][:, 1] - target[0, 1]) * 1000,
                data["normal_sum"],
                np.rad2deg(data["orientation_error"]),
                (data["position"][:, 2] - target[:, 2]) * 1000,
            )
            labels = (
                "World Y displacement (mm)",
                "Summed contact normal force (N)",
                "Orientation error (deg)",
                "Z tracking error: actual - target (mm)",
            )
        else:
            ft = data["ft_filtered" if filtered else "ft"]
            values = [ft[:, channel] for channel in PANEL_CHANNELS]
            labels = [
                f"{CHANNELS[channel]} ({'N' if channel < 3 else 'N m'})"
                for channel in PANEL_CHANNELS
            ]
        for ax, value in zip(axes.flat, values):
            ax.plot(
                data["time"],
                value,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                linestyle=SERIES_STYLES[index % len(SERIES_STYLES)],
                linewidth=1.5,
                label=f"k={data['parameters'][1]:g}",
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
    axes[0, 0].legend(fontsize=9, ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_hashes(config_path, cfg):
    return {
        **protected_hashes(config_path, cfg),
        str(Path(__file__).resolve()): file_digest(__file__),
    }


def run_stiffness_sweep(
    config_path,
    out,
    stiffnesses=DEFAULT_STIFFNESSES,
    mu=0.9,
    width=0.02,
    factory=CollectionWipe,
    make_plots=True,
):
    stiffnesses = validate_stiffnesses(stiffnesses)
    cfg, out = prepare_output(config_path, out, [mu], float(stiffnesses[0]), width)
    before = sweep_hashes(config_path, cfg)
    sources = experiment_provenance()
    report = {
        "schema_version": 1,
        "state": "running",
        "scope": "Isolated simulation, not training data",
        "sweep_parameter": "stiffness_direct",
        "controller": "normal IK+FF, gain=300",
        "config": cfg,
        "parameter_names": ["friction", "stiffness_direct", "width"],
        "parameters": [[float(mu), float(k), float(width)] for k in stiffnesses],
        "channels": CHANNELS,
        "units": ["N"] * 3 + ["N m"] * 3,
        "frame": "ft_frame local",
        "filter": cfg["filter"],
        "phase_slices": {"press": [0, 200], "forward": [200, 300], "reverse": [300, 400]},
        "notes": [
            "k is the simulation stiffness_direct parameter, not calibrated physical N/m",
            "Existing mapping solref=[-k,-2*sqrt(k)]: the damping term changes with k",
            "Raw local FT includes tool weight and dynamics; no bias subtraction or normalization",
            "contact_wrench is world-frame contact wrench ON tool, moment about TCP",
            "width is the solimp parameter, not geometric sponge width",
            "Each k uses a fresh environment and the original exploration and contact checks",
            "Motion acceptance failures are retained; completion is not successful wiping",
        ],
        "protected_hashes_before": before,
        "provenance": sources,
        "runs": [],
    }
    write_json(out / "summary.json", report)
    records = []
    try:
        for index, parameters in enumerate(report["parameters"]):
            k = parameters[1]
            folder = out / f"k_{k:g}"
            row = {
                "index": index,
                "parameters": parameters,
                "state": "running",
                "contact_solref": contact_parameters(*parameters)[1].tolist(),
            }
            report["runs"].append(row)
            write_json(out / "summary.json", report)
            print(
                f"Stiffness [{index + 1}/{len(stiffnesses)}]: k={k:g}, mu={mu:g}, width={width:g}",
                flush=True,
            )
            try:
                # A one-mu run reuses the verified recorder without altering the previous sweep.
                result = run_sweep(
                    config_path,
                    folder,
                    mus=(mu,),
                    stiffness=k,
                    width=width,
                    factory=factory,
                    make_plots=False,
                )
                row.update(
                    {key: value for key, value in result["runs"][0].items() if key != "index"}
                )
                if result["state"] != "completed":
                    raise RuntimeError(
                        result.get("error", row.get("error", "Condition did not complete"))
                    )
                path = folder / f"mu_{mu:g}" / "trajectory.npz"
                with np.load(path, allow_pickle=False) as saved:
                    data = {key: saved[key].copy() for key in saved.files}
                np.testing.assert_array_equal(data["parameters"], parameters)
                if records:
                    np.testing.assert_array_equal(data["time"], records[0]["time"])
                    np.testing.assert_array_equal(
                        data["target_position"], records[0]["target_position"]
                    )
                if make_plots:
                    plot_ft(folder / "ft.png", [data], cfg, True, single=True)
                records.append(data)
                row.update(state="completed", trajectory_path=str(path.relative_to(out)))
            except Exception as exc:
                row.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                print(row["error"], flush=True)
            write_json(out / "summary.json", report)
        if all(row["state"] == "completed" for row in report["runs"]):
            np.savez_compressed(
                out / "sweep.npz",
                **{key: np.stack([data[key] for data in records]) for key in records[0]},
            )
            if make_plots:
                plot_comparison(out / "ft_raw_comparison.png", records, filtered=False)
                plot_comparison(out / "ft_filtered_comparison.png", records)
                plot_comparison(out / "motion_contact.png", records, motion=True)
            report["state"] = "completed"
            report["sweep_sha256"] = file_digest(out / "sweep.npz")
        else:
            report["state"] = "failed"
    except Exception as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["protected_hashes_after"] = sweep_hashes(config_path, cfg)
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
    parser.add_argument("--stiffness", nargs="+", type=float, default=list(DEFAULT_STIFFNESSES))
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--width", type=float, default=0.02)
    args = parser.parse_args()
    result = run_stiffness_sweep(
        args.dataset_config, args.output, args.stiffness, args.mu, args.width
    )
    print(f"Stiffness sweep {result['state']}: {Path(args.output).resolve()}", flush=True)
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
