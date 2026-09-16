"""Paper-inspired signed mean-reference ratios; six-axis/time plots are extensions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.real_deploy.plot_comparison import load_demonstrations, load_run
from scripts.shared.common import file_digest


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def reference_statistics(demos):
    means, mean_squares = [], []
    for demo in demos:
        end = demo["duration"]
        # All bound programmed demonstrations contain exactly four seconds of wiping.
        mask = (demo["time"] >= end - 4 - 1e-9) & (demo["time"] < end - 1e-9)
        values = demo["ft"][mask]
        if values.shape != (400, 6) or not np.isfinite(values).all():
            raise ValueError("Expected 400 measured wiping samples per demonstration")
        means.append(values.mean(axis=0))
        mean_squares.append(np.mean(values ** 2, axis=0))
    mean = np.mean(means, axis=0)
    rms = np.sqrt(np.mean(mean_squares, axis=0))
    cancellation = np.divide(np.abs(mean), rms, out=np.zeros_like(mean), where=rms > 0)
    return mean, rms, cancellation, np.array(means)


def signed_ratio(values, reference):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(reference):
        raise ValueError("Ratio requires finite inputs")
    if abs(reference) <= 1e-12:
        return np.full_like(values, np.nan)
    return 100 * values / reference


def plot_ratio(run, reference, cancellation, channels, path):
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 3.1 * len(channels) + 1.5),
                             squeeze=False)
    for ax, channel in zip(axes.flat, channels):
        name = CHANNELS[channel]
        values = signed_ratio(run["tared_ft"][:, channel], reference[channel])
        ax.plot(run["time"], values, color="#c73535", lw=1.6, label="Deployment / demo mean")
        ax.axhline(100, color="#222222", ls=":", lw=1.5, label="Reference = 100%")
        ax.axhline(0, color="#999999", lw=0.7)
        ax.axvspan(0, 2, color="#eeeeee", zorder=-2)
        ax.axvspan(6.4, 10, color="#eeeeee", zorder=-2)
        ax.set(xlim=(0, 10), xticks=np.arange(0, 11), xlabel="Time from motion start (s)",
               ylabel=f"{name} / mean demo {name} (%)")
        unit = "N" if channel < 3 else "N m"
        title = f"{name}: signed reference = {reference[channel]:.6g} {unit}"
        if cancellation[channel] < 0.1:
            title += "  |  CAUTION: opposite signs nearly cancel in reference mean"
        ax.set_title(title, fontsize=11)
        if not np.isfinite(values).any():
            ax.text(0.5, 0.5, "Undefined: reference mean is zero", transform=ax.transAxes,
                    ha="center", color="#a33a00")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Applied wrench relative to demonstration mean: ratio (%)", fontsize=16, y=0.99)
    fig.text(0.5, 0.89 if len(channels) == 1 else 0.945,
             "100 * measured(t) / signed reference mean; not pointwise division or relative error",
             ha="center", fontsize=10)
    fig.text(0.5, 0.018,
             "Reference: equal-weight mean of 8 demos, each measured 4 s wiping segment; no padded samples.\n"
             "Tared sensor-local values. Gray: startup / final hold. Time traces and torque ratios extend the paper's Fz summary metric.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.085 if len(channels) == 1 else 0.06,
                           0.99, 0.82 if len(channels) == 1 else 0.89))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    run, session = load_run(args.events)
    demos = load_demonstrations(args.raw_data, session, args.evaluation)
    reference, rms, cancellation, means = reference_statistics(demos)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, channels in (("fz_ratio", (2,)), ("force_ratio", (0, 1, 2)),
                           ("torque_ratio", (3, 4, 5))):
        path = args.output / f"{name}.png"
        plot_ratio(run, reference, cancellation, channels, path)
        print(path)
    summaries = {}
    for name, start, end in (("full_0_10s", 0, 10), ("moving_2_6p4s", 2, 6.4),
                             ("feedback_2_10s", 2, 10)):
        mask = (run["time"] >= start - 1e-9) & (run["time"] < end - 1e-9)
        mean = run["tared_ft"][mask].mean(axis=0)
        summaries[name] = {channel: {"mean_si": float(mean[i]),
                                   "ratio_percent": float(signed_ratio(mean[i], reference[i]))
                                   if abs(reference[i]) > 1e-12 else None}
                           for i, channel in enumerate(CHANNELS)}
    report = {
        "paper_source": "Adaptive_wiping.pdf, page 6 Fig. 6 / Table II; page 5 Table I",
        "paper_metric": "mean deployed Fz / mean demonstration Fz * 100",
        "extension": "time-dependent values and other wrench channels; not paper-reported curves",
        "reference_scope": "eight bound same-sponge programmed demos; four measured wiping seconds each",
        "reference_mean_si": dict(zip(CHANNELS, reference.tolist())),
        "reference_rms_si_diagnostic_only": dict(zip(CHANNELS, rms.tolist())),
        "absolute_mean_over_rms": dict(zip(CHANNELS, cancellation.tolist())),
        "cancellation_warning_threshold": 0.1,
        "warning_threshold_note": "heuristic diagnostic, not a paper criterion; ratios are not clipped",
        "per_demo_signed_mean_si": {d["name"]: m.tolist() for d, m in zip(demos, means)},
        "deployment_summaries_half_open_windows": summaries,
        "inputs": {str(p): file_digest(p) for p in (args.events, args.raw_data, args.evaluation)},
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
