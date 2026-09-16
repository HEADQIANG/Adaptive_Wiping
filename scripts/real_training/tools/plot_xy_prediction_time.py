"""Plot saved offline XY predictions against the 25 policy sample times."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_predictions(source, output):
    keys = ("xy_target", "xy_prediction", "xy_baseline")
    with np.load(source, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in keys}
    shape = arrays["xy_target"].shape
    if len(shape) != 3 or shape[0] < 1 or shape[1:] != (25, 2):
        raise ValueError("Expected XY arrays with shape (episodes, 25, 2)")
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("XY arrays must have matching shapes and finite values")
    times = np.arange(1, 26, dtype=np.float64) * 0.4
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for component, axis_name in enumerate(("X", "Y")):
        fig, axes = plt.subplots(
            (shape[0] + 1) // 2, 2,
            figsize=(12, 2.7 * ((shape[0] + 1) // 2)),
            squeeze=False, sharex=True, sharey=True, constrained_layout=True,
        )
        for episode, ax in enumerate(axes.flat):
            if episode >= shape[0]:
                ax.set_visible(False)
                continue
            for key, label, color, style in (
                ("xy_target", "Demonstration", "#1f77b4", "-"),
                ("xy_prediction", "Prediction", "#ff7f0e", "-"),
                ("xy_baseline", "Train mean", "#2ca02c", "--"),
            ):
                ax.plot(
                    times, arrays[key][episode, :, component] * 1000,
                    label=label, color=color, linestyle=style, linewidth=1.8,
                )
            ax.set(
                title=f"Episode {episode + 1}", xlabel="Time (s)",
                ylabel=f"Base {axis_name} (mm)", xlim=(0, 10),
                xticks=np.arange(0, 11, 2),
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(
            f"{axis_name} trajectory vs time: offline training-set prediction\n"
            "25 samples, 0.4-10 s; held-last padding included; not closed-loop execution",
            fontsize=12,
        )
        path = output / f"predictions_{axis_name.lower()}_time.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output_dir = new_output(args.output_dir)
    for path in plot_predictions(args.predictions, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
