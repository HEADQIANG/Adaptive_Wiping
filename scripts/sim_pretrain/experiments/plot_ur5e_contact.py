"""Compare saved UR5e and AIRBOT trajectories without running physics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ur", required=True)
    parser.add_argument("--matched", required=True)
    parser.add_argument(
        "--airbot", default="archive/sim_pretrain/pretrain_paper_tabletop_compact_v2"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    ur = json.loads((Path(args.ur) / "report.json").read_text())
    matched = json.loads((Path(args.matched) / "report.json").read_text())
    air = json.loads((Path(args.airbot) / "sanity.json").read_text())
    cases = [(0, 1000, 0.02), (3.5, 1000, 0.02), (3.5, 0.5, 0.02)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for col, parameters in enumerate(cases):
        paths = []
        for index, row in enumerate(air["grid"]):
            if np.allclose(row["parameters"], parameters):
                paths.append(
                    (
                        "AIRBOT IK+FF 300",
                        Path(args.airbot) / "sanity_traces" / f"contact_{index:02d}.npz",
                    )
                )
        for mode, variant in ur["variants"].items():
            for index, row in enumerate(variant["contacts"]):
                if np.allclose(row["parameters"], parameters):
                    paths.append(
                        (
                            f"UR5e {mode} {row['gain']}",
                            Path(args.ur) / mode / f"contact_{index:02d}.npz",
                        )
                    )
        for row in matched["cases"]:
            if row["name"] != "free" and np.allclose(row["parameters"], parameters):
                paths.append(("UR5e IK+FF 300", Path(args.matched) / (row["name"] + ".npz")))
        for label, path in paths:
            with np.load(path) as data:
                t = data["time"]
                axes[0, col].plot(
                    t,
                    1000 * (data["position"][:, 1] - data["target_position"][199, 1]),
                    label=label,
                )
                axes[1, col].plot(t, np.rad2deg(data["orientation_error"]))
                axes[2, col].plot(
                    t, 1000 * (data["position"][:, 0] - data["target_position"][:, 0])
                )
        axes[0, col].plot([0, 2, 3, 4], [0, 0, 50, 0], "k--", label="target")
        axes[0, col].set_title(f"mu={parameters[0]}, k={parameters[1]}, width=.02")
        axes[1, col].axhline(2, color="black", linestyle="--")
        for limit in (-3, 3):
            axes[2, col].axhline(limit, color="black", linestyle="--")
        axes[2, col].set_xlabel("Time (s)")
    for row, label in enumerate(("Y displacement (mm)", "Orientation error (deg)", "X error (mm)")):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
