"""Plot logged post-send sensor samples without connecting to hardware."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.log.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    samples = [
        row for row in rows if row.get("event") == "sample" and row.get("phase") == "exploration"
    ]
    if not samples:
        raise ValueError("No exploration samples in log")
    time = np.array([row["protocol_time_s"] for row in samples], dtype=float)
    raw = np.array([row["force"]["raw_sensor_wrench_si"] for row in samples])
    corrected = np.array([row["force"]["bias_corrected_sensor_wrench_si"] for row in samples])
    if (
        raw.shape != (len(time), 6)
        or corrected.shape != raw.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(corrected).all()
        or not np.isfinite(time).all()
        or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Invalid sensor samples or protocol times")
    same = np.array_equal(raw, corrected)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    channels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for channel, ax in zip([0, 3, 1, 4, 2, 5], axes.flat):
        for start, end, color in [(0, 2, "#eef3f8"), (2, 3, "#eef7ed"), (3, 4, "#fcf1e9")]:
            ax.axvspan(start, end, color=color, zorder=0)
        ax.plot(time, raw[:, channel], color="#176b99", linewidth=1.25, label="Raw sensor")
        if not same:
            ax.plot(
                time, corrected[:, channel], color="#bd4934", linewidth=1, label="Bias corrected"
            )
        for boundary in (2, 3):
            ax.axvline(boundary, color="#747474", linestyle="--", linewidth=0.8)
        unit = "N" if channel < 3 else "N m"
        ax.set_ylabel(f"{channels[channel]} ({unit})")
        ax.set_title(
            f"{channels[channel]}  |  min {raw[:, channel].min():.3f}, max {raw[:, channel].max():.3f}",
            fontsize=11,
            loc="left",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(0, 4)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[0]:
        for center, label in [(1, "Press"), (2.5, "+Y slide"), (3.5, "-Y slide")]:
            ax.text(
                center,
                0.96,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#444444",
            )
    for ax in axes[-1]:
        ax.set_xlabel("Protocol time (s)")
    axes[1, 1].legend(loc="best", fontsize=9)
    fig.suptitle(f"{args.log.stem}: real exploration force / torque", fontsize=17, y=0.98)
    note = (
        "Raw = bias corrected (configured bias is zero)."
        if same
        else "Raw and bias-corrected samples shown."
    )
    fig.text(
        0.5,
        0.935,
        f"{len(time)} samples | sensor-local frame, sensor origin | unfiltered",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        note + " Gravity retained; stage labels describe commanded motion.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91), h_pad=1.6)
    output = args.output or args.log.with_name(args.log.stem + "_curves.png")
    from scripts.shared.run_paths import new_output

    output = new_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": len(time),
                "raw_equals_corrected": same,
                "min": dict(zip(channels, raw.min(axis=0).tolist())),
                "max": dict(zip(channels, raw.max(axis=0).tolist())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
