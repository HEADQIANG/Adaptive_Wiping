"""Offline overview of eight accepted AIRBOT demonstrations."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#8c6510", "#56b4e9", "#6c4ba5", "#555555"]


def load_demos(session):
    session = session.resolve()
    demos, seen = [], set()
    for number in range(1, 9):
        manifest = session / f"demo_{number:02}.json"
        record = json.loads(manifest.read_text(encoding="utf-8"))
        path = (session / record["raw_file"]).resolve()
        if path.parent != session or path in seen:
            raise ValueError("Invalid or duplicate raw log path")
        seen.add(path)
        content = path.read_bytes()
        if (
            record.get("accepted") is not True
            or record.get("source_kind") != "real"
            or record.get("quality", {}).get("passed") is not True
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Rejected or changed log: {manifest}")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if rows[0]["event"] != "start" or rows[-1]["event"] != "finished":
            raise ValueError(f"Incomplete log: {path}")
        start = float(rows[0]["start_perf_s"])
        samples = [r for r in rows if r.get("event") == "sample"]
        pose_t = np.array([r["pose"]["host_monotonic_s"] - start for r in samples])
        positions = np.array([r["pose"]["sdk_end_position_m"] for r in samples]) * 1000
        lag = np.array([r["observed_perf_s"] - r["pose"]["host_monotonic_s"] for r in samples])
        if np.any(lag < 0) or np.any(lag > 0.020):
            raise ValueError("Pose/observation clock mismatch")
        # latest() can repeat a received sample; do not count it as a new reading.
        received = {}
        for row in samples:
            force = row["ft"]
            stamp = float(force["sensor_receive_perf_s"])
            value = np.array(force["raw_sensor_wrench_si"], dtype=float)
            if stamp in received and not np.array_equal(value, received[stamp]):
                raise ValueError("Conflicting readings at the same receive timestamp")
            received[stamp] = value
        stamps = sorted(received)
        ft_t = np.array(stamps) - start
        ft = np.array([received[t] for t in stamps])
        if (
            positions.shape != (len(pose_t), 3)
            or ft.shape != (len(ft_t), 6)
            or any(not np.isfinite(a).all() for a in (pose_t, positions, ft_t, ft))
            or len(pose_t) < 2
            or len(ft_t) < 2
            or not np.all(np.diff(pose_t) > 0)
        ):
            raise ValueError(f"Invalid samples: {path}")
        demos.append(
            dict(
                name=manifest.stem,
                pose_t=pose_t,
                positions=positions,
                ft_t=ft_t,
                ft=ft,
                raw_file=path.name,
                sha256=record["sha256"],
            )
        )
    return demos


def decorate(fig, title, subtitle):
    fig.suptitle(title, fontsize=18, y=0.985)
    fig.text(0.5, 0.947, subtitle, ha="center", fontsize=10)
    handles = [
        Line2D([0], [0], color=color, lw=2, label=f"Demo {i:02}")
        for i, color in enumerate(COLORS, 1)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=8,
        frameon=False,
        fontsize=10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        default=Path(
            "archive/real_training/raw_data/manual_demonstrations/session_record_only_002"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/real_training/demonstration_overview_002")
    )
    args = parser.parse_args()
    demos = load_demos(args.session)
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    axes = [fig.add_subplot(2, 3, 1, projection="3d")]
    axes += [fig.add_subplot(2, 3, i) for i in range(2, 7)]
    for demo, color in zip(demos, COLORS):
        p, t = demo["positions"], demo["pose_t"]
        axes[0].plot(*p.T, color=color, lw=1, alpha=0.85)
        axes[0].scatter(*p[0], color=color, s=25, marker="o")
        axes[0].scatter(*p[-1], color=color, s=30, marker="x")
        for ax, (x, y) in zip(axes[1:3], [(0, 1), (0, 2)]):
            ax.plot(p[:, x], p[:, y], color=color, lw=1, alpha=0.85)
            ax.scatter(p[0, x], p[0, y], color=color, s=22, marker="o")
            ax.scatter(p[-1, x], p[-1, y], color=color, s=28, marker="x")
        for index, ax in enumerate(axes[3:]):
            ax.plot(t, p[:, index], color=color, lw=1, alpha=0.85)
    axes[0].set(
        xlabel="SDK X (mm)", ylabel="SDK Y (mm)", zlabel="SDK Z (mm)", title="3D trajectory"
    )
    extent = np.ptp(np.concatenate([d["positions"] for d in demos]), axis=0)
    axes[0].set_box_aspect(np.maximum(extent, 1))
    for axis in (axes[0].xaxis, axes[0].yaxis, axes[0].zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
        axis.labelpad = 9
    axes[0].tick_params(labelsize=8, pad=1)
    for ax, (x, y) in zip(axes[1:3], [("X", "Y"), ("X", "Z")]):
        ax.set(xlabel=f"SDK {x} (mm)", ylabel=f"SDK {y} (mm)", title=f"{x}{y} projection")
        ax.set_aspect("equal", adjustable="box")
    for ax, channel in zip(axes[3:], "XYZ"):
        ax.set(
            xlabel="Elapsed host time (s)",
            ylabel=f"SDK {channel} (mm)",
            title=f"{channel} position",
        )
    for ax in axes[1:]:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    decorate(
        fig,
        "Eight accepted demonstrations: trajectory overview",
        "SDK end position, not calibrated sponge TCP | Absolute coordinates | Circle: start; cross: end",
    )
    fig.text(
        0.5,
        0.015,
        "Each demonstration uses its own recorded start time; no spatial alignment or time warping.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.87), w_pad=2.5, h_pad=2.5)
    fig.savefig(args.output / "trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    channels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for channel, ax in zip([0, 3, 1, 4, 2, 5], axes.flat):
        for demo, color in zip(demos, COLORS):
            ax.plot(demo["ft_t"], demo["ft"][:, channel], color=color, lw=0.9, alpha=0.8)
        unit = "N" if channel < 3 else "N m"
        ax.set(title=channels[channel], ylabel=f"{channels[channel]} ({unit})")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Sensor host receive time relative to demo start (s)")
    decorate(
        fig,
        "Eight accepted demonstrations: six-axis force / torque",
        "Raw, unfiltered | Sensor-local axes at sensor origin | Gravity and electronic bias retained",
    )
    fig.text(
        0.5,
        0.015,
        "Repeated receive timestamps deduplicated; no interpolation. Sensor axes are not SDK/world axes.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.87), h_pad=1.8)
    fig.savefig(args.output / "force_torque.png", dpi=160)
    plt.close(fig)
    report = {
        "session": str(args.session),
        "demonstrations": [
            {
                "name": d["name"],
                "raw_file": d["raw_file"],
                "sha256": d["sha256"],
                "pose_samples": len(d["pose_t"]),
                "unique_ft_samples": len(d["ft_t"]),
                "pose_time_range_s": [float(d["pose_t"][0]), float(d["pose_t"][-1])],
                "position_span_mm": np.ptp(d["positions"], axis=0).tolist(),
                "ft_min": d["ft"].min(axis=0).tolist(),
                "ft_max": d["ft"].max(axis=0).tolist(),
            }
            for d in demos
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
