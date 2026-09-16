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


def load_demos(session, *, allow_partial=False):
    session = Path(session).resolve()
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    mode = metadata["config"].get("force_recording", "raw")
    if mode not in ("raw", "software_tared"):
        raise ValueError("Unknown manual force recording mode")
    field = "tared_sensor_wrench_si" if mode == "software_tared" else "raw_sensor_wrench_si"
    manifests = sorted(session.glob("demo_*.json"))
    count = len(manifests)
    if (count > 8 or (not allow_partial and count != 8)
            or [p.name for p in manifests] != [f"demo_{i:02}.json" for i in range(1, count + 1)]):
        raise ValueError("Expected consecutive accepted demo_01 through demo_08")
    demos, seen = [], set()
    for manifest in manifests:
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
        if not rows[-1].get("quality", {}).get("passed"):
            raise ValueError(f"Timing rejected log: {path}")
        baseline = None
        if mode == "software_tared":
            tare = rows[0].get("tare")
            if (record.get("force_recording") != mode or rows[0].get("force_recording") != mode
                    or not isinstance(tare, dict) or record.get("tare") != tare):
                raise ValueError("Missing or conflicting recorded tare; no raw fallback")
            baseline = np.asarray(tare["raw_baseline_si"], dtype=float)
            if baseline.shape != (6,) or not np.isfinite(baseline).all():
                raise ValueError("Invalid tare baseline")
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
            if field not in force:
                raise ValueError(f"Missing {field}; no raw fallback")
            value = np.array(force[field], dtype=float)
            if baseline is not None:
                raw = np.asarray(force["raw_sensor_wrench_si"], dtype=float)
                if (force.get("tare_id") != tare["id"] or raw.shape != (6,)
                        or value.shape != (6,) or not np.isfinite(raw).all()
                        or not np.allclose(value, raw - baseline, atol=1e-10, rtol=1e-10)):
                    raise ValueError("Tared wrench differs from recorded raw value and baseline")
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
                force_recording=mode,
                force_field=field,
                tare_id=record.get("tare", {}).get("id") if record.get("tare") else None,
            )
        )
    return demos


def decorate(fig, title, subtitle, count):
    fig.suptitle(title, fontsize=18, y=0.985)
    fig.text(0.5, 0.947, subtitle, ha="center", fontsize=10)
    handles = [
        Line2D([0], [0], color=color, lw=2, label=f"Demo {i:02}")
        for i, color in enumerate(COLORS[:count], 1)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=8,
        frameon=False,
        fontsize=10,
    )


def save_overview(session, output=None, *, allow_partial=False):
    """Export only accepted records; reserve a fresh directory, never replace plots."""
    from scripts.shared.paths import writable_path

    session = Path(session).resolve()
    demos = load_demos(session, allow_partial=allow_partial)
    if not demos:
        return None
    if output is None:
        index = 1
        while True:
            output = writable_path(session / ("plots" if index == 1 else f"plots_{index:03}"))
            try:
                output.mkdir()
                break
            except FileExistsError:
                index += 1
    else:
        output = writable_path(output)
        output.mkdir(parents=True, exist_ok=False)
    count = len(demos)
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
        f"{count}/8 accepted demonstrations: trajectory overview",
        "SDK end position | Absolute coordinates | Circle: start; cross: end",
        count,
    )
    fig.text(
        0.5,
        0.015,
        "Each demonstration uses its own recorded start time; no spatial alignment or time warping.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.87), w_pad=2.5, h_pad=2.5)
    try:
        fig.savefig(output / "trajectories.png", dpi=160)
    finally:
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
        f"{count}/8 accepted demonstrations: six-axis force / torque",
        ("Software-tared, unfiltered | Sensor-local axes | Fixed-pose baseline, not gravity compensation"
         if demos[0]["force_recording"] == "software_tared" else
         "Raw, unfiltered | Sensor-local axes at sensor origin | Gravity and electronic bias retained"),
        count,
    )
    fig.text(
        0.5,
        0.015,
        "Repeated receive timestamps deduplicated; no interpolation. Sensor axes are not SDK/world axes.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.87), h_pad=1.8)
    try:
        fig.savefig(output / "force_torque.png", dpi=160)
    finally:
        plt.close(fig)
    report = {
        "session": str(session),
        "accepted": count,
        "complete": count == 8,
        "force_recording": demos[0]["force_recording"],
        "force_field": demos[0]["force_field"],
        "demonstrations": [
            {
                "name": d["name"],
                "raw_file": d["raw_file"],
                "sha256": d["sha256"],
                "tare_id": d["tare_id"],
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
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {count}/8 accepted demonstration plots: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--output", type=Path,
                        help="new output directory (default: session/plots, with collision suffix)")
    parser.add_argument("--allow-partial", action="store_true", help="plot fewer than 8 accepted records")
    args = parser.parse_args(argv)
    output = args.output
    if args.session is None:
        args.session = Path("archive/real_training/raw_data/manual_demonstrations/session_record_only_002")
        output = output or Path("runs/real_demonstrations/analysis/demonstration_overview_002")
    if output is not None:
        from scripts.shared.run_paths import new_output

        output = new_output(output)
    save_overview(args.session, output, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
