"""Read-only plots of a completed fixed-setup run and its demonstration references."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from scripts.shared.common import file_digest
from scripts.shared.real_preprocessing import CausalFTFilter, finite_array


COLORS = ("#0072b2", "#56b4e9", "#009e73", "#cc79a7",
          "#669933", "#d55e00", "#998833", "#aa4499")


def load_run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [r for r in rows if r["event"] == "session_start"]
    if len(starts) != 1 or starts[0].get("shadow") is not False:
        raise ValueError("Expected one real motion session, not shadow inference")
    events = {r["event"] for r in rows}
    if "aborted" in events or not {"policy_complete", "session_complete"} <= events:
        raise ValueError("Expected a completed deployment")
    samples = [r for r in rows if r.get("phase") == "policy"]
    if [r["tick"] for r in samples] != list(range(1001)):
        raise ValueError("Expected contiguous 100 Hz policy ticks 0..1000")
    baseline = [r["bias_si"] for r in rows if r["event"] == "baseline_complete"]
    if len(baseline) != 1:
        raise ValueError("Expected one fresh unloaded baseline")
    run = {key: finite_array([r[key] for r in samples], key) for key in
           ("raw_ft", "tared_ft", "filtered_ft", "target_sdk_m", "tracking_target_sdk_m")}
    run["position"] = finite_array(
        [r["pose"]["sdk_end_position_m"] for r in samples], "position")
    start = samples[0]["due_perf_s"]
    run["time"] = np.array([r["due_perf_s"] - start for r in samples])
    run["pose_time"] = np.array([r["pose"]["host_monotonic_s"] - start for r in samples])
    if not np.allclose(run["time"], np.arange(1001) / 100, atol=1e-8, rtol=0):
        raise ValueError("Unexpected deployment time grid")
    if np.any(np.diff(run["pose_time"]) <= 0):
        raise ValueError("Deployment pose timestamps must increase")
    np.testing.assert_allclose(run["raw_ft"] - baseline[0], run["tared_ft"], atol=1e-12)
    np.testing.assert_allclose(
        CausalFTFilter().process(run["tared_ft"]), run["filtered_ft"], atol=1e-10)
    return run, starts[0]


def load_demonstrations(path, session, evaluation):
    report = json.loads(evaluation.read_text())
    if file_digest(path) != report["bindings"]["raw_sha256"]:
        raise ValueError("Demonstration file differs from the evaluated training data")
    demos = []
    with h5py.File(path, "r") as source:
        metadata = json.loads(source.attrs["metadata"])
        if (not source.attrs["complete"] or source.attrs["source_kind"] != "real"
                or metadata["compensation"] != "recorded_unloaded_baseline_subtracted"
                or metadata["derivation"] != "programmed_hold_last_10s_v1"):
            raise ValueError("Expected completed tared hold-last real demonstrations")
        binding = session["spec"]["collection_session_sha256"]
        matches = [digest for name, digest in metadata["source_hashes"].items()
                   if name.endswith("/session.json")]
        if matches != [binding]:
            raise ValueError("Reference demonstrations do not match the deployed session")
        if len(source["demonstrations"]) != 8:
            raise ValueError("Expected eight reference demonstrations")
        for name, group in sorted(source["demonstrations"].items()):
            time = group["ft_time"][:] - group.attrs["start_time"]
            np.testing.assert_allclose(time, np.arange(1001) / 100, atol=1e-10)
            np.testing.assert_allclose(group["pose_time"][:] - group.attrs["start_time"], time)
            ft = finite_array(group["ft"][:], name)
            np.testing.assert_allclose(
                group["ft_raw_before_baseline"][:] - group["recorded_unloaded_baseline"][:],
                ft, atol=1e-12)
            duration = float(group.attrs["measured_duration_s"])
            padding = time > duration + 1e-9
            np.testing.assert_array_equal(group["is_padding"][:].astype(bool), padding)
            position = finite_array(group["sdk_end_position"][:], name)
            if ft.shape != (1001, 6) or position.shape != (1001, 3):
                raise ValueError("Unexpected demonstration shape")
            demos.append(dict(name=name, time=time, ft=ft, position=position,
                              filtered=CausalFTFilter().process(ft), duration=duration,
                              depth_mm=float(group.attrs["initial_depth_m"]) * 1000))
    return demos


def reference_lines(ax, demos, key, channel, scale):
    for demo, color in zip(demos, COLORS):
        time, values = demo["time"], demo[key][:, channel] * scale
        end = int(round(demo["duration"] * 100))
        ax.plot(time[:end + 1], values[:end + 1], color=color, lw=1, alpha=0.75)
        ax.plot(time[end:], values[end:], color=color, lw=1, alpha=0.65, ls="--")


def decorate(fig, axes, demos, title, subtitle, *, position=False):
    handles = [Line2D([], [], color="#222222", lw=2.2, label="Deployment measured")]
    if position:
        handles.append(Line2D([], [], color="#e66101", lw=2, ls="--", label="Deployment command"))
    handles.extend(Line2D([], [], color=color, lw=1.5,
                          label=f"Demo {i:02} ({demo['depth_mm']:.0f} mm)")
                   for i, (demo, color) in enumerate(zip(demos, COLORS), 1))
    fig.suptitle(title, fontsize=17, y=0.99)
    fig.text(0.5, 0.949, subtitle, ha="center", fontsize=10)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.928),
               ncol=5, frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11))
        ax.set_xlabel("Time from motion start (s)")
        ax.grid(alpha=0.2)
        ax.axvline(2, color="#999999", ls=":", lw=0.8)
        ax.axvspan(5.6, 6.4, color="#eeeeee", zorder=-2)
        ax.axvspan(6.4, 10, color="#dddddd", alpha=0.6, zorder=-2)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.text(0.5, 0.016,
             "References: programmed demonstrations, not force setpoints. Dashed demo tails are held-last padding.\n"
             "Shading: some demos padded after 5.6 s; all after 6.4 s. No phase shift or lag correction.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.85), h_pad=1.5)


def make_plots(run, demos, output):
    paths = []
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for channel, ax in enumerate(axes):
        reference_lines(ax, demos, "position", channel, 1000)
        ax.plot(run["pose_time"], run["position"][:, channel] * 1000, color="#222222", lw=2.2)
        ax.plot(run["time"], run["target_sdk_m"][:, channel] * 1000,
                color="#e66101", lw=2, ls="--")
        ax.set_ylabel(f"Base {'XYZ'[channel]} (mm)")
    decorate(fig, axes, demos, "Real deployment: X / Y / Z vs references",
             "SDK end position in base frame; absolute coordinates, not sponge compression", position=True)
    path = output / "position_xyz.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    for kind, indices, filtered in (
        ("force", (0, 1, 2), False), ("torque", (3, 4, 5), False),
        ("wrench_filtered", (0, 1, 2, 3, 4, 5), True),
    ):
        fig, axes = plt.subplots(3, 2 if filtered else 1, figsize=(14, 10), squeeze=False)
        for channel in indices:
            ax = axes[channel % 3, channel // 3] if filtered else axes[channel % 3, 0]
            reference_lines(ax, demos, "filtered" if filtered else "ft", channel, 1)
            values = run["filtered_ft" if filtered else "tared_ft"]
            ax.plot(run["time"], values[:, channel], color="#222222", lw=1.8)
            label = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[channel]
            ax.set_ylabel(f"{label} ({'N' if channel < 3 else 'N m'})")
        suffix = "2nd-order causal 1 Hz low-pass, 100 Hz hold" if filtered else "No additional low-pass; 100 Hz held samples"
        decorate(fig, axes, demos, f"Real deployment: {kind.replace('_', ' ')} vs demonstration references",
                 "Each run's unloaded baseline removed; sensor-local axes and sign preserved. " + suffix)
        path = output / f"{kind}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


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
    args.output.mkdir(parents=True, exist_ok=False)
    paths = make_plots(run, demos, args.output)
    summary = {
        "scope": "one_completed_real_deployment_vs_eight_training_demonstrations",
        "reference_kind": "programmed_fixed_depth_not_hand_guided",
        "time_alignment": "motion_start_no_lag_or_phase_correction",
        "compensation": "each_run_unloaded_baseline_subtracted_sensor_local",
        "inputs": {str(p): file_digest(p) for p in (args.events, args.raw_data, args.evaluation)},
        "reference_durations_s": {d["name"]: d["duration"] for d in demos},
        "plots": [p.name for p in paths],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
