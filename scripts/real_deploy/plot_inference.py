"""Automatic Fz/height and reference-ratio plots after a completed manual policy."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def reference_from_training(spec, training, bindings):
    """Freeze the bound manual demonstrations' signed Fz mean before connecting."""
    import h5py
    import numpy as np
    from scripts.real_training.config import resolve
    from scripts.shared.common import file_digest
    from scripts.shared.real_preprocessing import finite_array
    from scripts.shared.sampling import previous_samples

    path = resolve(training["raw_data"]).resolve()
    expected = bindings[str(path)]
    if file_digest(path) != expected:
        raise ValueError("Plot reference raw data changed")
    means, squares, episodes = [], [], []
    grid = np.arange(1000) / 100
    with h5py.File(path, "r") as source:
        meta = json.loads(source.attrs["metadata"])
        session_hashes = [digest for name, digest in meta["source_hashes"].items()
                          if Path(name).name == "session.json"]
        if (not source.attrs["complete"] or source.attrs["source_kind"] != "real"
                or meta.get("derivation") != "manual_recorded_baseline_10s_v1"
                or meta.get("compensation") != "recorded_unloaded_baseline_subtracted"
                or meta.get("contains_derived_samples") is not False
                or session_hashes != [spec["collection_session_sha256"]]
                or len(source["demonstrations"]) != 8):
            raise ValueError("Expected eight bound, measured, tared manual demonstrations")
        for name, group in sorted(source["demonstrations"].items()):
            stamps = finite_array(group["ft_time"][:] - group.attrs["start_time"], "reference times")
            ft = finite_array(group["ft"][:], "reference wrench")
            raw = finite_array(group["ft_raw_before_baseline"][:], "reference raw wrench")
            bias = finite_array(group["recorded_unloaded_baseline"][:], "reference baseline")
            if (not group.attrs["complete"] or group.attrs["source_kind"] != "real"
                    or stamps.ndim != 1 or len(stamps) < 2 or np.any(np.diff(stamps) <= 0)
                    or stamps[0] > 1e-9 or stamps[-1] < 10 - 1e-9
                    or ft.shape != (len(stamps), 6) or raw.shape != ft.shape or bias.shape != (6,)
                    or not np.allclose(raw - bias, ft, atol=1e-9, rtol=0)):
                raise ValueError(f"Invalid measured reference episode: {name}")
            fz = previous_samples(stamps, ft, grid)[:, 2]
            mean, square = float(fz.mean()), float(np.mean(fz ** 2))
            means.append(mean)
            squares.append(square)
            episodes.append({"name": name, "mean_fz_n": mean, "sample_count": len(fz)})
    if file_digest(path) != expected:
        raise ValueError("Plot reference raw data changed while reading")
    mean, rms = float(np.mean(means)), float(np.sqrt(np.mean(squares)))
    return {"schema_version": 1, "mean_fz_n": mean, "rms_fz_n": rms,
            "absolute_mean_over_rms": abs(mean) / rms if rms else 0.0,
            "method": "equal_mean_of_8_manual_demos_tared_unfiltered_fz_100hz_0_to_10s_half_open",
            "window_s": [0, 10], "sample_hz": 100, "episodes": episodes,
            "raw_data": str(path), "raw_sha256": expected,
            "collection_session_sha256": spec["collection_session_sha256"]}


class AutomaticPlots:
    """Launch only after inference; never wait for plotting while holding the robot."""

    def __init__(self, events):
        self.events = Path(events).resolve()
        self.output = self.events.with_suffix(".plots")
        self.log = self.events.with_suffix(".plots.log")
        self.process = None
        self.error = None

    def start(self):
        try:
            from scripts.shared.paths import ROOT

            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with self.log.open("x", encoding="utf-8") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.real_deploy.plot_inference",
                     "--events", str(self.events), "--output", str(self.output), "--wait-seconds", "10"],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            print(f"Saving inference plots in background: {self.output}", flush=True)
        except Exception as exc:
            self.error = exc
            print(f"Automatic plots could not start: {exc}; robot remains in final hold.", file=sys.stderr)

    def finish(self):
        """Called after hardware cleanup, so a renderer cannot delay monitoring/handoff."""
        if self.error is not None:
            return False
        if self.process is None:
            return True
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"Plotting still running in background; check {self.log}", file=sys.stderr)
            return False
        if code != 0 or not (self.output / "summary.json").is_file():
            print(f"Automatic plots failed; events preserved. See {self.log}", file=sys.stderr)
            return False
        print(f"Saved inference plots: {self.output / 'fz_height.png'} and {self.output / 'fz_ratio.png'}")
        return True


def policy_snapshot(path, wait_seconds=0):
    """Read only through policy_complete, including when the log is still being written."""
    if not 0 <= wait_seconds <= 30:
        raise ValueError("wait-seconds must be in [0, 30]")
    deadline = time.monotonic() + wait_seconds
    while True:
        prefix = []
        for line in Path(path).read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            prefix.append(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") in ("fault", "aborted", "stop_error"):
                raise ValueError("Inference did not complete successfully; no automatic plots")
            if row.get("event") == "policy_complete":
                return b"".join(prefix)
        if time.monotonic() >= deadline:
            raise ValueError("No flushed policy_complete event; inference log is incomplete")
        time.sleep(0.05)


def _write_csv(path, header, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def generate(events, output, *, wait_seconds=0):
    import numpy as np
    from scripts.real_deploy.plot_run_comparison import load_run
    from scripts.real_deploy.plot_reference_ratio import signed_ratio

    events, output = Path(events).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Plot output already exists: {output}")
    snapshot = policy_snapshot(events, wait_seconds)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_path = output / "inference_events.jsonl"
    with snapshot_path.open("xb") as stream:
        stream.write(snapshot)
    run = load_run(snapshot_path, policy_only=True)
    if run["kind"] != "manual":
        raise ValueError("Automatic inference plots require manual runtime logs")
    reference = run["header"]["spec"]["plot_reference_fz"]
    ref = float(reference["mean_fz_n"])
    means = np.asarray([episode["mean_fz_n"] for episode in reference["episodes"]], dtype=float)
    if (reference.get("schema_version") != 1 or means.shape != (8,) or not np.isfinite(means).all()
            or not np.isfinite(ref) or not np.isclose(ref, means.mean(), atol=1e-12, rtol=1e-12)
            or reference["collection_session_sha256"] != run["header"]["spec"]["collection_session_sha256"]
            or reference["raw_sha256"] != run["header"]["bindings"].get(reference["raw_data"])):
        raise ValueError("Reference Fz mean or deployment binding mismatch")
    ticks = run["prediction_ticks"]
    prediction_time = run["time"][ticks]
    latest_fz = run["filtered_ft"][ticks, 2]
    axis, sign = run["normal_axis"], run["normal_sign"]
    axis_name = "xyz"[axis]
    next_normal = run["position"][ticks, axis] + sign * run["delta_h"]
    raw_ratio = signed_ratio(run["tared_ft"][:, 2], ref)
    filtered_ratio = signed_ratio(run["filtered_ft"][:, 2], ref)
    _render(run, reference, prediction_time, latest_fz, next_normal, raw_ratio, filtered_ratio, output)
    _write_csv(output / "fz_height_predictions.csv",
               ["prediction_time_s", "endpoint_time_s", "filtered_fz_n", "predicted_delta_h_m",
                f"measured_anchor_{axis_name}_m", f"predicted_next_{axis_name}_m",
                *[f"history_fz_{i}_n" for i in range(1, 6)], "delta_h_to_sdk_sign"],
               ([t, t + 0.4, fz, dh, run["position"][tick, axis], normal,
                 *run["filtered_ft"][np.arange(tick - 160, tick + 1, 40), 2], sign]
                for tick, t, fz, dh, normal in zip(ticks, prediction_time, latest_fz, run["delta_h"], next_normal)))
    _write_csv(output / "fz_reference_ratio.csv",
               ["time_s", "tared_fz_n", "filtered_fz_n", "reference_fz_n", "ratio_percent", "filtered_ratio_percent"],
               ([t, raw, filtered, ref, r if np.isfinite(r) else "", f if np.isfinite(f) else ""]
                for t, raw, filtered, r, f in zip(run["time"], run["tared_ft"][:, 2],
                                                 run["filtered_ft"][:, 2], raw_ratio, filtered_ratio)))
    summary = {"schema_version": 1, "scope": "completed_inference_only_not_handoff_or_session_outcome",
               "source_events": str(events), "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
               "duration_s": run["final_tick"] / 100, "prediction_count": len(ticks),
               "reference": reference, "ratio_formula": "100 * tared_sensor_Fz(t) / signed_mean_demo_Fz",
               "ratio_defined": abs(ref) > 1e-12,
               "near_cancellation": reference["absolute_mean_over_rms"] < 0.1,
               "wiping_frame": run["wiping_frame"],
               "height_units": f"m in CSV, mm in plots; SDK {axis_name.upper()}, not surface clearance",
               "pairing": f"latest sensor-local filtered Fz paired with delta_h and measured_{axis_name.upper()} + ({sign}) * delta_h; endpoint t+0.4s",
               "model_history": "five six-axis filtered frames spaced 0.4s; Fz alone does not determine height",
               "plots": ["fz_height.png", "fz_ratio.png"]}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary


def _render(run, reference, times, fz, next_normal, raw_ratio, filtered_ratio, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    end = run["time"][-1]
    axis_name = "XYZ"[run["normal_axis"]]

    def time_axis(ax):
        ax.set_xlim(0, end)
        ax.set_xticks(np.arange(0, end + 1))
        ax.set_xlabel("Time after s (s)")
        ax.axvspan(0, 2, color="#778899", alpha=0.10)
        ax.axvline(2, color="#68737d", ls=":", lw=1)
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), layout="constrained")
    axes[0].plot(run["time"], run["tared_ft"][:, 2], color="#a8c4cf", lw=0.8, label="Tared Fz")
    axes[0].plot(run["time"], run["filtered_ft"][:, 2], color="#147d92", lw=1.8, label="Filtered Fz")
    axes[0].scatter(times, fz, color="#147d92", s=24, zorder=3, label="Fz at prediction")
    axes[0].set_ylabel("Sensor-local Fz (N)")
    axes[0].legend(loc="best", ncol=3, fontsize=9)
    axes[1].plot(times, next_normal * 1000, "o-", color="#315ca8", ms=4,
                 label=f"Predicted {axis_name}(t+0.4s), shown at prediction time t")
    axes[1].set_ylabel(f"Predicted next SDK {axis_name} (mm)", color="#315ca8")
    delta_axis = axes[1].twinx()
    delta_axis.plot(times, run["delta_h"] * 1000, "s--", color="#bb592c", ms=4, label="Predicted delta h")
    delta_axis.set_ylabel("Predicted delta h (mm)", color="#bb592c")
    delta_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    lines = axes[1].lines + delta_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9)
    for ax in axes[:2]:
        time_axis(ax)
    dots = axes[2].scatter(fz, run["delta_h"] * 1000, c=times, cmap="viridis", s=45, zorder=3)
    axes[2].plot(fz, run["delta_h"] * 1000, color="#88959f", alpha=0.5, lw=0.8)
    axes[2].set(xlabel="Filtered Fz at prediction (N)", ylabel="Predicted delta h (mm)")
    axes[2].grid(alpha=0.2)
    axes[2].ticklabel_format(style="plain", useOffset=False)
    fig.colorbar(dots, ax=axes[2], label="Prediction time after s (s)")
    fig.suptitle(f"Sensor-local Fz and predicted normal position (SDK {axis_name})\n"
                 "Gray: first 2 s history collection; model uses five six-axis frames", fontsize=14)
    fig.savefig(output / "fz_height.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.plot(run["time"], raw_ratio, color="#b9c6d1", lw=0.9, label="Tared Fz / reference")
    ax.plot(run["time"], filtered_ratio, color="#147d92", lw=1.8, label="Filtered Fz / reference")
    ax.axhline(100, color="#222222", ls="--", lw=1, label="Reference = 100%")
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    time_axis(ax)
    ax.set_ylabel("Sensor-local Fz / reference Fz (%)")
    ax.legend(loc="best", fontsize=9)
    if not np.isfinite(raw_ratio).any():
        ax.text(0.5, 0.5, "Ratio undefined: reference Fz is zero", transform=ax.transAxes, ha="center")
    elif reference["absolute_mean_over_rms"] < 0.1:
        ax.text(0.02, 0.95, "Reference signs nearly cancel; ratios may be large", transform=ax.transAxes,
                va="top", color="#a54926")
    ax.set_title(f"Fz / signed demonstration mean | reference = {reference['mean_fz_n']:.6g} N\n"
                 "8 bound manual demos, each measured [0, 10) s; 100% means equal signed force", fontsize=12)
    fig.savefig(output / "fz_ratio.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Fresh directory; defaults to <events stem>.plots")
    parser.add_argument("--wait-seconds", type=float, default=0, help="Wait for background event writer, at most 30s")
    args = parser.parse_args(argv)
    output = args.output or args.events.with_suffix(".plots")
    try:
        generate(args.events, output, wait_seconds=args.wait_seconds)
    except Exception as exc:
        print(f"Automatic plots failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {output / 'fz_height.png'} and {output / 'fz_ratio.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
