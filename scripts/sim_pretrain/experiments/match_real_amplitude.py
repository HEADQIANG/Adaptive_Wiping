"""Diagnostic parameter search against one real tared exploration, never training."""

import argparse
import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.shared.common import file_digest, load_config, provenance, write_json
from scripts.shared.paths import ROOT

WINDOWS = ((1.8, 2.0), (2.6, 3.0), (3.7, 4.0))
TIMES = np.arange(1, 401) / 100


def random_grid(count, seed, bounds):
    """Latin hypercube: uniform mu, log-uniform gain/stiffness/width."""
    if count <= 0 or seed < 0:
        raise ValueError("Require positive count and nonnegative seed")
    rng = np.random.default_rng(seed)
    columns = []
    for j, values in enumerate(bounds):
        if len(values) != 2 or not np.isfinite(values).all():
            raise ValueError("Random sampling requires two finite endpoints per parameter")
        low, high = values
        if low >= high or low < 0 or (j != 1 and low == 0):
            raise ValueError("Invalid random sampling bounds")
        u = (rng.permutation(count) + rng.random(count)) / count
        columns.append(low + u*(high-low) if j == 1 else np.exp(np.log(low)+u*np.log(high/low)))
    return np.column_stack(columns).tolist()


def coverage_summary(traces, real):
    """Marginal envelopes are not joint support or a trajectory match."""
    if len(traces) == 0:
        return dict(count=0, status="no_candidates")
    traces = np.asarray(traces)
    if traces.shape[1:] != (400, 6) or not np.isfinite(traces).all():
        raise ValueError("Expected finite N x 400 x 6 traces")
    low, high = traces.min(axis=0), traces.max(axis=0)
    inside = (real >= low) & (real <= high)
    phase = np.asarray([features(ft) for ft in traces])
    phase_low, phase_high = phase.min(axis=0), phase.max(axis=0)
    scale = np.maximum(np.percentile(np.abs(real), 95, axis=0), [0.5]*3+[0.01]*3)
    # A single candidate must satisfy all six channels at the same time.
    errors = np.abs(traces-real)/scale
    joint_time = np.any(np.all(errors <= 0.1, axis=2), axis=0)
    return dict(count=len(traces), status="evaluated",
                curve_fraction_channels=inside.mean(axis=0).tolist(),
                curve_fraction_all_marginals=float(inside.all(axis=1).mean()),
                phase_low=phase_low.tolist(), phase_high=phase_high.tolist(),
                phase_inside=((features(real)>=phase_low)&(features(real)<=phase_high)).tolist(),
                joint_time_fraction_at_10pct=float(joint_time.mean()),
                whole_trace_matches_at_10pct=int(np.all(errors<=0.1, axis=(1,2)).sum()))


def save_coverage(output, report, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"all_completed": [], "no_saturation": [], "motion_accepted": []}
    for row in report["cases"]:
        if not row["completed"]:
            continue
        with np.load(output / row["trajectory"]) as data:
            ft = data["ft"].copy()
        groups["all_completed"].append(ft)
        if row["metrics"]["saturation_fraction"] == 0:
            groups["no_saturation"].append(ft)
        if row["acceptance"]["passed"]:
            groups["motion_accepted"].append(ft)
    result = {name: coverage_summary(values, real) for name, values in groups.items()}
    report["coverage"] = result
    report["coverage_definition"] = (
        "Exact per-time and phase marginal min/max; not joint support. "
        "Joint tolerance=10% of per-channel real abs P95, floored at 0.5N/0.01Nm; "
        "joint_time may use different candidates at different times; whole_trace uses one candidate."
    )
    arrays = {"time": TIMES, "real": real}
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for name, values in groups.items():
        if values:
            arrays[name+"_low"] = np.min(values, axis=0)
            arrays[name+"_high"] = np.max(values, axis=0)
    for ax, j in zip(axes.flat, (0,3,1,4,2,5)):
        for name, color in zip(groups, ("#a0a0a0", "#267db3", "#29924b")):
            if groups[name]:
                ax.fill_between(TIMES, arrays[name+"_low"][:,j], arrays[name+"_high"][:,j],
                                color=color, alpha=0.25, label=f"{name} n={len(groups[name])}")
        ax.plot(TIMES, real[:,j], color="black", lw=1.4, label="Real 003")
        ax.set_ylabel(f"{('Fx','Fy','Fz','Tx','Ty','Tz')[j]} ({'N' if j<3 else 'N m'})")
        ax.grid(alpha=0.2)
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Marginal envelopes, not joint coverage | motion accepted: "
                 + str(len(groups["motion_accepted"])))
    fig.savefig(output / "coverage.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(output / "coverage.npz", **arrays)


def real_trace(path):
    """Use the same deduplicated receive timestamps and tared field as the real PNG."""
    origin = last = None
    times, values = [], []
    complete = False
    sample_times = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "aborted":
                raise ValueError("Reference session aborted")
            if row.get("event") == "session_complete":
                complete = True
            if row.get("event") == "phase_start" and row.get("phase") == "exploration":
                if origin is not None:
                    raise ValueError("Expected one exploration per reference log")
                origin = row["perf_s"]
            if row.get("event") != "sample" or row.get("phase") != "exploration":
                continue
            force = row["force"]
            sample_times.append(row["protocol_time_s"])
            wrench = np.asarray(force["tared_sensor_wrench_si"], dtype=float)
            stamp = float(force["sensor_receive_perf_s"])
            if origin is None or wrench.shape != (6,) or not np.isfinite(wrench).all() or not np.isfinite(stamp):
                raise ValueError("Invalid real exploration sample")
            if last is None or stamp > last:
                times.append(max(0, stamp-origin))
                values.append(wrench)
                last = stamp
    if (not complete or len(sample_times) != 400 or not np.allclose(sample_times, TIMES, atol=1e-8)
            or len(times) < 100 or times[0] > 0.05 or times[-1] < 3.95
            or np.max(np.diff(times)) > 0.05):
        raise ValueError("Reference must contain a complete four-second exploration")
    times, values = np.asarray(times), np.asarray(values)
    aligned = np.column_stack([np.interp(TIMES, times, values[:, j]) for j in range(6)])
    return times, values, aligned


def features(ft):
    return np.asarray([ft[(TIMES >= a) & (TIMES <= b)].mean(axis=0) for a, b in WINDOWS])


def comparison(ft, real):
    scale = np.maximum(np.percentile(np.abs(real), 95, axis=0), [0.5]*3 + [0.01]*3)
    phase = float(np.sqrt(np.mean(((features(ft)-features(real))/scale)**2)))
    curve_channels = np.sqrt(np.mean(((ft-real)/scale)**2, axis=0))
    curve = float(np.sqrt(np.mean(curve_channels**2)))
    return dict(
        score=0.7*phase + 0.3*curve, phase_nrmse=phase, curve_nrmse=curve,
        curve_nrmse_channels=curve_channels.tolist(),
        phase_means=features(ft).tolist(),
        p95_abs=np.percentile(np.abs(ft), 95, axis=0).tolist(),
        normalization_scale=scale.tolist(),
    )


def run_case(job):
    from scripts.sim_pretrain.acceptance import contact_motion_acceptance
    from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics

    index, cfg, params, reference, output = job
    gain, mu, stiffness, width = params
    row = dict(index=index, gain=gain, mu=mu, stiffness=stiffness, width=width, completed=False)
    env = None
    try:
        env = PretrainingWipe(cfg, gain, (mu, stiffness, width))
        data = env.rollout()
        if data["ft"].shape != (400, 6) or not np.isfinite(data["ft"]).all():
            raise ValueError("Invalid simulation wrench")
        row.update(comparison(data["ft"], reference))
        row.update(metrics=rollout_metrics(data), acceptance=contact_motion_acceptance(data))
        path = Path(output) / f"case_{index:03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
        row.update(completed=True, trajectory=path.name)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return row


def plot_best(output, report, real_time, real_ft):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted((r for r in report["cases"] if r["completed"]), key=lambda r: r["score"])[:3]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    channels = (0, 3, 1, 4, 2, 5)
    for ax, j in zip(axes.flat, channels):
        ax.plot(real_time, real_ft[:, j], color="black", lw=1.5, label="Real 003")
        for row in rows:
            with np.load(output / row["trajectory"]) as data:
                label = f"#{row['index']} g={row['gain']:g} mu={row['mu']:g} k={row['stiffness']:g} w={row['width']:g}"
                ax.plot(data["time"], data["ft"][:, j], lw=1, label=label)
        ax.set_ylabel(f"{('Fx','Fy','Fz','Tx','Ty','Tz')[j]} ({'N' if j<3 else 'N m'})")
        ax.set_xlim(0, 4)
        ax.grid(alpha=0.2)
        ax.axvline(2, color="0.7", ls="--", lw=0.7)
        ax.axvline(3, color="0.7", ls="--", lw=0.7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Real vs simulation | tared FT | no output scaling or time shift")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"))
    parser.add_argument("--reference", default=str(ROOT / "runs/real_exploration/exploration_tared_plot_003.jsonl"))
    parser.add_argument("--gains", nargs="+", type=float, default=[300, 1000])
    parser.add_argument("--mu", nargs="+", type=float, default=[0.5, 0.8, 1.2])
    parser.add_argument("--stiffness", nargs="+", type=float, default=[100.25, 500.25, 1000])
    parser.add_argument("--width", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--random-samples", type=int, default=0,
                        help="Latin hypercube count; each parameter list must have two bounds")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args(argv)
    for name in ("gains", "mu", "stiffness", "width"):
        values = np.asarray(getattr(args, name))
        if (not np.isfinite(values).all() or np.any(values < 0)
                or (name != "mu" and np.any(values == 0))):
            parser.error("Require finite nonnegative mu and positive stiffness/width/gains")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be in [1,4]")
    if args.random_samples < 0 or args.seed < 0:
        parser.error("random-samples and seed must be nonnegative")
    if args.random_samples:
        try:
            grid = random_grid(args.random_samples, args.seed,
                               (args.gains, args.mu, args.stiffness, args.width))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        grid = list(itertools.product(args.gains, args.mu, args.stiffness, args.width))
    cfg = load_config(args.config)
    rt, rf, aligned = real_trace(args.reference)
    from scripts.shared.run_paths import new_output

    output = new_output(args.output or "runs/sim_data/match_real_003")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        state="running", purpose="diagnostic_only_not_training_or_material_calibration",
        config=cfg, config_file=str(args.config), reference=str(args.reference),
        reference_sha256=file_digest(args.reference), provenance=provenance(),
        windows=WINDOWS, real_phase_means=features(aligned).tolist(),
        real_p95_abs=np.percentile(np.abs(aligned), 95, axis=0).tolist(),
        score_definition="0.7 * phase NRMSE + 0.3 * curve NRMSE; equal channel weight; per-channel real p95 scale",
        interpolation="Real receive timestamps interpolated onto 0.01..4.00; endpoint hold; no lag fitting",
        cases=[], grid=grid, sampling=dict(method="latin_hypercube" if args.random_samples else "grid",
                                         seed=args.seed, random_samples=args.random_samples,
                                         parameters=[args.gains, args.mu, args.stiffness, args.width]),
    )
    write_json(output / "report.json", report)
    with (output / "real_reference.npz").open("xb") as stream:
        np.savez_compressed(stream, time=rt, ft=rf, aligned_time=TIMES, aligned_ft=aligned)
    jobs = [(i, cfg, p, aligned, str(output)) for i, p in enumerate(grid)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(run_case, jobs):
            report["cases"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps({key: row[key] for key in ("index", "gain", "mu", "stiffness", "width", "completed", "score", "error") if key in row}), flush=True)
    completed = sorted((r for r in report["cases"] if r["completed"]), key=lambda r: r["score"])
    report.update(state="complete", ranked_indices=[r["index"] for r in completed],
                  acceptance_passed_indices=[r["index"] for r in completed if r["acceptance"]["passed"]])
    if completed:
        plot_best(output, report, rt, rf)
    save_coverage(output, report, aligned)
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
