"""Run one diagnostic exploration with live, initially tared force/torque plots."""

import argparse
import copy
import json
import time
from datetime import datetime

import numpy as np

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT, writable_path
from scripts.sim_pretrain.simulation import (
    PretrainingWipe, rollout_metrics, FT_OUTPUT_FRAME, FT_OUTPUT_SIGNS, FT_PROCESSING,
)


class LivePlot:
    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.figure, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
        self.axes = list(axes.flat)
        self.channels = (0, 3, 1, 4, 2, 5)
        self.lines = []
        for ax, channel in zip(self.axes, self.channels):
            label = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[channel]
            ax.set_ylabel(f"{label} ({'N' if channel < 3 else 'N m'})")
            ax.set_xlim(0, 4)
            ax.grid(alpha=0.25)
            ax.axvline(2, color="0.6", linestyle="--", linewidth=0.8)
            ax.axvline(3, color="0.6", linestyle="--", linewidth=0.8)
            self.lines.append(ax.plot([], [], color=("#187c65" if channel < 3 else "#ae4860"))[0])
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        self.figure.suptitle("Initial-tared FT | Y/Z output reversed")
        self.figure.tight_layout()
        self.times, self.values = [], []
        self.started = None
        self.last_draw = -1.0
        plt.show(block=False)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __call__(self, timestamp, wrench):
        if not self.plt.fignum_exists(self.figure.number):
            raise RuntimeError("Plot closed before exploration completed; no complete data saved")
        if self.started is None:
            self.started = time.monotonic()
        self.times.append(timestamp)
        self.values.append(np.asarray(wrench).copy())
        # Wall-clock pacing does not alter the simulation clock or command trajectory.
        while time.monotonic() < self.started + timestamp:
            self.plt.pause(min(0.01, max(0.001, self.started + timestamp - time.monotonic())))
        if timestamp - self.last_draw >= 0.05 - 1e-9 or timestamp == 4.0:
            values = np.asarray(self.values)
            for ax, line, channel in zip(self.axes, self.lines, self.channels):
                line.set_data(self.times, values[:, channel])
                ax.relim()
                ax.autoscale_view(scalex=False)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            self.last_draw = timestamp


class LiveViewer:
    def __init__(self, env):
        import mujoco
        import mujoco.viewer
        from scripts.sim_pretrain.experiments.visualize_wiping import configure_view

        self.mujoco = mujoco
        # The viewer may edit its model/data through mouse and UI interactions.
        self.model = copy.copy(env.sim.model._model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, env.sim.data._data)
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        with self.handle.lock():
            configure_view(
                self.handle.cam, self.handle.opt, collisions=False,
                tabletop=env.pretrain_cfg["simulation"].get("base_mount", "stand") == "tabletop",
            )
        self.handle.sync()

    def update(self, env):
        if not self.handle.is_running():
            raise RuntimeError("MuJoCo viewer closed before exploration completed; no complete data saved")
        with self.handle.lock():
            self.mujoco.mj_copyData(self.data, self.model, env.sim.data._data)
        self.handle.sync()

    def close(self):
        self.handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"))
    parser.add_argument("--mu", type=float, default=1.2, help="Sliding friction coefficient")
    parser.add_argument("--stiffness", type=float, default=500.25, help="MuJoCo direct stiffness parameter, not calibrated N/m")
    parser.add_argument("--width", type=float, default=0.16, help="solimp transition width, not sponge geometry width")
    parser.add_argument("--gain", type=float, default=300, help="Joint controller gain")
    parser.add_argument("--ramp-s", type=float, default=None, help="Each acceleration/deceleration ramp in seconds, [0, 0.5]; 0 restores legacy steps")
    parser.add_argument("--output", default=None, help="New output directory; defaults to a timestamped run")
    parser.add_argument("--no-plot", action="store_true", help="Headless single-rollout verification")
    parser.add_argument("--no-viewer", action="store_true", help="Show FT curves without the MuJoCo window")
    parser.add_argument("--close-after-run", action="store_true", help="Close windows after saving instead of keeping the final state")
    args = parser.parse_args(argv)
    if not all(np.isfinite(v) for v in (args.mu, args.stiffness, args.width, args.gain)):
        parser.error("Parameters must be finite")
    if args.mu < 0 or min(args.stiffness, args.width, args.gain) <= 0:
        parser.error("Require mu >= 0 and positive stiffness/width/gain")
    cfg = load_config(args.config)
    if args.ramp_s is not None:
        if not np.isfinite(args.ramp_s) or not 0 <= args.ramp_s <= 0.5:
            parser.error("--ramp-s must be finite and in [0, 0.5]")
        cfg["simulation"]["exploration_ramp_s"] = args.ramp_s
    ramp_s = cfg["simulation"].get("exploration_ramp_s", 0.0)
    output = writable_path(args.output or ("runs/sim_pretrain/explore_once_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    plot = None
    if not args.no_plot:
        import matplotlib

        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "template", "cairo", "pgf"):
            parser.error("Live plotting needs a desktop backend; use MPLBACKEND=TkAgg or --no-plot")
        plot = LivePlot()
    output.mkdir(parents=True, exist_ok=False)
    env = None
    viewer = None
    try:
        print(f"Single exploration: mu={args.mu:g}, stiffness={args.stiffness:g}, width={args.width:g}", flush=True)
        print("Preparing non-contact start; then 4 s exploration. No batch collection or training.", flush=True)
        print(f"Velocity ramp: {ramp_s:g} s; phase durations and displacements unchanged.", flush=True)
        env = PretrainingWipe(cfg, args.gain, (args.mu, args.stiffness, args.width))

        def on_sample(timestamp, wrench):
            nonlocal viewer
            if plot is not None:
                plot(timestamp, wrench)
                if not args.no_viewer:
                    if viewer is None:
                        viewer = LiveViewer(env)
                        # Window initialization is not part of exploration wall time.
                        plot.started = time.monotonic()
                    viewer.update(env)

        data = env.rollout(sample_callback=on_sample if plot is not None else None)
        metadata = dict(
            config=cfg, gain=args.gain, parameters=[args.mu, args.stiffness, args.width],
            trajectory_profile="smoothstep_velocity_v1" if ramp_s > 0 else "legacy_piecewise_linear",
            ft_processing=FT_PROCESSING, frame=FT_OUTPUT_FRAME,
            raw_frame="ft_frame local", sensor_quat_frame="raw ft_frame local",
            ft_output_signs=FT_OUTPUT_SIGNS,
            channels=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            units=["N"] * 3 + ["N*m"] * 3, complete=True,
            purpose="single_diagnostic_not_training_acceptance",
            metrics=rollout_metrics(data),
        )
        with (output / "exploration.npz").open("xb") as stream:
            np.savez_compressed(stream, **data, metadata_json=json.dumps(metadata))
        if plot is not None:
            plot.figure.savefig(output / "ft.png", dpi=150)
        print(f"Saved one 400x6 exploration: {output / 'exploration.npz'}", flush=True)
        print(json.dumps(metadata["metrics"], indent=2), flush=True)
        if plot is not None and not args.close_after_run:
            print("Exploration complete. Close both windows to exit.", flush=True)
            while True:
                plot_open = plot.plt.fignum_exists(plot.figure.number)
                viewer_open = viewer is not None and viewer.handle.is_running()
                if not plot_open and not viewer_open:
                    break
                if viewer_open:
                    viewer.handle.sync()
                if plot_open:
                    plot.plt.pause(0.02)
                else:
                    time.sleep(0.02)
    finally:
        if viewer is not None:
            viewer.close()
        if plot is not None:
            plot.plt.close(plot.figure)
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
