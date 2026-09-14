"""Read-only, separate-process display of tared exploration JSONL samples."""

import json
import math
import multiprocessing
import signal
import sys
import time
from pathlib import Path


def save_exploration_plot(path):
    """Render exploration only, without a desktop or robot connection."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scripts.force_sensor.live_kwr75_plot import AXES, PANEL_ORDER

    tail = LogTail(path)
    origin = last_stamp = None
    times, values = [], []
    status = "Partial recording"
    try:
        while rows := tail.read():
            for row in rows:
                event = row.get("event")
                if event == "phase_start" and row.get("phase") == "exploration":
                    origin = row["perf_s"]
                elif event in ("motion_complete", "session_complete", "aborted"):
                    status = event.replace("_", " ")
                elif event == "sample" and row.get("phase") == "exploration" and origin is not None:
                    force = row.get("force") or {}
                    wrench = force.get("tared_sensor_wrench_si")
                    stamp = force.get("sensor_receive_perf_s")
                    if (
                        not isinstance(wrench, list) or len(wrench) != 6
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in wrench)
                        or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    ):
                        raise ValueError("Saved plot requires finite tared samples; no raw fallback")
                    if last_stamp is None or stamp > last_stamp:
                        times.append(max(0.0, stamp - origin))
                        values.append(wrench)
                        last_stamp = stamp
    finally:
        tail.close()
    if not times:
        return None
    fig = Figure(figsize=(11, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    panels = fig.subplots(3, 2, sharex=True)
    fig.suptitle(f"AIRBOT exploration | Tared force / torque | {status}")
    for channel, ax in zip(PANEL_ORDER, panels.flat):
        ax.plot(times, [v[channel] for v in values], linewidth=1.1,
                color="#137e79" if channel < 3 else "#bd572b")
        ax.set_ylabel(f"{AXES[channel]} ({'N' if channel < 3 else 'Nm'})")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, max(4.0, times[-1]))
    for ax in panels[-1]:
        ax.set_xlabel("Time since exploration start (s)")
    output = Path(path).with_suffix(".png")
    # Never replace an existing image, including a user's manual export.
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150)
    return output


class LogTail:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def read(self, limit=1000):
        if self.stream is None:
            try:
                self.stream = self.path.open(encoding="utf-8")
            except FileNotFoundError:
                return []
        rows = []
        for _ in range(limit):
            position = self.stream.tell()
            line = self.stream.readline()
            if not line.endswith("\n"):
                self.stream.seek(position)
                break
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def close(self):
        if self.stream is not None:
            self.stream.close()


class ExplorationTrace:
    def __init__(self, plot):
        self.plot = plot
        self.origin = None
        self.elapsed = 0.0
        self.last_stamp = None
        self.last_arrival = None
        self.active = False
        self.status = "Waiting for software tare"
        plot.fig.canvas.manager.set_window_title("AIRBOT exploration | Tared force / torque")
        plot.fig.suptitle("AIRBOT exploration | Tared force / torque", fontsize=14, y=0.98)
        for ax in plot.panels[-1]:
            ax.set_xlabel("Time since exploration start (s)")

    def ingest(self, row):
        event = row.get("event")
        if event == "tare_start":
            self.status = "Software tare: stationary non-contact start"
        elif event == "tare_complete":
            self.status = "Tare complete; waiting for exploration"
        elif event == "phase_start":
            if row["phase"] == "exploration":
                self.origin = row["perf_s"]
                self.status = "Press"
            else:
                self.status = "Retract"
            self.active = row["phase"] == "exploration"
            self.last_arrival = time.perf_counter()
        elif event in ("motion_complete", "session_complete", "aborted"):
            self.active = False
            self.status = {
                "motion_complete": "Motion complete; awaiting supported IDLE handoff",
                "session_complete": "Session complete",
                "aborted": "ABORTED: check terminal and robot safety state",
            }[event]
        elif event == "sample" and row.get("phase") == "exploration" and self.origin is not None:
            force = row.get("force") or {}
            values = force.get("tared_sensor_wrench_si")
            stamp = force.get("sensor_receive_perf_s")
            if (
                not isinstance(values, list) or len(values) != 6
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
            ):
                raise ValueError("Live plot requires finite tared six-axis samples; no raw fallback")
            self.last_arrival = time.perf_counter()
            if row["phase"] == "exploration":
                t = row["protocol_time_s"]
                self.status = "Press" if t <= 2 else "+Y slide" if t <= 3 else "-Y return"
            else:
                self.status = "Retract"
            if self.last_stamp is None or stamp > self.last_stamp:
                self.last_stamp = stamp
                self.elapsed = max(0.0, stamp - self.origin)
                self.plot.update(self.elapsed, (self.elapsed, values))

    def refresh(self):
        stalled = (
            self.active and self.last_arrival is not None
            and time.perf_counter() - self.last_arrival > 0.5
        )
        label = "No new logged samples" if stalled else self.status
        self.plot.update(self.elapsed, None, force_draw=True)
        self.plot.status.set_text(f"{label} | {self.elapsed:.2f} s | tared, sensor-local")
        self.plot.status.set_color("#b42318" if stalled or "ABORTED" in label else "#137e79")
        self.plot.fig.canvas.draw_idle()
        self.plot.process_events()


def _plot_worker(path, window_s, refresh_hz, stop, ready):
    # Ctrl-C belongs to the controller's attended stop/handoff flow.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    plot = None
    tail = LogTail(path)
    try:
        from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot

        plot = LiveWrenchPlot(window_s=window_s, refresh_hz=refresh_hz, net=True)
        trace = ExplorationTrace(plot)
        plot.open()
        if plot.closed:
            raise RuntimeError("Live plot window closed before startup")
        ready.send((True, None))
        while not stop.is_set() and not plot.closed:
            for row in tail.read():
                trace.ingest(row)
            trace.refresh()
            stop.wait(1 / refresh_hz)
    except Exception as exc:
        message = f"Live exploration plot failed: {exc}"
        print(message, file=sys.stderr, flush=True)
        try:
            ready.send((False, message))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        tail.close()
        ready.close()
        if plot is not None:
            plot.close()


class ExplorationLivePlot:
    def __init__(self, path, window_s=10.0, refresh_hz=10.0):
        self.path = Path(path).resolve()
        self.save_on_close = False
        context = multiprocessing.get_context("spawn")
        self.stop = context.Event()
        self.ready, sender = context.Pipe(duplex=False)
        self.sender = sender
        self.process = context.Process(
            target=_plot_worker,
            args=(str(Path(path).resolve()), window_s, refresh_hz, self.stop, sender),
            daemon=True,
        )

    def start(self):
        try:
            self.process.start()
            self.sender.close()
            if not self.ready.poll(15):
                raise RuntimeError("Live plot startup timed out; hardware has not been connected")
            ok, message = self.ready.recv()
            if not ok:
                raise RuntimeError(message)
            self.save_on_close = True
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.ready.close()
        self.sender.close()
        if self.save_on_close:
            self.save_on_close = False
            try:
                output = save_exploration_plot(self.path)
                if output is not None:
                    print(f"Saved tared force/torque plot: {output}", flush=True)
            except Exception as exc:
                print(f"Could not save exploration PNG (JSONL unchanged): {exc}",
                      file=sys.stderr, flush=True)
