"""Separate-process, read-only display of newly recorded manual attempts."""

import math
import multiprocessing
import signal
import sys
import time
from pathlib import Path

from scripts.real_training.exploration_live_plot import LogTail


class ManualTrace:
    def __init__(self, plot):
        self.plot = plot
        self.origin = None
        self.elapsed = 0.0
        self.last_stamp = None
        self.last_arrival = None
        self.active = False
        self.status = "Waiting for tare and recording"
        plot.fig.canvas.manager.set_window_title("AIRBOT manual | Tared force / torque")
        plot.fig.suptitle("AIRBOT manual | Tared force / torque", fontsize=14, y=0.98)
        for ax in plot.panels[-1]:
            ax.set_xlabel("Time since first recording (s)")

    def ingest(self, row):
        event = row.get("event")
        if event == "start":
            if not row.get("tare") or row.get("force_recording") != "software_tared":
                raise ValueError("Live plot requires a recorded tare; no raw fallback")
            if self.origin is None:
                self.origin = row["start_perf_s"]
            else:
                self.plot.update(self.elapsed, None, stale=True)
            self.active = True
            self.last_arrival = time.perf_counter()
            self.status = "Recording"
        elif event == "finished":
            self.active = False
            self.status = ("Timing passed; awaiting operator acceptance"
                           if row["quality"]["passed"] else "Timing rejected")
        elif event == "sample" and self.active:
            force = row["ft"]
            values, stamp = force.get("tared_sensor_wrench_si"), force.get("sensor_receive_perf_s")
            if (not isinstance(values, list) or len(values) != 6
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                    or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)):
                raise ValueError("Live plot requires finite tared six-axis samples; no raw fallback")
            self.last_arrival = time.perf_counter()
            if self.last_stamp is None or stamp > self.last_stamp:
                self.last_stamp = stamp
                self.elapsed = max(0.0, stamp - self.origin)
                self.plot.update(self.elapsed, (self.elapsed, values))

    def refresh(self):
        stalled = (self.active and self.last_arrival is not None
                   and time.perf_counter() - self.last_arrival > 0.5)
        label = "No new logged samples; check terminal" if stalled else self.status
        self.plot.update(self.elapsed, None, force_draw=True)
        self.plot.status.set_text(f"{label} | {self.elapsed:.2f} s | tared, sensor-local")
        self.plot.status.set_color("#b42318" if stalled else "#137e79")
        self.plot.fig.canvas.draw_idle()
        self.plot.process_events()


def _worker(folder, existing, stop, ready):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    plot = tail = None
    seen = set(existing)
    try:
        from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot

        plot = LiveWrenchPlot(window_s=10, refresh_hz=10, net=True)
        trace = ManualTrace(plot)
        plot.open()
        if plot.closed:
            raise RuntimeError("Live plot closed before startup")
        ready.send((True, None))
        while not stop.is_set() and not plot.closed:
            if tail is not None:
                for row in tail.read():
                    trace.ingest(row)
            new = [p for p in Path(folder).glob("attempt_*.jsonl") if p.name not in seen]
            if new:
                path = min(new, key=lambda p: p.stat().st_mtime_ns)
                if tail is not None:
                    tail.close()
                tail = LogTail(path)
                seen.add(path.name)
            trace.refresh()
            stop.wait(0.1)
        if not stop.is_set():
            print("Manual plot closed; acquisition continues. Use terminal q/Ctrl+C to exit safely.",
                  file=sys.stderr, flush=True)
    except Exception as exc:
        message = f"Manual live plot failed: {exc}"
        print(message, file=sys.stderr, flush=True)
        try:
            ready.send((False, message))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if tail is not None:
            tail.close()
        ready.close()
        if plot is not None:
            plot.close()


class ManualLivePlot:
    def __init__(self, folder):
        context = multiprocessing.get_context("spawn")
        self.stop = context.Event()
        self.ready, self.sender = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_worker,
            args=(str(Path(folder).resolve()), [p.name for p in Path(folder).glob("attempt_*.jsonl")],
                  self.stop, self.sender),
            daemon=True,
        )

    def start(self):
        try:
            self.process.start()
            self.sender.close()
            if not self.ready.poll(15):
                raise RuntimeError("Manual plot startup timed out; hardware has not been connected")
            ok, message = self.ready.recv()
            if not ok:
                raise RuntimeError(message)
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
