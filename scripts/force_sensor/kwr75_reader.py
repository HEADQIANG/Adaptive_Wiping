#!/usr/bin/env python3
"""KWR75 wrist F/T sensor — in-process background reader (Path B, read-only).

Reusable latest-frame reader built on the constants verified in ``kwr75_probe.py``
(see NOTES.md 'KWR75 wrist F/T sensor'). A daemon thread drains the serial stream and
keeps the most recent 6-axis reading; callers poll ``latest()`` at their own clock and
log it alongside arm state. Touches ONLY the F/T sensor on /dev/ttyUSB0 — it does NOT
talk to the arm (USB-CAN on ttyACM0) and issues no motion, so it is safe to run any time.

Protocol (verified live): 8-N-1, no flow control, baud 460800. 28-byte frame =
head 0x48 0xAA | 6x float32 LE (Fx Fy Fz Tx Ty Tz) at offsets 2/6/10/14/18/22 |
tail 0x0D 0x0A. Each value x 9.81 (raw kgf/kgf-m -> N/N-m). No checksum. ~1024 Hz.
START = 48 AA 0D 0A, STOP = 43 AA 0D 0A.

Standalone demo: ``python scripts/force_sensor/kwr75_reader.py --tare --secs 60 --plot
--csv runs/force_sensor/kwr75.csv``. See ``docs/force_sensor/kwr75_reader.md`` for setup and timestamp semantics.
"""

import csv
import math
import queue
import signal
import struct
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import serial

G = 9.81
BAUD = 460800
_START = bytes([0x48, 0xAA, 0x0D, 0x0A])
_STOP = bytes([0x43, 0xAA, 0x0D, 0x0A])
_HEAD = bytes([0x48, 0xAA])
_TAIL = bytes([0x0D, 0x0A])
_FRAME = 28
AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


class _CsvRecorder:
    """Bounded batch queue: serial thread enqueues, caller writes to disk."""

    def __init__(self, path, bias):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._queue = queue.Queue(maxsize=1024)
        self._bias = bias.copy()
        self._t0 = time.perf_counter()
        self._last_flush = self._t0
        self.error = None
        self.rows = 0
        units = ("N", "N", "N", "Nm", "Nm", "Nm")
        try:
            self._writer.writerow(
                ["frame_index", "timestamp_utc", "host_time_ns", "elapsed_s"]
                + [
                    f"{kind}_{axis}_{unit}"
                    for kind in ("raw", "net")
                    for axis, unit in zip(AXES, units)
                ]
            )
            self._file.flush()
        except BaseException:
            self._file.close()
            raise

    def enqueue(self, t_perf, host_time_ns, first_index, frames):
        if self.error is not None:
            return
        try:
            self._queue.put_nowait((t_perf, host_time_ns, first_index, frames))
        except queue.Full:
            self.error = RuntimeError("CSV queue full: recording incomplete; disk writer too slow")

    def drain(self):
        # Bound each drain so continuous input cannot starve display or shutdown.
        for _ in range(self._queue.qsize()):
            t_perf, host_time_ns, first_index, frames = self._queue.get_nowait()
            utc = datetime.fromtimestamp(host_time_ns / 1e9, timezone.utc).isoformat(
                timespec="microseconds"
            )
            for offset, frame in enumerate(frames):
                raw = np.asarray(frame, dtype=float) * G
                self._writer.writerow(
                    [first_index + offset, utc, host_time_ns, f"{t_perf - self._t0:.9f}"]
                    + raw.tolist()
                    + (raw - self._bias).tolist()
                )
                self.rows += 1
        now = time.perf_counter()
        if now - self._last_flush >= 1.0:
            self._file.flush()
            self._last_flush = now
        if self.error is not None:
            raise self.error

    def close(self):
        try:
            self.drain()
        finally:
            self._file.close()


class _BackgroundCsvRecorder(_CsvRecorder):
    """Single disk-writing owner; producer and control calls never write files."""

    def __init__(self, path, bias):
        super().__init__(path, bias)
        self._stop_writer = threading.Event()
        self._ready = threading.Event()
        self._worker = threading.Thread(target=self._run_writer, name="kwr75-csv", daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self._file.close()
            raise

    def enqueue(self, *args):
        super().enqueue(*args)
        self._ready.set()

    def drain(self):
        if self.error is not None:
            raise self.error
        if not self._worker.is_alive():
            raise RuntimeError("CSV writer stopped; recording incomplete")

    def _run_writer(self):
        try:
            while not self._stop_writer.is_set():
                self._ready.wait(0.01)
                self._ready.clear()
                _CsvRecorder.drain(self)
            _CsvRecorder.drain(self)
        except BaseException as exc:
            if self.error is None:
                self.error = exc
        finally:
            try:
                self._file.close()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def close(self):
        # The reader detaches the producer before asking the writer to finish.
        self._stop_writer.set()
        self._ready.set()
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            raise RuntimeError("CSV writer shutdown timed out; recording incomplete")
        if self.error is not None:
            raise self.error


class Kwr75Reader:
    """Background-thread reader keeping the latest KWR75 frame.

    Usage:
        r = Kwr75Reader().start()
        r.tare(1.0)                       # optional: zero out mount self-weight at rest
        t, w = r.latest()                 # w = np.array([Fx,Fy,Fz,Tx,Ty,Tz]) in N / N-m
        r.stop()
    """

    def __init__(self, port="/dev/ttyUSB0", baud=BAUD):
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None  # (t_perf, np.ndarray[6]) — raw (pre-tare)
        self._count = 0  # total frames parsed since start
        self._last_frame_t = None  # wall-clock of most recent frame (staleness check)
        self._bias = np.zeros(6)
        self._csv = None
        self._read_error = None
        # short ring of recent (t, raw6) for window-averaged 1kHz->100Hz decimation
        self._ring = deque(maxlen=256)  # ~256 ms at 1 kHz

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self._ser = serial.Serial(
            self.port, self.baud, bytesize=8, parity="N", stopbits=1, timeout=0.1
        )
        self._ser.reset_input_buffer()
        self._ser.write(_START)
        self._ser.flush()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kwr75", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.write(_STOP)
                self._ser.flush()
            except Exception:
                pass
            try:
                self._ser.close()
            finally:
                self._ser = None
                self._close_csv()
        else:
            self._close_csv()

    def start_csv(self, path, *, bias=None, background=False):
        """Record subsequent parsed frames, freezing the current tare bias."""
        with self._lock:
            if self._csv is not None:
                raise RuntimeError("CSV recording already started")
            selected = self._bias if bias is None else np.asarray(bias, dtype=float)
            if selected.shape != (6,) or not np.isfinite(selected).all():
                raise ValueError("CSV bias must contain six finite values")
            recorder_type = _BackgroundCsvRecorder if background else _CsvRecorder
            self._csv = recorder_type(path, selected)
            return self._csv

    def stop_csv(self):
        """Detach the recorder before closing; the serial reader remains active."""
        with self._lock:
            recorder, self._csv = self._csv, None
        if recorder is not None:
            recorder.close()

    def flush_csv(self):
        """Drain synchronous CSV; background mode only checks recording health."""
        if self._csv is not None:
            self._csv.drain()
            if self._read_error is not None:
                raise RuntimeError(f"Sensor read failed: {self._read_error}") from self._read_error

    def _close_csv(self):
        self.stop_csv()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- reader thread ----------------------------------------------------
    def _run(self):
        # Small reads keep latency low: ~9 frames (~9 ms of data) per chunk rather than
        # blocking until a 4 KB buffer fills (~140 ms stale). At 1 kHz the sensor emits
        # ~28 KB/s, well within a 256-byte-per-loop drain.
        buf = bytearray()
        self._ser.timeout = 0.02
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(256)
            except Exception as exc:
                self._read_error = exc
                break
            if not chunk:
                continue
            buf += chunk
            i, got = 0, []
            while True:
                j = buf.find(_HEAD, i)
                if j < 0 or j + _FRAME > len(buf):
                    break
                fr = buf[j : j + _FRAME]
                if fr[26:28] == _TAIL:
                    got.append(struct.unpack("<6f", fr[2:26]))
                    i = j + _FRAME
                else:
                    i = j + 1
            if got:
                vals = np.asarray(got[-1], dtype=float) * G
                with self._lock:
                    now = time.perf_counter()
                    host_time_ns = time.time_ns()
                    if self._csv is not None:
                        self._csv.enqueue(now, host_time_ns, self._count + 1, got)
                    self._latest = (now, vals)
                    self._count += len(got)  # count FRAMES, not chunks -> true rate
                    self._last_frame_t = now
                    # chunk spans ~9 ms; tag each frame with chunk-end time (fine for a
                    # 10 ms window average). Ring feeds window_mean() decimation.
                    for f in got:
                        self._ring.append((now, np.asarray(f, dtype=float) * G))
            # drop everything we consumed; keep only the trailing partial frame
            if i:
                del buf[:i]
            if len(buf) > (1 << 16):  # runaway guard if delimiters ever desync
                del buf[:-64]

    # -- polling ----------------------------------------------------------
    def latest(self, net=True):
        """Return (t_perf, wrench[6]) or None if no frame yet. net=True subtracts tare."""
        with self._lock:
            if self._latest is None:
                return None
            t, vals = self._latest
        return t, (vals - self._bias if net else vals.copy())

    def window_mean(self, dur=0.01, net=True):
        """Box-average of frames from the last ``dur`` seconds -> (t_end, wrench[6], n).

        This is the 1 kHz -> 100 Hz decimation used to time-align with the 100 Hz arm
        stream: averaging ~10 frames per 10 ms window cuts the (white) shear noise ~3x
        (verified: Fx 1.5 N @1kHz -> 0.48 N @100Hz). Falls back to the single latest
        frame if the ring is momentarily empty. Returns None if no frame yet.
        """
        with self._lock:
            if not self._ring:
                return None
            t_end = self._ring[-1][0]
            sel = [w for (t, w) in self._ring if t >= t_end - dur]
        arr = np.array(sel)
        w = arr.mean(axis=0)
        return t_end, (w - self._bias if net else w), len(sel)

    def latest_before(self, timestamp, net=False):
        """Last received frame at/before a policy deadline, without future leakage."""
        if not np.isfinite(timestamp):
            raise ValueError("Deadline must be finite")
        with self._lock:
            for stamp, value in reversed(self._ring):
                if stamp <= timestamp:
                    return stamp, value.copy() - self._bias if net else value.copy()
        return None

    def is_stale(self, max_age=0.05):
        """True if the newest frame is older than max_age s (sensor dropped / unplugged)."""
        with self._lock:
            last = self._last_frame_t
        return last is None or (time.perf_counter() - last) > max_age

    @property
    def count(self):
        with self._lock:
            return self._count

    @property
    def bias(self):
        return self._bias.copy()

    def tare(self, dur=1.0):
        """Average distinct rest frames over dur seconds; store as bias (net-zero at rest).

        Fixed-pose only — like virtual_ft --tare, this zeros the mount self-weight at the
        current wrist orientation. It does NOT gravity-compensate across poses.
        """
        samples, seen = [], -1
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < dur:
            with self._lock:
                c, latest = self._count, self._latest
            if latest is not None and c != seen:
                samples.append(latest[1])
                seen = c
            time.sleep(0.0005)
        if samples:
            self._bias = np.mean(samples, axis=0)
        return self._bias, len(samples)


def _fmt(w):
    return (
        f"F=({w[0]:+7.2f},{w[1]:+7.2f},{w[2]:+7.2f})N  T=({w[3]:+6.3f},{w[4]:+6.3f},{w[5]:+6.3f})Nm"
    )


class _PlotClosed(Exception):
    pass


def _create_plot(window_s, refresh_hz, net):
    if __package__:
        from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot
    else:
        from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot
    return LiveWrenchPlot(window_s=window_s, refresh_hz=refresh_hz, net=net)


def _saved_plot_path(csv_path, net):
    mode = "net" if net else "raw"
    return csv_path.with_name(f"{csv_path.stem}_{mode}.png")


def _save_recorded_plot(csv_path, net):
    """Render the entire closed CSV, not the rolling display's polling buffer."""
    if __package__:
        from scripts.force_sensor.plot_kwr75_csv import load_csv, make_figure
    else:
        from scripts.force_sensor.plot_kwr75_csv import load_csv, make_figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = "net" if net else "raw"
    times, indices, data = load_csv(csv_path, (mode,))
    output = _saved_plot_path(csv_path, net)
    fig = make_figure(times, indices, data, csv_path.name)
    try:
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160)
    finally:
        plt.close(fig)
    return output


def main():
    import argparse

    ap = argparse.ArgumentParser(description="KWR75 read-only reader demo")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--tare", action="store_true", help="zero out rest reading first")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--csv", type=Path, help="save every parsed frame to a new CSV file")
    ap.add_argument(
        "--no-save-plot",
        action="store_true",
        help="disable automatic full-recording PNG export when --csv is used",
    )
    ap.add_argument("--plot", action="store_true", help="show live six-axis rolling curves")
    ap.add_argument(
        "--plot-window",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="visible rolling window, 0.5-300 seconds (default: 10)",
    )
    ap.add_argument(
        "--plot-hz",
        type=float,
        default=10.0,
        metavar="HZ",
        help="plot refresh limit, 1-30 Hz (default: 10); not sensor rate",
    )
    args = ap.parse_args()
    if not math.isfinite(args.secs) or args.secs <= 0:
        ap.error("--secs must be finite and greater than zero")
    from scripts.shared.run_paths import new_output

    args.csv = new_output(args.csv)
    if args.csv is not None and args.csv.exists():
        ap.error(f"CSV file already exists (will not overwrite): {args.csv}")
    save_plot = args.csv is not None and not args.no_save_plot
    if save_plot and _saved_plot_path(args.csv, args.tare).exists():
        ap.error(
            f"PNG file already exists (will not overwrite): {_saved_plot_path(args.csv, args.tare)}"
        )
    if not math.isfinite(args.plot_window) or not 0.5 <= args.plot_window <= 300:
        ap.error("--plot-window must be between 0.5 and 300 seconds")
    if not math.isfinite(args.plot_hz) or not 1 <= args.plot_hz <= 30:
        ap.error("--plot-hz must be between 1 and 30 Hz")

    r = Kwr75Reader(args.port)
    recorder = None
    plot = None
    exit_code = 0
    interrupted = threading.Event()
    # Defer SIGINT to polling boundaries so a batch is never abandoned mid-write.
    old_sigint = signal.signal(signal.SIGINT, lambda *_: interrupted.set())

    def check_interrupt():
        if interrupted.is_set():
            raise KeyboardInterrupt
        if plot is not None:
            plot.process_events()
            if plot.closed:
                raise _PlotClosed

    try:
        if save_plot:
            try:
                import matplotlib
            except ImportError as exc:
                raise RuntimeError(
                    "PNG export requires matplotlib: python -m pip install matplotlib; "
                    "or use --no-save-plot to record CSV only"
                ) from exc
        if args.plot:
            plot = _create_plot(args.plot_window, args.plot_hz, args.tare)
            plot.open()
            check_interrupt()
        r.start()
        # wait for first frame
        t0 = time.perf_counter()
        while r.latest() is None and time.perf_counter() - t0 < 2.0:
            check_interrupt()
            time.sleep(0.01)
        check_interrupt()
        if r.latest() is None:
            raise RuntimeError("no frames within 2 s - check power / port")
        if args.tare:
            print("Taring for 1 s; keep the tool still and unloaded ...", flush=True)
            bias, n = r.tare(1.0)
            print(f"tared over {n} frames: bias {_fmt(bias)}")
        check_interrupt()
        if args.csv is not None:
            if r.is_stale():
                raise RuntimeError("Sensor data is stale; CSV recording not started")
            recorder = r.start_csv(args.csv)
            print(f"CSV recording: {recorder.path}")

        print(f"streaming {args.secs:.0f}s (net={'yes' if args.tare else 'raw'}) ...")
        c0, t0 = r.count, time.perf_counter()
        peak = np.zeros(6)
        raw_s, win_s, seen = [], [], -1
        while time.perf_counter() - t0 < args.secs:
            check_interrupt()
            r.flush_csv()
            if plot is not None and r._read_error is not None:
                raise RuntimeError(f"Sensor read failed: {r._read_error}") from r._read_error
            got = r.latest(net=args.tare)
            if got is not None:
                w = got[1]
                peak = np.maximum(peak, np.abs(w))
                if r.count != seen:  # de-dup: one raw sample per new batch
                    raw_s.append(w)
                    seen = r.count
            win = r.window_mean(0.01, net=args.tare)  # 100 Hz decimated reading
            if win is not None:
                win_s.append(win[1])
            if plot is not None:
                sample = (got[0] - t0, got[1]) if got is not None else None
                plot.update(time.perf_counter() - t0, sample, stale=r.is_stale())
            time.sleep(0.01)
            print(
                "  " + _fmt(w) + ("  [STALE]" if r.is_stale() else "        "), end="\r", flush=True
            )
        check_interrupt()
        dt = time.perf_counter() - t0
        print()
        print(f"rate ~{(r.count - c0) / dt:.0f} Hz over {dt:.1f}s")
        print("peak |.|: " + _fmt(peak))
        if raw_s:
            s1 = np.std(np.array(raw_s), axis=0)
            print("std polled: " + "  ".join(f"{a} {v:.3f}" for a, v in zip(AXES, s1)))
        if win_s:
            s2 = np.std(np.array(win_s), axis=0)
            print(
                "std window: "
                + "  ".join(f"{a} {v:.3f}" for a, v in zip(AXES, s2))
                + "  (window_mean 10ms)"
            )
    except _PlotClosed:
        print("\nPlot closed; stopping acquisition and saving captured frames ...")
    except KeyboardInterrupt:
        print("\nInterrupted; saving captured CSV frames ..." if recorder else "\nInterrupted.")
        exit_code = 130
    except (OSError, RuntimeError, ImportError, serial.SerialException) as exc:
        print(f"\nError: {exc}")
        exit_code = 1
    finally:
        try:
            r.stop()
            if recorder is not None and r._read_error is not None:
                print(f"\nError: sensor disconnected; CSV may be incomplete: {r._read_error}")
                exit_code = 1
        except (OSError, RuntimeError, serial.SerialException) as exc:
            print(f"\nError while saving/closing: {exc}; CSV may be incomplete")
            exit_code = 1
        finally:
            try:
                if recorder is not None:
                    print(f"\nCSV: {recorder.path} ({recorder.rows} rows written)")
                if plot is not None:
                    plot.close()
                if save_plot and recorder is not None:
                    if exit_code not in (0, 130):
                        print("PNG skipped: acquisition/save failed; check CSV completeness.")
                    elif recorder.rows == 0:
                        print("PNG skipped: CSV has no data rows.")
                    else:
                        try:
                            print("Saving full-recording curve PNG ...", flush=True)
                            output = _save_recorded_plot(recorder.path, args.tare)
                            print(f"PNG: {output}", flush=True)
                        except (OSError, RuntimeError, ImportError, ValueError, csv.Error) as exc:
                            print(f"PNG export failed: {exc}; recorded CSV is preserved.")
                            exit_code = 1
            finally:
                signal.signal(signal.SIGINT, old_sigint)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
