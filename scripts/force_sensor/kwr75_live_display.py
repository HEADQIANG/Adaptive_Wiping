#!/usr/bin/env python3
"""Live numeric and trend display for a KWR75 six-axis F/T sensor.

Windows example:
    python kwr75_live_display.py --port COM3

The display is read-only: it only starts and stops the KWR75 stream. It does
not communicate with, or command, any connected robot arm.
"""

import argparse
import time
from collections import deque

import numpy as np
import serial

from scripts.force_sensor.kwr75_reader import AXES, Kwr75Reader

FORCE_COLORS = ("#d92d20", "#1570ef", "#039855")
TORQUE_COLORS = ("#f79009", "#175cd3", "#12b76a")
UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")


def _show_error(message):
    import tkinter as tk

    root = tk.Tk()
    root.title("KWR75 Connection Error")
    root.configure(padx=24, pady=20, background="#f5f7fa")
    tk.Label(
        root,
        text="KWR75 connection failed",
        font=("Segoe UI", 15, "bold"),
        background="#f5f7fa",
        foreground="#b42318",
    ).pack(anchor="w")
    tk.Label(
        root,
        text=message,
        justify="left",
        wraplength=520,
        font=("Segoe UI", 10),
        background="#f5f7fa",
    ).pack(anchor="w", pady=(10, 16))
    tk.Button(root, text="Close", command=root.destroy, width=12).pack(anchor="e")
    root.mainloop()


class LiveDisplay:
    """Tk + Matplotlib display refreshed from Kwr75Reader's latest frame."""

    def __init__(self, reader, history, tare):
        import tkinter as tk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.reader = reader
        self.history = history
        self.tare_requested = tare
        self.tared = not tare
        self.last_sample_t = None
        self.started_at = time.perf_counter()
        self.times = deque()
        self.samples = deque()
        self.display_running = False

        self.root = tk.Tk()
        self.root.title("KWR75 Live Force / Torque")
        self.root.minsize(1000, 700)
        self.root.configure(background="#f5f7fa")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        header = tk.Frame(self.root, background="#ffffff", padx=24, pady=16)
        header.pack(fill="x")
        tk.Label(
            header,
            text="KWR75 Live Force / Torque",
            font=("Segoe UI", 17, "bold"),
            background="#ffffff",
            foreground="#101828",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 12))

        self.value_labels = []
        for col, (axis, unit) in enumerate(zip(AXES, UNITS)):
            cell = tk.Frame(header, background="#ffffff")
            cell.grid(row=1, column=col, sticky="ew", padx=(0, 20) if col < 5 else 0)
            header.grid_columnconfigure(col, weight=1)
            tk.Label(
                cell,
                text=axis,
                font=("Segoe UI", 10, "bold"),
                background="#ffffff",
                foreground="#475467",
            ).pack(anchor="w")
            value = tk.Label(
                cell,
                text="--",
                font=("Consolas", 16, "bold"),
                background="#ffffff",
                foreground="#101828",
            )
            value.pack(anchor="w", pady=(2, 0))
            tk.Label(
                cell, text=unit, font=("Segoe UI", 9), background="#ffffff", foreground="#667085"
            ).pack(anchor="w")
            self.value_labels.append(value)

        self.status = tk.Label(
            header,
            text="Paused | click 开始显示数据",
            anchor="w",
            font=("Segoe UI", 10),
            background="#ffffff",
            foreground="#9a6700",
        )
        self.status.grid(row=2, column=0, columnspan=6, sticky="w", pady=(14, 0))

        controls = tk.Frame(header, background="#ffffff")
        controls.grid(row=3, column=0, columnspan=6, sticky="w", pady=(12, 0))
        self.start_button = tk.Button(
            controls,
            text="开始显示数据",
            command=self._start_display,
            font=("Segoe UI", 10, "bold"),
            width=14,
            cursor="hand2",
            background="#1570ef",
            foreground="#ffffff",
            activebackground="#175cd3",
            activeforeground="#ffffff",
            relief="flat",
            padx=8,
            pady=5,
        )
        self.start_button.pack(side="left", padx=(0, 8))
        self.pause_button = tk.Button(
            controls,
            text="暂停显示数据",
            command=self._pause_display,
            font=("Segoe UI", 10, "bold"),
            width=14,
            cursor="hand2",
            background="#ffffff",
            foreground="#344054",
            activebackground="#eaecf0",
            activeforeground="#101828",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            state="disabled",
        )
        self.pause_button.pack(side="left")

        figure = Figure(figsize=(10, 6.2), dpi=100, facecolor="#f5f7fa")
        self.force_ax = figure.add_subplot(211)
        self.torque_ax = figure.add_subplot(212)
        figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.09, hspace=0.38)

        self.force_lines = self._setup_axis(self.force_ax, "Force", "N", FORCE_COLORS)
        self.torque_lines = self._setup_axis(self.torque_ax, "Torque", "Nm", TORQUE_COLORS)
        self.torque_ax.set_xlabel("Time (s)")

        canvas = FigureCanvasTkAgg(figure, master=self.root)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=18, pady=16)
        self.canvas = canvas
        self.closed = False

    @staticmethod
    def _setup_axis(axis, title, unit, colors):
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#101828", pad=8)
        axis.set_ylabel(unit)
        axis.grid(True, color="#d0d5dd", linewidth=0.7, alpha=0.75)
        axis.set_facecolor("#ffffff")
        axis.tick_params(colors="#475467", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#98a2b3")
        indices = range(3) if title == "Force" else range(3, 6)
        return [
            axis.plot([], [], color=color, linewidth=1.7, label=AXES[index])[0]
            for index, color in zip(indices, colors)
        ]

    @staticmethod
    def _limits(data):
        low, high = float(np.min(data)), float(np.max(data))
        span = high - low
        padding = max(0.02, span * 0.15, max(abs(low), abs(high)) * 0.03)
        return low - padding, high + padding

    def _append_sample(self, sample_t, wrench):
        if self.last_sample_t is not None and sample_t <= self.last_sample_t:
            return
        self.last_sample_t = sample_t
        now = sample_t - self.started_at
        self.times.append(now)
        self.samples.append(np.asarray(wrench, dtype=float).copy())
        while self.times and now - self.times[0] > self.history:
            self.times.popleft()
            self.samples.popleft()

    def _redraw_plot(self):
        if not self.samples:
            return
        ts = np.asarray(self.times)
        data = np.asarray(self.samples)
        for line, values in zip(self.force_lines, data[:, :3].T):
            line.set_data(ts, values)
        for line, values in zip(self.torque_lines, data[:, 3:].T):
            line.set_data(ts, values)

        right = max(self.history, ts[-1])
        left = max(0.0, right - self.history)
        self.force_ax.set_xlim(left, right)
        self.torque_ax.set_xlim(left, right)
        self.force_ax.set_ylim(*self._limits(data[:, :3]))
        self.torque_ax.set_ylim(*self._limits(data[:, 3:]))
        self.force_ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
        self.torque_ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
        self.canvas.draw_idle()

    def _start_display(self):
        self.display_running = True
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.status.configure(text="Starting display...", foreground="#1570ef")

    def _pause_display(self):
        self.display_running = False
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.status.configure(
            text="Paused | acquisition continues in background", foreground="#667085"
        )

    def _refresh(self):
        if self.closed:
            return
        if not self.display_running:
            self.root.after(50, self._refresh)
            return
        latest = self.reader.latest(net=self.tared)
        if latest is None:
            self.status.configure(text="Waiting for sensor data...", foreground="#9a6700")
        else:
            if self.tare_requested and not self.tared:
                self.status.configure(
                    text="Taring at the current fixed pose...", foreground="#9a6700"
                )
                self.reader.tare(1.0)
                self.tared = True
                latest = self.reader.latest(net=True)

            _, wrench = latest
            for label, value in zip(self.value_labels, wrench):
                label.configure(text=f"{value:+.3f}")

            averaged = self.reader.window_mean(0.01, net=self.tared)
            if averaged is not None:
                sample_t, sample, _ = averaged
                self._append_sample(sample_t, sample)
                self._redraw_plot()

            if self.reader.is_stale(max_age=0.25):
                self.status.configure(text="No new host data for over 250 ms", foreground="#b42318")
            else:
                self.status.configure(
                    text=f"Receiving  |  {self.reader.count} frames  |  {self.history:.0f} s history",
                    foreground="#027a48",
                )
        self.root.after(50, self._refresh)

    def _close(self):
        self.closed = True
        self.root.quit()
        self.root.destroy()

    def run(self):
        self._refresh()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="KWR75 live numeric and trend display")
    parser.add_argument("--port", default="COM3", help="sensor serial port (default: COM3)")
    parser.add_argument(
        "--history", type=float, default=10.0, help="visible plot history in seconds (default: 10)"
    )
    parser.add_argument(
        "--tare",
        action="store_true",
        help="zero the current fixed-pose reading after the first frame",
    )
    args = parser.parse_args()
    if args.history <= 0:
        parser.error("--history must be greater than zero")

    reader = Kwr75Reader(args.port)
    try:
        reader.start()
    except serial.SerialException as exc:
        _show_error(f"Could not open {args.port}.\n\n{exc}")
        return 1

    try:
        LiveDisplay(reader, args.history, args.tare).run()
        return 0
    finally:
        reader.stop()


if __name__ == "__main__":
    raise SystemExit(main())
