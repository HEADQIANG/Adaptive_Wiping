"""Main-thread rolling display of polled wrench samples; no serial or file I/O."""

import math
from collections import deque

import numpy as np

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
PANEL_ORDER = (0, 3, 1, 4, 2, 5)


class LiveWrenchPlot:
    def __init__(self, window_s=10.0, refresh_hz=10.0, net=True):
        if not math.isfinite(window_s) or not 0.5 <= window_s <= 300:
            raise ValueError("Plot window must be between 0.5 and 300 seconds")
        if not math.isfinite(refresh_hz) or not 1 <= refresh_hz <= 30:
            raise ValueError("Plot refresh must be between 1 and 30 Hz")
        import matplotlib.pyplot as plt

        self._plt = plt
        self.window_s = window_s
        self._period = 1.0 / refresh_hz
        self._samples = deque(maxlen=math.ceil(window_s * 200) + 2)
        self._last_sample_t = None
        self._gap_pending = False
        self._last_draw = -math.inf
        self.closed = False
        self.fig, panels = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
        self.panels = panels
        self.fig.subplots_adjust(
            left=0.085, right=0.97, top=0.86, bottom=0.09, hspace=0.22, wspace=0.25
        )
        self.fig.canvas.manager.set_window_title("KWR75 live force / torque")
        self.fig.suptitle(f"KWR75 | {'Net' if net else 'Raw'} force / torque", fontsize=14, y=0.98)
        self.status = self.fig.text(0.085, 0.91, "Waiting for sensor", fontsize=10, color="#606060")
        self.lines = []
        for channel, ax in zip(PANEL_ORDER, panels.flat):
            unit = "N" if channel < 3 else "Nm"
            ax.set_ylabel(f"{AXES[channel]} ({unit})")
            ax.set_xlim(0, window_s)
            ax.set_ylim((-1, 1) if channel < 3 else (-0.1, 0.1))
            ax.set_autoscaley_on(True)
            ax.margins(y=0.15)
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            color = "#137e79" if channel < 3 else "#bd572b"
            (line,) = ax.plot([], [], color=color, linewidth=1.1)
            self.lines.append(line)
        for ax in panels[-1]:
            ax.set_xlabel("Elapsed time (s)")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event):
        self.closed = True

    def open(self):
        if self.fig.canvas.required_interactive_framework is None:
            raise RuntimeError(
                "--plot requires a desktop Tk/Qt backend; install python3.10-tk "
                "and run in a desktop terminal (MPLBACKEND=TkAgg), or omit --plot"
            )
        self._plt.show(block=False)
        self.fig.canvas.draw()
        self.process_events()

    def process_events(self):
        if not self.closed:
            self.fig.canvas.flush_events()
            if not self._plt.fignum_exists(self.fig.number):
                self.closed = True

    def update(self, elapsed, sample, stale=False, force_draw=False):
        if self.closed:
            return
        if stale:
            self._gap_pending = True
        invalid = False
        if sample is not None:
            t, wrench = sample
            invalid = not np.isfinite(wrench).all()
            if t >= 0 and (self._last_sample_t is None or t > self._last_sample_t):
                if self._last_sample_t is not None and self._gap_pending:
                    self._samples.append((t, np.full(6, np.nan)))
                self._samples.append((t, np.array(wrench, dtype=float, copy=True)))
                self._last_sample_t = t
                self._gap_pending = stale
        left = max(0, elapsed - self.window_s)
        while self._samples and self._samples[0][0] < left:
            self._samples.popleft()
        if not force_draw and elapsed - self._last_draw < self._period:
            return
        self._last_draw = elapsed
        if self._samples:
            times, values = zip(*self._samples)
            values = np.asarray(values)
        else:
            times, values = [], np.empty((0, 6))
        for channel, ax, line in zip(PANEL_ORDER, self.panels.flat, self.lines):
            line.set_data(times, values[:, channel])
            ax.set_xlim(left, left + self.window_s)
            if np.isfinite(values[:, channel]).any():
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
        state = (
            "STALE: no new frame for >50 ms"
            if stale
            else "INVALID: non-finite sensor value"
            if invalid
            else "Streaming"
        )
        self.status.set_text(f"{state} | {elapsed:.1f} s")
        self.status.set_color("#b42318" if stale or invalid else "#137e79")
        self.fig.canvas.draw_idle()

    def close(self):
        self.closed = True
        self._plt.close(self.fig)
