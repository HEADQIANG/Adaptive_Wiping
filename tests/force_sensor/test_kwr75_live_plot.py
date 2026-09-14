"""Rolling display tests on a noninteractive canvas; no hardware is used."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backend_bases import CloseEvent

from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot


class LivePlotTests(unittest.TestCase):
    def make_plot(self, **kwargs):
        plot = LiveWrenchPlot(**kwargs)
        self.addCleanup(plot.close)
        return plot

    def test_six_axes_and_nonblank_render(self):
        plot = self.make_plot()
        plot.update(0.1, (0.1, np.arange(6, dtype=float)))
        plot.update(0.3, (0.3, np.arange(6, dtype=float) + 1))
        plot.fig.canvas.draw()
        self.assertEqual(len(plot.fig.axes), 6)
        self.assertEqual(
            [ax.get_ylabel() for ax in plot.panels.flat],
            ["Fx (N)", "Tx (Nm)", "Fy (N)", "Ty (Nm)", "Fz (N)", "Tz (Nm)"],
        )
        np.testing.assert_allclose(plot.lines[1].get_ydata(), [3, 4])
        pixels = np.asarray(plot.fig.canvas.buffer_rgba())[:, :, :3]
        self.assertGreater(float(pixels.std()), 10)

    def test_window_scrolls_and_expires_old_data_during_stale(self):
        plot = self.make_plot(window_s=1)
        plot.update(0.1, (0.1, np.ones(6)))
        plot.update(0.5, (0.5, np.ones(6) * 2))
        plot.update(1.2, (1.2, np.ones(6) * 3))
        np.testing.assert_allclose(plot.panels[0, 0].get_xlim(), [0.2, 1.2])
        np.testing.assert_allclose(plot.lines[0].get_xdata(), [0.5, 1.2])
        plot.update(3, (1.2, np.ones(6) * 3), stale=True)
        self.assertEqual(len(plot._samples), 0)
        self.assertIn("STALE", plot.status.get_text())

    def test_repeated_samples_not_duplicated_and_stale_recovery_breaks_line(self):
        plot = self.make_plot()
        values = np.ones(6)
        plot.update(0.1, (0.1, values))
        values[:] = 99
        plot.update(0.3, (0.1, values), stale=True)
        self.assertEqual(len(plot._samples), 1)
        self.assertEqual(plot._samples[0][1][0], 1)
        plot.update(0.5, (0.5, values))
        self.assertTrue(np.isnan(plot.lines[0].get_ydata()[1]))
        self.assertIn("Streaming", plot.status.get_text())

    def test_refresh_is_throttled_but_samples_are_buffered(self):
        plot = self.make_plot(refresh_hz=10)
        with patch.object(plot.fig.canvas, "draw_idle") as draw:
            plot.update(0, (0, np.zeros(6)))
            plot.update(0.01, (0.01, np.ones(6)))
            self.assertEqual(draw.call_count, 1)
            self.assertEqual(len(plot._samples), 2)
            plot.update(0.11, (0.11, np.ones(6)))
            self.assertEqual(draw.call_count, 2)

    def test_bounded_buffer(self):
        plot = self.make_plot(window_s=1, refresh_hz=1)
        with patch.object(plot.fig.canvas, "draw_idle"):
            for index in range(1000):
                plot.update(index / 100, (index / 100, np.ones(6)))
        self.assertLessEqual(len(plot._samples), 102)

    def test_force_draw_flushes_last_buffered_sample(self):
        plot = self.make_plot(refresh_hz=10)
        plot.update(0, (0, np.zeros(6)))
        plot.update(0.01, (0.01, np.ones(6)))
        self.assertEqual(len(plot.lines[0].get_xdata()), 1)
        plot.update(0.01, None, force_draw=True)
        np.testing.assert_allclose(plot.lines[0].get_xdata(), [0, 0.01])
        np.testing.assert_allclose(plot.lines[0].get_ydata(), [0, 1])

    def test_nonfinite_values_are_flagged(self):
        plot = self.make_plot()
        plot.update(0.1, (0.1, np.full(6, np.nan)))
        self.assertIn("INVALID", plot.status.get_text())

    def test_close_event_and_headless_failure(self):
        plot = self.make_plot()
        with self.assertRaisesRegex(RuntimeError, "desktop"):
            plot.open()
        event = CloseEvent("close_event", plot.fig.canvas)
        plot.fig.canvas.callbacks.process("close_event", event)
        self.assertTrue(plot.closed)


if __name__ == "__main__":
    unittest.main()
