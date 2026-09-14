"""Read-only log display, process lifecycle and hardware-free plotting checks."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np

from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot
from scripts.real_training.exploration_live_plot import (
    ExplorationLivePlot,
    ExplorationTrace,
    LogTail,
    save_exploration_plot,
)


def sample(t, phase="exploration"):
    return {
        "event": "sample", "phase": phase, "protocol_time_s": t if phase == "exploration" else None,
        "force": {
            "sensor_receive_perf_s": 100 + t,
            "tared_sensor_wrench_si": [np.sin(t + axis) for axis in range(6)],
            "raw_sensor_wrench_si": [1000] * 6,
            "bias_corrected_sensor_wrench_si": [900] * 6,
        },
    }


class ExplorationPlotTests(unittest.TestCase):
    def setUp(self):
        self.plot = LiveWrenchPlot(window_s=10, refresh_hz=10, net=True)
        self.addCleanup(self.plot.close)
        self.trace = ExplorationTrace(self.plot)

    def start(self):
        self.trace.ingest({"event": "phase_start", "phase": "exploration", "perf_s": 100})

    def test_tared_six_axes_and_nonblank_render(self):
        self.start()
        for t in np.arange(1, 401) / 100:
            self.trace.ingest(sample(float(t)))
        self.trace.refresh()
        self.plot.fig.canvas.draw()
        self.assertEqual(len(self.plot.fig.axes), 6)
        self.assertIn("Tared", self.plot.fig._suptitle.get_text())
        self.assertEqual(len(self.plot.lines[0].get_xdata()), 400)
        self.assertAlmostEqual(self.plot.lines[0].get_xdata()[-1], 4.0)
        for line, axis in zip(self.plot.lines, (0, 3, 1, 4, 2, 5)):
            np.testing.assert_allclose(line.get_ydata(), np.sin(np.asarray(line.get_xdata()) + axis))
        pixels = np.asarray(self.plot.fig.canvas.buffer_rgba())[:, :, :3]
        self.assertGreater(float(pixels.std()), 10)

    def test_phases_retract_and_handoff_keep_curves(self):
        self.trace.ingest({"event": "tare_start"})
        self.assertIn("tare", self.trace.status)
        self.trace.ingest({"event": "tare_complete"})
        self.start()
        for t, expected in ((1, "Press"), (2.5, "+Y slide"), (3.5, "-Y return")):
            self.trace.ingest(sample(t))
            self.assertEqual(self.trace.status, expected)
        self.trace.ingest({"event": "phase_start", "phase": "retract", "perf_s": 104})
        self.trace.ingest(sample(5, "retract"))
        self.assertEqual(self.trace.status, "Retract")
        self.assertEqual(self.trace.elapsed, 3.5)
        self.assertFalse(self.trace.active)
        self.trace.ingest({"event": "motion_complete"})
        self.assertFalse(self.trace.active)
        self.assertIn("handoff", self.trace.status)
        self.assertEqual(len(self.plot._samples), 3)
        self.trace.ingest({"event": "aborted"})
        self.trace.refresh()
        self.assertIn("ABORTED", self.plot.status.get_text())

    def test_repeated_sensor_timestamps_are_not_duplicated(self):
        self.start()
        row = sample(0.1)
        self.trace.ingest(row)
        self.trace.ingest(copy.deepcopy(row))
        self.assertEqual(len(self.plot._samples), 1)

    def test_slow_run_uses_receive_time_not_protocol_time(self):
        self.start()
        row = sample(1.0)
        row["force"]["sensor_receive_perf_s"] = 105.0
        self.trace.ingest(row)
        self.trace.refresh()
        self.assertEqual(self.trace.elapsed, 5.0)
        self.assertEqual(self.trace.status, "Press")
        self.assertEqual(self.plot.lines[0].get_xdata()[-1], 5.0)

    def test_missing_or_invalid_tared_values_never_fall_back_to_raw(self):
        self.start()
        for values in (None, [], [0] * 5, [float("nan")] * 6, ["bad"] * 6):
            row = sample(0.1)
            row["force"]["tared_sensor_wrench_si"] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "no raw fallback"):
                self.trace.ingest(row)
        self.assertEqual(len(self.plot._samples), 0)

    def test_no_new_log_status_and_recovery(self):
        self.start()
        with patch("scripts.real_training.exploration_live_plot.time.perf_counter", return_value=1):
            self.trace.ingest(sample(0.1))
        with patch("scripts.real_training.exploration_live_plot.time.perf_counter", return_value=2):
            self.trace.refresh()
            self.assertIn("No new logged samples", self.plot.status.get_text())
            self.trace.ingest(sample(0.2))
            self.trace.refresh()
            self.assertIn("Press", self.plot.status.get_text())

    def test_tail_waits_for_file_and_complete_lines_without_modifying_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            tail = LogTail(path)
            self.addCleanup(tail.close)
            self.assertEqual(tail.read(), [])
            encoded = json.dumps(sample(0.1))
            path.write_text(encoded[:30])
            self.assertEqual(tail.read(), [])
            with path.open("a") as stream:
                stream.write(encoded[30:] + "\n" + json.dumps(sample(0.2)) + "\n")
            before = path.read_bytes()
            self.assertEqual(tail.read(limit=1), [sample(0.1)])
            self.assertEqual(tail.read(), [sample(0.2)])
            self.assertEqual(tail.read(), [])
            self.assertEqual(path.read_bytes(), before)


class PlotProcessTests(unittest.TestCase):
    def test_export_excludes_retract_and_keeps_last_exploration_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            rows = [{"event": "phase_start", "phase": "exploration", "perf_s": 100},
                    sample(0.01), sample(4), sample(6, "retract"),
                    {"event": "session_complete"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            original = path.read_bytes()
            from matplotlib.axes import Axes
            original_plot = Axes.plot
            observed = []

            def capture(ax, x, y, **kwargs):
                observed.append((list(x), list(y)))
                return original_plot(ax, x, y, **kwargs)

            with patch.object(Axes, "plot", capture):
                output = save_exploration_plot(path)
            self.assertEqual(output, path.with_suffix(".png"))
            self.assertEqual(len(observed), 6)
            np.testing.assert_allclose(observed[0][0], [0.01, 4])
            np.testing.assert_allclose(observed[0][1], np.sin([0.01, 4]))
            import matplotlib.image as mpimg
            self.assertGreater(float(mpimg.imread(output).std()), 0.05)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                save_exploration_plot(path)

    def test_empty_recording_does_not_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.jsonl"
            self.assertIsNone(save_exploration_plot(path))
            self.assertFalse(path.with_suffix(".png").exists())

    def test_headless_failure_cleans_up_without_creating_log(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, MPLBACKEND="Agg"):
            path = Path(folder) / "not-created.jsonl"
            plot = ExplorationLivePlot(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "desktop"):
                    plot.start()
                self.assertFalse(plot.process.is_alive())
                self.assertFalse(path.exists())
            finally:
                plot.close()

    @unittest.skipUnless(os.environ.get("EXPLORATION_PLOT_GUI_TEST") == "1", "Opt-in desktop test")
    def test_desktop_process_reads_synthetic_log_and_closes(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, MPLBACKEND="TkAgg"):
            path = Path(folder) / "synthetic.jsonl"
            plot = ExplorationLivePlot(path)
            try:
                plot.start()
                self.assertTrue(plot.process.is_alive())
                rows = [{"event": "phase_start", "phase": "exploration", "perf_s": 100}]
                rows.extend(sample(i / 100) for i in range(1, 401))
                rows.append({"event": "motion_complete"})
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                # Let the GUI handle a frame; this process has no hardware objects.
                plot.process.join(timeout=1)
                self.assertTrue(plot.process.is_alive())
            finally:
                plot.close()
            self.assertFalse(plot.process.is_alive())
            self.assertTrue(path.with_suffix(".png").exists())


if __name__ == "__main__":
    unittest.main()
