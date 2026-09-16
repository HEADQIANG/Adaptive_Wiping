"""Software-only tests for manual tared curves and CSV bias freezing."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np

from scripts.force_sensor.kwr75_reader import G, Kwr75Reader
from scripts.force_sensor.live_kwr75_plot import LiveWrenchPlot
from scripts.real_training.manual_live_plot import ManualLivePlot, ManualTrace


def start(t=100):
    return {"event": "start", "start_perf_s": t, "force_recording": "software_tared",
            "tare": {"id": "test", "raw_baseline_si": [10] * 6}}


def sample(t):
    return {"event": "sample", "ft": {"sensor_receive_perf_s": 100 + t,
            "tared_sensor_wrench_si": [float(np.sin(t + i)) for i in range(6)],
            "raw_sensor_wrench_si": [1000] * 6}}


class ManualPlotTests(unittest.TestCase):
    def setUp(self):
        self.plot = LiveWrenchPlot(net=True)
        self.addCleanup(self.plot.close)
        self.trace = ManualTrace(self.plot)

    def test_tared_axes_and_nonblank_render(self):
        self.trace.ingest(start())
        for t in np.arange(1, 1001) / 100:
            self.trace.ingest(sample(float(t)))
        self.trace.refresh()
        self.plot.fig.canvas.draw()
        self.assertEqual(len(self.plot.fig.axes), 6)
        for line, channel in zip(self.plot.lines, (0, 3, 1, 4, 2, 5)):
            np.testing.assert_allclose(line.get_ydata(), np.sin(np.asarray(line.get_xdata()) + channel),
                                       atol=1e-12)
        self.assertGreater(np.asarray(self.plot.fig.canvas.buffer_rgba()).std(), 10)

    def test_tare_missing_or_invalid_never_falls_back_to_raw(self):
        with self.assertRaises(ValueError):
            self.trace.ingest({**start(), "tare": None})
        self.trace.ingest(start())
        for value in (None, [], [float("nan")] * 6):
            row = sample(0.1)
            row["ft"]["tared_sensor_wrench_si"] = value
            with self.assertRaises(ValueError):
                self.trace.ingest(row)

    def test_completion_new_attempt_gap_and_stall(self):
        self.trace.ingest(start())
        self.trace.ingest(sample(0.1))
        self.trace.ingest(sample(0.1))
        self.assertEqual(len(self.plot._samples), 1)
        self.trace.ingest({"event": "finished", "quality": {"passed": False}})
        self.assertFalse(self.trace.active)
        self.assertEqual(self.trace.status, "Timing rejected")
        self.trace.ingest(start(102))
        self.trace.ingest(sample(2.1))
        self.assertTrue(np.isnan(self.plot._samples[-2][1]).all())
        with patch("scripts.real_training.manual_live_plot.time.perf_counter", return_value=1e12):
            self.trace.refresh()
        self.assertIn("No new logged samples", self.plot.status.get_text())


class ManualProcessTests(unittest.TestCase):
    def test_headless_failure_closes_worker(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, MPLBACKEND="Agg"):
            plot = ManualLivePlot(folder)
            try:
                with self.assertRaisesRegex(RuntimeError, "desktop"):
                    plot.start()
                self.assertFalse(plot.process.is_alive())
            finally:
                plot.close()

    @unittest.skipUnless(os.environ.get("MANUAL_PLOT_GUI_TEST") == "1", "Opt-in desktop test")
    def test_desktop_reads_new_attempt_and_closes(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, MPLBACKEND="TkAgg"):
            plot = ManualLivePlot(folder)
            try:
                plot.start()
                path = Path(folder) / "attempt_test.jsonl"
                rows = [start(), *[sample(i / 100) for i in range(1001)],
                        {"event": "finished", "quality": {"passed": True}}]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                plot.process.join(timeout=2)
                self.assertTrue(plot.process.is_alive())
            finally:
                plot.close()
            self.assertFalse(plot.process.is_alive())

    def test_csv_net_matches_tare_and_each_file_freezes_its_bias(self):
        reader = Kwr75Reader()
        self.addCleanup(reader.stop)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sensor.csv"
            bias = [10] * 6
            recorder = reader.start_csv(path, bias=bias)
            bias[0] = 999
            recorder.enqueue(100, 1_000_000_000, 1, [[12 / G] * 6])
            reader.stop_csv()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["raw_Fx_N"]), 12)
            self.assertAlmostEqual(float(row["net_Fx_N"]), 2)
            self.assertEqual(reader.bias.tolist(), [0] * 6)
            reader.start_csv(Path(folder) / "next.csv", bias=[20] * 6)
            reader.stop_csv()
            with self.assertRaises(ValueError):
                reader.start_csv(Path(folder) / "invalid.csv", bias=[float("nan")] * 6)


if __name__ == "__main__":
    unittest.main()
