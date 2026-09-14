"""Offline plotting tests; fixtures are synthetic and no serial port is opened."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scripts.force_sensor.plot_kwr75_csv as plotter


class PlotTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.csv = Path(folder.name) / "sensor.csv"
        self.png = Path(folder.name) / "plots" / "sensor.png"
        self.header = ["frame_index", "elapsed_s"] + [
            f"{mode}_{axis}_{unit}"
            for mode in ("raw", "net")
            for axis, unit in zip(plotter.AXES, plotter.UNITS)
        ]
        self.samples = [
            [i + 1, (i // 9 + 1) * 0.009, *([i * 0.01 + 1] * 6), *([i * 0.01] * 6)]
            for i in range(90)
        ]
        self.write_csv()

    def write_csv(self, rows=None, header=None):
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header if header is None else header)
            writer.writerows(self.samples if rows is None else rows)

    def run_cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = plotter.main([str(self.csv), "--output", str(self.png), *args])
        return result, output.getvalue()

    def test_load_preserves_all_frames_and_repeated_times(self):
        times, indices, data = plotter.load_csv(self.csv, ("raw", "net"))
        self.assertEqual(len(times), 90)
        self.assertEqual(len(np.unique(times)), 10)
        np.testing.assert_array_equal(indices, np.arange(1, 91))
        np.testing.assert_allclose(data["raw"] - data["net"], 1)

    def test_trailing_mean_is_frame_weighted_and_batch_consistent(self):
        times = np.array([0, 0, 0.005, 0.014, 0.014, 0.03])
        values = np.arange(6, dtype=float)[:, None]
        ends, means = plotter.window_mean(times, values, 0.01)
        np.testing.assert_allclose(ends, [0, 0.005, 0.014, 0.03])
        np.testing.assert_allclose(means[:, 0], [0.5, 1, 3, 5])

    def test_export_png_and_preserve_csv(self):
        original = self.csv.read_bytes()
        result, output = self.run_cli("--mode", "both", "--window-ms", "10")
        self.assertEqual(result, 0, output)
        self.assertTrue(self.png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        image = plt.imread(self.png)
        self.assertGreater(image.shape[0], 1000)
        self.assertGreater(np.std(image[:, :, :3]), 0.03)
        self.assertEqual(self.csv.read_bytes(), original)

    def test_six_panels_and_data_gaps(self):
        times = np.array([0.01, 0.02, 0.2])
        indices = np.array([1, 2, 4])
        fig = plotter.make_figure(times, indices, {"net": np.ones((3, 6))}, "test.csv", 10)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual(
            [ax.get_ylabel() for ax in fig.axes],
            ["Fx (N)", "Tx (Nm)", "Fy (N)", "Ty (Nm)", "Fz (N)", "Tz (Nm)"],
        )
        for ax in fig.axes:
            self.assertTrue(np.isnan(ax.lines[0].get_xdata()).any())
            self.assertTrue(np.isnan(ax.lines[1].get_xdata()).any())

    def test_time_range_and_single_frame(self):
        self.write_csv(rows=[self.samples[0]])
        result, output = self.run_cli("--start", "0.009", "--end", "0.009", "--window-ms", "10")
        self.assertEqual(result, 0, output)
        self.assertIn("Frames: 1", output)

    def test_no_frames_selected(self):
        result, output = self.run_cli("--start", "10")
        self.assertEqual(result, 1)
        self.assertIn("No frames", output)
        self.assertFalse(self.png.exists())

    def test_invalid_csv_rejected(self):
        invalid_rows = (
            [],
            [[1, "nan", *([0] * 12)]],
            [[1, 0.1, *([float("inf")] * 12)]],
            [[1, 0.1, 0]],
            [[1, 0.2, *([0] * 12)], [2, 0.1, *([0] * 12)]],
            [[2, 0.1, *([0] * 12)], [2, 0.2, *([0] * 12)]],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows):
                self.write_csv(rows=rows)
                with self.assertRaises(ValueError):
                    plotter.load_csv(self.csv, ("net",))

    def test_missing_columns_rejected(self):
        self.write_csv(header=["frame_index", "elapsed_s"], rows=[[1, 0.1]])
        result, output = self.run_cli()
        self.assertEqual(result, 1)
        self.assertIn("Missing CSV columns", output)

    def test_output_not_overwritten(self):
        result, output = self.run_cli()
        self.assertEqual(result, 0, output)
        original = self.png.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli()
        self.assertEqual(self.png.read_bytes(), original)

    def test_invalid_options(self):
        for args in (
            ("--window-ms", "nan"),
            ("--window-ms", "-1"),
            ("--start", "-1"),
            ("--end", "inf"),
            ("--start", "2", "--end", "1"),
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_cli(*args)

    def test_show_without_gui_still_saves_png(self):
        result, output = self.run_cli("--show")
        self.assertEqual(result, 0, output)
        self.assertTrue(self.png.exists())
        self.assertIn("no interactive backend", output)


if __name__ == "__main__":
    unittest.main()
