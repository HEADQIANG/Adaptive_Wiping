"""Read-only signed extrema for synthetic KWR75 CSV records."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import scripts.force_sensor.kwr75_net_peaks as peaks


class NetPeakTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "sensor.csv"
        self.header = ["frame_index", "elapsed_s"] + [
            f"net_{axis}_{unit}" for axis, unit in zip(peaks.AXES, peaks.UNITS)
        ]
        self.rows = [
            [1001, 0.01, -10, 2, -18, -1.1, 0.8, -0.12],
            [1002, 0.01, 10, -12, 0.2, 0.7, -0.7, 0.1],
            [1003, 0.02, 1, 1, -7, 0, 0, 0],
        ]
        self.write_csv()

    def write_csv(self):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header)
            writer.writerows(self.rows)

    def test_signed_maxima_and_minima_for_each_axis(self):
        result = peaks.summarize(self.path)
        self.assertEqual([row["axis"] for row in result], list(peaks.AXES))
        self.assertEqual([row["unit"] for row in result], list(peaks.UNITS))
        self.assertEqual([row["max"] for row in result], [10, 2, 0.2, 0.7, 0.8, 0.1])
        self.assertEqual([row["min"] for row in result], [-10, -12, -18, -1.1, -0.7, -0.12])

    def test_single_nonzero_frame_does_not_include_zero_or_take_absolute_value(self):
        self.rows = [self.rows[0]]
        self.write_csv()
        result = peaks.summarize(self.path)
        self.assertEqual([row["max"] for row in result], [-10, 2, -18, -1.1, 0.8, -0.12])
        self.assertEqual([row["min"] for row in result], [-10, 2, -18, -1.1, 0.8, -0.12])

    def test_single_zero_frame(self):
        self.rows = [[1, 0, 0, 0, 0, 0, 0, 0]]
        self.write_csv()
        result = peaks.summarize(self.path)
        self.assertTrue(all(row["max"] == row["min"] == 0 for row in result))

    def test_cli_preserves_input_and_labels_values(self):
        original = self.path.read_bytes()
        output = io.StringIO()
        with redirect_stdout(output):
            result = peaks.main([str(self.path)])
        self.assertEqual(result, 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[0].split(), ["Axis", "Unit", "Max", "Min"])
        self.assertEqual(lines[3].split(), ["Fz", "N", "+0.200000", "-18.000000"])
        self.assertIn("-18.000000", output.getvalue())
        self.assertEqual(self.path.read_bytes(), original)

    def test_missing_file_reports_failure(self):
        output = io.StringIO()
        with redirect_stderr(output):
            result = peaks.main([str(self.path.parent / "missing.csv")])
        self.assertEqual(result, 1)
        self.assertIn("Error:", output.getvalue())

    def test_empty_and_nonfinite_data_rejected(self):
        for rows in ([], [[1, 0.01, float("nan"), 0, 0, 0, 0, 0]]):
            with self.subTest(rows=rows):
                self.rows = rows
                self.write_csv()
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(peaks.main([str(self.path)]), 1)


if __name__ == "__main__":
    unittest.main()
