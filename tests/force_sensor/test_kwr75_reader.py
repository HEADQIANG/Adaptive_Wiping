"""CSV recording tests using fake serial streams; never open hardware ports."""

import csv
import io
import queue
import signal
import struct
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.force_sensor.kwr75_reader as sensor


def frame(values):
    return sensor._HEAD + struct.pack("<6f", *values) + sensor._TAIL


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")
        self.closed = False
        self.commands = []

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.commands.append(data)

    def flush(self):
        pass

    def read(self, size):
        threading.Event().wait(0.001)
        return frame([1, 2, 3, 4, 5, 6]) * 9

    def close(self):
        self.closed = True


class FakePlot:
    def __init__(self):
        self.closed = False
        self.samples = []
        self.close_on_update = False

    def open(self):
        pass

    def process_events(self):
        pass

    def update(self, elapsed, sample, stale=False):
        self.samples.append((elapsed, sample, stale))
        if self.close_on_update:
            self.closed = True

    def close(self):
        self.closed = True


class CsvTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "nested" / "sensor.csv"

    def rows(self):
        with self.path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_all_frames_including_split_frames_and_tare_values(self):
        reader = sensor.Kwr75Reader()
        reader._bias = np.arange(6, dtype=float)
        reader._count = 20
        recorder = reader.start_csv(self.path)
        values = [tuple(range(i, i + 6)) for i in range(30)]
        stream = b"noise" + b"".join(frame(v) for v in values)
        chunks = iter(stream[i : i + 17] for i in range(0, len(stream), 17))

        def read(_size):
            chunk = next(chunks, b"")
            if not chunk:
                reader._stop.set()
            return chunk

        port = FakeSerial()
        port.read = read
        reader._ser = port
        reader._run()
        reader.stop()
        rows = self.rows()
        self.assertEqual(len(rows), 30)
        self.assertEqual(recorder.rows, 30)
        self.assertTrue(port.closed)
        for i, row in enumerate(rows):
            self.assertEqual(int(row["frame_index"]), 21 + i)
            self.assertGreaterEqual(float(row["elapsed_s"]), 0)
            self.assertGreater(int(row["host_time_ns"]), 0)
            self.assertTrue(row["timestamp_utc"].endswith("+00:00"))
            self.assertAlmostEqual(float(row["raw_Fy_N"]), (i + 1) * sensor.G)
            self.assertAlmostEqual(float(row["net_Fy_N"]), (i + 1) * sensor.G - 1)

    def test_batches_share_timestamp_and_bias_is_frozen(self):
        bias = np.ones(6)
        recorder = sensor._CsvRecorder(self.path, bias)
        self.addCleanup(lambda: recorder._file.close())
        bias[:] = 99
        stamp = time.time_ns()
        recorder.enqueue(time.perf_counter(), stamp, 1, [(1,) * 6, (2,) * 6])
        recorder.close()
        rows = self.rows()
        self.assertEqual(rows[0]["host_time_ns"], rows[1]["host_time_ns"])
        self.assertEqual(rows[0]["elapsed_s"], rows[1]["elapsed_s"])
        self.assertAlmostEqual(float(rows[0]["net_Fx_N"]), 8.81)

    def test_existing_file_is_not_overwritten(self):
        recorder = sensor._CsvRecorder(self.path, np.zeros(6))
        recorder.close()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            sensor._CsvRecorder(self.path, np.zeros(6))
        self.assertEqual(self.path.read_bytes(), original)

    def test_queue_overflow_is_reported_and_file_closed(self):
        recorder = sensor._CsvRecorder(self.path, np.zeros(6))
        recorder._queue = queue.Queue(maxsize=1)
        for index in (1, 2):
            recorder.enqueue(time.perf_counter(), time.time_ns(), index, [(0,) * 6])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            recorder.close()
        self.assertTrue(recorder._file.closed)
        self.assertEqual(len(self.rows()), 1)

    def run_cli(self, extra=(), interrupt=False):
        output = io.StringIO()
        port = FakeSerial()
        args = [
            "scripts.force_sensor.kwr75_reader.py",
            "--secs",
            "0.04",
            "--csv",
            str(self.path),
            *extra,
        ]
        with (
            patch("sys.argv", args),
            patch.object(sensor.serial, "Serial", return_value=port),
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            if interrupt:
                original = sensor.Kwr75Reader.flush_csv

                def interrupted(reader):
                    threading.Event().wait(0.01)
                    original(reader)
                    raise KeyboardInterrupt

                with patch.object(sensor.Kwr75Reader, "flush_csv", interrupted):
                    result = sensor.main()
            else:
                result = sensor.main()
        return result, port, output.getvalue()

    def test_cli_records_without_tare_and_stops_sensor(self):
        result, port, output = self.run_cli()
        self.assertEqual(result, 0, output)
        rows = self.rows()
        self.assertGreater(len(rows), 9)
        self.assertEqual(rows[0]["raw_Fx_N"], rows[0]["net_Fx_N"])
        self.assertEqual(port.commands, [sensor._START, sensor._STOP])
        self.assertTrue(port.closed)
        self.assertIn(f"{len(rows)} rows written", output)
        self.assertTrue(self.path.with_name("sensor_raw.png").is_file())

    def test_ctrl_c_drains_queue_and_closes_port(self):
        result, port, output = self.run_cli(interrupt=True)
        self.assertEqual(result, 130, output)
        self.assertGreater(len(self.rows()), 0)
        self.assertTrue(port.closed)
        self.assertIn(sensor._STOP, port.commands)
        self.assertTrue(self.path.with_name("sensor_raw.png").is_file())

    def test_sigint_during_drain_preserves_entire_batch(self):
        original = sensor._CsvRecorder.drain
        previous_handler = signal.getsignal(signal.SIGINT)

        def signalled(recorder):
            signal.raise_signal(signal.SIGINT)
            original(recorder)

        with patch.object(sensor._CsvRecorder, "drain", signalled):
            result, port, output = self.run_cli()
        self.assertEqual(result, 130, output)
        self.assertGreater(len(self.rows()), 0)
        self.assertEqual(len(self.rows()) % 9, 0)
        self.assertTrue(port.closed)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)

    def test_disk_error_reports_failure_and_closes_port(self):
        with patch.object(sensor._CsvRecorder, "drain", side_effect=OSError("disk full")):
            result, port, output = self.run_cli()
        self.assertEqual(result, 1, output)
        self.assertIn("disk full", output)
        self.assertIn("incomplete", output)
        self.assertTrue(port.closed)

    def test_cli_existing_csv_does_not_open_serial(self):
        recorder = sensor._CsvRecorder(self.path, np.zeros(6))
        recorder.close()
        with (
            patch(
                "sys.argv",
                ["scripts.force_sensor.kwr75_reader.py", "--csv", str(self.path)],
            ),
            patch.object(sensor.serial, "Serial") as serial_open,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            sensor.main()
        serial_open.assert_not_called()

    def test_cli_tare_runs_before_recording(self):
        def tare(reader, dur):
            self.assertIsNone(reader._csv)
            reader._bias = np.ones(6)
            return reader.bias, 10

        with patch.object(sensor.Kwr75Reader, "tare", tare):
            result, _, output = self.run_cli(extra=["--tare"])
        self.assertEqual(result, 0, output)
        self.assertAlmostEqual(float(self.rows()[0]["net_Fx_N"]), 8.81)
        self.assertTrue(self.path.with_name("sensor_net.png").is_file())

    def test_invalid_duration_does_not_open_serial(self):
        for duration in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(duration=duration),
                patch(
                    "sys.argv", ["scripts.force_sensor.kwr75_reader.py", "--secs", duration]
                ),
                patch.object(sensor.serial, "Serial") as serial_open,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                sensor.main()
            serial_open.assert_not_called()

    def test_serial_failure_surfaces_to_csv_caller(self):
        reader = sensor.Kwr75Reader()
        recorder = reader.start_csv(self.path)
        port = FakeSerial()

        def disconnected(_size):
            raise sensor.serial.SerialException("disconnected")

        port.read = disconnected
        reader._ser = port
        reader._run()
        with self.assertRaisesRegex(RuntimeError, "Sensor read failed"):
            reader.flush_csv()
        reader.stop()
        self.assertTrue(recorder._file.closed)

    def test_plot_receives_net_samples_while_csv_records_every_frame(self):
        plot = FakePlot()

        def tare(reader, dur):
            reader._bias = np.ones(6)
            return reader.bias, 10

        with (
            patch.object(sensor, "_create_plot", return_value=plot) as create,
            patch.object(sensor.Kwr75Reader, "tare", tare),
        ):
            result, port, output = self.run_cli(extra=["--plot", "--tare"])
        self.assertEqual(result, 0, output)
        create.assert_called_once_with(10.0, 10.0, True)
        self.assertTrue(plot.closed)
        self.assertTrue(port.closed)
        self.assertGreater(len(self.rows()), len(plot.samples))
        self.assertAlmostEqual(plot.samples[-1][1][1][0], 8.81)
        indices = [int(row["frame_index"]) for row in self.rows()]
        self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))

    def test_closing_plot_stops_sensor_and_saves_csv(self):
        plot = FakePlot()
        plot.close_on_update = True
        with patch.object(sensor, "_create_plot", return_value=plot):
            result, port, output = self.run_cli(extra=["--plot"])
        self.assertEqual(result, 0, output)
        self.assertIn("Plot closed", output)
        self.assertGreater(len(self.rows()), 0)
        self.assertTrue(port.closed)
        self.assertIn(sensor._STOP, port.commands)
        self.assertTrue(self.path.with_name("sensor_raw.png").is_file())

    def test_no_save_plot_keeps_csv_only(self):
        with patch.object(sensor, "_save_recorded_plot") as export:
            result, port, output = self.run_cli(extra=["--no-save-plot"])
        self.assertEqual(result, 0, output)
        export.assert_not_called()
        self.assertGreater(len(self.rows()), 0)
        self.assertTrue(port.closed)
        self.assertFalse(self.path.with_name("sensor_raw.png").exists())

    def test_export_runs_after_csv_is_closed_and_uses_all_rows(self):
        closed = []
        original_close = sensor._CsvRecorder.close
        original_export = sensor._save_recorded_plot

        def close(recorder):
            original_close(recorder)
            closed.append((recorder._file.closed, recorder.rows))

        def export(path, net):
            self.assertTrue(closed[0][0])
            self.assertEqual(len(self.rows()), closed[0][1])
            return original_export(path, net)

        with (
            patch.object(sensor._CsvRecorder, "close", close),
            patch.object(sensor, "_save_recorded_plot", side_effect=export),
        ):
            result, _, output = self.run_cli()
        self.assertEqual(result, 0, output)
        self.assertIn("PNG:", output)

    def test_png_failure_preserves_csv_and_returns_error(self):
        with patch.object(sensor, "_save_recorded_plot", side_effect=OSError("PNG disk error")):
            result, port, output = self.run_cli()
        self.assertEqual(result, 1)
        self.assertIn("PNG export failed", output)
        self.assertGreater(len(self.rows()), 0)
        self.assertTrue(port.closed)

    def test_csv_failure_skips_png(self):
        with (
            patch.object(sensor._CsvRecorder, "drain", side_effect=OSError("disk full")),
            patch.object(sensor, "_save_recorded_plot") as export,
        ):
            result, port, output = self.run_cli()
        self.assertEqual(result, 1)
        self.assertIn("PNG skipped", output)
        export.assert_not_called()
        self.assertTrue(port.closed)

    def test_empty_csv_skips_png(self):
        with (
            patch.object(sensor._CsvRecorder, "enqueue"),
            patch.object(sensor, "_save_recorded_plot") as export,
        ):
            result, _, output = self.run_cli()
        self.assertEqual(result, 0, output)
        self.assertIn("PNG skipped: CSV has no data rows", output)
        export.assert_not_called()

    def test_existing_png_rejected_before_serial_open(self):
        result, _, output = self.run_cli()
        self.assertEqual(result, 0, output)
        self.path.rename(self.path.with_name("previous.csv"))
        png = self.path.with_name("sensor_raw.png")
        original = png.read_bytes()
        with (
            patch(
                "sys.argv",
                ["scripts.force_sensor.kwr75_reader.py", "--csv", str(self.path)],
            ),
            patch.object(sensor.serial, "Serial") as serial_open,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            sensor.main()
        serial_open.assert_not_called()
        self.assertEqual(png.read_bytes(), original)

    def test_missing_gui_does_not_open_serial(self):
        plot = FakePlot()
        with (
            patch.object(sensor, "_create_plot", return_value=plot),
            patch.object(plot, "open", side_effect=RuntimeError("no desktop backend")),
        ):
            result, port, output = self.run_cli(extra=["--plot"])
        self.assertEqual(result, 1, output)
        self.assertEqual(port.commands, [])
        self.assertFalse(self.path.exists())
        self.assertTrue(plot.closed)

    def test_without_plot_does_not_load_gui(self):
        with patch.object(sensor, "_create_plot") as create:
            result, _, output = self.run_cli()
        self.assertEqual(result, 0, output)
        create.assert_not_called()

    def test_ctrl_c_with_plot_saves_csv_and_closes_window(self):
        plot = FakePlot()
        with patch.object(sensor, "_create_plot", return_value=plot):
            result, port, output = self.run_cli(extra=["--plot"], interrupt=True)
        self.assertEqual(result, 130, output)
        self.assertGreater(len(self.rows()), 0)
        self.assertTrue(plot.closed)
        self.assertTrue(port.closed)

    def test_invalid_plot_options_do_not_open_serial(self):
        for option, value in (
            ("--plot-window", "0"),
            ("--plot-window", "nan"),
            ("--plot-hz", "0"),
            ("--plot-hz", "31"),
        ):
            with (
                self.subTest(option=option, value=value),
                patch(
                    "sys.argv",
                    ["scripts.force_sensor.kwr75_reader.py", "--plot", option, value],
                ),
                patch.object(sensor.serial, "Serial") as serial_open,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                sensor.main()
            serial_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
