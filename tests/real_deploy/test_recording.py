"""Background disk ownership and bounded queues, without hardware."""

import csv
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.force_sensor.kwr75_reader import Kwr75Reader
from scripts.real_deploy.recording import EventWriter
from scripts.robot_control.safety import write_event


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_events_are_complete_ordered_and_exclusive(self):
        path = self.folder / "events.jsonl"
        with EventWriter(path) as stream:
            for tick in range(1001):
                write_event(stream, {"event": "sample", "tick": tick})
        with path.open() as stream:
            self.assertEqual([json.loads(line)["tick"] for line in stream], list(range(1001)))
        with self.assertRaises(FileExistsError):
            EventWriter(path)

    def test_blocked_event_disk_never_blocks_producer_and_queue_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        disk = Mock()

        def blocked(text):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test writer not released")

        disk.write.side_effect = blocked
        with patch.object(Path, "open", return_value=disk):
            stream = EventWriter(self.folder / "events.jsonl")
        try:
            stream.write("first\n")
            self.assertTrue(entered.wait(1))
            for _ in range(1024):
                stream.write("queued\n")
                stream.flush()
            with self.assertRaisesRegex(RuntimeError, "Event queue full"):
                stream.write("overflow\n")
        finally:
            release.set()
            with self.assertRaisesRegex(RuntimeError, "Event queue full"):
                stream.close()
        self.assertEqual(disk.write.call_count, 1025)
        disk.close.assert_called_once()

    def test_event_disk_failure_is_reported_and_does_not_mask_primary(self):
        disk = Mock()
        disk.write.side_effect = OSError("disk full")
        with patch.object(Path, "open", return_value=disk):
            stream = EventWriter(self.folder / "events.jsonl")
        stream.write("record\n")
        stream._worker.join(timeout=1)
        with self.assertRaisesRegex(OSError, "disk full"):
            stream.flush()
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaisesRegex(ValueError, "primary"):
            with stream:
                raise ValueError("primary")
        self.assertIn("Event cleanup failed", output.getvalue())
        disk.close.assert_called_once()

    def test_event_thread_start_failure_closes_file(self):
        disk = Mock()
        with patch.object(Path, "open", return_value=disk), patch.object(
                threading.Thread, "start", side_effect=RuntimeError("thread unavailable")):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                EventWriter(self.folder / "events.jsonl")
        disk.close.assert_called_once()

    def test_event_shutdown_timeout_leaves_file_with_writer(self):
        entered, release = threading.Event(), threading.Event()
        disk = Mock()

        def blocked(text):
            entered.set()
            release.wait(2)

        disk.write.side_effect = blocked
        with patch.object(Path, "open", return_value=disk):
            stream = EventWriter(self.folder / "events.jsonl")
        try:
            stream.write("record\n")
            self.assertTrue(entered.wait(1))
            with patch.object(stream._worker, "join") as join:
                with self.assertRaisesRegex(RuntimeError, "shutdown timed out"):
                    stream.close()
                join.assert_called_once_with(timeout=2.0)
            disk.close.assert_not_called()
        finally:
            release.set()
            stream.close()
        disk.close.assert_called_once()

    def test_background_csv_preserves_all_frames_and_frozen_bias(self):
        reader = Kwr75Reader()
        path = self.folder / "sensor.csv"
        recorder = reader.start_csv(path, bias=[1] * 6, background=True)
        try:
            for index in range(400):
                recorder.enqueue(1, 1_000_000_000, index * 9 + 1, [(1,) * 6] * 9)
                reader.flush_csv()
        finally:
            reader.stop_csv()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["frame_index"]) for row in rows], list(range(1, 3601)))
        self.assertAlmostEqual(float(rows[-1]["net_Fx_N"]), 8.81)
        self.assertTrue(recorder._file.closed)
        self.assertFalse(recorder._worker.is_alive())

    def test_blocked_csv_disk_does_not_block_control_health_check(self):
        reader = Kwr75Reader()
        recorder = reader.start_csv(self.folder / "sensor.csv", background=True)
        entered, release = threading.Event(), threading.Event()
        original = recorder._writer

        def blocked(row):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test writer not released")
            original.writerow(row)

        recorder._writer = Mock()
        recorder._writer.writerow.side_effect = blocked
        try:
            recorder.enqueue(1, 1_000_000_000, 1, [(0,) * 6])
            self.assertTrue(entered.wait(1))
            reader.flush_csv()
            for index in range(1024):
                recorder.enqueue(1, 1_000_000_000, index + 2, [(0,) * 6])
            reader.flush_csv()
            recorder.enqueue(1, 1_000_000_000, 1026, [(0,) * 6])
            with self.assertRaisesRegex(RuntimeError, "CSV queue full"):
                reader.flush_csv()
        finally:
            release.set()
            with self.assertRaisesRegex(RuntimeError, "CSV queue full"):
                reader.stop_csv()
        self.assertTrue(recorder._file.closed)

    def test_csv_disk_failure_surfaces_and_reader_detaches(self):
        reader = Kwr75Reader()
        recorder = reader.start_csv(self.folder / "sensor.csv", background=True)
        recorder._writer = Mock()
        recorder._writer.writerow.side_effect = OSError("disk full")
        recorder.enqueue(1, 1_000_000_000, 1, [(0,) * 6])
        recorder._worker.join(timeout=1)
        with self.assertRaisesRegex(OSError, "disk full"):
            reader.flush_csv()
        with self.assertRaisesRegex(OSError, "disk full"):
            reader.stop_csv()
        self.assertIsNone(reader._csv)
        self.assertTrue(recorder._file.closed)

    def test_csv_thread_start_failure_closes_file(self):
        reader = Kwr75Reader()
        disk = io.StringIO()
        with patch.object(Path, "open", return_value=disk), patch.object(
                threading.Thread, "start", side_effect=RuntimeError("thread unavailable")):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                reader.start_csv(self.folder / "sensor.csv", background=True)
        self.assertTrue(disk.closed)
        self.assertIsNone(reader._csv)

    def test_csv_shutdown_timeout_detaches_without_concurrent_file_close(self):
        reader = Kwr75Reader()
        recorder = reader.start_csv(self.folder / "sensor.csv", background=True)
        entered, release = threading.Event(), threading.Event()
        original = recorder._writer

        def blocked(row):
            entered.set()
            release.wait(2)
            original.writerow(row)

        recorder._writer = Mock()
        recorder._writer.writerow.side_effect = blocked
        try:
            recorder.enqueue(1, 1_000_000_000, 1, [(0,) * 6])
            self.assertTrue(entered.wait(1))
            with patch.object(recorder._worker, "join") as join:
                with self.assertRaisesRegex(RuntimeError, "shutdown timed out"):
                    reader.stop_csv()
                join.assert_called_once_with(timeout=2.0)
            self.assertIsNone(reader._csv)
            self.assertFalse(recorder._file.closed)
        finally:
            release.set()
            recorder.close()
        self.assertTrue(recorder._file.closed)
