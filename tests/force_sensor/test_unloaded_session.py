import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.force_sensor.tools.record_unloaded_session import (
    interactive,
    main,
    next_path,
    session_status,
)


class SessionTests(unittest.TestCase):
    def test_numbering_preserves_incomplete_and_old_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "pose_002.json").write_text("incomplete")
            (path / "connection_001.json").write_text("{}")
            self.assertEqual(next_path(path).name, "pose_003.json")
            status = session_status(path)
            self.assertEqual(status["accepted"], [])
            self.assertEqual(len(status["rejected"]), 1)
            self.assertFalse(status["complete_calibration"])

    def test_only_explicit_r_records_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            commands = iter(["", "no", "s", "r", "r", "q"])
            seen = []

            def fake(path, *, confirmed):
                seen.append((path.name, confirmed))
                path.write_text(json.dumps({"complete": False}))
                return {"eligible_unloaded_pose": False, "errors": ["fixture"]}

            with patch("builtins.print"):
                interactive(Path(folder), read=lambda _: next(commands), recorder=fake)
            self.assertEqual(seen, [("pose_001.json", True), ("pose_002.json", True)])

    def test_status_does_not_open_devices_or_create_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "absent"
            with (
                patch("builtins.print"),
                patch(
                    "scripts.force_sensor.tools.record_unloaded_session.capture",
                    side_effect=AssertionError("hardware"),
                ),
            ):
                self.assertEqual(main(["--status", "--directory", str(target)]), 0)
            self.assertFalse(target.exists())

    def test_noninteractive_recording_rejected(self):
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main([])


if __name__ == "__main__":
    unittest.main()
