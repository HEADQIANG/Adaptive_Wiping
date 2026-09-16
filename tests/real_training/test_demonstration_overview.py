"""Synthetic files exercise accepted-log export; no hardware or real collection."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np

from scripts.real_training import airbot_demonstrations as demo
from scripts.real_training.tools.plot_demonstration_overview import load_demos, main, save_overview


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def fixture(self, count=8, mode="software_tared"):
        cfg = json.loads((demo.ROOT / "configs/real_training/airbot_demonstrations.json").read_text())
        cfg.update(force_recording=mode, live_plot=False)
        (self.folder / "session.json").write_text(json.dumps({"config": cfg}))
        for i in range(1, count + 1):
            tare = {"id": f"test-{i}", "raw_baseline_si": [10] * 6}
            rows = [{"event": "start", "start_perf_s": 100,
                     "force_recording": mode, "tare": tare}]
            for t in (0, 5, 10):
                ft = {"sensor_receive_perf_s": 100 + t,
                      "raw_sensor_wrench_si": [10 + axis + t for axis in range(6)]}
                if mode == "software_tared":
                    ft.update(tared_sensor_wrench_si=[axis + t for axis in range(6)], tare_id=tare["id"])
                rows.append({"event": "sample", "observed_perf_s": 100 + t + 0.001,
                             "pose": {"host_monotonic_s": 100 + t,
                                      "sdk_end_position_m": [0.2 + t / 1000, 0.01 * i + t / 100, 0.1 + t / 1000]},
                             "ft": ft})
            rows.append({"event": "finished", "quality": {"passed": True}})
            path = self.folder / f"attempt_{i}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            record = {"accepted": True, "source_kind": "real", "training_ready": False,
                      "quality": {"passed": True}, "raw_file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "force_recording": mode}
            if mode == "software_tared":
                record["tare"] = tare
            (self.folder / f"demo_{i:02}.json").write_text(json.dumps(record))
        return cfg

    def test_eight_tared_trajectories_and_export_nonblank_without_changing_sources(self):
        self.fixture()
        before = {p: p.read_bytes() for p in self.folder.iterdir()}
        demos = load_demos(self.folder)
        np.testing.assert_array_equal(demos[0]["ft"][:, 0], [0, 5, 10])
        np.testing.assert_allclose(demos[0]["positions"][:, 0], [200, 205, 210])
        with redirect_stdout(io.StringIO()):
            output = save_overview(self.folder)
        self.assertEqual(output, self.folder / "plots")
        for name in ("trajectories.png", "force_torque.png"):
            self.assertGreater(float(mpimg.imread(output / name).std()), 0.05)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["accepted"], 8)
        self.assertTrue(report["complete"])
        self.assertEqual(report["force_field"], "tared_sensor_wrench_si")
        self.assertEqual([d["name"] for d in report["demonstrations"]], [f"demo_{i:02}" for i in range(1, 9)])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_partial_rejected_attempt_excluded_and_repeated_export_preserved(self):
        self.fixture(count=1)
        (self.folder / "attempt_rejected.jsonl").write_text("invalid rejected attempt")
        with self.assertRaises(ValueError):
            load_demos(self.folder)
        self.assertEqual(len(load_demos(self.folder, allow_partial=True)), 1)
        with redirect_stdout(io.StringIO()):
            first = save_overview(self.folder, allow_partial=True)
            before = (first / "force_torque.png").read_bytes()
            second = save_overview(self.folder, allow_partial=True)
        self.assertEqual(second.name, "plots_002")
        self.assertEqual(before, (first / "force_torque.png").read_bytes())
        self.assertFalse(json.loads((second / "summary.json").read_text())["complete"])
        with self.assertRaises(FileExistsError):
            save_overview(self.folder, first, allow_partial=True)

    def test_empty_session_no_plot_and_legacy_raw_field(self):
        self.fixture(count=0, mode="raw")
        self.assertIsNone(save_overview(self.folder, allow_partial=True))
        self.assertFalse((self.folder / "plots").exists())
        self.fixture(count=1, mode="raw")
        np.testing.assert_array_equal(load_demos(self.folder, allow_partial=True)[0]["ft"][:, 0], [10, 15, 20])

    def test_corruption_and_missing_tare_never_fall_back_to_raw(self):
        self.fixture(count=1)
        path = self.folder / "attempt_1.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed log"):
            load_demos(self.folder, allow_partial=True)
        for invalid in (None, [100] * 6):
            self.fixture(count=1)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if invalid is None:
                del rows[1]["ft"]["tared_sensor_wrench_si"]
            else:
                rows[1]["ft"]["tared_sensor_wrench_si"] = invalid
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = self.folder / "demo_01.json"
            record = json.loads(manifest.read_text())
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_demos(self.folder, allow_partial=True)

    def test_completed_run_exports_without_connecting_and_status_does_not_export(self):
        cfg = self.fixture()
        config = self.folder / "config.json"
        config.write_text(json.dumps(cfg))
        with (
            patch.dict("sys.modules", {"arm_sdk": SimpleNamespace(Controller=object())}),
            patch.object(demo.sys.stdin, "isatty", return_value=True),
            patch.object(demo, "open_client") as connect,
            patch.object(demo, "save_session_plots") as export,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(demo.main(["run", "--execute", "--config", str(config),
                                        "--output", str(self.folder)]), 0)
            export.assert_called_once_with(self.folder)
            export.reset_mock()
            self.assertEqual(demo.main(["status", "--config", str(config), "--output", str(self.folder)]), 0)
            export.assert_not_called()
        connect.assert_not_called()

    def test_cli_preserves_legacy_default_and_explicit_session_uses_local_plots(self):
        with patch("scripts.real_training.tools.plot_demonstration_overview.save_overview") as save, patch(
            "scripts.shared.run_paths.new_output", side_effect=lambda path: path
        ):
            main([])
            save.assert_called_once_with(
                Path("archive/real_training/raw_data/manual_demonstrations/session_record_only_002"),
                Path("runs/real_demonstrations/analysis/demonstration_overview_002"), allow_partial=False,
            )
            save.reset_mock()
            main(["--session", str(self.folder), "--allow-partial"])
            save.assert_called_once_with(self.folder, None, allow_partial=True)

    def test_plot_failure_is_reported_without_changing_raw_data(self):
        self.fixture(count=1)
        before = (self.folder / "attempt_1.jsonl").read_bytes()
        error = io.StringIO()
        with patch("scripts.real_training.tools.plot_demonstration_overview.save_overview",
                   side_effect=OSError("Disk full")), redirect_stderr(error):
            self.assertIsNone(demo.save_session_plots(self.folder))
        self.assertIn("recorded data unchanged", error.getvalue())
        self.assertEqual(before, (self.folder / "attempt_1.jsonl").read_bytes())

    def test_normal_interrupt_and_error_export_after_resource_cleanup(self):
        cfg = self.fixture(count=0)
        config = self.folder / "config.json"
        config.write_text(json.dumps(cfg))
        for failure, expected in ((None, 1), (KeyboardInterrupt(), 130), (RuntimeError("test fault"), 1)):
            calls = []
            client = SimpleNamespace(_stub=object(),
                                     get_firmware_info=lambda: SimpleNamespace(arm_sn=cfg["robot_sn"],
                                                                              eef_type=cfg["expected_eef_type"]),
                                     close=lambda: calls.append("client_closed"))

            def session(*args):
                calls.append("session_finished")
                if failure is not None:
                    raise failure
                return 0

            with (
                self.subTest(failure=failure), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                patch.dict("sys.modules", {"arm_sdk": SimpleNamespace(Controller=SimpleNamespace(idle="idle"))}),
                patch.object(demo.sys.stdin, "isatty", return_value=True),
                patch.object(demo, "open_client", return_value=client),
                patch.object(demo, "DeadlineStub", return_value=object()),
                patch.object(demo, "asdict", return_value={}),
                patch("scripts.force_sensor.kwr75_reader.Kwr75Reader") as sensor,
                patch.object(demo, "read_demo_force", return_value={}),
                patch.object(demo, "session", side_effect=session),
                patch.object(demo, "save_session_plots", side_effect=lambda folder: calls.append("export")),
            ):
                sensor.return_value.stop.side_effect = lambda: calls.append("sensor_stopped")
                self.assertEqual(demo.main(["run", "--execute", "--config", str(config),
                                            "--output", str(self.folder)]), expected)
            self.assertEqual(calls, ["session_finished", "client_closed", "sensor_stopped", "export"])


if __name__ == "__main__":
    unittest.main()
