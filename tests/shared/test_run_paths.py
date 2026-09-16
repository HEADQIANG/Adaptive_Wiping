"""Timestamp allocation and CLI wiring; no hardware or real training."""

import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scripts.shared import paths
from scripts.shared.run_paths import RunOutputs, new_output, new_run_config, timestamped


class RunPathTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.runs = self.folder / "runs"
        self.stack.enter_context(patch.object(paths, "RUNS", self.runs))
        self.messages = io.StringIO()
        self.stack.enter_context(redirect_stderr(self.messages))
        self.stack.enter_context(redirect_stdout(self.messages))
        self.now = datetime(2026, 9, 15, 14, 22, 5)

    def test_exact_format_and_related_files(self):
        outputs = RunOutputs(now=self.now)
        base = self.runs / "real_exploration"
        log = outputs.path(base / "manual_exploration_001.jsonl")
        png = outputs.path(base / "manual_exploration_001.png")
        self.assertEqual(log.parent, base / "0915_142205")
        self.assertEqual(png.parent, log.parent)
        self.assertFalse(log.exists())
        self.assertIn(str(log), self.messages.getvalue())

    def test_existing_legacy_log_unchanged_and_repeat_unique(self):
        base = self.runs / "real_exploration"
        base.mkdir(parents=True)
        old = base / "manual_exploration_001.jsonl"
        old.write_text("original")
        first = RunOutputs(now=self.now).path(old)
        first.write_text("first")
        second = RunOutputs(now=self.now).path(old)
        self.assertEqual(second.parent.name, "0915_142206")
        self.assertEqual(old.read_text(), "original")
        self.assertEqual(first.read_text(), "first")

    def test_concurrent_reservations_are_exclusive(self):
        output = self.runs / "sim_training/experiment"
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: RunOutputs(now=self.now).path(output), range(16)))
        self.assertEqual(len(set(results)), 16)
        for result in results:
            self.assertRegex(result.parent.name, r"^\d{4}_\d{6}$")
            self.assertFalse(result.exists())

    def test_collision_rolls_over_midnight(self):
        base = self.runs / "force_sensor"
        (base / "1231_235959").mkdir(parents=True)
        result = RunOutputs(now=datetime(2026, 12, 31, 23, 59, 59)).path(base / "ft.csv")
        self.assertEqual(result.parent.name, "0101_000000")

    def test_explicit_timestamp_paths_do_not_nest(self):
        base = self.runs / "real_training/0915_142205"
        for output in (base, base / "session", base / "session/log.jsonl"):
            self.assertEqual(new_output(output), output)
        self.assertFalse(base.exists())

    def test_invalid_calendar_names_are_not_timestamps(self):
        for name in ("1331_142205", "0230_142205", "0915_250000", "20260915_142205"):
            self.assertFalse(timestamped(self.runs / name / "log.jsonl"))

    def test_external_paths_and_resume_are_unchanged(self):
        external = self.folder / "external/log.jsonl"
        self.assertEqual(new_output(external), external)
        legacy = self.runs / "real_training/session"
        self.assertEqual(new_output(legacy, resume=True), legacy)
        self.assertIsNone(new_output(None))
        self.assertFalse(self.runs.exists())

    def test_archive_and_runs_root_are_rejected(self):
        for value in (paths.ARCHIVE / "test.json", "outputs/test.json"):
            with self.assertRaises(PermissionError):
                new_output(value)
        with self.assertRaises(ValueError):
            new_output(self.runs)

    def config(self):
        return {"output_dir": str(self.runs / "real_training/models/manual"),
                "raw_data": str(self.runs / "real_training/derived/raw.h5"),
                "encoder": "archive/encoder.pt", "collection": {"source": "archive/dataset"}}

    def test_snapshot_preserves_inputs_and_reuses_paths(self):
        import yaml

        original = self.config()
        saved = copy.deepcopy(original)
        cfg = new_run_config(original, include_raw=True)
        self.assertEqual(original, saved)
        self.assertEqual(cfg["encoder"], original["encoder"])
        self.assertEqual(cfg["collection"], original["collection"])
        self.assertNotEqual(cfg["raw_data"], original["raw_data"])
        out = Path(cfg["output_dir"])
        self.assertFalse(out.exists())
        snapshot = out.parent / "run_config.yaml"
        loaded = yaml.safe_load(snapshot.read_text())
        self.assertEqual(loaded, cfg)
        self.assertEqual(new_run_config(loaded), cfg)
        self.assertEqual(new_run_config(loaded, include_raw=True), cfg)

    def test_prepare_keeps_raw_input_and_new_template_means_new_run(self):
        original = self.config()
        first = new_run_config(original)
        second = new_run_config(original)
        self.assertEqual(first["raw_data"], original["raw_data"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(new_run_config(original, resume=True), original)

    def test_external_config_has_no_snapshot_side_effect(self):
        cfg = self.config()
        cfg["output_dir"] = str(self.folder / "external")
        self.assertIs(new_run_config(cfg, include_raw=True), cfg)
        self.assertFalse(self.runs.exists())

    def test_exploration_cli_maps_before_dispatch_and_repeated_launch(self):
        from scripts.real_training import airbot_exploration, manual_exploration

        output = self.runs / "real_exploration/manual_exploration_001.jsonl"
        with patch.object(airbot_exploration.sys.stdin, "isatty", return_value=True), patch.object(
                manual_exploration, "main", return_value=0) as run:
            for _ in range(2):
                self.assertEqual(airbot_exploration.main([
                    "run", "--execute", "--plot", "--output", str(output)]), 0)
            targets = [call.args[0].output for call in run.call_args_list]
        self.assertNotEqual(*targets)
        self.assertTrue(all(timestamped(target) for target in targets))
        self.assertTrue(all(target.name == output.name for target in targets))

    def test_exploration_check_and_preview_do_not_allocate(self):
        from scripts.real_training import airbot_exploration, manual_exploration

        with patch.object(manual_exploration, "main", return_value=0):
            for action in ("preview", "check"):
                airbot_exploration.main([action, "--output", str(self.runs / "robot/log.jsonl")])
        self.assertFalse(self.runs.exists())

    def test_exploration_authorization_precedes_allocation(self):
        from scripts.real_training import airbot_exploration

        for tty, execute in ((False, True), (True, False)):
            with patch.object(airbot_exploration.sys.stdin, "isatty", return_value=tty):
                with self.assertRaises(SystemExit):
                    airbot_exploration.main(["run", "--output", str(self.runs / "robot/log.jsonl")]
                                           + (["--execute"] if execute else []))
        self.assertFalse(self.runs.exists())

    def test_programmed_new_session_and_explicit_resume(self):
        from scripts.real_training import airbot_demonstrations, airbot_programmed_demonstrations as program

        base = self.runs / "real_training/programmed/session"
        with patch.object(program, "load_configuration", return_value=({}, {}, {})), patch.object(
                program, "readiness", return_value=[]), patch.object(
                program, "freeze_session") as freeze, patch.object(
                program, "accepted_records", return_value=[None] * 8), patch.object(
                program.sys.stdin, "isatty", return_value=True), patch.object(
                program, "open_client", side_effect=AssertionError("No hardware")):
            self.assertEqual(airbot_demonstrations.main([
                "run", "--mode", "programmed", "--execute", "--output", str(base)]), 0)
            actual = freeze.call_args.args[0]
            self.assertTrue(timestamped(actual))
            base.mkdir(parents=True)
            (base / "session.json").write_text(json.dumps({"protocol": program.PROTOCOL}))
            self.assertEqual(airbot_demonstrations.main([
                "run", "--mode", "programmed", "--execute", "--output", str(base)]), 0)
            self.assertEqual(freeze.call_args.args[0], base)

    def test_sensor_csv_and_png_use_one_directory_before_device_open(self):
        from scripts.force_sensor import kwr75_reader as reader

        base = self.runs / "force_sensor/reading.csv"
        with patch.object(sys, "argv", ["reader", "--csv", str(base)]), patch.object(
                reader, "_saved_plot_path", wraps=reader._saved_plot_path) as plot_path, patch.object(
                reader, "Kwr75Reader", side_effect=RuntimeError("fake connection boundary")):
            with self.assertRaisesRegex(RuntimeError, "fake connection"):
                reader.main()
        actual = plot_path.call_args.args[0]
        self.assertTrue(timestamped(actual))
        self.assertEqual(reader._saved_plot_path(actual, True).parent, actual.parent)
        self.assertFalse(base.exists())

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Simulation dependencies unavailable")
    def test_simulation_sanity_snapshot_is_shared_with_collect(self):
        from scripts.sim_pretrain import collection, pretrain

        with patch.object(pretrain, "load_config", return_value=self.config()), patch.object(
                sys, "argv", ["pretrain", "sanity"]), patch.object(
                collection, "sanity", return_value={"passed": True}) as sanity:
            self.assertEqual(pretrain.main(), 0)
            cfg = sanity.call_args.args[0]
        self.assertTrue(timestamped(cfg["output_dir"]))
        with patch.object(pretrain, "load_config", return_value=cfg), patch.object(
                sys, "argv", ["pretrain", "collect"]), patch.object(
                collection, "collect", return_value={"complete": True}) as collect:
            self.assertEqual(pretrain.main(), 0)
            self.assertEqual(collect.call_args.args[0], cfg)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "Offline import dependencies unavailable")
    def test_import_audit_does_not_allocate_but_import_freezes_config(self):
        from scripts.real_training import import_airbot

        cfg = self.config()
        with patch.object(import_airbot, "load_config", return_value=cfg), patch.object(
                import_airbot, "import_dataset", return_value={}) as importer:
            import_airbot.main(["--audit-only"])
            self.assertEqual(importer.call_args.args[0], cfg)
            self.assertFalse(self.runs.exists())
            import_airbot.main(["--subtract-recorded-baseline"])
            actual = importer.call_args.args[0]
            self.assertTrue(timestamped(actual["raw_data"]))
            self.assertTrue(timestamped(actual["output_dir"]))
            self.assertTrue(importer.call_args.kwargs["subtract_recorded_baseline"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Offline training dependencies unavailable")
    def test_prepare_and_training_use_the_same_snapshot(self):
        from scripts.real_training import data, training, training_cli

        cfg = self.config()
        info = dict.fromkeys(("source_kind", "raw_sha256", "encoder_sha256", "demo_ids",
                            "encoder_epoch", "encoder_randomization", "valid_windows_total",
                            "exploration_out_of_range_per_channel", "warnings"))
        with patch.object(training_cli, "load_config", return_value=cfg), patch.object(
                data, "prepare", return_value=info) as prepare:
            self.assertEqual(training_cli.main(["prepare"]), 0)
            actual = prepare.call_args.args[0]
        self.assertTrue(timestamped(actual["output_dir"]))
        with patch.object(training_cli, "load_config", return_value=actual), patch.object(
                training, "train", return_value={}) as train:
            self.assertEqual(training_cli.main(["train", "--resume"]), 0)
            self.assertEqual(train.call_args.args[0], actual)
            self.assertTrue(train.call_args.kwargs["resume"])


if __name__ == "__main__":
    unittest.main()
