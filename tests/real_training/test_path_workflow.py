"""Offline CLI path selection, isolated outputs and collection-to-export dispatch."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
import h5py

from scripts.real_training import training, training_cli
from scripts.real_training.config import load_config
from scripts.shared.common import ROOT, file_digest
from tests.real_training import test_manual_tared_import as fixtures


class PathWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ManualTaredTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.folder = self.fixture.folder
        self.output = self.folder / "new training"
        self.args = ["--demonstrations", str(self.fixture.session), "--exploration",
                     str(self.fixture.exploration), "--confirm-same-setup"]

    def cli(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = training_cli.main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_pair_conflicts_and_resume_are_rejected_before_work(self):
        for args in (["train", "--session", "missing"], ["train", "--exploration", "missing"],
                     ["train", *self.args, "--resume"], ["train", *self.args, "--raw-data", "raw.h5"],
                     ["export", *self.args], ["train", "--resume", "--encoder", "encoder.pt"],
                     ["smoke-test", "--output-dir", str(self.output)],
                     ["train", "--confirm-same-setup"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                self.cli(args)
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_confirmation_and_programmed_mismatch_fail_without_output(self):
        for args in (self.args[:-1], [*self.args, "--programmed-hold-last"]):
            code, _, error = self.cli(["train", *args, "--output-dir", str(self.output)])
            self.assertEqual(code, 2, error)
            self.assertFalse(self.output.exists())

    def test_inspect_and_prepare_use_actual_recordings_and_preserve_sources(self):
        encoder = ROOT / "runs/sim_training/pretrain_wide_1200_v1/encoder.pt"
        if not encoder.is_file():
            self.skipTest("Real frozen encoder audit fixture not installed")
        before = {p: file_digest(p) for p in self.fixture.session.iterdir()}
        before[self.fixture.exploration] = file_digest(self.fixture.exploration)
        code, output, error = self.cli(["inspect", *self.args, "--output-dir", str(self.output)])
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)["valid_windows_total"], 160)
        self.assertFalse(self.output.exists())
        args = ["prepare", *self.args, "--encoder", str(encoder), "--output-dir", str(self.output)]
        code, output, error = self.cli(args)
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        cfg = load_config(result["run_config"])
        self.assertEqual(cfg["output_dir"], str(self.output))
        self.assertEqual(cfg["profile"], "airbot_native_tared_offline")
        self.assertEqual(cfg["encoder"], str(encoder))
        self.assertEqual(cfg["recordings"]["demonstrations"], str(self.fixture.session))
        self.assertTrue((self.output / "prepared.h5").is_file())
        self.assertTrue(Path(cfg["raw_data"]).is_file())
        self.assertEqual(before, {p: file_digest(p) for p in before})
        self.assertEqual(self.cli(args)[0], 2)
        second = self.folder / "prepared from existing raw"
        code, output, error = self.cli([
            "prepare", "--config", result["run_config"], "--raw-data", cfg["raw_data"],
            "--output-dir", str(second)])
        self.assertEqual(code, 0, error)
        second_cfg = load_config(json.loads(output)["run_config"])
        self.assertEqual(second_cfg["output_dir"], str(second))
        self.assertEqual(second_cfg["raw_data"], cfg["raw_data"])

    def test_train_runs_all_stages_with_one_resolved_snapshot(self):
        calls = []

        def action(label):
            def run(cfg):
                calls.append((label, cfg.copy()))
                if label == "export":
                    return {"policy": str(Path(cfg["output_dir"]) / "policy.pt")}
                return {}
            return run

        with patch("scripts.real_training.import_airbot.import_dataset", return_value={}), patch(
            "scripts.real_training.data.prepare", return_value={"valid_windows_total": 160, "warnings": []}
        ), patch.object(training, "train", side_effect=action("train")), patch.object(
            training, "evaluate", side_effect=action("evaluate")
        ), patch.object(training, "export", side_effect=action("export")):
            code, output, error = self.cli(["train", *self.args, "--output-dir", str(self.output)])
        self.assertEqual(code, 0, error)
        self.assertEqual([name for name, _ in calls], ["train", "evaluate", "export"])
        snapshot = yaml.safe_load((self.output / "run_config.yaml").read_text())
        self.assertTrue(all(cfg == snapshot for _, cfg in calls))
        self.assertEqual(json.loads(output)["policy"], str(self.output / "policy.pt"))

    def test_real_data_preserves_non_normal_sponge_and_rejects_mixed_ids(self):
        from scripts.real_training.data import _load_raw, write_raw_log

        self.fixture.meta["config"]["sponge_id"] = "sponge_batch_02"
        self.fixture.save("session.json", self.fixture.meta)
        rows = [json.loads(line) for line in self.fixture.exploration.read_text().splitlines()]
        rows[0]["config"]["sponge_id"] = "sponge_batch_02"
        self.fixture.exploration.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        meta, exp, demos = self.fixture.assemble()
        self.assertEqual(next(iter(exp.values()))["attrs"]["sponge_id"], "sponge_batch_02")
        path = self.folder / "batch_02.h5"
        write_raw_log(path, meta, exp, demos, source_kind="real")
        cfg = load_config("configs/real_training/real_training_manual_tared.yaml")
        cfg["raw_data"] = str(path)
        self.assertEqual(_load_raw(cfg)[0]["xy"].shape, (8, 25, 2))
        with h5py.File(path, "r+") as h5:
            h5["demonstrations/demo_01"].attrs["sponge_id"] = "different_sponge"
        with self.assertRaisesRegex(ValueError, "share one sponge_id"):
            _load_raw(cfg)

    def test_training_failure_stops_before_evaluation_or_export(self):
        with patch("scripts.real_training.import_airbot.import_dataset", return_value={}), patch(
            "scripts.real_training.data.prepare", return_value={"valid_windows_total": 160, "warnings": []}
        ), patch.object(training, "train", side_effect=RuntimeError("training failed")), patch.object(
            training, "evaluate"
        ) as evaluate, patch.object(training, "export") as export:
            code, _, error = self.cli(["train", *self.args, "--output-dir", str(self.output)])
        self.assertEqual(code, 2)
        self.assertIn("training failed", error)
        self.assertTrue((self.output / "run_config.yaml").is_file())
        evaluate.assert_not_called()
        export.assert_not_called()

    def test_explicit_encoder_config_and_session_file_are_resolved(self):
        from argparse import Namespace
        from scripts.real_training.workflow import recording_config

        cfg, session, exploration, options = recording_config(Namespace(
            demonstrations=str(self.fixture.session / "session.json"),
            exploration=str(self.fixture.exploration), programmed_hold_last=False,
            confirm_same_setup=True, config="configs/real_training/real_training_manual_tared.yaml",
            encoder=str(self.folder / "custom encoder.pt"), calibration=None))
        self.assertEqual(session, self.fixture.session)
        self.assertEqual(exploration, self.fixture.exploration)
        self.assertEqual(cfg["encoder"], str(self.folder / "custom encoder.pt"))
        self.assertTrue(options["manual_tared"])
        self.assertTrue(options["subtract_recorded_baseline"])

    def test_existing_raw_paths_override_configuration_without_editing_it(self):
        config = ROOT / "configs/real_training/real_training_manual_tared.yaml"
        before = file_digest(config)
        with patch("scripts.real_training.data.inspect", return_value=(None, {
            k: None for k in ("source_kind", "raw_sha256", "encoder_sha256", "demo_ids", "encoder_epoch",
                             "encoder_randomization", "valid_windows_total",
                             "exploration_out_of_range_per_channel", "warnings")
        })) as inspect:
            code, _, error = self.cli(["inspect", "--config", str(config), "--raw-data",
                                      str(self.folder / "raw.h5"), "--output-dir", str(self.output)])
        self.assertEqual(code, 0, error)
        self.assertEqual(inspect.call_args.args[0]["raw_data"], str(self.folder / "raw.h5"))
        self.assertEqual(file_digest(config), before)


if __name__ == "__main__":
    unittest.main()
