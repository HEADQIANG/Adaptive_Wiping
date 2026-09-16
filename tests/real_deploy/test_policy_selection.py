"""Path-only policy selection never commands hardware or silently selects defaults."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.real_deploy import manual_setup, policy_selection as selection
from scripts.real_deploy.airbot_deploy import main
from scripts.real_training.config import load_config
from scripts.shared.common import ROOT, file_digest


class SelectionCliTests(unittest.TestCase):
    def test_mode_is_selected_from_profile_and_derivation(self):
        from types import SimpleNamespace

        cases = (("airbot_native_tared_offline", "manual_recorded_baseline_10s_v1", "manual-tared"),
                 ("airbot_native_tared_offline", "programmed_hold_last_10s_v1", "fixed-setup"),
                 ("airbot_sensor_calibrated_offline", None, "calibrated"))
        for profile, derivation, expected in cases:
            policy = SimpleNamespace(source_kind="real", metadata={
                "training_contract": {"profile": profile},
                "data": {"metadata": {"derivation": derivation}}})
            with self.subTest(profile=profile, derivation=derivation), patch.object(
                selection, "_load_policy", return_value=policy
            ):
                self.assertEqual(selection.policy_mode("selected.pt"), expected)

    def test_missing_and_incompatible_policies_never_connect(self):
        with patch("scripts.real_deploy.airbot_deploy.open_client") as connect, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--policy", "/missing/policy.pt", "--execute"]), 2)
            with patch.object(selection, "policy_mode", return_value="manual-tared"):
                self.assertEqual(main(["run", "--policy", "x.pt", "--mode", "fixed-setup", "--execute"]), 2)
        connect.assert_not_called()

    def test_inferred_mode_routes_selected_path(self):
        with patch.object(selection, "policy_mode", return_value="manual-tared"), patch.object(
            manual_setup, "main", return_value=0
        ) as dispatch:
            self.assertEqual(main(["preflight", "--policy", "my/policy.pt"]), 0)
        args = dispatch.call_args.args[0]
        self.assertEqual(args.mode, "manual-tared")
        self.assertEqual(args.policy, str(ROOT / "my/policy.pt"))

    def test_training_config_cannot_override_policy_binding(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["preflight", "--policy", "x.pt", "--training-config", "other.yaml"])
        self.assertEqual(raised.exception.code, 2)

    def test_corrupt_policy_returns_clean_error_without_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.pt"
            path.write_bytes(b"not a PyTorch policy")
            with patch("scripts.real_deploy.airbot_deploy.open_client") as connect, redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--policy", str(path), "--execute"]), 2)
            connect.assert_not_called()


class RealSelectionAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ROOT / "configs/real_deploy/airbot_manual_tared.json"
        cls.spec = json.loads(cls.config.read_text())
        cls.policy = ROOT / cls.spec["policy"]
        if not cls.policy.is_file():
            raise unittest.SkipTest("Real policy audit fixture not installed")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="policy-selection-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_legacy_policy_derives_all_paths_without_mutating_configuration(self):
        before = file_digest(self.config)
        stale = {**self.spec, "policy": "/missing/old.pt", "policy_sha256": "old",
                 "training_config": "/missing/old.yaml", "collection_session": "/missing/session",
                 "exploration": "/missing/exploration.jsonl"}
        spec, cfg, bindings = selection.select_setup(stale, self.policy)
        self.assertEqual(spec["policy"], str(self.policy))
        self.assertEqual(spec["collection_session_sha256"], self.spec["collection_session_sha256"])
        self.assertEqual(spec["exploration_sha256"], self.spec["exploration_sha256"])
        self.assertEqual(cfg["output_dir"], str(self.policy.parent))
        self.assertEqual(bindings[str(self.policy)], file_digest(self.policy))
        self.assertEqual(stale["policy_sha256"], "old")
        self.assertEqual(before, file_digest(self.config))

    def test_new_embedded_config_and_renamed_policy_need_no_run_sidecar(self):
        payload = torch.load(self.policy, weights_only=True, map_location="cpu")
        payload["metadata"]["training_config"] = load_config(self.spec["training_config"])
        path = self.folder / "different policy.pt"
        torch.save(payload, path)
        selected, _, cfg, _ = selection.training_inputs(path)
        self.assertEqual(selected, path)
        self.assertEqual(cfg["profile"], "airbot_native_tared_offline")
        loaded = manual_setup.load_setup(self.config, policy_path=path)
        self.assertEqual(loaded[0]["policy"], str(path))
        self.assertEqual(loaded[0]["policy_sha256"], file_digest(path))

    def test_missing_legacy_sidecar_or_wrong_binding_is_not_guessed(self):
        path = self.folder / "policy.pt"
        shutil.copyfile(self.policy, path)
        with self.assertRaisesRegex(ValueError, "final/run.json"):
            selection.training_inputs(path)
        record = json.loads((self.policy.parent / "final/run.json").read_text())
        record["bindings"]["raw_sha256"] = "wrong"
        (self.folder / "final").mkdir()
        (self.folder / "final/run.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            selection.training_inputs(path)

    def test_wrong_embedded_configuration_and_prepared_data_rejected(self):
        payload = torch.load(self.policy, weights_only=True, map_location="cpu")
        cfg = load_config(self.spec["training_config"])
        cfg["training"]["seed"] += 1
        payload["metadata"]["training_config"] = cfg
        path = self.folder / "policy.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            selection.training_inputs(path)
        payload["metadata"]["training_config"]["training"]["seed"] -= 1
        torch.save(payload, path)
        (self.folder / "prepared.h5").write_bytes(b"wrong prepared data")
        shutil.copyfile(self.policy.parent / "prepared_integrity.json", self.folder / "prepared_integrity.json")
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            selection.training_inputs(path)

    def test_selected_policy_preflight_is_offline_and_reports_identity(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(manual_setup, "open_client") as connect, redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["preflight", "--policy", str(self.policy.parent)])
        self.assertEqual(code, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["hardware_connected"])
        self.assertEqual(report["policy_sha256"], file_digest(self.policy))
        self.assertEqual(report["mode"], manual_setup.MODE)
        connect.assert_not_called()

    def test_current_code_binding_is_checked_before_connection(self):
        from scripts.real_deploy.fixed_setup import assert_unchanged

        loaded = manual_setup.load_setup(self.config, policy_path=self.policy)
        bindings = loaded[-1]
        source = str(ROOT / "scripts/real_training/training.py")
        self.assertEqual(bindings[source], file_digest(source))
        self.assertIn("scripts/real_training/training.py", loaded[0]["training_software_differences"])
        bindings[source] = "changed after preflight"
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            assert_unchanged(bindings)

    def test_selected_synthetic_policy_is_rejected(self):
        payload = torch.load(self.policy, weights_only=True, map_location="cpu")
        payload["source_kind"] = "synthetic"
        payload["warnings"] = ["SYNTHETIC_SOFTWARE_TEST_ONLY"]
        path = self.folder / "policy.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            selection.policy_mode(path)
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            selection.training_inputs(path)


if __name__ == "__main__":
    unittest.main()
