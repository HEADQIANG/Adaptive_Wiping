"""Documented module entrypoints work without an installation or hardware."""

import importlib
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from scripts.shared.paths import ROOT


ENTRYPOINTS = (
    "scripts.sim_pretrain",
    "scripts.force_sensor",
    "scripts.real_training",
    "scripts.real_deploy",
    "scripts.robot_control",
    "scripts.robot_control.airbot_initial_pose",
)


class ModuleEntrypointTests(unittest.TestCase):
    def test_help_without_pythonpath_or_installed_project(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        # -S disables site-packages, including any editable project installation.
        for module_name in ENTRYPOINTS:
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [sys.executable, "-S", "-m", module_name, "--help"],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_arguments_and_exit_code_are_forwarded_unchanged(self):
        routes = (
            ("scripts.sim_pretrain", "retrain-wide-vae",
             "scripts.sim_pretrain.experiments.retrain_wide_vae", []),
            ("scripts.force_sensor", "read", "scripts.force_sensor.kwr75_reader", []),
            ("scripts.real_training", "train", "scripts.real_training.training_cli", ["train"]),
            ("scripts.real_deploy", "run", "scripts.real_deploy.airbot_deploy", ["run"]),
        )
        for entrypoint, command, target, prefix in routes:
            with self.subTest(module=entrypoint):
                entry = importlib.import_module(entrypoint + ".__main__")
                module = types.ModuleType(target)
                seen = []

                def main():
                    seen.extend(sys.argv[1:])
                    return 23

                module.main = main
                args = ["--execute", "--output", "runs/path with spaces/file.json"]
                with patch.dict(sys.modules, {target: module}), patch.object(
                    sys, "argv", [entrypoint, command, *args]
                ):
                    original_argv = sys.argv
                    self.assertEqual(entry.main(), 23)
                    self.assertIs(sys.argv, original_argv)
                    self.assertEqual(seen, [*prefix, *args])
                    seen.clear()
                    self.assertEqual(entry.main([command, *args]), 23)
                    self.assertIs(sys.argv, original_argv)
                    self.assertEqual(seen, [*prefix, *args])

    def test_import_does_not_parse_arguments_or_load_device_dependencies(self):
        for entrypoint in ENTRYPOINTS:
            module = entrypoint if entrypoint.endswith("airbot_initial_pose") else entrypoint + ".__main__"
            code = f"""
import importlib
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'mujoco', 'robosuite', 'arm_sdk', 'serial'):
            raise RuntimeError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['entrypoint', 'not-a-command', '--execute']
importlib.import_module({module!r})
"""
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_command_is_rejected(self):
        for module_name in ENTRYPOINTS:
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [sys.executable, "-S", "-m", module_name, "not-a-command"],
                    cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
