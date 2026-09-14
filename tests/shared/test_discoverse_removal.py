"""The retired vendor tree is historical evidence, never a runtime dependency."""

import subprocess
import sys
import unittest

from scripts.shared.assets import DISCOVERSE_INERTIA_SOURCE, verify
from scripts.shared.paths import ARCHIVE, ROOT, read_path, writable_path


class DiscoverseRemovalTests(unittest.TestCase):
    def test_old_paths_resolve_to_read_only_source_evidence(self):
        self.assertFalse((ROOT / "DISCOVERSE").exists())
        retired = ARCHIVE / "sim_pretrain/discoverse_source_20260911/source"
        for name in (
            ".git/HEAD",
            "LICENSE",
            "discoverse/doc/usage.md",
            "examples/force_control/joint_impedance_control.py",
            "models/mjcf/manipulator/new_airbot_play/mjx_airbot_play.xml",
        ):
            path = read_path("DISCOVERSE/" + name, historical=True)
            self.assertEqual(path, retired / name)
            self.assertTrue(path.is_file())
            with self.assertRaises(PermissionError):
                writable_path(path)
        with self.assertRaises(PermissionError):
            writable_path("DISCOVERSE/models/new.xml")
        source = read_path("DISCOVERSE/models/mjcf/manipulator/airbot_play/airbot_play.xml")
        self.assertNotEqual(source, DISCOVERSE_INERTIA_SOURCE)
        self.assertEqual(source.read_bytes(), DISCOVERSE_INERTIA_SOURCE.read_bytes())
        self.assertTrue(verify(sources=True)["passed"])

    def test_simulation_cannot_read_or_import_retired_vendor(self):
        code = r"""
import sys, os
from pathlib import Path
root = Path.cwd()
blocked = (root / "DISCOVERSE", root / "archive/sim_pretrain/discoverse_source_20260911")
def audit(event, args):
    if event in ("open", "os.listdir", "os.scandir") and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(folder) for folder in blocked):
            raise RuntimeError("Unexpected retired vendor access: " + str(path))
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0].lower() == "discoverse":
            raise RuntimeError("Unexpected retired vendor import: " + fullname)
sys.addaudithook(audit)
sys.meta_path.insert(0, Block())
from scripts.shared.assets import verify
from scripts.shared.common import load_config, provenance
from scripts.sim_pretrain.simulation import PretrainingWipe
assert verify()["passed"]
assert not any(p.startswith(("DISCOVERSE/", "archive/")) for p in provenance()["sources"])
env = PretrainingWipe(load_config("configs/sim_pretrain/pretrain_paper.yaml"), 300)
try:
    for _ in range(10):
        env.sim.step()
finally:
    env.close()
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
