import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.shared.paths import ROOT

spec = importlib.util.spec_from_file_location(
    "frame_audit", ROOT / "scripts/robot_control/tools/audit_airbot_frames.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
fk, read_dh = audit.fk, audit.read_dh


class FrameAuditTests(unittest.TestCase):
    def test_dh_order(self):
        dh = {k: np.zeros(7) for k in ("alpha", "a", "d", "theta_offset")}
        dh["a"][0] = 1
        q = np.array([np.pi / 2, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            fk(q, dh, "standard", np.eye(4), "none")[:3, 3], [0, 1, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            fk(q, dh, "modified", np.eye(4), "none")[:3, 3], [1, 0, 0], atol=1e-12
        )

    def test_end_conversion(self):
        dh = {k: np.zeros(7) for k in ("alpha", "a", "d", "theta_offset")}
        end = np.array([[0.0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(fk(np.zeros(6), dh, "modified", end, "right"), end)
        np.testing.assert_allclose(fk(np.zeros(6), dh, "modified", end, "right_inverse"), end.T)

    def test_log_validation(self):
        block = "".join(f"{k}: 0 0 0 0 0 0 0\n" for k in ("alpha", "a", "d", "theta_offset"))
        block += "end_convert:\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.txt"
            path.write_text(block * 2)
            dh, end = read_dh(path)
            self.assertEqual(dh["a"].shape, (7,))
            np.testing.assert_array_equal(end, np.eye(4))
            path.write_text(block + block.replace("a: 0 0", "a: 1 0"))
            with self.assertRaises(ValueError):
                read_dh(path)
            path.write_text(block.replace("end_convert:\n1", "end_convert:\n2"))
            with self.assertRaises(ValueError):
                read_dh(path)


if __name__ == "__main__":
    unittest.main()
