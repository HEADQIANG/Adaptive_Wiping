"""Manual tare import contracts; generated fixtures never leave temporary folders."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.real_training.airbot_demonstrations import quality
from scripts.real_training.config import load_config
from scripts.real_training.data import _load_raw, load_prepared, prepare, write_raw_log
from scripts.real_training.import_airbot import assemble, events, import_dataset, main
from scripts.shared.common import ROOT, file_digest
from tests.real_training.test_manual_exploration import Fixture

OPTIONS = dict(manual_tared=True, subtract_recorded_baseline=True, confirm_same_setup=True)


class ManualTaredTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.exploration = self.log_file()
        self.session = self.folder / "manual"
        self.session.mkdir()
        self.meta = {
            "force_recording": "software_tared", "primary_force_field": "tared_sensor_wrench_si",
            "config": {**self.cfg, "force_recording": "software_tared", "demonstrations": 8,
                       "duration_s": 10, "sample_hz": 100, "training_hz": 2.5, "surface_id": "test"},
        }
        self.save("session.json", self.meta)
        self.tare = {
            "id": "a" * 32, "method": "mean_raw_sensor_wrench_noncontact_1s",
            "noncontact_confirmed_by": "operator_z", "stationary_check": False,
            "raw_baseline_si": [1, 2, 3, 4, 5, 6], "unique_samples": 100, "receive_span_s": 0.99,
        }
        baseline = [self.sample(50 + i / 100, self.tare["raw_baseline_si"]) for i in range(100)]
        self.save("tare_" + self.tare["id"] + ".json", {**self.tare, "samples": baseline})
        for i in range(1, 9):
            samples = [self.sample(100 * i + j / 100, [j / 100, 2, 4, 4, 5, 6]) for j in range(1003)]
            report = quality(samples, 100 * i)
            rows = [{"event": "start", "start_perf_s": 100 * i,
                     "force_recording": "software_tared", "tare": self.tare},
                    *samples, {"event": "finished", "quality": report}]
            raw = self.session / f"attempt_{i}.jsonl"
            raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.save(f"attempt_{i}.start.json", {"tare": self.tare})
            self.save(f"demo_{i:02}.json", {
                "source_kind": "real", "accepted": True, "training_ready": False,
                "force_recording": "software_tared", "tare": self.tare, "quality": report,
                "raw_file": raw.name, "sha256": file_digest(raw),
            })

    def save(self, name, value):
        (self.session / name).write_text(json.dumps(value))

    def sample(self, stamp, raw):
        return {"event": "sample", "observed_perf_s": stamp,
                "pose": {"host_monotonic_s": stamp - 0.0001, "read_duration_s": 0.001,
                         "sdk_end_position_m": [stamp / 1000, 0, 0.2],
                         "sdk_end_orientation_xyzw": [0, 0, 0, 1]},
                "ft": {"sensor_receive_perf_s": stamp, "sensor_age_s": 0,
                       "raw_sensor_wrench_si": raw, "tare_id": self.tare["id"],
                       "tared_sensor_wrench_si": (np.asarray(raw) - self.tare["raw_baseline_si"]).tolist()}}

    def mutate_raw(self, callback, *, refresh_hash=True):
        path = self.session / "attempt_1.jsonl"
        rows = events(path)
        callback(rows)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if refresh_hash:
            record = json.loads((self.session / "demo_01.json").read_text())
            record["sha256"] = file_digest(path)
            self.save("demo_01.json", record)

    def assemble(self, **options):
        return assemble(self.exploration, self.session, **{**OPTIONS, **options})

    def test_measured_streams_preserved_and_shapes(self):
        before = {p: file_digest(p) for p in self.session.iterdir()}
        meta, exp, demos = self.assemble()
        self.assertEqual(meta["derivation"], "manual_recorded_baseline_10s_v1")
        self.assertFalse(meta["contains_derived_samples"])
        self.assertFalse(meta["setup_confirmation"]["bound_before_collection"])
        self.assertEqual(meta["setup_confirmation"]["confirmation_stage"], "offline_import_after_collection")
        self.assertEqual(len(meta["source_hashes"]), 27)
        original = events(self.session / "attempt_1.jsonl")[1:-1]
        np.testing.assert_array_equal(demos["demo_01"]["ft_time"], [r["ft"]["sensor_receive_perf_s"] for r in original])
        np.testing.assert_array_equal(demos["demo_01"]["sdk_end_position"], [r["pose"]["sdk_end_position_m"] for r in original])
        for name, demo in demos.items():
            self.assertNotIn("is_padding", demo)
            np.testing.assert_array_equal(demo["ft"], demo["ft_raw_before_baseline"] - self.tare["raw_baseline_si"])
        cfg = load_config(ROOT / "configs/real_training/real_training_manual_tared.yaml")
        raw = self.folder / "raw.h5"
        write_raw_log(raw, meta, exp, demos, source_kind="real")
        arrays, _ = _load_raw({**cfg, "raw_data": str(raw)})
        self.assertEqual(arrays["ft"].shape, (8, 25, 6))
        self.assertEqual(arrays["xy"].shape, (8, 25, 2))
        self.assertEqual(arrays["exploration"].shape, (1, 400, 6))
        self.assertEqual(before, {p: file_digest(p) for p in self.session.iterdir()})
        with h5py.File(raw, "r+") as h5:
            h5["demonstrations/demo_01/ft"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Tared data differs"):
            _load_raw({**cfg, "raw_data": str(raw)})

    def test_explicit_opt_ins_required_and_padding_forbidden(self):
        for option in ({"manual_tared": False}, {"subtract_recorded_baseline": False},
                       {"confirm_same_setup": False}, {"programmed_hold_last": True}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.assemble(**option)
        with self.assertRaisesRegex(ValueError, "Software-tared"):
            assemble(self.exploration, self.session)
        with self.assertRaisesRegex(ValueError, "programmed protocol"):
            assemble(self.exploration, self.session, programmed_hold_last=True, subtract_recorded_baseline=True)

    def test_confirmation_does_not_override_recorded_setup(self):
        for key in ("robot_sn", "sensor_port", "expected_eef_type", "sponge_id", "exploration_id",
                    "table_normal_sdk", "slide_direction_sdk"):
            meta = copy.deepcopy(self.meta)
            meta["config"][key] = "mismatch"
            self.save("session.json", meta)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "setup mismatch"):
                self.assemble()

    def test_changed_raw_hash_rejected(self):
        self.mutate_raw(lambda rows: rows[1]["ft"].update(tared_sensor_wrench_si=[99] * 6), refresh_hash=False)
        with self.assertRaisesRegex(ValueError, "changed demonstration"):
            self.assemble()

    def test_double_tare_rejected_even_with_updated_hash(self):
        self.mutate_raw(lambda rows: rows[1]["ft"].update(tared_sensor_wrench_si=[99] * 6))
        with self.assertRaisesRegex(ValueError, "raw minus recorded baseline"):
            self.assemble()

    def test_wrong_tare_id_rejected(self):
        self.mutate_raw(lambda rows: rows[1]["ft"].update(tare_id="b" * 32))
        with self.assertRaisesRegex(ValueError, "tare id"):
            self.assemble()

    def test_missing_sidecar_rejected(self):
        (self.session / "attempt_1.start.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.assemble()

    def test_sidecar_tare_mismatch_rejected(self):
        self.save("attempt_1.start.json", {"tare": {**self.tare, "raw_baseline_si": [0] * 6}})
        with self.assertRaisesRegex(ValueError, "sidecar differs"):
            self.assemble()

    def test_shared_baseline_hash_cannot_change_between_demos(self):
        from scripts.real_training import manual_tared_import

        counts = {}
        def changing_hash(path):
            key = str(path)
            counts[key] = counts.get(key, 0) + 1
            if path.name.startswith("tare_") and counts[key] > 1:
                return "changed"
            return file_digest(path)
        with patch.object(manual_tared_import, "file_digest", side_effect=changing_hash):
            with self.assertRaisesRegex(ValueError, "sidecar changed"):
                self.assemble()

    def test_baseline_raw_mean_recomputed(self):
        name = "tare_" + self.tare["id"] + ".json"
        baseline = json.loads((self.session / name).read_text())
        baseline["samples"][0]["ft"]["raw_sensor_wrench_si"][0] += 1
        self.save(name, baseline)
        with self.assertRaisesRegex(ValueError, "mean does not match"):
            self.assemble()

    def test_baseline_coverage_recomputed(self):
        name = "tare_" + self.tare["id"] + ".json"
        baseline = json.loads((self.session / name).read_text())
        baseline["samples"] = baseline["samples"][:30]
        self.save(name, baseline)
        with self.assertRaisesRegex(ValueError, "count, coverage"):
            self.assemble()

    def test_stale_force_and_nonfinite_clock_rejected(self):
        raw = (self.session / "attempt_1.jsonl").read_text()
        for field, value in (("sensor_age_s", 0.021), ("sensor_receive_perf_s", float("nan"))):
            (self.session / "attempt_1.jsonl").write_text(raw)
            self.mutate_raw(lambda rows: rows[1]["ft"].update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.assemble()

    def test_internal_gap_rejected(self):
        self.mutate_raw(lambda rows: rows.__delitem__(slice(40, 45)))
        with self.assertRaisesRegex(ValueError, "gap exceeds"):
            self.assemble()

    def test_aborted_demo_and_exploration_rejected(self):
        self.mutate_raw(lambda rows: rows[-1].update(event="aborted"))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.assemble()
        with self.exploration.open("a") as stream:
            stream.write('{"event":"aborted"}\n')
        with self.assertRaisesRegex(ValueError, "completed"):
            self.assemble()

    def test_raw_profile_cannot_discard_baseline(self):
        cfg = load_config(ROOT / "configs/real_training/real_training_airbot_native.yaml")
        with self.assertRaisesRegex(ValueError, "raw profiles forbid"):
            import_dataset(cfg, self.exploration, self.session, audit_only=True, **OPTIONS)

    def test_cli_forwards_audit_flags_without_output_allocation(self):
        with patch("scripts.real_training.import_airbot.import_dataset", return_value={}) as imported, patch(
            "scripts.shared.run_paths.new_run_config", side_effect=AssertionError("audit must not allocate")):
            main(["--config", "configs/real_training/real_training_manual_tared.yaml",
                  "--manual-tared", "--subtract-recorded-baseline", "--confirm-same-setup", "--audit-only"])
        for key, value in {**OPTIONS, "audit_only": True}.items():
            self.assertEqual(imported.call_args.kwargs[key], value)


class RealManualTaredAudit(unittest.TestCase):
    def test_real_import_prepare_and_source_integrity(self):
        session = ROOT / "runs/real_demonstrations/manual/0915_163046/session_001"
        exploration = ROOT / "runs/real_exploration/0915_152957/manual_exploration_001.jsonl"
        cfg = load_config(ROOT / "configs/real_training/real_training_manual_tared.yaml")
        if not (session / "demo_08.json").exists() or not exploration.exists() or not (ROOT / cfg["encoder"]).exists():
            self.skipTest("Real manual collection and encoder audit fixtures not installed")
        with tempfile.TemporaryDirectory() as temp:
            cfg = {**cfg, "raw_data": str(Path(temp) / "raw.h5"), "output_dir": str(Path(temp) / "prepared")}
            report = import_dataset(cfg, exploration, session, audit_only=True, **OPTIONS)
            self.assertFalse(Path(cfg["raw_data"]).exists())
            self.assertEqual(report["valid_windows_total"], 160)
            import_dataset(cfg, exploration, session, **OPTIONS)
            with self.assertRaises(FileExistsError):
                import_dataset(cfg, exploration, session, **OPTIONS)
            prepare(cfg)
            arrays, info = load_prepared(cfg)
            self.assertEqual(arrays["ft"].shape, (8, 25, 6))
            self.assertEqual(info["valid_windows_total"], 160)
            self.assertNotIn("is_padding", arrays)
            for path, expected in report["source_hashes"].items():
                self.assertEqual(file_digest(path), expected)


if __name__ == "__main__":
    unittest.main()
