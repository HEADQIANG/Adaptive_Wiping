"""Reference timing and score tests use synthetic fixtures, not training data."""

import json
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from scripts.sim_pretrain.experiments.match_real_amplitude import (
    TIMES, comparison, coverage_summary, random_grid, real_trace,
)


def log_rows():
    return [
        {"event": "phase_start", "phase": "exploration", "perf_s": 100},
        *[dict(event="sample", phase="exploration", protocol_time_s=float(t),
               force=dict(sensor_receive_perf_s=100 + (i//2*2+1)/100,
                          tared_sensor_wrench_si=[float(i//2)]*6))
          for i, t in enumerate(TIMES)],
        {"event": "session_complete"},
    ]


class MatchRealAmplitudeTests(unittest.TestCase):
    def test_random_grid_reproducible_and_stratified(self):
        bounds = ((2000,10000),(.4,1),(100,10000),(.001,.1))
        a = np.asarray(random_grid(32, 42, bounds))
        np.testing.assert_array_equal(a, random_grid(32, 42, bounds))
        for j, (low, high) in enumerate(bounds):
            u = (a[:,j]-low)/(high-low) if j == 1 else np.log(a[:,j]/low)/np.log(high/low)
            np.testing.assert_array_equal(np.sort((u*32).astype(int)), np.arange(32))
        with self.assertRaises(ValueError):
            random_grid(2, 1, ((0,1),(.4,1),(1,2),(1,2)))

    def test_marginal_coverage_does_not_imply_joint_coverage(self):
        real = np.zeros((400,6))
        result = coverage_summary([np.ones_like(real), -np.ones_like(real)], real)
        self.assertEqual(result["curve_fraction_channels"], [1]*6)
        self.assertEqual(result["joint_time_fraction_at_10pct"], 0)
        self.assertEqual(result["whole_trace_matches_at_10pct"], 0)
        self.assertEqual(coverage_summary([], real)["status"], "no_candidates")
        result = coverage_summary([real], real)
        self.assertEqual(result["whole_trace_matches_at_10pct"], 1)
        self.assertTrue(np.asarray(result["phase_inside"]).all())

    def read(self, rows):
        with patch("pathlib.Path.open", mock_open(read_data="\n".join(json.dumps(r) for r in rows))):
            return real_trace("synthetic.jsonl")

    def test_duplicate_sensor_timestamps_follow_plot_semantics(self):
        times, ft, aligned = self.read(log_rows())
        self.assertEqual(len(times), 200)
        self.assertEqual(aligned.shape, (400, 6))
        np.testing.assert_allclose(times[:2], [.01, .03])
        np.testing.assert_allclose(aligned[1], .5)
        np.testing.assert_allclose(aligned[-1], ft[-1])

    def test_incomplete_aborted_and_invalid_references_rejected(self):
        for kind in ("incomplete", "aborted", "missing", "nan"):
            rows = log_rows()
            if kind == "incomplete":
                rows.pop()
            elif kind == "aborted":
                rows.append({"event": "aborted"})
            elif kind == "missing":
                rows.pop(100)
            else:
                rows[100]["force"]["tared_sensor_wrench_si"][0] = float("nan")
            with self.assertRaises(ValueError):
                self.read(rows)

    def test_matching_score_zero_and_no_amplitude_or_sign_fitting(self):
        reference = np.tile([4, -6, -12, .2, .1, -.05], (400, 1))
        self.assertEqual(comparison(reference, reference)["score"], 0)
        self.assertAlmostEqual(comparison(reference/2, reference)["score"], .5)
        self.assertAlmostEqual(comparison(-reference, reference)["score"], 2)
        self.assertTrue(np.isfinite(comparison(np.zeros((400, 6)), np.zeros((400, 6)))["score"]))


if __name__ == "__main__":
    unittest.main()
