"""Signed reference ratios and exclusion of synthetic tails."""

import unittest

import numpy as np

from scripts.real_deploy.plot_reference_ratio import reference_statistics, signed_ratio


class RatioTests(unittest.TestCase):
    def test_negative_force_reference(self):
        np.testing.assert_allclose(signed_ratio([-5, -10, -15, 5], -10), [50, 100, 150, -50])

    def test_zero_reference_is_undefined(self):
        self.assertTrue(np.isnan(signed_ratio([0, 1], 0)).all())

    def test_small_reference_not_clipped(self):
        np.testing.assert_allclose(signed_ratio([0.1], 0.001), [10000])

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            signed_ratio([np.nan], 1)

    def test_reference_excludes_startup_and_padding(self):
        demos = []
        for duration, level in ((5.6, 2), (6.4, 4)):
            time = np.arange(1001) / 100
            ft = np.full((1001, 6), 999.0)
            mask = (time >= duration - 4 - 1e-9) & (time < duration - 1e-9)
            ft[mask] = level
            demos.append(dict(time=time, ft=ft, duration=duration))
        mean, rms, _, per_demo = reference_statistics(demos)
        np.testing.assert_allclose(mean, 3)
        np.testing.assert_allclose(rms, np.sqrt(10))
        np.testing.assert_allclose(per_demo[:, 0], [2, 4])


if __name__ == "__main__":
    unittest.main()
