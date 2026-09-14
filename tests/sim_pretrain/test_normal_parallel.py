"""Ordered prefetch without changing the original recorder's failure semantics."""

import sys
import unittest
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.collect_normal_parallel import (
    OrderedRollouts,
    RolloutProxy,
)


class FakeExecutor:
    def __init__(self, **kwargs):
        self.jobs = []
        self.shutdown_called = False

    def submit(self, fn, parameters):
        self.jobs.append(parameters)
        future = Future()
        future.set_result(
            {
                "data": {"parameters": np.array(parameters)},
                "unloaded": [0] * 6,
                "unload_error": None,
            }
        )
        return future

    def shutdown(self, **kwargs):
        self.shutdown_called = True


class NormalParallelTests(unittest.TestCase):
    def provider(self):
        rows = [(index / 20, 100.0, 0.02) for index in range(12)]
        provider = OrderedRollouts(
            {},
            {"assignments": {"train": rows[:8], "validation": rows[8:10], "test": rows[10:]}},
            2,
            executor_factory=FakeExecutor,
        )
        self.addCleanup(provider.close)
        return provider, rows

    def test_prefetch_is_bounded_and_preserves_requested_order_and_resume(self):
        provider, rows = self.provider()
        for index in (3, 4, 7, 9, 10, 11):
            proxy = provider.factory({}, rows[index], None)
            np.testing.assert_array_equal(proxy.rollout()["parameters"], rows[index])
            self.assertEqual(proxy.unload(None), [0] * 6)
            self.assertLess(len(provider.jobs), provider.window)
            proxy.close()
        self.assertEqual(provider.jobs, {})
        self.assertEqual(len(provider.pool.jobs), len(set(provider.pool.jobs)))
        self.assertEqual(provider.pool.jobs[0], rows[3])

    def test_unload_failure_keeps_rollout_available_and_wrong_mode_rejected(self):
        provider, rows = self.provider()
        proxy = RolloutProxy(provider, rows[0])
        data = proxy.rollout()
        proxy.result["unload_error"] = "synthetic unload failure"
        with self.assertRaisesRegex(RuntimeError, "unload failure"):
            proxy.unload(data)
        np.testing.assert_array_equal(data["parameters"], rows[0])
        with self.assertRaisesRegex(ValueError, "normal mode"):
            provider.factory({}, rows[0], {"mode": "impedance"})

    def test_duplicate_assignments_and_worker_bounds_rejected(self):
        manifest = {"assignments": {"train": [(0, 1, 0.02), (0, 1, 0.02)]}}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            OrderedRollouts({}, manifest, 2, executor_factory=FakeExecutor)
        for workers in (0, 9):
            with self.assertRaisesRegex(ValueError, "workers"):
                OrderedRollouts({}, manifest, workers, executor_factory=FakeExecutor)


if __name__ == "__main__":
    unittest.main()
