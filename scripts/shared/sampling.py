"""Causal timestamp alignment shared by calibration and data import."""

import numpy as np

from .real_preprocessing import finite_array


def previous_samples(time, values, targets, max_age=0.02):
    time = finite_array(time, "sample timestamps")
    targets = finite_array(targets, "target timestamps")
    indices = np.searchsorted(time, targets + 1e-9, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("No past sample at requested time")
    if np.any(targets - time[indices] > max_age + 1e-9):
        raise ValueError("Requested sample is stale")
    return np.asarray(values)[indices]
