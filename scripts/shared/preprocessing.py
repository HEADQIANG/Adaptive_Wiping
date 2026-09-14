"""Offline filtering and training-set-only normalization."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


class Preprocessor:
    def __init__(self, order=2, cutoff_hz=10.0, sample_hz=100):
        self.order, self.cutoff_hz, self.sample_hz = order, cutoff_hz, sample_hz
        self.minimum = self.maximum = self.constant = None

    def filtered(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 3 or data.shape[1:] != (400, 6) or not np.isfinite(data).all():
            raise ValueError("Expected finite force trajectories [batch,400,6]")
        sos = butter(self.order, self.cutoff_hz, fs=self.sample_hz, output="sos")
        return sosfiltfilt(sos, data, axis=1)

    def fit(self, training_data):
        if len(training_data) == 0:
            raise ValueError("Empty training set")
        data = self.filtered(training_data)
        self.minimum = data.min(axis=(0, 1))
        self.maximum = data.max(axis=(0, 1))
        self.constant = self.maximum - self.minimum < 1e-8
        return self

    def transform(self, data):
        if self.minimum is None:
            raise RuntimeError("Preprocessor is not fitted")
        span = np.where(self.constant, 1.0, self.maximum - self.minimum)
        return (0.9 * (self.filtered(data) - self.minimum) / span).astype(np.float32)

    def inverse(self, data):
        span = np.where(self.constant, 1.0, self.maximum - self.minimum)
        return np.asarray(data) / 0.9 * span + self.minimum

    def state(self):
        return {
            "order": self.order,
            "cutoff_hz": self.cutoff_hz,
            "sample_hz": self.sample_hz,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "constant": self.constant.tolist(),
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(state["order"], state["cutoff_hz"], state["sample_hz"])
        obj.minimum, obj.maximum = np.asarray(state["minimum"]), np.asarray(state["maximum"])
        obj.constant = np.asarray(state["constant"], dtype=bool)
        return obj
