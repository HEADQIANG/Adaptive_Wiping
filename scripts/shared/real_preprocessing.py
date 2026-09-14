"""Causal FT filtering and fold-local physical-unit scaling."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def finite_array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


class CausalFTFilter:
    def __init__(self, sample_hz=100.0, order=2, cutoff_hz=1.0):
        if not np.isfinite(sample_hz) or sample_hz < 100:
            raise ValueError("FT nominal sample rate must be at least 100 Hz")
        self.sample_hz, self.order, self.cutoff_hz = float(sample_hz), order, cutoff_hz
        self.sos = butter(order, cutoff_hz, fs=sample_hz, output="sos")
        self.reset()

    def reset(self):
        self.zi = None

    def process(self, samples):
        samples = finite_array(samples, "FT")
        if samples.ndim != 2 or samples.shape[1] != 6:
            raise ValueError("Expected FT [time,6]")
        if not len(samples):
            return samples.copy()
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * samples[0][None, None, :]
        result, self.zi = sosfilt(self.sos, samples, axis=0, zi=self.zi)
        return result

    def push(self, sample):
        sample = finite_array(sample, "FT sample")
        if sample.shape != (6,):
            raise ValueError("Expected FT sample [6]")
        return self.process(sample[None])[0]


class ChannelScaler:
    def fit(self, data):
        data = finite_array(data, "scaler data")
        if data.ndim < 2 or not data.size:
            raise ValueError("Scaler expects nonempty [...,channels]")
        axes = tuple(range(data.ndim - 1))
        self.minimum = data.min(axis=axes)
        self.maximum = data.max(axis=axes)
        self.constant = (self.maximum - self.minimum) < 1e-8
        return self

    @property
    def span(self):
        return np.where(self.constant, 1.0, self.maximum - self.minimum)

    def _array(self, data):
        data = finite_array(data, "scaled data")
        if data.ndim < 1 or data.shape[-1] != len(self.minimum):
            raise ValueError("Scaler channel count mismatch")
        return data

    def transform(self, data):
        return (0.9 * (self._array(data) - self.minimum) / self.span).astype(np.float32)

    def inverse(self, data):
        return self._array(data) / 0.9 * self.span + self.minimum

    def state(self):
        return {name: getattr(self, name).tolist() for name in ("minimum", "maximum", "constant")}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.minimum = finite_array(state["minimum"], "minimum")
        obj.maximum = finite_array(state["maximum"], "maximum")
        obj.constant = np.asarray(state["constant"], dtype=bool)
        if (
            obj.minimum.ndim != 1
            or obj.minimum.shape != obj.maximum.shape
            or obj.minimum.shape != obj.constant.shape
            or np.any(obj.maximum < obj.minimum)
            or not np.array_equal(obj.constant, obj.maximum - obj.minimum < 1e-8)
        ):
            raise ValueError("Invalid scaler state")
        return obj


def out_of_range(data):
    data = np.asarray(data)
    return np.mean((data < 0) | (data > 0.9), axis=tuple(range(data.ndim - 1))).tolist()


def make_windows(ft, height):
    ft, height = finite_array(ft, "FT"), finite_array(height, "height")
    if ft.ndim != 3 or ft.shape[1:] != (25, 6) or height.shape != ft.shape[:2]:
        raise ValueError("Expected FT [episodes,25,6] and height [episodes,25]")
    windows = np.stack([ft[:, k - 4 : k + 1] for k in range(4, 24)], axis=1)
    targets = (height[:, 5:] - height[:, 4:-1])[..., None]
    return windows, targets
