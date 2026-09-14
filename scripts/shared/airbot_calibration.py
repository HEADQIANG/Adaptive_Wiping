"""Measured AIRBOT frame contract shared by offline conversion and deployment."""

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.shared.common import CHANNELS, file_digest
from scripts.shared.paths import read_path as resolve
from scripts.shared.real_preprocessing import finite_array
from scripts.shared.sampling import previous_samples

FRAME_KEYS = ("base_from_sdk", "end_from_tcp", "sensor_to_ft_frame", "sensor_bias_si")
VERIFICATIONS = (
    "units_and_axes_verified",
    "electronic_bias_verified",
    "tcp_geometry_verified",
    "applies_to_recorded_setup",
)
CONVENTIONS = {
    "position_frame": "base_link",
    "position_units": "m",
    "orientation_order": "xyzw",
    "ft_frame": "ft_frame local",
    "ft_reference_point": "ft_frame origin",
    "compensation": "sensor_bias_only_gravity_retained",
    "channels": CHANNELS,
    "ft_units": ["N"] * 3 + ["N*m"] * 3,
}


def rigid(value, name):
    value = finite_array(value, name)
    if (
        value.shape != (4, 4)
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a measured right-handed rigid transform")
    return value


def quaternion(value):
    value = finite_array(value, "SDK quaternion")
    if value.shape != (4,) or not np.isclose(np.linalg.norm(value), 1, atol=1e-3, rtol=0):
        raise ValueError("Expected unit SDK xyzw quaternion")
    return Rotation.from_quat(value)


class Frames:
    def __init__(self, cfg):
        self.base = rigid(cfg["base_from_sdk"], "base_from_sdk")
        self.tcp = rigid(cfg["end_from_tcp"], "end_from_tcp")
        self.ft = rigid(cfg["sensor_to_ft_frame"], "sensor_to_ft_frame")
        self.bias = finite_array(cfg["sensor_bias_si"], "sensor_bias_si")
        if self.bias.shape != (6,):
            raise ValueError("Expected six measured electronic bias values")

    def to_tcp(self, state):
        r = quaternion(state["sdk_end_orientation_xyzw"]).as_matrix()
        end = finite_array(state["sdk_end_position_m"], "SDK position")
        if end.shape != (3,):
            raise ValueError("Expected SDK end position [3]")
        return self.base[:3, :3] @ (end + r @ self.tcp[:3, 3]) + self.base[:3, 3]

    def tcp_quaternion(self, sdk_quaternion):
        r = self.base[:3, :3] @ quaternion(sdk_quaternion).as_matrix() @ self.tcp[:3, :3]
        return Rotation.from_matrix(r).as_quat()

    def to_end(self, target, sdk_quaternion):
        target = finite_array(target, "TCP position")
        if target.shape != (3,):
            raise ValueError("Expected TCP position [3]")
        r = quaternion(sdk_quaternion).as_matrix()
        return self.base[:3, :3].T @ (target - self.base[:3, 3]) - r @ self.tcp[:3, 3]

    def wrench(self, raw):
        raw = finite_array(raw, "sensor FT")
        if raw.shape != (6,):
            raise ValueError("Expected sensor FT [6]")
        value = raw - self.bias
        force = self.ft[:3, :3] @ value[:3]
        torque = self.ft[:3, :3] @ value[3:] + np.cross(self.ft[:3, 3], force)
        return np.r_[force, torque]

    def wrenches(self, raw):
        raw = finite_array(raw, "sensor FT stream")
        if raw.ndim != 2 or raw.shape[1] != 6 or not len(raw):
            raise ValueError("Expected nonempty sensor FT [N,6]")
        return np.array([self.wrench(row) for row in raw])


def checked_received(time_s, values, duration):
    time_s = finite_array(time_s, "received times")
    values = finite_array(values, "received FT")
    if time_s.ndim != 1 or len(time_s) < 2 or values.shape != (len(time_s), 6):
        raise ValueError("Invalid received stream dimensions")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Received timestamps must be strictly increasing")
    if time_s[0] > 1e-9 or time_s[-1] < duration - 1e-9:
        raise ValueError("Received stream does not cover the full segment")
    first = max(0, np.searchsorted(time_s, 1e-9, side="right") - 1)
    last = min(len(time_s), np.searchsorted(time_s, duration, side="left") + 1)
    times, ft = time_s[first:last], values[first:last]
    if -times[0] > 0.020 + 1e-9 or np.any(np.diff(times) > 0.020 + 1e-9):
        raise ValueError("Received FT exceeds the 20 ms age/gap limit")
    grid = np.arange(round(duration * 100) + 1, dtype=np.float64) / 100
    return grid, previous_samples(times, ft, grid)


def calibration_issues(record):
    issues = []
    expected = {
        "schema_version": 1,
        "artifact_kind": "airbot_physical_calibration",
        "source_kind": "real",
    }
    for key, value in {**expected, **CONVENTIONS}.items():
        if record.get(key) != value or (
            key == "schema_version" and type(record.get(key)) is not int
        ):
            issues.append(f"Expected calibration {key}={value!r}")
    for key in (
        "calibration_id",
        "robot_id",
        "sensor_id",
        "tcp_definition",
        "sensor_frame",
        "method_notes",
    ):
        if not isinstance(record.get(key), str) or not record[key].strip():
            issues.append(f"Missing measured calibration field: {key}")
    verification = record.get("verification")
    for flag in VERIFICATIONS:
        if not isinstance(verification, dict) or verification.get(flag) is not True:
            issues.append(f"Physical verification required: {flag}")
    for key in FRAME_KEYS:
        try:
            if key == "sensor_bias_si":
                if finite_array(record.get(key), key).shape != (6,):
                    raise ValueError("sensor_bias_si requires six measured values")
            else:
                rigid(record.get(key), key)
        except (ValueError, TypeError) as exc:
            issues.append(str(exc))
    if not isinstance(record.get("evidence"), list) or not record["evidence"]:
        issues.append("Measured calibration evidence files and SHA256 values are required")
    return issues


class Calibration:
    def __init__(self, path):
        self.path = resolve(path).resolve()
        self.sha256 = file_digest(self.path)
        self.record = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.record, dict):
            raise ValueError("Calibration must be a JSON object")
        issues = calibration_issues(self.record)
        if issues:
            raise ValueError("Calibration incomplete: " + "; ".join(issues))
        self.evidence = {}
        for entry in self.record["evidence"]:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
            ):
                raise ValueError("Invalid calibration evidence entry")
            source = Path(entry["path"])
            source = source if source.is_absolute() else self.path.parent / source
            if source.resolve() == self.path or not source.is_file() or source.stat().st_size == 0:
                raise ValueError("Calibration evidence must be a distinct nonempty file")
            self.evidence[str(source.resolve())] = entry["sha256"]
        self.frames = Frames(self.record)
        self.assert_unchanged()

    def assert_unchanged(self):
        for path, expected in {str(self.path): self.sha256, **self.evidence}.items():
            if file_digest(path) != expected:
                raise ValueError("Calibration record or physical evidence changed")

    def metadata(self):
        fields = ("robot_id", "sensor_id", "calibration_id", "tcp_definition", "sensor_frame")
        return {
            **CONVENTIONS,
            **{key: self.record[key] for key in fields},
            "sensor_to_ft_frame": self.record["sensor_to_ft_frame"],
            "calibration_status": "operator_measured_evidence_supplied_not_software_certified",
            "calibration_sha256": self.sha256,
            "calibration_evidence_sha256": dict(self.evidence),
            "frame_calibration": {key: self.record[key] for key in FRAME_KEYS},
        }

    def bind_deployment(self, cfg, training_metadata):
        expected = self.metadata()
        for key in ("calibration_sha256", "robot_id", "sensor_id", "calibration_id", *CONVENTIONS):
            if training_metadata.get(key) != expected[key]:
                raise ValueError(f"Training/calibration mismatch: {key}")
        for key in FRAME_KEYS:
            if training_metadata.get("frame_calibration", {}).get(key) != self.record[key]:
                raise ValueError(f"Training frame calibration mismatch: {key}")
            if cfg.get(key) is not None and cfg[key] != self.record[key]:
                raise ValueError(f"Deployment override differs from calibration: {key}")
            cfg[key] = self.record[key]
        return self.frames
