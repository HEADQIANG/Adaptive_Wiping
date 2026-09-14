"""Shared robot and force guards; no learning or simulation dependency."""

import json
import math
import time

from .client import StateError, vector

MODES = ("force-guarded", "contact-no-ft", "air")
MAX_FT_AGE_S = 0.020


def norm(values):
    return math.sqrt(sum(x * x for x in values))


def quaternion_angle(a, b):
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b))) / (norm(a) * norm(b))))


def positive(value, name, maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Set a finite, measured/approved {name}")
    if not 0 < value <= maximum:
        raise ValueError(f"Invalid {name}: expected 0 < value <= {maximum}")
    return value


def checked_vector(value, size, name):
    if not isinstance(value, list):
        raise ValueError(f"Set {name}: {size} finite values")
    return vector(value, size, name)


def no_ft(cfg):
    mode = cfg.get("mode", "force-guarded")
    if mode not in MODES:
        raise ValueError(f"Unknown exploration mode: {mode}")
    return mode != "force-guarded"


def measured_joint_speed_limit(cfg):
    command = positive(cfg.get("joint_speed_limit_rad_s"), "joint_speed_limit_rad_s", 0.5)
    limit = positive(
        cfg.get("measured_joint_speed_stop_rad_s", command), "measured_joint_speed_stop_rad_s", 1.2
    )
    if limit < command:
        raise ValueError("Measured joint speed stop must not be below command speed limit")
    if no_ft(cfg) and limit != command:
        raise ValueError("Separate measured speed stop requires force-guarded mode")
    return limit


def check_bounds(state, cfg):
    """Check finite pose data and joint limits, without a project XYZ envelope."""
    vector(state["sdk_end_position_m"], 3, "SDK end position")
    joints = vector(state["joint_position_rad"], 6, "joint position")
    if any(
        not lo <= x <= hi
        for x, lo, hi in zip(
            joints, cfg["joint_min_rad"], cfg["joint_max_rad"]
        )
    ):
        raise StateError("Joint position outside approved limits")


def read_force(sensor, cfg, *, initial=False):
    if no_ft(cfg):
        return None
    if sensor is None:
        raise StateError(
            "Force-guarded exploration requires a live force sensor; no automatic bypass"
        )
    # Preserve gravity and raw sensor-origin axes. Never tare a contacting tool.
    item = sensor.latest(net=False)
    if item is None:
        raise StateError("No force sensor data")
    stamp, raw = item
    age = time.perf_counter() - stamp
    if not math.isfinite(stamp) or not 0 <= age <= MAX_FT_AGE_S:
        raise StateError(f"Force sensor stale/invalid: age={age:.6f}s")
    raw = vector(raw, 6, "raw sensor wrench")
    wrench = [x - b for x, b in zip(raw, cfg["sensor_bias_si"])]
    limit = cfg["max_initial_force_n"] if initial else cfg["max_force_n"]
    if norm(wrench[:3]) > limit or norm(wrench[3:]) > cfg["max_torque_nm"]:
        raise StateError(f"Force/torque limit exceeded: {wrench}")
    return {
        "sensor_receive_perf_s": stamp,
        "sensor_age_s": age,
        "raw_sensor_wrench_si": raw,
        "bias_corrected_sensor_wrench_si": wrench,
    }


def write_event(stream, event):
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()


def interrupt_on_signal(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")
