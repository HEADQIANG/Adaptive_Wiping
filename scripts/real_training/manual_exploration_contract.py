"""Stdlib-only validation and immutable links for manual-start exploration logs."""

import hashlib
import json
import math
from pathlib import Path

from scripts.shared.exploration_protocol import PRESS_DURATION_S, PRESS_SPEED_M_S

PROTOCOL = "airbot_manual_start_exploration_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_vector(value, size):
    if (not isinstance(value, list) or len(value) != size
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f"Expected {size} finite values")
    return value


def close(a, b, tolerance=1e-9):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def validate(rows):
    if not rows:
        raise ValueError("Empty manual exploration log")
    header = rows[0]
    if (header.get("event") != "session_start" or header.get("mode") != "manual-start"
            or header.get("source_kind") != "real" or header.get("acquisition_protocol") != PROTOCOL
            or header.get("force_monitoring") is not True or header.get("force_limits_enforced") is not False
            or header.get("stationary_validated") is not False
            or rows[-1] != {"event": "session_complete", "idle_confirmed": True}
            or any(r.get("event") == "aborted" for r in rows)):
        raise ValueError("Expected a completed real manual-start exploration")

    def one(event):
        found = [r for r in rows if r.get("event") == event]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {event}")
        return found[0]

    one("session_start")
    one("session_complete")
    reference, tare_start, tare, initial, done = [one(name) for name in
        ("reference_captured", "tare_start", "tare_complete", "initial", "motion_complete")]
    if not (rows.index(reference) < rows.index(tare_start) < rows.index(tare) < rows.index(initial) < rows.index(done)):
        raise ValueError("Manual exploration events reordered")
    if (reference.get("noncontact_operator_confirmed") is not True
            or tare.get("noncontact_operator_confirmed") is not True
            or tare.get("stationary_validated") is not False):
        raise ValueError("Missing operator-confirmed non-contact baseline semantics")
    pose = reference["runtime_reference_pose"]
    for key, size in (("sdk_end_position_m", 3), ("sdk_end_orientation_xyzw", 4), ("joint_position_rad", 6)):
        finite_vector(pose[key], size)
    if abs(math.sqrt(sum(x * x for x in pose["sdk_end_orientation_xyzw"])) - 1) > 0.01:
        raise ValueError("Invalid reference quaternion")
    if (done.get("mode") != "manual-start" or done.get("acquisition_protocol") != PROTOCOL
            or done.get("nominal_protocol_match") is not True or done.get("time_scale") != 1
            or done.get("exploration_samples") != 400 or done.get("software_tare_applied") is not True
            or done.get("force_limits_enforced") is not False
            or done.get("press_speed_m_s") != PRESS_SPEED_M_S
            or done.get("press_duration_s") != PRESS_DURATION_S):
        raise ValueError("Manual exploration requires the current original-speed 400-frame protocol")
    bias = finite_vector(tare["tare_bias_si"], 6)
    baseline_samples = tare["samples"]
    stamps = [r["sensor_receive_perf_s"] for r in baseline_samples]
    if (len(stamps) < 40 or tare.get("distinct_samples") != len(stamps)
            or any(not math.isfinite(t) for t in stamps)
            or any(b <= a for a, b in zip(stamps, stamps[1:]))
            or stamps[-1] - stamps[0] < 0.9
            or stamps[0] < tare["start_perf_s"] or stamps[-1] > tare["end_perf_s"]):
        raise ValueError("Invalid baseline sample count or coverage")
    for row in baseline_samples:
        finite_vector(row["raw_sensor_wrench_si"], 6)
        if not 0 <= row["sensor_age_s"] <= 0.020:
            raise ValueError("Stale baseline sample")
    mean = [math.fsum(r["raw_sensor_wrench_si"][i] for r in baseline_samples) / len(stamps) for i in range(6)]
    if not all(close(a, b) for a, b in zip(mean, bias)):
        raise ValueError("Recorded baseline differs from raw sample mean")
    samples = [r for r in rows if r.get("event") == "sample"]
    if len(samples) != 600 or [r.get("phase") for r in samples] != ["exploration"] * 400 + ["retract"] * 200:
        raise ValueError("Missing or reordered exploration/retraction samples")
    starts = [r for r in rows if r.get("event") == "phase_start" and r.get("phase") in ("exploration", "retract")]
    if [r["phase"] for r in starts] != ["exploration", "retract"]:
        raise ValueError("Invalid motion phase boundaries")
    if not rows.index(initial) < rows.index(starts[0]) < rows.index(samples[0]):
        raise ValueError("Initial observation must precede exploration")
    if not rows.index(samples[399]) < rows.index(starts[1]) < rows.index(samples[400]):
        raise ValueError("Retraction must follow exploration")
    if rows.index(samples[-1]) >= rows.index(done):
        raise ValueError("Completion precedes final retraction sample")
    for segment, start in ((samples[:400], starts[0]), (samples[400:], starts[1])):
        for index, row in enumerate(segment, 1):
            if (row.get("index") != index or not close(row["due_perf_s"], start["perf_s"] + index / 100, 1e-8)
                    or not 0 <= row["lateness_s"] <= 0.005 + 1e-9
                    or not close(row["sample_perf_s"] - row["due_perf_s"], row["lateness_s"])
                    or not start["perf_s"] <= row["send_perf_s"] <= row["due_perf_s"]
                    or (start["phase"] == "exploration" and not close(row["protocol_time_s"], index / 100))):
                raise ValueError("Manual exploration sample timing/sequence invalid")
    unique = {}
    last_stamp = None
    for row in [initial, *samples]:
        for field in ("pre_send_force", "force"):
            force = row.get(field)
            if force is None:
                if field == "force" or row is not initial:
                    raise ValueError("Missing force observation")
                continue
            stamp = force["sensor_receive_perf_s"]
            raw = finite_vector(force["raw_sensor_wrench_si"], 6)
            tared = finite_vector(force["tared_sensor_wrench_si"], 6)
            if (not math.isfinite(stamp) or not 0 <= force["sensor_age_s"] <= 0.020
                    or (last_stamp is not None and stamp < last_stamp)
                    or (stamp in unique and unique[stamp] != raw)):
                raise ValueError("Invalid force timing or conflicting frames")
            if not all(close(a - b, c) for a, b, c in zip(raw, bias, tared)):
                raise ValueError("Tared force differs from raw minus baseline; no double subtraction")
            unique[stamp], last_stamp = raw, stamp
    stamps = sorted(unique)
    start = starts[0]["perf_s"]
    if (stamps[0] > start or stamps[-1] < start + 4
            or any(b - a > 0.020 + 1e-9 for a, b in zip(stamps, stamps[1:]))):
        raise ValueError("Incomplete force coverage or gap over 20 ms")
    return header


def load_completed(path):
    payload = Path(path).read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return validate(rows), hashlib.sha256(payload).hexdigest()


def verify_setup(header, cfg):
    for key in ("robot_sn", "expected_eef_type", "sensor_port", "table_normal_sdk",
                "slide_direction_sdk", "sponge_id", "exploration_id"):
        if not cfg.get(key) or cfg[key] != header["config"].get(key):
            raise ValueError(f"Exploration/program setup mismatch: {key}")


def verify_binding(binding, header, sha256, cfg):
    if (not isinstance(binding, dict) or binding.get("sha256") != sha256
            or binding.get("acquisition_protocol") != PROTOCOL
            or binding.get("exploration_id") != cfg["exploration_id"]
            or binding.get("same_setup_confirmed") is not True):
        raise ValueError("Manual exploration requires a new demonstration session bound before collection")
    verify_setup(header, cfg)
