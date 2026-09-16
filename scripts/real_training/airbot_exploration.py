"""Attended AIRBOT 5.2.2 exploration. Default preview never connects to hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import select
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts.robot_control.adapter import DeadlineStub, Robot
from scripts.robot_control.airbot_initial_pose import validate_stationary
from scripts.robot_control.client import (
    SDK_VERSION,
    StateError,
    health,
    open_client,
    snapshot,
    vector,
    wait_controller,
)
from scripts.robot_control.safety import (
    check_bounds,
    checked_vector,
    interrupt_on_signal,
    measured_joint_speed_limit,
    no_ft,
    norm,
    positive,
    quaternion_angle,
    read_force as read_guarded_force,
    write_event,
)
from scripts.shared.exploration_protocol import (
    INITIAL_GAP_M,
    PRESS_DEPTH_M,
    PRESS_DURATION_S,
    PRESS_SPEED_M_S,
    SAMPLE_HZ,
    exploration_offset,
)
from scripts.shared.paths import ROOT

START_POSITION_TOLERANCE_M = 0.002
START_JOINT_TOLERANCE_RAD = 0.03
ORIENTATION_TOLERANCE_RAD = 0.02
TRACKING_LIMIT_M = 0.005
MAX_LATENESS_S = 0.005
MAX_FT_AGE_S = 0.020
MODES = ("force-guarded", "contact-no-ft", "air")
MIN_AIR_START_CLEARANCE_M = 0.050
MIN_AIR_RUNNING_CLEARANCE_M = 0.020
TARE_DURATION_S = 1.0
MIN_TARE_SAMPLES = 20


def read_force(sensor, cfg, *, initial=False, tare_bias=None):
    result = read_guarded_force(sensor, cfg, initial=initial)
    if result is not None and tare_bias is not None:
        result["tared_sensor_wrench_si"] = [
            value - bias for value, bias in zip(result["raw_sensor_wrench_si"], tare_bias)
        ]
    return result


def collect_tare(robot, sensor, cfg, pose, stream):
    if no_ft(cfg):
        return None
    if cfg.get("initial_gap_m") != INITIAL_GAP_M:
        raise ValueError("Software tare requires the verified 1 mm non-contact start gap")
    print("Software tare: keep the tool stationary and clear of the table for 1 second.", flush=True)
    start = time.perf_counter()
    write_event(stream, {"event": "tare_start", "perf_s": start, "duration_s": TARE_DURATION_S})
    states, forces = [], []
    while time.perf_counter() - start < TARE_DURATION_S:
        state = robot.read("idle")
        check_start(state, pose, cfg)
        states.append(state)
        force = read_guarded_force(sensor, cfg, initial=True)
        stamp = force["sensor_receive_perf_s"]
        if stamp > start:
            if forces and stamp < forces[-1]["sensor_receive_perf_s"]:
                raise StateError("Force sensor timestamp moved backwards during tare")
            if not forces or stamp > forces[-1]["sensor_receive_perf_s"]:
                forces.append(force)
        time.sleep(1 / SAMPLE_HZ)
    validate_stationary(states)
    if len(forces) < MIN_TARE_SAMPLES or (
        forces[-1]["sensor_receive_perf_s"] - forces[0]["sensor_receive_perf_s"]
        < 0.9 * TARE_DURATION_S
    ):
        raise StateError("Insufficient fresh force samples or time coverage for software tare")
    # Match fixed-pose tare semantics without blocking robot and raw-load checks.
    bias = [
        math.fsum(force["raw_sensor_wrench_si"][axis] for force in forces) / len(forces)
        for axis in range(6)
    ]
    write_event(
        stream,
        {
            "event": "tare_complete",
            "start_perf_s": start,
            "end_perf_s": time.perf_counter(),
            "distinct_samples": len(forces),
            "tare_bias_si": bias,
            "samples": forces,
            "method": "mean of distinct raw sensor frames at the stationary non-contact start",
            "safety_wrench_field": "bias_corrected_sensor_wrench_si",
            "gravity_compensated": False,
        },
    )
    return bias


class ObservationError(StateError):
    def __init__(self, message, observation):
        super().__init__(message)
        self.observation = observation


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tracking_limit(cfg):
    return cfg["position_error_limit_m"] if cfg.get("mode") == "contact-no-ft" else TRACKING_LIMIT_M


def tracking_error_policy(cfg):
    policy = cfg.get("tracking_error_policy", "stop")
    if policy not in ("stop", "record-only"):
        raise ValueError("tracking_error_policy must be stop or record-only")
    if policy == "record-only" and no_ft(cfg):
        raise ValueError("record-only tracking requires force-guarded mode")
    return policy


def running_orientation_limit(cfg):
    limit = positive(
        cfg.get("orientation_error_limit_rad", ORIENTATION_TOLERANCE_RAD),
        "orientation_error_limit_rad",
        0.15,
    )
    if no_ft(cfg) and limit != ORIENTATION_TOLERANCE_RAD:
        raise ValueError("Custom orientation error limit requires force-guarded mode")
    return limit


def lateral_tracking_policy(cfg):
    policy = cfg.get("lateral_tracking_policy", "stop")
    if policy not in ("stop", "record-only"):
        raise ValueError("lateral_tracking_policy must be stop or record-only")
    if policy == "record-only" and no_ft(cfg):
        raise ValueError("record-only lateral tracking requires force-guarded mode")
    return policy


def tracking_errors(position, target, cfg):
    delta = [p - t for p, t in zip(position, target)]
    normal, slide = cfg["table_normal_sdk"], cfg["slide_direction_sdk"]
    cross = [
        normal[1] * slide[2] - normal[2] * slide[1],
        normal[2] * slide[0] - normal[0] * slide[2],
        normal[0] * slide[1] - normal[1] * slide[0],
    ]
    return {
        "position_error_m": norm(delta),
        "normal_error_m": sum(d * n for d, n in zip(delta, normal)),
        "slide_error_m": sum(d * s for d, s in zip(delta, slide)),
        "cross_slide_error_m": sum(d * c for d, c in zip(delta, cross)),
    }


def validate_air_setup(cfg, record=None):
    if cfg.get("mode") != "air":
        return
    if cfg.get("air_path_clear_confirmed") is not True:
        raise ValueError(
            "air requires air_path_clear_confirmed for the entire robot/tool swept path"
        )
    if "initial_gap_m" in cfg:
        raise ValueError("Air setup must use a new elevated start, not a contact-gap configuration")
    positive(cfg.get("air_start_clearance_m"), "air_start_clearance_m")
    if cfg["air_start_clearance_m"] < MIN_AIR_START_CLEARANCE_M:
        raise ValueError(
            "Air start needs at least 50 mm verified tool clearance over the entire lateral path"
        )
    if record is not None and record.get("metadata", {}).get("label") != "air_motion_start":
        raise ValueError(
            "Record a NEW elevated pose with --label air_motion_start; do not reuse the contact start"
        )


def check_air_clearance(state, pose, cfg):
    if cfg.get("mode") != "air":
        return None
    travel = sum(
        (p0 - p) * n
        for p0, p, n in zip(
            pose["sdk_end_position_m"], state["sdk_end_position_m"], cfg["table_normal_sdk"]
        )
    )
    # The declared lower bound includes geometry uncertainty and permitted tool
    # rotation. This is a plane-clearance estimate, not a collision detector.
    clearance = cfg["air_start_clearance_m"] - travel
    if clearance < MIN_AIR_RUNNING_CLEARANCE_M:
        raise StateError(
            "Estimated air clearance below 20 mm; stop without continuing toward contact"
        )
    return clearance


def validate_compression_budget(cfg):
    if cfg.get("mode") != "contact-no-ft":
        return
    if cfg.get("no_force_contact_confirmed") is not True:
        raise ValueError("contact-no-ft requires explicit no_force_contact_confirmed")
    if cfg.get("initial_gap_m") != INITIAL_GAP_M:
        raise ValueError("The recorded start must have a measured 1 mm gap, not precompression")
    positive(cfg.get("sponge_thickness_m"), "sponge_thickness_m")
    positive(
        cfg.get("verified_compression_allowance_m"),
        "verified_compression_allowance_m",
        cfg["sponge_thickness_m"],
    )
    positive(cfg.get("position_error_limit_m"), "position_error_limit_m", TRACKING_LIMIT_M)
    positive(cfg.get("contact_geometry_uncertainty_m"), "contact_geometry_uncertainty_m")
    positive(cfg.get("compression_reserve_m"), "compression_reserve_m")
    # The geometry bound must include gap error, table height variation along the
    # slide, SDK position error and tool-edge motion under the permitted orientation error.
    required = (
        PRESS_DEPTH_M
        - INITIAL_GAP_M
        + cfg["position_error_limit_m"]
        + cfg["contact_geometry_uncertainty_m"]
        + cfg["compression_reserve_m"]
    )
    if required > cfg["verified_compression_allowance_m"] + 1e-12:
        raise ValueError(
            f"Compression budget requires {required * 1000:.3f} mm, exceeds approved "
            f"{cfg['verified_compression_allowance_m'] * 1000:.3f} mm; do not relax unverified bounds"
        )


def check_compression(state, pose, cfg):
    if cfg.get("mode") != "contact-no-ft":
        return None
    travel = sum(
        (p0 - p) * n
        for p0, p, n in zip(
            pose["sdk_end_position_m"], state["sdk_end_position_m"], cfg["table_normal_sdk"]
        )
    )
    upper_bound = max(0.0, travel - INITIAL_GAP_M + cfg["contact_geometry_uncertainty_m"])
    if upper_bound > cfg["verified_compression_allowance_m"] - cfg["compression_reserve_m"]:
        raise StateError(
            "Estimated compression reached the reserved travel boundary; force is NOT measured"
        )
    return upper_bound


def load_setup(path, mode="force-guarded"):
    cfg = read_json(path)
    if cfg.get("mode", "force-guarded") != mode:
        raise ValueError("Configuration mode and explicit --mode must match")
    motion_only = no_ft(cfg)
    tracking_error_policy(cfg)
    running_orientation_limit(cfg)
    lateral_tracking_policy(cfg)
    validate_compression_budget(cfg)
    validate_air_setup(cfg)
    if cfg.get("schema_version") != 1 or cfg.get("calibration_confirmed") is not True:
        raise ValueError("Complete on-site calibration first; calibration_confirmed must be true")
    keys = ("calibration_id", "robot_sn", "sdk_frame_and_tool_note")
    for key in keys + (() if motion_only else ("sensor_port",)):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Set {key}; example configuration is deliberately not executable")
    for key, size in (
        ("table_normal_sdk", 3),
        ("slide_direction_sdk", 3),
        ("joint_min_rad", 6),
        ("joint_max_rad", 6),
        ("joint_current_limits", 6),
    ):
        cfg[key] = checked_vector(cfg.get(key), size, key)
    if not motion_only:
        cfg["sensor_bias_si"] = checked_vector(cfg.get("sensor_bias_si"), 6, "sensor_bias_si")
    normal, slide = cfg["table_normal_sdk"], cfg["slide_direction_sdk"]
    if (
        abs(norm(normal) - 1) > 1e-6
        or abs(norm(slide) - 1) > 1e-6
        or abs(sum(a * b for a, b in zip(normal, slide))) > 1e-6
    ):
        raise ValueError(
            "Table normal and +Y slide direction must be orthogonal unit vectors in SDK axes"
        )
    if mode != "air":
        if cfg.get("initial_gap_m") != INITIAL_GAP_M:
            raise ValueError(
                "Simulation requires an independently measured 0.001 m sponge/table gap"
            )
        positive(cfg.get("verified_compression_allowance_m"), "verified_compression_allowance_m")
        if cfg["verified_compression_allowance_m"] < PRESS_DEPTH_M - INITIAL_GAP_M:
            raise ValueError(
                "The full trajectory needs 9 mm compression allowance; do not run it on this tool"
            )
    if any(a >= b for a, b in zip(cfg["joint_min_rad"], cfg["joint_max_rad"])):
        raise ValueError("Every joint_min_rad must be below joint_max_rad")
    positive(cfg.get("joint_speed_limit_rad_s"), "joint_speed_limit_rad_s", 0.5)
    measured_joint_speed_limit(cfg)
    for value in cfg["joint_current_limits"]:
        positive(value, "joint_current_limits", 20)
    if cfg.get("expected_eef_type") != "NULL":
        positive(cfg.get("eef_current_limit"), "eef_current_limit", 20)
    if not motion_only:
        for key in ("max_force_n", "max_torque_nm", "max_initial_force_n"):
            positive(cfg.get(key), key)
        if cfg["max_initial_force_n"] > cfg["max_force_n"]:
            raise ValueError("Initial force limit must not exceed running force limit")
    pose_path = Path(cfg["initial_pose_file"])
    if not pose_path.is_absolute():
        pose_path = ROOT / pose_path
    record = read_json(pose_path)
    if record.get("metadata", {}).get("source_kind") == "synthetic":
        raise ValueError("A fake control pose cannot be used for real exploration")
    validate_air_setup(cfg, record)
    if (
        record.get("kind") != "airbot_manual_initial_pose"
        or record.get("schema_version") != 1
        or record.get("sdk_version") != SDK_VERSION
    ):
        raise ValueError("Expected a 5.2.2 airbot_initial_pose teach record")
    pose = record["initial_pose"]
    for key, size in (
        ("sdk_end_position_m", 3),
        ("sdk_end_orientation_xyzw", 4),
        ("joint_position_rad", 6),
    ):
        pose[key] = checked_vector(pose.get(key), size, key)
    if abs(norm(pose["sdk_end_orientation_xyzw"]) - 1) > 0.001:
        raise ValueError("Initial quaternion must be normalized xyzw")
    firmware = record.get("metadata", {}).get("firmware_cached") or {}
    if firmware.get("arm_sn") != cfg["robot_sn"]:
        raise ValueError("Recorded start robot serial number differs from configuration")
    if (
        cfg.get("expected_eef_type") is not None
        and firmware.get("eef_type") != cfg["expected_eef_type"]
    ):
        raise ValueError("Recorded start end-effector type differs from configuration")
    cfg["initial_pose_sha256"] = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    check_bounds(pose, cfg)
    for index in range(401):
        target = target_position(pose, cfg, index / SAMPLE_HZ)
        check_air_clearance({"sdk_end_position_m": target}, pose, cfg)
    return cfg, record


def target_position(pose, cfg, t):
    _, y, z = exploration_offset(t)
    return [
        p + y * sy + z * nz
        for p, sy, nz in zip(
            pose["sdk_end_position_m"], cfg["slide_direction_sdk"], cfg["table_normal_sdk"]
        )
    ]


def check_start(state, pose, cfg):
    check_bounds(state, cfg)
    error = math.dist(state["sdk_end_position_m"], pose["sdk_end_position_m"])
    angle = quaternion_angle(state["sdk_end_orientation_xyzw"], pose["sdk_end_orientation_xyzw"])
    joint_error = max(
        abs(a - b) for a, b in zip(state["joint_position_rad"], pose["joint_position_rad"])
    )
    tolerance = min(START_POSITION_TOLERANCE_M, tracking_limit(cfg))
    if (
        error > tolerance
        or angle > ORIENTATION_TOLERANCE_RAD
        or joint_error > START_JOINT_TOLERANCE_RAD
    ):
        raise StateError(
            f"Start mismatch: {error * 1000:.3f} mm, {angle:.4f} rad, joint {joint_error:.4f} rad; "
            "manually reposition and recheck; no automatic return"
        )
    check_compression(state, pose, cfg)
    check_air_clearance(state, pose, cfg)


def observe(
    robot, sensor, cfg, pose, target, controller="servo", initial=False, sliding=False,
    tare_bias=None,
):
    robot.owned()
    state = robot.read(controller)
    errors = tracking_errors(state["sdk_end_position_m"], target, cfg)
    orientation_error = quaternion_angle(
        state["sdk_end_orientation_xyzw"], pose["sdk_end_orientation_xyzw"]
    )
    orientation_limit = running_orientation_limit(cfg)
    record_all = not initial and tracking_error_policy(cfg) == "record-only"
    record_slide = record_all or (
        sliding and not initial and lateral_tracking_policy(cfg) == "record-only"
    )
    observation = {
        "state": state,
        "force": None,
        "target_position_m": list(target),
        "observation_perf_s": time.perf_counter(),
        **errors,
        "orientation_error_rad": orientation_error,
        "orientation_error_limit_rad": orientation_limit,
        "tracking_error_record_only": record_all,
        "position_tracking_exceeds_limit": errors["position_error_m"] > tracking_limit(cfg),
        "orientation_tracking_exceeds_limit": orientation_error > orientation_limit,
        "slide_error_record_only": record_slide,
        "slide_tracking_exceeds_5mm": abs(errors["slide_error_m"]) > TRACKING_LIMIT_M,
    }
    try:
        check_bounds(state, cfg)
        speed_stop = measured_joint_speed_limit(cfg)
        exceeded = [
            f"J{i + 1}={v:.9g} rad/s"
            for i, v in enumerate(state["joint_velocity_rad_s"])
            if abs(v) > speed_stop
        ]
        if exceeded:
            raise StateError(
                "Measured joint speed exceeded configured limit "
                f"{speed_stop:g} rad/s: " + ", ".join(exceeded)
            )
        # Relative tracking can be record-only; absolute bounds above always apply.
        monitored_error = (
            math.hypot(errors["normal_error_m"], errors["cross_slide_error_m"])
            if record_slide
            else errors["position_error_m"]
        )
        if not record_all and monitored_error > tracking_limit(cfg):
            label = "Normal/cross-slide" if record_slide else "Position"
            raise StateError(
                f"{label} tracking error exceeds {tracking_limit(cfg) * 1000:g} mm "
                f"(measured {monitored_error * 1000:.3f} mm)"
            )
        if not record_all and orientation_error > orientation_limit:
            raise StateError(
                f"Orientation changed by more than {orientation_limit:g} rad "
                f"(measured {orientation_error:.9g} rad)"
            )
        observation["force"] = read_force(sensor, cfg, initial=initial, tare_bias=tare_bias)
        observation["estimated_compression_upper_bound_m"] = check_compression(state, pose, cfg)
        observation["estimated_air_clearance_m"] = check_air_clearance(state, pose, cfg)
    except StateError as exc:
        # Carry already-read state to the abort log; stop before any extra I/O.
        raise ObservationError(str(exc), observation) from exc
    return observation


def wait_for_supported_idle(check):
    print(
        "Retraction commands complete; actual return is NOT guaranteed. Servo remains active.",
        flush=True,
    )
    print(
        "Arrange safe support outside pinch points, then type IDLE to release control.", flush=True
    )
    while True:
        check()
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if ready:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("Terminal closed")
            if line.strip() == "IDLE":
                return


def execute(robot, sensor, cfg, record, stream, time_scale=1.0, finish=wait_for_supported_idle):
    positive(time_scale, "time_scale", 10)
    if time_scale < 1:
        raise ValueError("Acceleration above simulation speed is forbidden")
    validate_compression_budget(cfg)
    validate_air_setup(cfg, record)
    lateral_tracking_policy(cfg)
    tracking_error_policy(cfg)
    pose = record["initial_pose"]
    measured_joint_speed_limit(cfg)
    acquired = False
    running_orientation_limit(cfg)
    mode_attempted = False
    context = {"phase": "preflight"}
    try:
        samples = []
        for _ in range(11):
            state = robot.read("idle")
            check_start(state, pose, cfg)
            read_force(sensor, cfg, initial=True)
            samples.append(state)
            time.sleep(0.05)
        validate_stationary(samples)
        context = {"phase": "tare"}
        tare_bias = collect_tare(robot, sensor, cfg, pose, stream)
        context = {"phase": "preflight"}
        robot.acquire()
        acquired = True
        check_start(robot.read("idle"), pose, cfg)
        read_force(sensor, cfg, initial=True)
        mode_attempted = True
        robot.enter()
        check_start(robot.read("servo"), pose, cfg)
        first = observe(
            robot, sensor, cfg, pose, pose["sdk_end_position_m"], initial=True,
            tare_bias=tare_bias,
        )
        write_event(stream, {"event": "initial", **first})
        quaternion = pose["sdk_end_orientation_xyzw"]
        robot.send(pose["sdk_end_position_m"], quaternion)
        errors, lateness, slide_errors = [], [], []
        previous_target = pose["sdk_end_position_m"]
        for phase in ("exploration", "retract"):
            # Retraction is separate from the 400 exploration samples. It reverses
            # only the protocol press, not the simulation's unvalidated 80 mm lift.
            count = round((4 if phase == "exploration" else 2) * SAMPLE_HZ * time_scale)
            t0 = time.perf_counter()
            write_event(stream, {"event": "phase_start", "phase": phase, "perf_s": t0})
            for index in range(1, count + 1):
                due = t0 + index / SAMPLE_HZ
                sliding = phase == "exploration" and index > count / 2
                context = {"phase": phase, "index": index, "check": "pre_send"}
                before = observe(
                    robot, sensor, cfg, pose, previous_target, sliding=sliding,
                    tare_bias=tare_bias,
                )
                if time.perf_counter() > due:
                    raise StateError("Control iteration overrun before send; no catch-up burst")
                if phase == "exploration":
                    protocol_t = 4 * index / count
                    target = target_position(pose, cfg, protocol_t)
                else:
                    protocol_t = None
                    target = [
                        p - PRESS_DEPTH_M * (1 - index / count) * n
                        for p, n in zip(pose["sdk_end_position_m"], cfg["table_normal_sdk"])
                    ]
                send_time = time.perf_counter()
                robot.send(target, quaternion)
                time.sleep(max(0, due - time.perf_counter()))
                context = {"phase": phase, "index": index, "check": "post_send"}
                measured = observe(
                    robot, sensor, cfg, pose, target, sliding=sliding, tare_bias=tare_bias,
                )
                sample_time = time.perf_counter()
                late = sample_time - due
                error = math.dist(measured["state"]["sdk_end_position_m"], target)
                write_event(
                    stream,
                    {
                        "event": "sample",
                        "phase": phase,
                        "index": index,
                        "protocol_time_s": protocol_t,
                        "due_perf_s": due,
                        "send_perf_s": send_time,
                        "sample_perf_s": sample_time,
                        "lateness_s": late,
                        "position_error_m": error,
                        "target_position_m": target,
                        "target_quaternion_xyzw": quaternion,
                        "pre_send_force": before["force"],
                        **measured,
                    },
                )
                if late > MAX_LATENESS_S:
                    raise StateError(
                        "100 Hz deadline missed by over 5 ms; no skipped samples or catch-up"
                    )
                if phase == "exploration":
                    errors.append(error)
                    lateness.append(late)
                    slide_errors.append(measured["slide_error_m"])
                previous_target = target
        rms = math.sqrt(sum(e * e for e in errors) / len(errors))
        write_event(
            stream,
            {
                "event": "motion_complete",
                "exploration_samples": len(errors),
                "mode": cfg.get("mode", "force-guarded"),
                "force_monitoring": not no_ft(cfg),
                "contact_expected": cfg.get("mode") != "air",
                "software_tare_applied": tare_bias is not None,
                "time_scale": time_scale,
                "position_rms_m": rms,
                "max_lateness_s": max(lateness),
                "nominal_protocol_match": time_scale == 1,
                "press_speed_m_s": PRESS_SPEED_M_S,
                "press_duration_s": PRESS_DURATION_S,
                "position_rms_within_simulation_1mm": rms <= 0.001,
                "lateral_tracking_policy": lateral_tracking_policy(cfg),
                "tracking_error_policy": tracking_error_policy(cfg),
                "completion_semantics": "command sequence completed, not measured trajectory acceptance",
                "return_position_error_m": measured["position_error_m"],
                "return_orientation_error_rad": measured["orientation_error_rad"],
                "return_within_tracking_limits": not (
                    measured["position_tracking_exceeds_limit"]
                    or measured["orientation_tracking_exceeds_limit"]
                ),
                "max_abs_slide_error_m": max(abs(e) for e in slide_errors),
                "slide_error_over_5mm_samples": sum(
                    abs(e) > TRACKING_LIMIT_M for e in slide_errors
                ),
                "encoder_ready": False,
            },
        )
        context = {"phase": "handoff"}
        finish(
            lambda: observe(
                robot, sensor, cfg, pose, pose["sdk_end_position_m"], tare_bias=tare_bias,
            )
        )
        robot.idle()
        write_event(stream, {"event": "session_complete", "idle_confirmed": True})
    except BaseException as exc:
        stop_error = None
        if acquired and mode_attempted:
            try:
                robot.abort()
            except BaseException as failure:
                stop_error = str(failure)
                print(f"STOP FAILURE: {failure}", file=sys.stderr, flush=True)
        try:
            write_event(
                stream,
                {
                    "event": "aborted",
                    "error": f"{type(exc).__name__}: {exc}",
                    "mode": cfg.get("mode", "force-guarded"),
                    "force_monitoring": not no_ft(cfg),
                    "software_stop_requested": acquired and mode_attempted,
                    "context": context,
                    "fault_observation": getattr(exc, "observation", None),
                    "stop_error": stop_error,
                    "encoder_ready": False,
                },
            )
        except Exception:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="preview", choices=("preview", "check", "run"))
    parser.add_argument("--mode", choices=("manual-start", *MODES), default="manual-start")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, help="JSONL log; runs outputs get an automatic MMDD_HHMMSS directory")
    parser.add_argument("--plot", action="store_true", help="Show tared F/T in a separate read-only process")
    parser.add_argument("--plot-window", type=float, default=10.0, help="Rolling plot window, 0.5..300 seconds")
    parser.add_argument("--plot-hz", type=float, default=10.0, help="Display refresh, 1..30 Hz; control remains 100 Hz")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="1 uses the 4-second real protocol; 2..10 slower commissioning, not encoder data",
    )
    args = parser.parse_args(argv)
    if args.plot and args.mode not in ("manual-start", "force-guarded"):
        parser.error("--plot requires manual-start or force-guarded measured, tared force data")
    if not math.isfinite(args.plot_window) or not 0.5 <= args.plot_window <= 300:
        parser.error("plot-window must be finite and in [0.5, 300]")
    if not math.isfinite(args.plot_hz) or not 1 <= args.plot_hz <= 30:
        parser.error("plot-hz must be finite and in [1, 30]")
    if args.config is None:
        filename = {
            "manual-start": "real_training/airbot_exploration_manual.json",
            "contact-no-ft": "real_training/airbot_contact_motion.json",
            "air": "robot_control/airbot_air_motion.json",
            "force-guarded": "real_training/airbot_exploration.json",
        }[args.mode]
        args.config = ROOT / "configs" / filename
    if not math.isfinite(args.time_scale) or not 1 <= args.time_scale <= 10:
        parser.error("time-scale must be finite and in [1, 10]")
    if args.action == "preview":
        print(
            json.dumps(
                {
                    "hardware_connected": False,
                    "workspace_bounds_enforced": False,
                    "live_plot_requested": args.plot,
                    "live_plot_opened": False,
                    "sample_hz": SAMPLE_HZ,
                    "mode": args.mode,
                    "force_monitoring": args.mode in ("manual-start", "force-guarded"),
                    "force_limits_enforced": args.mode == "force-guarded",
                    "joint_position_limits_enforced": args.mode != "manual-start",
                    **({"drag_measured_joint_speed_stop_enforced": False,
                        "servo_measured_joint_speed_stop_enforced": True}
                       if args.mode == "manual-start" else {}),
                    "start_matching_required": args.mode != "manual-start",
                    "stationary_acceptance_required": args.mode != "manual-start",
                    "keyboard_flow": ["h: hold", "s: tare and explore", "IDLE: release after return"]
                    if args.mode == "manual-start" else None,
                    "software_tare_duration_s": TARE_DURATION_S
                    if args.mode in ("manual-start", "force-guarded") else None,
                    "contact_expected": args.mode != "air",
                    "minimum_air_start_clearance_m": MIN_AIR_START_CLEARANCE_M
                    if args.mode == "air"
                    else None,
                    "encoder_ready": False,
                    "duration_s": 4 * args.time_scale,
                    "initial_gap_m": None if args.mode == "air" else INITIAL_GAP_M,
                    "nominal_protocol_match": args.time_scale == 1,
                    "table_frame_keyframes_m": {
                        str(t): exploration_offset(t) for t in (0, 2, 3, 4)
                    },
                    "press_speed_m_s": PRESS_SPEED_M_S,
                    "press_duration_s": PRESS_DURATION_S,
                    "retract": "separate +10 mm normal, then attended servo-to-idle handoff",
                },
                indent=2,
            )
        )
        return 0
    if args.action == "run":
        if not args.execute or not sys.stdin.isatty() or args.output is None:
            parser.error("run requires --execute, an interactive terminal and --output")
        from scripts.shared.run_paths import new_output

        args.output = new_output(args.output)
        if args.output.exists():
            parser.error("Output already exists")
    if args.mode == "manual-start":
        from scripts.real_training.manual_exploration import main as manual_main

        return manual_main(args)
    client = sensor = live_plot = None
    previous_sigterm = None
    try:
        cfg, record = load_setup(args.config, args.mode)
        if args.action == "run":
            # Resolve dependencies before any control-mode change.
            if not no_ft(cfg):
                from scripts.force_sensor.kwr75_reader import Kwr75Reader

            if args.mode == "air":
                print(
                    "AIR ONLY: verify server --no-return, a NEW elevated start and at least 50 mm tool clearance."
                )
            else:
                print(
                    "Verify server --no-return, fixed tool orientation/table setup, 9 mm compression allowance,"
                )
            print(
                "approved limits, physical emergency stop, support and a clear entire swept path."
            )
            print("No project XYZ workspace boundary is enforced; verify the entire swept path on site.")
            print(
                "Faults request SOFTWARE STOP without retract; it is NOT a safety-rated stop or hold."
            )
            print(
                f"Exploration: {4 * args.time_scale:g}s, then {2 * args.time_scale:g}s retract. No automatic homing."
            )
            if not no_ft(cfg):
                print(
                    "Before motion: 1 s software tare at the stationary, non-contact 1 mm gap. "
                    "Raw/configured-bias force limits remain active; "
                    "tare is not gravity compensation."
                )
            print(
                f"Joint command speed: {cfg['joint_speed_limit_rad_s']:g} rad/s; "
                f"measured speed stop: {measured_joint_speed_limit(cfg):g} rad/s."
            )
            print(
                f"Running tracking policy: {tracking_error_policy(cfg)}; "
                f"orientation reference threshold: {running_orientation_limit(cfg):g} rad; "
                f"start orientation tolerance: {ORIENTATION_TOLERANCE_RAD:g} rad."
            )
            if args.mode == "air":
                print(
                    "No force sensor. Keep the ENTIRE robot/tool/cable swept path clear; no table contact permitted."
                )
                print(
                    "Fixed 10 mm downward travel and 50 mm lateral round trip; no automatic lift from contact."
                )
            elif no_ft(cfg):
                print(
                    "CONTACT TEST WITHOUT FORCE MEASUREMENT OR OVERFORCE STOP. Geometric bounds are NOT force limits."
                )
                print(
                    "Fixed 10 mm downward travel; estimated nominal compression 9 mm from the measured 1 mm gap."
                )
            print("--execute authorizes motion after device and start checks.")
            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_signal)
            if args.plot:
                from scripts.real_training.exploration_live_plot import ExplorationLivePlot

                live_plot = ExplorationLivePlot(args.output, args.plot_window, args.plot_hz)
                live_plot.start()
                print("Tared F/T plot opened. Closing it only closes the display, not robot motion.")
        client = open_client(args.host, args.port)
        client._stub = DeadlineStub(client._stub)
        if args.action == "check":
            state = snapshot(client, "idle")
            firmware = client.get_firmware_info()
            if firmware is None or firmware.arm_sn != cfg["robot_sn"]:
                raise StateError("Connected robot serial number differs from approved setup")
            if (
                cfg.get("expected_eef_type") is not None
                and firmware.eef_type != cfg["expected_eef_type"]
            ):
                raise StateError("Connected end-effector type differs from approved setup")
            check_start(state, record["initial_pose"], cfg)
            print(
                json.dumps(
                    {
                        "read_only": True,
                        "workspace_bounds_enforced": False,
                        "start_matches": True,
                        "state": state,
                        "mode": args.mode,
                        "force_monitoring": False,
                        "contact_expected": args.mode != "air",
                        "force_checked": False,
                        "calibration_physically_verified": False,
                        "sdk_keyframes_m": {
                            str(t): target_position(record["initial_pose"], cfg, t)
                            for t in (0, 2, 3, 4)
                        },
                    },
                    indent=2,
                )
            )
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            write_event(
                stream,
                {
                    "event": "session_start",
                    "workspace_bounds_enforced": False,
                    "source_kind": "real",
                    "sdk_version": SDK_VERSION,
                    "mode": args.mode,
                    "force_monitoring": not no_ft(cfg),
                    "contact_expected": args.mode != "air",
                    "live_plot_requested": args.plot,
                    "config": cfg,
                    "initial_pose_record": record,
                    "firmware": asdict(client.get_firmware_info()),
                    "clock": "host perf_counter; SDK snapshots also carry host monotonic",
                    "ft_frame": None
                    if no_ft(cfg)
                    else "sensor local at sensor origin; NOT simulation ft_frame",
                    "encoder_ready": False,
                },
            )
            if not no_ft(cfg):
                sensor = Kwr75Reader(port=cfg["sensor_port"])
                sensor.start()
                deadline = time.perf_counter() + 2
                while sensor.latest(net=False) is None and time.perf_counter() < deadline:
                    time.sleep(0.01)
            robot = Robot(client, cfg)
            execute(robot, sensor, cfg, record, stream, args.time_scale)
        print(f"Completed and idle confirmed. Log: {args.output}")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted. Check physical support and emergency-stop state.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}. No automatic retry; check hardware safety state.", file=sys.stderr)
        return 1
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            if sensor is not None:
                sensor.stop()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if live_plot is not None:
                    live_plot.close()


if __name__ == "__main__":
    raise SystemExit(main())
