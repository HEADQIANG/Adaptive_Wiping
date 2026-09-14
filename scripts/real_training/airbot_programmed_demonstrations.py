"""Attended fixed-depth demonstrations with exploration guards and raw force logging."""

import hashlib
import json
import math
import os
import select
import signal
import statistics
import sys
import termios
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from scripts.real_training.airbot_demonstrations import write_json
from scripts.real_training.airbot_exploration import (
    ORIENTATION_TOLERANCE_RAD, START_POSITION_TOLERANCE_M,
    check_start, load_setup, observe as exploration_observe, tracking_limit,
)
from scripts.robot_control.adapter import DeadlineStub, Robot
from scripts.robot_control.airbot_initial_pose import NotStationary, validate_stationary
from scripts.robot_control.client import StateError, open_client, snapshot
from scripts.robot_control.safety import (
    check_bounds, interrupt_on_signal, measured_joint_speed_limit, quaternion_angle,
    read_force, write_event,
)
from scripts.shared.exploration_protocol import PRESS_SPEED_M_S, SAMPLE_HZ, exploration_offset
from scripts.shared.paths import ROOT

PROTOCOL = "airbot_programmed_fixed_depth_v2"
SEQUENCE = ("nominal", "nominal", "under", "over", "under", "over", "under", "over")
DEPTHS = {"nominal": 0.010, "under": 0.008, "over": 0.012}
SLIDE_SPEED = exploration_offset(3.0)[1]
DT = 1 / SAMPLE_HZ


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def readiness(cfg):
    errors = []
    if cfg.get("setup_confirmed") is not True:
        errors.append("setup_confirmed must be true after onsite verification of the new fixed-depth path")
    if cfg.get("startup_approach", {}).get("direct_path_confirmed") is not True:
        errors.append("startup_approach.direct_path_confirmed must be true after operator approval")
    return errors


def load_configuration(path):
    path = resolve(path)
    source_hash = digest(path)
    spec = json.loads(path.read_text())
    for key, value in {
        "schema_version": 1, "protocol": PROTOCOL, "sponge_id": "normal",
        "motion_policy": "fixed-depth", "initial_depth_m": DEPTHS, "max_depth_m": 0.020,
        "stationary_speed_limit_rad_s": 0.1, "baseline_retry_timeout_s": 10.0,
        "retraction_lateness_limit_s": 0.01, "retraction_max_extension_s": 1.0,
    }.items():
        if spec.get(key) != value or isinstance(spec.get(key), bool):
            raise ValueError(f"{key} must be {value!r}")
    if any(key in spec for key in ("control", "target_force_n", "force_direction_confirmed")):
        raise ValueError("Closed-loop fields are not supported by the fixed-depth protocol; use a new configuration/session")
    for key in ("surface_id", "exploration_id", "setup_note", "exploration_config"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"Missing {key}")
    approach = spec.get("startup_approach", {})
    if approach.get("mode") != "direct" or not approach.get("confirmation_note"):
        raise ValueError("startup_approach requires direct mode and confirmation_note")
    for key, cap in {"speed_m_s": 0.05, "angular_speed_rad_s": 0.3,
                     "max_duration_s": 60, "tracking_tolerance_m": 0.005,
                     "orientation_tolerance_rad": 0.02}.items():
        value = approach.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value <= cap:
            raise ValueError(f"startup_approach.{key} must be finite and in (0, {cap}]")
    exp_path = resolve(spec["exploration_config"])
    exp_hash = digest(exp_path)
    cfg, record = load_setup(exp_path, "force-guarded")
    pose_path = resolve(cfg["initial_pose_file"])
    hashes = {str(path): source_hash, str(exp_path): exp_hash, str(pose_path): cfg["initial_pose_sha256"]}
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError("Configuration changed during read")
    if cfg["verified_compression_allowance_m"] + cfg["initial_gap_m"] < max(DEPTHS.values()) - 1e-12:
        raise ValueError("Exploration setup does not allow the full 12 mm commanded press")
    # Only program-specific fields override the copied exploration setup.
    for key in ("protocol", "sponge_id", "surface_id", "exploration_id", "initial_depth_m",
                "motion_policy", "max_depth_m", "setup_confirmed", "setup_note", "startup_approach",
                "stationary_speed_limit_rad_s", "baseline_retry_timeout_s",
                "retraction_lateness_limit_s", "retraction_max_extension_s"):
        cfg[key] = spec.get(key)
    frozen = {
        "protocol": PROTOCOL, "demonstration_source": "programmed", "source_kind": "real",
        "training_ready": False, "config": cfg, "initial_pose_record": record,
        "motion_policy": "fixed-depth", "force_feedback_enabled": False,
        "source_hashes": hashes, "sequence": list(SEQUENCE),
        "motion_protocol": {"sample_hz": SAMPLE_HZ, "press_speed_m_s": PRESS_SPEED_M_S,
                            "slide_speed_m_s": SLIDE_SPEED, "slide_offsets_m": [0, 0.05, -0.05, 0],
                            "startup_timing_revision": 2,
                            "startup_cycle_budget_s": DT, "startup_wakeup_limit_s": 0.005,
                            "postroll_s": 2 * DT},
        "force_semantics": "raw and unloaded-baseline-subtracted sensor wrench; not calibrated normal force; no force target",
        "position_frame": "SDK configured end; not calibrated sponge TCP",
        "clock": "host monotonic (pose) / perf_counter (FT); no hardware timestamps",
    }
    return cfg, record["initial_pose"], frozen


def slide_offset(t):
    t1 = 0.05 / SLIDE_SPEED
    if not 0 <= t <= 4 * t1 + 1e-9:
        raise ValueError("Slide time outside protocol")
    if t <= t1:
        return SLIDE_SPEED * t
    if t <= 3 * t1:
        return 0.05 - SLIDE_SPEED * (t - t1)
    return -0.05 + SLIDE_SPEED * (t - 3 * t1)


def duration(condition):
    return DEPTHS[condition] / PRESS_SPEED_M_S + 0.20 / SLIDE_SPEED


def target(pose, cfg, lateral, depth):
    if not math.isfinite(depth) or not 0 <= depth <= cfg["max_depth_m"]:
        raise StateError("Commanded total depth outside [0,20 mm]")
    return [p + lateral * s - depth * n for p, s, n in zip(
        pose["sdk_end_position_m"], cfg["slide_direction_sdk"], cfg["table_normal_sdk"])]


def observe(robot, sensor, cfg, pose, position, baseline=None, *, initial=False, sliding=False):
    observation = exploration_observe(
        robot, sensor, cfg, pose, position, initial=initial, sliding=sliding, tare_bias=baseline,
    )
    state = observation.pop("state")
    ft = observation.pop("force")
    depth = sum((p - x) * n for p, x, n in zip(pose["sdk_end_position_m"], state["sdk_end_position_m"], cfg["table_normal_sdk"]))
    if depth > cfg["max_depth_m"]:
        raise StateError("Measured total depth exceeded 20 mm")
    return {**observation, "pose": state, "ft": ft, "observed_perf_s": time.perf_counter(),
            "target_quaternion_xyzw": pose["sdk_end_orientation_xyzw"], "measured_depth_m": depth}


class Terminal:
    """Use unbuffered fd reads so queued lines cannot trigger a later motion."""

    def discard(self):
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

    def ask(self, prompt, monitor):
        self.discard()
        print(prompt, flush=True)
        pending = b""
        while True:
            monitor()
            if select.select([sys.stdin.fileno()], [], [], DT)[0]:
                chunk = os.read(sys.stdin.fileno(), 4096)
                if not chunk:
                    raise EOFError("Terminal closed")
                pending += chunk
                if len(pending) > 4096:
                    raise ValueError("Terminal command too long")
                if b"\n" in pending:
                    result = pending.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
                    self.discard()
                    return result


def check_return_pose(state, cfg, pose):
    try:
        check_start(state, pose, cfg)
    except StateError as exc:
        raise ValueError(f"Return/start pose not confirmed: {exc}") from exc


def approach_plan(state, cfg, pose):
    check_bounds(state, cfg)
    depth = sum((p - x) * n for p, x, n in zip(
        pose["sdk_end_position_m"], state["sdk_end_position_m"], cfg["table_normal_sdk"]))
    if depth > cfg["max_depth_m"]:
        raise StateError("Startup approach measured depth exceeded 20 mm")
    distance = math.dist(state["sdk_end_position_m"], pose["sdk_end_position_m"])
    angle = quaternion_angle(state["sdk_end_orientation_xyzw"], pose["sdk_end_orientation_xyzw"])
    settings = cfg["startup_approach"]
    # Quintic blend has peak derivative 1.875; both speed limits apply to peaks.
    seconds = 1.875 * max(distance / settings["speed_m_s"], angle / settings["angular_speed_rad_s"])
    if seconds > settings["max_duration_s"]:
        raise StateError("Direct startup approach exceeds maximum duration; no movement permitted")
    return {"distance_m": distance, "angle_rad": angle, "duration_s": math.ceil(seconds / DT) * DT}


def approach_observe(robot, sensor, cfg, pose, position, quaternion):
    robot.owned()
    state = robot.read("servo")
    check_bounds(state, cfg)
    if state["read_duration_s"] > 0.020:
        raise StateError("Startup approach state read exceeded 20 ms")
    if any(abs(v) > measured_joint_speed_limit(cfg) for v in state["joint_velocity_rad_s"]):
        raise StateError("Startup approach joint speed exceeded limit")
    settings = cfg["startup_approach"]
    if math.dist(state["sdk_end_position_m"], position) > settings["tracking_tolerance_m"]:
        raise StateError("Startup approach position tracking exceeded limit")
    if quaternion_angle(state["sdk_end_orientation_xyzw"], quaternion) > settings["orientation_tolerance_rad"]:
        raise StateError("Startup approach orientation tracking exceeded limit")
    depth = sum((p - x) * n for p, x, n in zip(
        pose["sdk_end_position_m"], state["sdk_end_position_m"], cfg["table_normal_sdk"]))
    if depth > cfg["max_depth_m"]:
        raise StateError("Startup approach measured depth exceeded 20 mm")
    ft = read_force(sensor, cfg, initial=True)
    return {"pose": state, "ft": ft, "target_position_m": list(position),
            "target_quaternion_xyzw": list(quaternion), "observed_perf_s": time.perf_counter()}


class ApproachTimingError(StateError):
    def __init__(self, phase, timing):
        self.approach_timing = dict(timing, failed_phase=phase)
        super().__init__(f"Startup approach {phase} deadline missed: "
                         + json.dumps(self.approach_timing, sort_keys=True))


def startup_approach(robot, sensor, cfg, pose, state, stream):
    from scipy.spatial.transform import Rotation, Slerp

    plan = approach_plan(state, cfg, pose)
    position = list(state["sdk_end_position_m"])
    quaternion = list(state["sdk_end_orientation_xyzw"])
    write_event(stream, {"event": "approach_start", "initial_state": state, **plan})
    rotation = Slerp([0, 1], Rotation.from_quat([quaternion, pose["sdk_end_orientation_xyzw"]]))
    # Drain preflight backlog before the fresh safety observation and initial hold.
    sensor.flush_csv()
    # Check for drift during mode switching before issuing even the initial hold.
    hold_start = time.perf_counter()
    approach_observe(robot, sensor, cfg, pose, position, quaternion)
    observed = time.perf_counter()
    if observed - hold_start > DT:
        raise ApproachTimingError("hold_observation", {"observation_s": observed - hold_start})
    robot.send(position, quaternion)
    if time.perf_counter() - hold_start > DT:
        raise ApproachTimingError("hold_command", {"observation_s": observed - hold_start,
                                                   "command_s": time.perf_counter() - observed})
    count = round(plan["duration_s"] / DT)
    start = time.perf_counter()
    previous_logging = None
    for tick in range(1, count + 1):
        deadline = start + tick * DT
        time.sleep(max(0, deadline - time.perf_counter()))
        woke = time.perf_counter()
        timing = {"tick": tick, "scheduled_perf_s": deadline, "cycle_budget_s": DT,
                  "wakeup_lateness_s": woke - deadline, "previous_logging_s": previous_logging}
        if timing["wakeup_lateness_s"] > 0.005:
            raise ApproachTimingError("scheduling", timing)
        row = approach_observe(robot, sensor, cfg, pose, position, quaternion)
        observed = time.perf_counter()
        timing["observation_s"] = observed - woke
        timing["state_read_s"] = row["pose"]["read_duration_s"]
        alpha = tick / count
        alpha = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
        position = [p + alpha * (q - p) for p, q in zip(state["sdk_end_position_m"], pose["sdk_end_position_m"])]
        quaternion = rotation(alpha).as_quat().tolist()
        prepared = time.perf_counter()
        timing["target_calculation_s"] = prepared - observed
        timing["elapsed_s"] = prepared - deadline
        if timing["elapsed_s"] > DT:
            raise ApproachTimingError("observation_or_target", timing)
        robot.send(position, quaternion)
        sent = time.perf_counter()
        timing["command_s"] = sent - prepared
        timing["elapsed_s"] = sent - deadline
        if timing["elapsed_s"] > DT:
            raise ApproachTimingError("command", timing)
        # Disk work is after the command, but still charged to this 10 ms cycle.
        sensor.flush_csv()
        flushed = time.perf_counter()
        timing["csv_flush_s"] = flushed - sent
        timing["elapsed_s"] = flushed - deadline
        if timing["elapsed_s"] > DT:
            raise ApproachTimingError("csv_logging", timing)
        write_event(stream, {"event": "sample", "phase": "approach", **row,
                             "commanded_position_m": position, "commanded_quaternion_xyzw": quaternion,
                             "timing": timing})
        finished = time.perf_counter()
        timing["json_write_s"] = finished - flushed
        timing["elapsed_s"] = finished - deadline
        previous_logging = {"tick": tick, "csv_flush_s": timing["csv_flush_s"],
                            "json_write_s": timing["json_write_s"], "elapsed_s": timing["elapsed_s"]}
        if timing["elapsed_s"] > DT:
            raise ApproachTimingError("json_logging", timing)
    write_event(stream, {"event": "approach_motion_complete", "last_logging": previous_logging})
    settle(robot, sensor, cfg, pose, None, stream, "approach_settle")
    write_event(stream, {"event": "approach_complete", "return_confirmed": True})


def verify_return(rows, cfg, pose):
    for row in rows:
        check_return_pose(row["pose"], cfg, pose)
    validate_stationary([r["pose"] for r in rows], max_joint_speed=cfg["stationary_speed_limit_rad_s"])


def settle(robot, sensor, cfg, pose, baseline, stream, phase, *, deadline=None):
    baseline_wait = deadline is not None
    if deadline is None:
        deadline = time.perf_counter() + 2.0
    rows = []
    while time.perf_counter() < deadline:
        row = observe(robot, sensor, cfg, pose, pose["sdk_end_position_m"], baseline, initial=True)
        rows.append(row)
        rows = rows[-11:]
        write_event(stream, {"event": "sample", "phase": phase, **row})
        sensor.flush_csv()
        if baseline_wait:
            check_return_pose(row["pose"], cfg, pose)
        if len(rows) == 11:
            try:
                verify_return(rows, cfg, pose)
                return rows[-1]
            except ValueError as exc:
                if baseline_wait and not isinstance(exc, NotStationary):
                    raise
        time.sleep(0.05)
    raise StateError("Stationary start/return not confirmed within allowed time")


def collect_baseline(robot, sensor, cfg, pose, stream):
    deadline = time.perf_counter() + cfg["baseline_retry_timeout_s"]
    attempt = 0
    while time.perf_counter() < deadline:
        # Only stationarity failures are retryable. Pose, force and stream guards
        # propagate immediately; the existing servo start target remains unchanged.
        settle(robot, sensor, cfg, pose, None, stream, "start_check", deadline=deadline)
        attempt += 1
        start = time.perf_counter()
        unique, states = {}, []
        write_event(stream, {"event": "tare_start", "baseline_attempt": attempt,
                             "start_perf_s": start, "deadline_perf_s": deadline})
        try:
            while time.perf_counter() - start < 1.0:
                if time.perf_counter() >= deadline:
                    raise StateError("Baseline acquisition exceeded total retry timeout")
                row = observe(robot, sensor, cfg, pose, pose["sdk_end_position_m"], initial=True)
                states.append(row)
                write_event(stream, {"event": "sample", "phase": "tare", "baseline_attempt": attempt, **row})
                sensor.flush_csv()
                check_return_pose(row["pose"], cfg, pose)
                if len(states) >= 11:
                    verify_return(states[-11:], cfg, pose)
                ft = row["ft"]
                stamp = ft["sensor_receive_perf_s"]
                if unique and stamp < max(unique):
                    raise StateError("Baseline receive clock reversed")
                if stamp >= start:
                    if stamp in unique and unique[stamp] != ft["raw_sensor_wrench_si"]:
                        raise StateError("Conflicting baseline frames share a timestamp")
                    unique[stamp] = ft["raw_sensor_wrench_si"]
                time.sleep(DT)
            verify_return(states, cfg, pose)
        except NotStationary as exc:
            write_event(stream, {"event": "tare_discarded", "baseline_attempt": attempt,
                                 "reason": str(exc), "samples": len(states)})
            print("Baseline discarded: not stationary. Holding monitored start; retrying within total timeout.", flush=True)
            continue
        if time.perf_counter() >= deadline:
            raise StateError("Baseline acquisition exceeded total retry timeout")
        if len(unique) < 40 or max(unique) - min(unique) < 0.9:
            raise StateError("Insufficient fresh unloaded baseline samples")
        baseline = [statistics.mean(v[i] for v in unique.values()) for i in range(6)]
        write_event(stream, {"event": "tare_complete", "baseline_attempt": attempt,
                             "tare_bias_si": baseline, "distinct_samples": len(unique),
                             "gravity_compensated": False})
        return baseline
    raise StateError("Baseline acquisition exceeded total retry timeout")


def timing_quality(rows, start, end):
    errors = []
    pose = [r["pose"]["host_monotonic_s"] for r in rows]
    ft = [r["ft"]["sensor_receive_perf_s"] for r in rows]
    gaps = [b - a for a, b in zip(pose, pose[1:])]
    unique = sorted(set(ft))
    ft_gaps = [b - a for a, b in zip(unique, unique[1:])]
    if len(rows) < 2 or pose[0] > start or pose[-1] < end or ft[0] > start or ft[-1] < end:
        errors.append("incomplete measured episode coverage")
    if not gaps or min(gaps) <= 0 or max(gaps) > 0.020 + 1e-9:
        errors.append("pose gap >20ms/nonmonotonic")
    if not ft_gaps or max(ft_gaps) > 0.020 + 1e-9 or any(b < a for a, b in zip(ft, ft[1:])):
        errors.append("FT gap >20ms/nonmonotonic")
    median = statistics.median(gaps) if gaps else None
    if median is None or not 0.009 <= median <= 0.011:
        errors.append("pose median period not 100Hz +/-10%")
    if any(r["pose"]["read_duration_s"] > 0.020 for r in rows):
        errors.append("pose read >20ms")
    if any(not 0 <= r["observed_perf_s"] - r["pose"]["host_monotonic_s"] <= 0.020 for r in rows):
        errors.append("pose/FT host clock convention or observation latency mismatch")
    return {"passed": not errors, "errors": errors, "samples": len(rows),
            "duration_s": end - start, "median_pose_period_s": median,
            "max_pose_gap_s": max(gaps, default=0), "max_ft_receive_gap_s": max(ft_gaps, default=0)}


def force_report(rows, initial_ft):
    report = {"force_feedback_enabled": False, "initial_ft": initial_ft}
    for name, field in (("raw", "raw_sensor_wrench_si"), ("tared", "tared_sensor_wrench_si")):
        values = [r["ft"][field] for r in rows]
        norms = [math.sqrt(sum(v * v for v in wrench[:3])) for wrench in values]
        report[name] = {
            "mean_si": [statistics.mean(v[i] for v in values) for i in range(6)],
            "min_si": [min(v[i] for v in values) for i in range(6)],
            "max_si": [max(v[i] for v in values) for i in range(6)],
            "force_norm_mean_n": statistics.mean(norms), "force_norm_max_n": max(norms),
        }
    return report


def record_episode(robot, sensor, cfg, pose, condition, path, terminal):
    with path.open("x", encoding="utf-8") as stream:
        write_event(stream, {"event": "attempt_start", "protocol": PROTOCOL,
                             "condition": condition, "initial_depth_m": DEPTHS[condition],
                             "source_kind": "real", "demonstration_source": "programmed",
                             "training_ready": False, "motion_policy": "fixed-depth",
                             "force_feedback_enabled": False,
                             "max_depth_m": cfg["max_depth_m"]})
        baseline = collect_baseline(robot, sensor, cfg, pose, stream)
        terminal.discard()
        position = pose["sdk_end_position_m"]
        first = observe(robot, sensor, cfg, pose, position, baseline)
        start = first["observed_perf_s"]
        end = start + duration(condition)
        rows = [first]
        write_event(stream, {"event": "episode_start", "start_perf_s": start, "end_perf_s": end})
        write_event(stream, {"event": "sample", "phase": "press", "index": 0, **first})
        press_steps = round(DEPTHS[condition] / PRESS_SPEED_M_S * SAMPLE_HZ)
        wipe_steps = round(duration(condition) * SAMPLE_HZ)
        sliding = []
        initial_ft = None
        print(f"RECORDING {condition}: {duration(condition):g}s; do not touch the robot.", flush=True)
        # Send before each deadline and observe afterwards, as in exploration.
        for index in range(1, wipe_steps + 3):
            due = start + index * DT
            before = observe(robot, sensor, cfg, pose, position, baseline,
                             sliding=press_steps < index <= wipe_steps)
            if time.perf_counter() >= due:
                raise StateError("Control iteration overrun before send; no catch-up")
            if index <= press_steps:
                phase = "press"
                depth, lateral = min(DEPTHS[condition], index * DT * PRESS_SPEED_M_S), 0.0
            elif index <= wipe_steps:
                phase = "slide"
                if initial_ft is None:
                    initial_ft = before["ft"]
                depth = DEPTHS[condition]
                lateral = slide_offset((index - press_steps) * DT)
            else:
                phase = "postroll"
                depth, lateral = DEPTHS[condition], 0.0
            position = target(pose, cfg, lateral, depth)
            robot.send(position, pose["sdk_end_orientation_xyzw"])
            time.sleep(max(0, due - time.perf_counter()))
            row = observe(robot, sensor, cfg, pose, position, baseline, sliding=phase == "slide")
            late = row["observed_perf_s"] - due
            if late > 0.005:
                raise StateError("Control sample more than 5ms late")
            row.update(commanded_depth_m=depth, lateral_offset_m=lateral,
                       due_perf_s=due, lateness_s=late, pre_send_ft=before["ft"])
            rows.append(row)
            if phase == "slide":
                sliding.append(row)
            write_event(stream, {"event": "sample", "phase": phase, "index": index, **row})
            sensor.flush_csv()
            terminal.discard()
        quality = timing_quality(rows, start, end)
        forces = force_report(sliding, initial_ft)
        write_event(stream, {"event": "episode_end", "quality": quality, "force_report": forces})
        retract_start = time.perf_counter()
        initial_depth = DEPTHS[condition]
        retract_steps = max(1, math.ceil(initial_depth / PRESS_SPEED_M_S / DT))
        shift, last_send = 0.0, None
        retract_deadline = retract_start + retract_steps * DT + cfg["retraction_max_extension_s"]
        def check_retraction_time(stage, due, **timing):
            now = time.perf_counter()
            if now > retract_deadline or now - due > cfg["retraction_lateness_limit_s"]:
                exc = StateError(f"Retraction {stage} timeout: lateness={now - due:.6f}s, "
                                 f"limit={cfg['retraction_lateness_limit_s']:.3f}s, "
                                 f"elapsed={now - retract_start:.6f}s")
                exc.retraction_timing = {"phase": stage, "index": index, "lateness_s": now - due,
                                         "elapsed_s": now - retract_start, "schedule_shift_s": shift,
                                         "total_deadline_perf_s": retract_deadline, **timing}
                raise exc
        for index in range(1, retract_steps + 1):
            due = retract_start + index * DT + shift
            # Never compensate for a late cycle with closely spaced targets.
            if last_send is not None:
                time.sleep(max(0, last_send + DT - time.perf_counter()))
            check_retraction_time("before_observe", due)
            observe_start = time.perf_counter()
            observe(robot, sensor, cfg, pose, position, baseline)
            observed = time.perf_counter()
            check_retraction_time("before_send", due, observation_s=observed - observe_start)
            depth = max(0, initial_depth - index * DT * PRESS_SPEED_M_S)
            position = target(pose, cfg, 0, depth)
            last_send = time.perf_counter()
            robot.send(position, pose["sdk_end_orientation_xyzw"])
            sent = time.perf_counter()
            check_retraction_time("command", due, command_s=sent - last_send)
            time.sleep(max(0, due - time.perf_counter()))
            row = observe(robot, sensor, cfg, pose, position, baseline)
            check_retraction_time("sample", due)
            write_event(stream, {"event": "sample", "phase": "retract", "commanded_depth_m": depth,
                                 "index": index, "due_perf_s": due, "send_perf_s": last_send,
                                 "schedule_shift_s": shift, "lateness_s": row["observed_perf_s"] - due,
                                 "pre_observation_s": observed - observe_start, "command_s": sent - last_send, **row})
            sensor.flush_csv()
            terminal.discard()
            check_retraction_time("logging", due)
            # Rebase on completion rather than issuing catch-up steps on the old clock.
            shift += max(0, time.perf_counter() - due)
        returned = settle(robot, sensor, cfg, pose, baseline, stream, "return_check")
        report = {"quality": quality, "force_report": forces, "return_confirmed": True,
                  "return_position_error_m": math.dist(returned["pose"]["sdk_end_position_m"], pose["sdk_end_position_m"]),
                  "baseline_si": baseline}
        write_event(stream, {"event": "finished", **report})
        stream.flush()
        os.fsync(stream.fileno())
        terminal.discard()
        return report


def accepted_records(folder):
    records = []
    paths = sorted(folder.glob("demo_*.json"))
    if [p.name for p in paths] != [f"demo_{i:02d}.json" for i in range(1, len(paths) + 1)] or len(paths) > 8:
        raise ValueError("Accepted program demos must be consecutive 01..08")
    raw_paths = set()
    for index, path in enumerate(paths):
        entry = json.loads(path.read_text())
        raw = (folder / entry["raw_file"]).resolve()
        if raw.parent != folder.resolve() or raw in raw_paths or digest(raw) != entry["sha256"]:
            raise ValueError("Program demonstration raw integrity failure")
        raw_paths.add(raw)
        if (entry.get("protocol") != PROTOCOL or entry.get("condition") != SEQUENCE[index]
                or entry.get("accepted") is not True or entry.get("training_ready") is not False
                or entry.get("demonstration_source") != "programmed" or entry.get("source_kind") != "real"
                or entry.get("motion_policy") != "fixed-depth" or entry.get("force_feedback_enabled") is not False
                or entry.get("return_confirmed") is not True or not entry.get("quality", {}).get("passed")):
            raise ValueError("Invalid accepted program demonstration")
        events = [json.loads(line) for line in raw.read_text().splitlines()]
        if (events[0].get("protocol") != PROTOCOL or events[0].get("condition") != SEQUENCE[index]
                or events[0].get("event") != "attempt_start"
                or events[0].get("motion_policy") != "fixed-depth" or events[0].get("force_feedback_enabled") is not False
                or events[0].get("initial_depth_m") != DEPTHS[SEQUENCE[index]]
                or events[-1].get("event") != "finished" or events[-1].get("return_confirmed") is not True
                or events[-1].get("quality") != entry["quality"]):
            raise ValueError("Incomplete program attempt")
        starts = [r for r in events if r.get("event") == "episode_start"]
        ends = [r for r in events if r.get("event") == "episode_end"]
        samples = [r for r in events if r.get("event") == "sample"
                   and r.get("phase") in ("press", "slide", "postroll")]
        expected_steps = round(duration(SEQUENCE[index]) * SAMPLE_HZ)
        if (len(starts) != 1 or len(ends) != 1
                or [r.get("index") for r in samples] != list(range(expected_steps + 3))
                or abs(starts[0]["end_perf_s"] - starts[0]["start_perf_s"] - duration(SEQUENCE[index])) > 1e-8):
            raise ValueError("Program episode duration or sample sequence mismatch")
        checked = timing_quality(samples, starts[0]["start_perf_s"], starts[0]["end_perf_s"])
        if not checked["passed"] or checked != entry["quality"]:
            raise ValueError("Program episode timing failed revalidation")
        records.append(entry)
    return records


def freeze_session(folder, frozen):
    manifest = folder / "session.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != frozen:
            raise ValueError("Configuration/start changed; use a new program session directory")
    else:
        if any(folder.glob("demo_*.json")):
            raise ValueError("Accepted files exist without session metadata")
        write_json(manifest, frozen)


def session(robot, sensor, cfg, pose, folder, terminal, recorder=record_episode):
    count = len(accepted_records(folder))
    baseline = None
    mode_attempted = False
    raw = None
    try:
        # Import before requesting servo; first SciPy import can take hundreds of ms.
        from scipy.spatial.transform import Rotation, Slerp  # noqa: F401

        robot.acquire()
        states = []
        for _ in range(11):
            robot.owned()
            state = robot.read("idle")
            approach_plan(state, cfg, pose)
            read_force(sensor, cfg, initial=True)
            states.append(state)
            time.sleep(0.05)
        validate_stationary(states, max_joint_speed=cfg["stationary_speed_limit_rad_s"])
        mode_attempted = True
        robot.enter()
        raw = folder / f"approach_{uuid.uuid4().hex}.jsonl"
        with raw.open("x") as stream:
            startup_approach(robot, sensor, cfg, pose, state, stream)
        terminal.discard()
        raw = None

        def monitor():
            row = observe(robot, sensor, cfg, pose, pose["sdk_end_position_m"], baseline, initial=True)
            check_return_pose(row["pose"], cfg, pose)
            sensor.flush_csv()
            return row

        while count < len(SEQUENCE):
            cmd = terminal.ask(
                f"{count}/8 accepted; next={SEQUENCE[count]}. Verify non-contact start: s + Enter=start, q=finish.", monitor)
            if cmd == "q":
                break
            if cmd != "s":
                continue
            raw = folder / f"attempt_{uuid.uuid4().hex}.jsonl"
            report = recorder(robot, sensor, cfg, pose, SEQUENCE[count], raw, terminal)
            baseline = report["baseline_si"]
            print(json.dumps(report, indent=2), flush=True)
            if not report["return_confirmed"]:
                raise StateError("Recorder did not confirm physical return")
            if not report["quality"]["passed"]:
                print("Timing rejected; attempt retained, not counted.", flush=True)
                continue
            cmd = terminal.ask("Returned and stationary. a + Enter=accept only if error-free; q=finish; other=reject.", monitor)
            if cmd == "q":
                break
            if cmd != "a":
                continue
            entry = {"protocol": PROTOCOL, "accepted": True, "source_kind": "real",
                     "demonstration_source": "programmed", "training_ready": False,
                     "motion_policy": "fixed-depth", "force_feedback_enabled": False,
                     "condition": SEQUENCE[count], "initial_depth_m": DEPTHS[SEQUENCE[count]],
                     "raw_file": raw.name, "sha256": digest(raw), **report}
            write_json(folder / f"demo_{count + 1:02d}.json", entry)
            count += 1
        while terminal.ask("Servo active. Arrange safe support; type IDLE + Enter to release, NOT position hold.", monitor) != "IDLE":
            pass
        robot.idle()
        print("Idle confirmed.", flush=True)
        return count
    except BaseException as exc:
        if mode_attempted:
            try:
                robot.abort()
                print("Software stop acknowledged; use onsite support procedure. No automatic retract.", file=sys.stderr)
            except BaseException as stop_error:
                print(f"STOP NOT CONFIRMED: {stop_error}; use physical emergency stop NOW", file=sys.stderr)
        try:
            write_json(folder / f"fault_{uuid.uuid4().hex}.json",
                       {"event": "aborted", "error": str(exc), "mode_attempted": mode_attempted,
                        "approach_timing": getattr(exc, "approach_timing", None),
                        "retraction_timing": getattr(exc, "retraction_timing", None),
                        "raw_file": raw.name if raw is not None else None})
        except Exception:
            pass
        raise


def main(args):
    args.config = args.config or ROOT / "configs/real_training/airbot_programmed_demonstrations.json"
    args.output = args.output or ROOT / "runs/real_training/programmed_demonstrations/fixed_depth_session_001"
    client = sensor = lock = None
    previous_signal = None
    try:
        if args.action == "status":
            frozen = json.loads((args.output / "session.json").read_text())
            if frozen.get("protocol") != PROTOCOL:
                raise ValueError("Not a programmed demonstration session")
            records = accepted_records(args.output)
            print(json.dumps({"accepted": len(records), "required": 8, "training_ready": False,
                              "hardware_connected": False, "conditions": [r["condition"] for r in records],
                              "episodes": [{"condition": r["condition"], "quality": r["quality"],
                                            "force_report": r["force_report"],
                                            "return_position_error_m": r["return_position_error_m"]}
                                           for r in records]}, indent=2))
            return 0 if len(records) == 8 else 1
        cfg, pose, frozen = load_configuration(args.config)
        errors = readiness(cfg)
        if args.action == "preview":
            print(json.dumps({"protocol": PROTOCOL, "hardware_connected": False, "training_ready": False,
                              "ready": not errors, "setup_errors": errors, "sequence": SEQUENCE,
                              "episodes": {c: {"initial_depth_m": DEPTHS[c], "duration_s": duration(c)} for c in DEPTHS},
                              "slide_offsets_m": [0, 0.05, -0.05, 0], "slide_speed_m_s": SLIDE_SPEED,
                              "press_speed_m_s": PRESS_SPEED_M_S, "max_depth_m": cfg["max_depth_m"],
                              "motion_policy": "fixed-depth", "force_feedback_enabled": False,
                              "force_policy": "inherited exploration force/torque stop thresholds",
                              "tracking_error_policy": cfg.get("tracking_error_policy", "stop"),
                              "lateral_tracking_policy": cfg.get("lateral_tracking_policy", "stop"),
                              "return_position_tolerance_m": min(START_POSITION_TOLERANCE_M, tracking_limit(cfg)),
                              "return_orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
                              "initial_pose": pose,
                              "startup_approach": cfg["startup_approach"],
                              "stationary_speed_limit_rad_s": cfg["stationary_speed_limit_rad_s"],
                              "baseline_retry_timeout_s": cfg["baseline_retry_timeout_s"],
                              "retraction_lateness_limit_s": cfg["retraction_lateness_limit_s"],
                              "retraction_max_extension_s": cfg["retraction_max_extension_s"],
                              "sdk_nominal_keyframes_m": {
                                  c: [target(pose, cfg, s, DEPTHS[c]) for s in (0, 0.05, -0.05, 0)]
                                  for c in DEPTHS},
                              "source_hashes": frozen["source_hashes"]}, indent=2))
            return 0 if not errors else 2
        if errors:
            raise ValueError("Program setup not ready: " + "; ".join(errors))
        if args.action == "run":
            if not args.execute or not sys.stdin.isatty():
                raise ValueError("run requires --execute and an attended terminal")
            import fcntl

            args.output.mkdir(parents=True, exist_ok=True)
            lock = (args.output / ".lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            freeze_session(args.output, frozen)
            if len(accepted_records(args.output)) == 8:
                print("8/8 already accepted; no hardware connection made.")
                return 0
            terminal = Terminal()
            print("Automatic DIRECT startup move before first s. Verify clear current-to-start path, --no-return, full +/-50mm path, 12mm press, 20mm boundary and emergency stop. No collision avoidance; no force feedback.")
            if terminal.ask("Type PROGRAMMED + Enter to confirm onsite readiness.", lambda: None) != "PROGRAMMED":
                return 0
            if any(digest(p) != h for p, h in frozen["source_hashes"].items()):
                raise ValueError("Configuration/start changed after preview; restart with verified setup")
            previous_signal = signal.signal(signal.SIGTERM, interrupt_on_signal)
        client = open_client(args.host, args.port)
        client._stub = DeadlineStub(client._stub)
        robot = Robot(client, cfg)
        state = snapshot(client, "idle")
        plan = approach_plan(state, cfg, pose)
        if args.action == "check":
            try:
                check_start(state, pose, cfg)
                matches = True
            except StateError:
                matches = False
            print(json.dumps({"read_only": True, "start_matches": matches, "startup_approach": plan,
                              "force_checked": False,
                              "calibration_physically_verified": False, "state": state}, indent=2))
            return 0
        from scripts.force_sensor.kwr75_reader import Kwr75Reader

        write_json(args.output / f"connection_{uuid.uuid4().hex}.json", asdict(client.get_firmware_info()))
        sensor = Kwr75Reader(cfg["sensor_port"])
        sensor.start()
        sensor.start_csv(args.output / f"sensor_{uuid.uuid4().hex}.csv")
        time.sleep(0.1)
        states = []
        for _ in range(11):
            state = snapshot(client, "idle")
            approach_plan(state, cfg, pose)
            read_force(sensor, cfg, initial=True)
            states.append(state)
            time.sleep(0.05)
        validate_stationary(states, max_joint_speed=cfg["stationary_speed_limit_rad_s"])
        return 0 if session(robot, sensor, cfg, pose, args.output, terminal) == 8 else 1
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; verify hardware stop and support state.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            try:
                if sensor is not None:
                    sensor.stop()
            finally:
                if lock is not None:
                    lock.close()
                if previous_signal is not None:
                    signal.signal(signal.SIGTERM, previous_signal)
