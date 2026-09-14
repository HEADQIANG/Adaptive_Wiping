"""Hash-bound fixed-installation deployment; no claim of calibrated generalization."""

import json
import math
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.real_deploy.airbot_deploy import PolicyLoop, guard_segment
from scripts.real_training import airbot_programmed_demonstrations as program
from scripts.real_training.config import load_config, output_lock, resolve, training_contract
from scripts.real_training.data import load_prepared
from scripts.robot_control.adapter import DeadlineStub, Robot
from scripts.robot_control.airbot_initial_pose import NotStationary, validate_stationary
from scripts.robot_control.client import StateError, open_client
from scripts.robot_control.safety import (
    check_bounds, interrupt_on_signal, measured_joint_speed_limit, positive,
    quaternion_angle, read_force, write_event,
)
from scripts.shared.common import file_digest, write_json
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_preprocessing import finite_array, make_windows

DT = 0.01
MODE = "fixed_setup_tared_v1"
SCOPE = "same_sponge_same_installation_only"
CONFIRMATIONS = (
    "motion_limits_confirmed", "startup_contact_confirmed", "policy_path_confirmed",
    "physical_estop_verified", "support_handoff_confirmed", "server_no_return_verified",
)


class FixedSetupLoop(PolicyLoop):
    """The 10 mm bootstrap occupies the original 0..2 s history, not extra time."""

    def __init__(self, policy, embedding, initial_position):
        super().__init__(policy, embedding, initial_position)
        self.segment_end[2] = self.initial[2] - 0.4 * 0.005

    def push(self, tick, raw_ft, measured_position):
        filtered = super().push(tick, raw_ft, measured_position)
        if 0 < tick < 200 and tick % 40 == 0:
            self.segment_end[2] = self.initial[2] - (tick + 40) * DT * 0.005
        return filtered


def load_setup(path):
    path = resolve(path)
    bindings = {str(path): file_digest(path)}
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1 or spec.get("mode") != MODE or spec.get("scope") != SCOPE:
        raise ValueError("Expected fixed-setup deployment configuration")
    if spec.get("same_hardware_and_sponge_confirmed") is not True:
        raise ValueError("Fixed installation and same sponge must be confirmed")
    training_path = resolve(spec["training_config"])
    training = load_config(training_path)
    if training["profile"] != "airbot_native_tared_offline":
        raise ValueError("Fixed setup requires tared native training profile")
    arrays, info = load_prepared(training)
    if not np.all(arrays["ft_hz"] == 100):
        raise ValueError("Fixed setup requires training FT filtered on a 100 Hz grid")
    bindings[str(training_path)] = file_digest(training_path)
    policy_path = resolve(spec["policy"])
    policy_hash = file_digest(policy_path)
    if policy_hash != spec["policy_sha256"]:
        raise ValueError("Policy hash differs from reviewed fixed setup")
    bindings[str(policy_path)] = policy_hash
    policy = OfflinePolicy(policy_path)
    if policy.source_kind != "real" or policy.hardware_ready:
        raise ValueError("Expected real, offline-only policy artifact")
    for key, expected in {"prepared_sha256": info["sha256"], "raw_sha256": info["raw_sha256"],
                          "encoder_sha256": info["encoder_sha256"]}.items():
        if policy.metadata["bindings"].get(key) != expected:
            raise ValueError(f"Policy/training {key} mismatch")
    if (policy.metadata["training_contract"] != training_contract(training)
            or policy.metadata["data"] != info):
        raise ValueError("Policy preprocessing or provenance differs from prepared data")
    meta = info["metadata"]
    if (meta.get("compensation") != "recorded_unloaded_baseline_subtracted"
            or meta.get("position_frame") != "SDK configured reference"
            or meta.get("derivation") != "programmed_hold_last_10s_v1"
            or policy.encoder.metadata.get("frame") != "ft_frame local with output Y/Z reversed"):
        raise ValueError("Unexpected fixed-setup input/position convention")
    session_path = resolve(spec["collection_session"]) / "session.json"
    session_hash = file_digest(session_path)
    if (session_hash != spec["collection_session_sha256"]
            or meta["source_hashes"].get(str(session_path)) != session_hash):
        raise ValueError("Policy must be bound to the reviewed collection session")
    frozen = json.loads(session_path.read_text())
    cfg = dict(frozen["config"])
    pose = frozen["initial_pose_record"]["initial_pose"]
    if (cfg["robot_sn"] != meta["robot_id"] or cfg["sensor_port"] != meta["sensor_id"]
            or cfg["expected_eef_type"] != "NULL" or cfg.get("mode", "force-guarded") != "force-guarded"
            or cfg["table_normal_sdk"] != [0, 0, 1] or cfg["slide_direction_sdk"] != [0, 1, 0]
            or cfg["sensor_bias_si"] != [0] * 6):
        raise ValueError("Fixed setup requires the recorded +Z/+Y installation and raw-load guards")
    cfg["mode"] = "force-guarded"
    exception = spec.get("encoder_quality_exception", {})
    if "source_encoder_quality_warning" in policy.warnings and (
        exception.get("scope") != SCOPE or not exception.get("reason")
        or exception.get("policy_sha256") != policy_hash
        or exception.get("encoder_sha256") != info["encoder_sha256"]
        or exception.get("collection_session_sha256") != session_hash
    ):
        raise ValueError("Encoder warning exception is not bound to this exact policy/encoder/session")
    if "exploration_outside_encoder_normalization_range_not_clipped" in policy.warnings:
        raise ValueError("Exploration out of encoder range remains a blocker")
    if spec.get("initialization") != "nominal_10mm_during_first_2s":
        raise ValueError("Explicit reviewed contact initialization is required")
    for key, cap in {"max_cartesian_speed_m_s": 0.05, "max_delta_h_m": 0.003,
                     "max_tracking_error_m": 0.005, "orientation_error_limit_rad": 0.02}.items():
        cfg[key] = positive(spec.get(key), key, cap)
    for key in ("tracking_error_policy", "orientation_error_policy"):
        cfg[key] = spec.get(key, "stop")
        if cfg[key] not in ("stop", "record-only"):
            raise ValueError(f"{key} must be stop or record-only")
    if cfg["max_depth_m"] != 0.02 or cfg["stationary_speed_limit_rad_s"] != 0.1:
        raise ValueError("Collection boundary/stationary contract changed")
    normalized = policy.encoder.preprocessor.transform(arrays["exploration"])
    if np.any((normalized < 0) | (normalized > 0.9)):
        raise ValueError("Frozen same-sponge exploration outside encoder range")
    embedding = policy.encode_exploration(arrays["exploration"])
    np.testing.assert_array_equal(arrays["sponge"], np.repeat(embedding, 8, axis=0))
    bindings.update(meta["source_hashes"])
    for relative, expected in policy.metadata["bindings"]["software"]["sources"].items():
        bindings[str(resolve(relative))] = expected
    bindings.update({str(resolve(training["raw_data"])): info["raw_sha256"],
                     str(resolve(training["encoder"])): info["encoder_sha256"],
                     str(resolve(training["output_dir"]) / "prepared.h5"): info["sha256"],
                     str(Path(__file__).resolve()): file_digest(__file__)})
    bindings[str(Path(__file__).with_name("airbot_deploy.py").resolve())] = file_digest(Path(__file__).with_name("airbot_deploy.py"))
    for folder in ("scripts/robot_control", "scripts/force_sensor", "scripts/real_deploy"):
        for source in sorted(resolve(folder).rglob("*.py")):
            bindings[str(source)] = file_digest(source)
    assert_unchanged(bindings)
    return spec, cfg, pose, policy, embedding, arrays, training, bindings


def assert_unchanged(bindings):
    if any(file_digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Fixed-setup inputs changed; review and rerun preflight/replay")


def envelope(position, pose, *, measured=False):
    position = finite_array(position, "SDK position")
    if position.shape != (3,):
        raise StateError("Expected SDK XYZ")
    offset = position - pose["sdk_end_position_m"]
    if (abs(offset[0]) > 0.005 or abs(offset[1]) > 0.05
            or -offset[2] > 0.02 or offset[2] > (0.002 if measured else 1e-9)):
        raise StateError("Fixed-setup relative XYZ/depth boundary exceeded; no clipping")


def guard_loop(loop, cfg, pose):
    guard_segment(loop, cfg)
    envelope(loop.segment_start, pose)
    envelope(loop.segment_end, pose)


def replay(spec, cfg, pose, policy, embedding, arrays, training, bindings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    trajectories, predictions, errors, violations = [], [], [], []
    with h5py.File(resolve(training["raw_data"]), "r") as h5:
        for index, name in enumerate(sorted(h5["demonstrations"])):
            group = h5[f"demonstrations/{name}"]
            raw = group["ft_raw_before_baseline"][:]
            bias = group["recorded_unloaded_baseline"][:]
            xyz = group["sdk_end_position"][:]
            loop = FixedSetupLoop(policy, embedding, pose["sdk_end_position_m"])
            targets, delta = [], []
            for tick in range(1001):
                loop.push(tick, raw[tick] - bias, xyz[tick])
                try:
                    guard_loop(loop, cfg, pose)
                except StateError as exc:
                    violations.append({"demo": name, "tick": tick, "error": str(exc)})
                targets.append(loop.target(tick))
                if loop.last_delta is not None:
                    delta.append(loop.last_delta)
            windows, _ = make_windows(arrays["ft"][index:index + 1], arrays["height"][index:index + 1])
            expected = policy.predict_delta_h(np.repeat(embedding, 20, axis=0), windows.reshape(20, 5, 6))[:, 0]
            # Single/batch inference may differ by a few float32 rounding ulps.
            np.testing.assert_allclose(delta, expected, atol=1e-8, rtol=0)
            errors.append(float(np.max(np.abs(np.asarray(delta) - expected))))
            trajectories.append(targets)
            predictions.append(delta)
    assert_unchanged(bindings)
    np.savez_compressed(output / "replay.npz", target_sdk_m=trajectories, delta_h_m=predictions,
                        time_s=np.arange(1001) / 100, hardware_ready=False)
    report = {"mode": MODE, "scope": "recorded_force_teacher_forced_not_closed_loop",
              "passed": not violations, "bindings": bindings, "violations": violations,
              "max_stream_batch_error_m": max(errors), "feedback_predictions": 160,
              "bootstrap": spec["initialization"], "hardware_ready": False,
              "warnings": policy.warnings}
    write_json(output / "report.json", report)
    return report


def run_blockers(spec, bindings):
    blockers = [f"Onsite confirmation required: {key}" for key in CONFIRMATIONS if spec.get(key) is not True]
    path = resolve(spec["replay_report"])
    if not path.is_file():
        blockers.append("Fixed-setup replay report is missing")
    else:
        report = json.loads(path.read_text())
        if report.get("mode") != MODE or report.get("passed") is not True or report.get("bindings") != bindings:
            blockers.append("Fixed-setup replay failed or is stale")
    return blockers


def observation(robot, sensor, cfg, pose, target, *, shadow=False, initial=False,
                approach=False, target_quaternion=None):
    if not shadow:
        robot.owned()
    state = robot.read("idle" if shadow else "servo")
    check_bounds(state, cfg)
    if state["read_duration_s"] > 0.02:
        raise StateError("Robot state read exceeded 20 ms")
    if max(abs(v) for v in state["joint_velocity_rad_s"]) > measured_joint_speed_limit(cfg):
        raise StateError("Measured joint speed exceeded")
    quaternion = pose["sdk_end_orientation_xyzw"] if target_quaternion is None else target_quaternion
    measured_quaternion = finite_array(state["sdk_end_orientation_xyzw"], "SDK orientation")
    if measured_quaternion.shape != (4,) or np.linalg.norm(measured_quaternion) < 1e-9:
        raise StateError("Invalid SDK orientation")
    angle = quaternion_angle(state["sdk_end_orientation_xyzw"], quaternion)
    position_error = math.dist(state["sdk_end_position_m"], target)
    position_limit = cfg["startup_approach"]["tracking_tolerance_m"] if approach else cfg["max_tracking_error_m"]
    angle_limit = cfg["startup_approach"]["orientation_tolerance_rad"] if approach else cfg["orientation_error_limit_rad"]
    position_policy = cfg.get("tracking_error_policy", "stop")
    angle_policy = cfg.get("orientation_error_policy", "stop")
    if angle > angle_limit and angle_policy != "record-only":
        raise StateError("Fixed tool orientation changed")
    if approach:
        # The direct approach starts outside the task envelope; depth remains guarded.
        if pose["sdk_end_position_m"][2] - state["sdk_end_position_m"][2] > cfg["max_depth_m"]:
            raise StateError("Startup approach measured depth exceeded 20 mm")
    else:
        envelope(state["sdk_end_position_m"], pose, measured=True)
    if position_error > position_limit and position_policy != "record-only":
        raise StateError("Fixed-setup position tracking exceeded limit")
    force = read_force(sensor, cfg, initial=initial)
    return {"pose": state, "force": force, "observed_perf_s": time.perf_counter(),
            "tracking_target_sdk_m": list(target), "tracking_target_quaternion_xyzw": list(quaternion),
            "position_error_m": position_error, "orientation_error_rad": angle,
            "position_error_reference_m": position_limit, "orientation_error_reference_rad": angle_limit,
            "position_tracking_exceeds_reference": position_error > position_limit,
            "orientation_tracking_exceeds_reference": angle > angle_limit,
            "tracking_error_policy": position_policy, "orientation_error_policy": angle_policy}


def startup_approach(robot, sensor, cfg, pose, state, stream):
    """Same direct quintic trajectory, with deployment-local tracking policies."""
    from scipy.spatial.transform import Rotation, Slerp

    plan = program.approach_plan(state, cfg, pose)
    position = list(state["sdk_end_position_m"])
    quaternion = list(state["sdk_end_orientation_xyzw"])
    rotation = Slerp([0, 1], Rotation.from_quat([quaternion, pose["sdk_end_orientation_xyzw"]]))
    write_event(stream, {"event": "approach_start", "initial_state": state, **plan})
    sensor.flush_csv()
    begun = time.perf_counter()
    row = observation(robot, sensor, cfg, pose, position, initial=True, approach=True,
                      target_quaternion=quaternion)
    if time.perf_counter() - begun > DT:
        raise StateError("Startup approach hold observation timeout")
    robot.send(position, quaternion)
    if time.perf_counter() - begun > DT:
        raise StateError("Startup approach hold command timeout")
    write_event(stream, {"event": "sample", "phase": "approach_hold", **row})
    if time.perf_counter() - begun > DT:
        raise StateError("Startup approach hold logging timeout")
    count = round(plan["duration_s"] / DT)
    begun = time.perf_counter()
    for tick in range(1, count + 1):
        due = begun + tick * DT
        time.sleep(max(0, due - time.perf_counter()))
        if time.perf_counter() - due > 0.005:
            raise StateError("Startup approach scheduling deadline missed")
        row = observation(robot, sensor, cfg, pose, position, initial=True, approach=True,
                          target_quaternion=quaternion)
        alpha = tick / count
        alpha = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
        position = [p + alpha * (q - p) for p, q in zip(state["sdk_end_position_m"], pose["sdk_end_position_m"])]
        quaternion = rotation(alpha).as_quat().tolist()
        if time.perf_counter() - due > DT:
            raise StateError("Startup approach observation/target timeout")
        robot.send(position, quaternion)
        if time.perf_counter() - due > DT:
            raise StateError("Startup approach command timeout")
        sensor.flush_csv()
        write_event(stream, {"event": "sample", "phase": "approach", **row,
                             "tick": tick, "due_perf_s": due,
                             "commanded_position_m": position, "commanded_quaternion_xyzw": quaternion})
        if time.perf_counter() - due > DT:
            raise StateError("Startup approach logging timeout")
    settle_at_start(robot, sensor, cfg, pose, stream, "approach_settle")
    write_event(stream, {"event": "approach_complete", "return_confirmed": True})


def baseline(robot, sensor, cfg, pose, stream, *, shadow=False):
    deadline = time.perf_counter() + 10
    while time.perf_counter() < deadline:
        stationary = []
        while len(stationary) < 11 and time.perf_counter() < deadline:
            row = observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"], shadow=shadow, initial=True)
            program.check_return_pose(row["pose"], cfg, pose)
            stationary.append(row["pose"])
            write_event(stream, {"event": "sample", "phase": "baseline_settle", **row})
            sensor.flush_csv()
            time.sleep(0.05)
        try:
            validate_stationary(stationary, max_joint_speed=0.1)
            start, unique, states = time.perf_counter(), {}, []
            while time.perf_counter() - start < 1:
                if time.perf_counter() >= deadline:
                    raise StateError("Baseline total timeout")
                row = observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"], shadow=shadow, initial=True)
                program.check_return_pose(row["pose"], cfg, pose)
                states.append(row["pose"])
                write_event(stream, {"event": "sample", "phase": "baseline", **row})
                if len(states) >= 11:
                    validate_stationary(states[-11:], max_joint_speed=0.1)
                force = row["force"]
                stamp, wrench = force["sensor_receive_perf_s"], force["raw_sensor_wrench_si"]
                if unique and stamp < max(unique):
                    raise StateError("Baseline sensor clock reversed")
                if stamp >= start:
                    if stamp in unique and unique[stamp] != wrench:
                        raise StateError("Conflicting baseline sensor frames")
                    unique[stamp] = wrench
                sensor.flush_csv()
                time.sleep(DT)
            validate_stationary(states, max_joint_speed=0.1)
        except NotStationary as exc:
            write_event(stream, {"event": "baseline_discarded", "reason": str(exc)})
            continue
        if time.perf_counter() > deadline or len(unique) < 40 or max(unique) - min(unique) < 0.9:
            raise StateError("Incomplete fresh 1s baseline within 10s timeout")
        bias = np.mean(list(unique.values()), axis=0)
        write_event(stream, {"event": "baseline_complete", "bias_si": bias.tolist(), "distinct_samples": len(unique)})
        return bias
    raise StateError("Baseline stationary timeout")


def run_loop(robot, sensor, policy, embedding, cfg, pose, bias, stream, *, shadow=False):
    loop = FixedSetupLoop(policy, embedding, pose["sdk_end_position_m"])
    target = list(pose["sdk_end_position_m"])
    start = time.perf_counter()
    last = None
    for tick in range(1001):
        due = start + tick * DT
        time.sleep(max(0, due - time.perf_counter()))
        if time.perf_counter() - due > 0.005:
            raise StateError("Fixed-setup scheduling deadline missed; no catch-up")
        last = observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"] if shadow else target, shadow=shadow)
        item = sensor.latest_before(due, net=False)
        if item is None or not 0 <= due - item[0] <= 0.02:
            raise StateError("No fresh causal FT at policy time")
        raw = finite_array(item[1], "raw sensor FT")
        if (raw.shape != (6,) or np.linalg.norm(raw[:3]) > cfg["max_force_n"]
                or np.linalg.norm(raw[3:]) > cfg["max_torque_nm"]):
            raise StateError("Causal raw force/torque limit exceeded")
        ft = raw - bias
        filtered = loop.push(tick, ft, last["pose"]["sdk_end_position_m"])
        guard_loop(loop, cfg, pose)
        target = loop.target(tick).tolist()
        envelope(target, pose)
        read_force(sensor, cfg)
        if time.perf_counter() - due > 0.009:
            raise StateError("Fixed-setup inference deadline exceeded before send")
        if not shadow:
            robot.send(target, pose["sdk_end_orientation_xyzw"])
        write_event(stream, {"event": "sample", "phase": "shadow" if shadow else "policy", "tick": tick,
                             "due_perf_s": due, "sensor_receive_perf_s": item[0], "raw_ft": raw.tolist(),
                             "tared_ft": ft.tolist(), "filtered_ft": filtered.tolist(),
                             "target_sdk_m": target, "delta_h_m": loop.last_delta, **last})
        sensor.flush_csv()
        if time.perf_counter() - due > DT:
            raise StateError("Fixed-setup command/logging exceeded 10 ms cycle")
    write_event(stream, {"event": "policy_complete", "shadow": shadow, "hardware_ready": False})
    return observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"] if shadow else target, shadow=shadow)


def retract(robot, sensor, cfg, pose, state, stream):
    position = np.array(state["sdk_end_position_m"])
    start = np.array(pose["sdk_end_position_m"])
    # First lift at the measured lateral position, then return laterally in air.
    for destination in (np.r_[position[:2], start[2]], start):
        origin = position.copy()
        count = max(1, math.ceil(np.linalg.norm(destination - origin) / 0.005 / DT))
        begun = time.perf_counter()
        for index in range(1, count + 1):
            cycle = time.perf_counter()
            row = observation(robot, sensor, cfg, pose, position)
            if time.perf_counter() - cycle > 0.01:
                raise StateError("Return observation timeout")
            position = origin + (destination - origin) * index / count
            envelope(position, pose)
            robot.send(position.tolist(), pose["sdk_end_orientation_xyzw"])
            sensor.flush_csv()
            write_event(stream, {"event": "return_target", "target_sdk_m": position.tolist(), **row})
            if time.perf_counter() - cycle > 0.02 or time.perf_counter() - begun > count * DT + 1:
                raise StateError("Return command/logging timeout")
            time.sleep(max(0, cycle + DT - time.perf_counter()))
    settle_at_start(robot, sensor, cfg, pose, stream, "return_check")
    write_event(stream, {"event": "return_confirmed"})


def settle_at_start(robot, sensor, cfg, pose, stream, phase):
    deadline, states = time.perf_counter() + 2, []
    while time.perf_counter() < deadline:
        row = observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"], initial=True)
        states.append(row["pose"])
        states = states[-11:]
        write_event(stream, {"event": "sample", "phase": phase, **row})
        sensor.flush_csv()
        if len(states) == 11:
            try:
                for state in states:
                    program.check_return_pose(state, cfg, pose)
                validate_stationary(states, max_joint_speed=0.1)
            except ValueError:
                pass
            else:
                return
        time.sleep(0.05)
    raise StateError("Fixed-setup return/stationary timeout")


def execute(robot, sensor, policy, embedding, cfg, pose, stream, terminal, *, shadow=False):
    attempted = False
    try:
        if not shadow:
            # Load interpolation dependencies before entering servo.
            from scipy.spatial.transform import Rotation, Slerp  # noqa: F401

            robot.acquire()
            states = []
            for _ in range(11):
                robot.owned()
                state = robot.read("idle")
                program.approach_plan(state, cfg, pose)
                read_force(sensor, cfg, initial=True)
                states.append(state)
                time.sleep(0.05)
            validate_stationary(states, max_joint_speed=0.1)
            attempted = True
            robot.enter()
            startup_approach(robot, sensor, cfg, pose, state, stream)
        def monitor():
            row = observation(robot, sensor, cfg, pose, pose["sdk_end_position_m"], shadow=shadow, initial=True)
            program.check_return_pose(row["pose"], cfg, pose)
            sensor.flush_csv()
            return row
        command = terminal.ask("At non-contact start. s=start one policy trial, q=finish.", monitor)
        while command not in ("s", "q"):
            command = terminal.ask("s=start one policy trial, q=finish.", monitor)
        if command == "s":
            bias = baseline(robot, sensor, cfg, pose, stream, shadow=shadow)
            terminal.discard()
            last = run_loop(robot, sensor, policy, embedding, cfg, pose, bias, stream, shadow=shadow)
            if not shadow:
                retract(robot, sensor, cfg, pose, last["pose"], stream)
        if not shadow:
            while terminal.ask("Servo active. Arrange support; type IDLE to release.", monitor) != "IDLE":
                pass
            robot.idle()
        write_event(stream, {"event": "session_complete", "shadow": shadow, "hardware_ready": False})
    except BaseException as exc:
        stop_error = None
        if attempted:
            try:
                robot.abort()
            except BaseException as failure:
                stop_error = str(failure)
                print(f"STOP NOT CONFIRMED: {failure}; use physical emergency stop", file=sys.stderr)
        write_event(stream, {"event": "aborted", "error": str(exc), "stop_error": stop_error,
                             "no_automatic_retract": True, "shadow": shadow})
        raise


def main(args):
    client = sensor = None
    previous_signal = None
    torch.set_num_threads(1)
    try:
        loaded = load_setup(args.config or "configs/real_deploy/airbot_fixed_setup.json")
        spec, cfg, pose, policy, embedding, arrays, training, bindings = loaded
        if args.action == "replay":
            if args.output is None:
                raise ValueError("replay requires a fresh --output directory")
            report = replay(*loaded, args.output)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 2
        blockers = run_blockers(spec, bindings)
        if args.action == "preflight":
            print(json.dumps({"mode": MODE, "offline_inputs_valid": True, "hardware_connected": False,
                              "run_enabled": not blockers, "blockers": blockers, "scope": SCOPE,
                              "tracking_error_policy": cfg["tracking_error_policy"],
                              "orientation_error_policy": cfg["orientation_error_policy"],
                              "start_return_pose_checks": "required",
                              "initialization": spec["initialization"], "warnings": policy.warnings}, indent=2))
            return 2 if blockers else 0
        shadow = args.action == "shadow"
        if not shadow and blockers:
            raise ValueError("Run blocked: " + "; ".join(blockers))
        if not args.execute or not sys.stdin.isatty() or args.output is None:
            raise ValueError("shadow/run require --execute, an attended terminal and a fresh --output JSONL")
        output = Path(args.output)
        csv = output.with_suffix(".sensor.csv")
        if output.exists() or csv.exists():
            raise FileExistsError("Output log/CSV already exists")
        terminal = program.Terminal()
        token = "SHADOW" if shadow else "FIXED-SETUP"
        print("Shadow sends NO motion; support the idle arm." if shadow else
              "Direct startup move after confirmation, then s starts a 10s policy including 10mm press. Independent emergency stop required.")
        if terminal.ask(f"Type {token} to confirm this fixed installation and clear path.", lambda: None) != token:
            return 0
        assert_unchanged(bindings)
        if not shadow and run_blockers(spec, bindings):
            raise ValueError("Replay/confirmations changed before connection")
        from scripts.force_sensor.kwr75_reader import Kwr75Reader

        with output_lock(output.parent), output.open("x") as stream:
            write_event(stream, {"event": "session_start", "mode": MODE, "shadow": shadow,
                                 "spec": spec, "bindings": bindings, "warnings": policy.warnings})
            previous_signal = signal.signal(signal.SIGTERM, interrupt_on_signal)
            client = open_client(args.host, args.port)
            client._stub = DeadlineStub(client._stub)
            robot = Robot(client, cfg)
            sensor = Kwr75Reader(cfg["sensor_port"])
            sensor.start()
            sensor.start_csv(csv)
            time.sleep(0.1)
            execute(robot, sensor, policy, embedding, cfg, pose, stream, terminal, shadow=shadow)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; verify stop/support state. No automatic recovery.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        try:
            if sensor is not None:
                sensor.stop()
        finally:
            if client is not None:
                client.close()
