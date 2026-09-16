"""Attended manual-start exploration; measured loads are recorded, never limited."""

import copy
import json
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import asdict

from scripts.robot_control.adapter import DeadlineStub
from scripts.robot_control.client import SDK_VERSION, StateError, open_client
from scripts.robot_control.hardware import HardwareRobot
from scripts.robot_control.safety import (
    checked_vector, interrupt_on_signal, norm, positive,
    quaternion_angle, write_event,
)
from scripts.shared.exploration_protocol import (
    INITIAL_GAP_M, PRESS_DEPTH_M, PRESS_DURATION_S, PRESS_SPEED_M_S,
    SAMPLE_HZ, exploration_offset,
)
from scripts.real_training.manual_exploration_contract import PROTOCOL

DT = 1 / SAMPLE_HZ
MAX_FT_AGE_S = 0.020
MAX_LATENESS_S = 0.005


def load_setup(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1 or cfg.get("mode") != "manual-start":
        raise ValueError("Expected a manual-start exploration configuration")
    for key in ("initial_pose_file", "initial_pose_sha256", "max_force_n",
                "max_initial_force_n", "max_torque_nm"):
        if key in cfg:
            raise ValueError(f"manual-start does not use {key}; select its dedicated configuration")
    for key in ("robot_sn", "expected_eef_type", "sensor_port", "sdk_frame_and_tool_note",
                "sponge_id", "exploration_id"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Set {key}")
    for key, size in (("table_normal_sdk", 3), ("slide_direction_sdk", 3),
                      ("joint_current_limits", 6), ("sensor_bias_si", 6)):
        cfg[key] = checked_vector(cfg.get(key), size, key)
    normal, slide = cfg["table_normal_sdk"], cfg["slide_direction_sdk"]
    if (abs(norm(normal) - 1) > 1e-6 or abs(norm(slide) - 1) > 1e-6
            or abs(sum(a * b for a, b in zip(normal, slide))) > 1e-6):
        raise ValueError("Table normal and slide direction must be orthogonal unit vectors")
    if cfg.get("initial_gap_m") != INITIAL_GAP_M:
        raise ValueError("Operator must establish the 1 mm non-contact start gap")
    if positive(cfg.get("verified_compression_allowance_m"), "compression allowance") < PRESS_DEPTH_M - INITIAL_GAP_M:
        raise ValueError("The nominal trajectory requires at least 9 mm compression allowance")
    command = positive(cfg.get("joint_speed_limit_rad_s"), "command speed", 0.4)
    measured = positive(cfg.get("measured_joint_speed_stop_rad_s"), "measured speed", 1.2)
    if measured < command:
        raise ValueError("Measured speed stop must not be below command speed")
    for value in cfg["joint_current_limits"]:
        positive(value, "joint current parameter", 20)
    if cfg["expected_eef_type"] != "NULL":
        positive(cfg.get("eef_current_limit"), "eef current parameter", 20)
    return cfg


def check_state(state, cfg, *, check_speed=True):
    checked_vector(state["sdk_end_position_m"], 3, "SDK end position")
    checked_vector(state["joint_position_rad"], 6, "joint position")
    velocity = checked_vector(state["joint_velocity_rad_s"], 6, "joint velocity")
    quaternion = checked_vector(state["sdk_end_orientation_xyzw"], 4, "orientation")
    if abs(norm(quaternion) - 1) > 0.01:
        raise StateError("Invalid measured unit quaternion")
    if check_speed and max(map(abs, velocity)) > cfg["measured_joint_speed_stop_rad_s"]:
        raise StateError("Measured joint speed exceeded configured limit")


class ForceReader:
    def __init__(self, sensor, cfg):
        self.sensor, self.cfg = sensor, cfg
        self.last_stamp = None
        self.last_raw = None

    def read(self, baseline=None):
        item = self.sensor.latest(net=False)
        if item is None:
            raise StateError("No force sensor data")
        stamp, values = item
        age = time.perf_counter() - stamp
        if not math.isfinite(stamp) or not 0 <= age <= MAX_FT_AGE_S:
            raise StateError("Force sensor stale/invalid")
        raw = checked_vector(list(values), 6, "raw sensor wrench")
        if self.last_stamp is not None:
            if stamp < self.last_stamp:
                raise StateError("Force receive timestamp moved backwards")
            if stamp == self.last_stamp and raw != self.last_raw:
                raise StateError("Conflicting force values share a receive timestamp")
        self.last_stamp, self.last_raw = stamp, raw
        result = {"sensor_receive_perf_s": stamp, "sensor_age_s": age,
                  "raw_sensor_wrench_si": raw,
                  "bias_corrected_sensor_wrench_si": [a - b for a, b in zip(raw, self.cfg["sensor_bias_si"])]}
        if baseline is not None:
            result["tared_sensor_wrench_si"] = [a - b for a, b in zip(raw, baseline)]
        return result


class Keyboard:
    def wait(self, key, monitor, prompt):
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd, termios.TCSANOW)
            termios.tcflush(fd, termios.TCIFLUSH)
            print(prompt, flush=True)
            buffer = ""
            while True:
                monitor()
                ready, _, _ = select.select([fd], [], [], DT)
                if not ready:
                    continue
                data = os.read(fd, 1)
                if not data or data == b"\x04":
                    raise EOFError("Terminal closed")
                char = data.decode("ascii", errors="ignore")
                if len(key) == 1:
                    if char.lower() == key:
                        return
                elif char in ("\n", "\r"):
                    if buffer == key:
                        return
                    buffer = ""
                elif char in ("\x7f", "\b"):
                    buffer = buffer[:-1]
                else:
                    buffer = (buffer + char)[-32:]
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, original)


def observe(robot, reader, cfg, mode, reference=None, target=None, baseline=None, *, check_speed=True):
    robot.owned()
    state = robot.read(mode)
    result = {"state": state, "force": None,
              "observation_perf_s": time.perf_counter()}
    try:
        check_state(state, cfg, check_speed=check_speed)
        result["force"] = reader.read(baseline)
        result["observation_perf_s"] = time.perf_counter()
    except (StateError, ValueError) as exc:
        # Preserve already-read feedback without delaying the stop for additional I/O.
        exc.fault_observation = result
        raise
    if reference is not None:
        from scripts.real_training.airbot_exploration import tracking_errors

        errors = tracking_errors(state["sdk_end_position_m"], target, cfg)
        angle = quaternion_angle(state["sdk_end_orientation_xyzw"], reference["sdk_end_orientation_xyzw"])
        result.update(errors, target_position_m=list(target), orientation_error_rad=angle,
                      orientation_error_limit_rad=0.15, tracking_error_record_only=True,
                      position_tracking_exceeds_limit=errors["position_error_m"] > 0.005,
                      orientation_tracking_exceeds_limit=angle > 0.15,
                      slide_error_record_only=True,
                      slide_tracking_exceeds_5mm=abs(errors["slide_error_m"]) > 0.005)
    return result


def collect_tare(hold, stream):
    start = time.perf_counter()
    write_event(stream, {"event": "tare_start", "perf_s": start, "duration_s": 1.0})
    samples = []
    while time.perf_counter() - start < 1.0:
        force = hold()["force"]
        stamp = force["sensor_receive_perf_s"]
        if stamp >= start and (not samples or stamp > samples[-1]["sensor_receive_perf_s"]):
            samples.append(force)
        time.sleep(DT)
    if len(samples) < 40 or samples[-1]["sensor_receive_perf_s"] - samples[0]["sensor_receive_perf_s"] < 0.9:
        raise StateError("Insufficient fresh baseline samples or coverage")
    bias = [math.fsum(row["raw_sensor_wrench_si"][i] for row in samples) / len(samples) for i in range(6)]
    write_event(stream, {"event": "tare_complete", "tare_bias_si": bias,
                         "distinct_samples": len(samples), "samples": samples,
                         "start_perf_s": start, "end_perf_s": time.perf_counter(),
                         "stationary_validated": False, "noncontact_operator_confirmed": True,
                         "gravity_compensated": False, "method": "mean of distinct raw sensor frames"})
    return bias


def execute(robot, sensor, cfg, stream, time_scale=1.0, keyboard=None):
    positive(time_scale, "time_scale", 10)
    if time_scale < 1:
        raise ValueError("Accelerated exploration is forbidden")
    keyboard = keyboard or Keyboard()
    reader = ForceReader(sensor, cfg)
    acquired = attempted = False
    phase, index = "preflight", None
    try:
        check_state(robot.read("idle"), cfg)
        reader.read()
        robot.acquire()
        acquired = True
        attempted = True
        robot.switch("gravity_comp")
        phase = "drag"
        write_event(stream, {"event": "phase_start", "phase": phase, "perf_s": time.perf_counter(),
                             "measured_joint_speed_stop_enforced": False})
        keyboard.wait("h", lambda: observe(robot, reader, cfg, "gravity_comp", check_speed=False),
                      "Drag to a clear 1 mm non-contact gap. Support the arm; press h to confirm setup and hold.")
        # The final reference read still belongs to gravity-compensation dragging.
        reference = copy.deepcopy(observe(robot, reader, cfg, "gravity_comp", check_speed=False)["state"])
        robot.switch("servo")
        robot.send_joint(reference["joint_position_rad"])
        write_event(stream, {"event": "reference_captured", "runtime_reference_pose": reference,
                             "noncontact_operator_confirmed": True, "stationary_validated": False})
        phase = "hold"
        write_event(stream, {"event": "phase_start", "phase": phase, "perf_s": time.perf_counter(),
                             "measured_joint_speed_stop_enforced": True})
        baseline = None

        def hold():
            row = observe(robot, reader, cfg, "servo", reference, reference["sdk_end_position_m"], baseline)
            robot.send_joint(reference["joint_position_rad"])
            return row

        keyboard.wait("s", hold, "Servo hold active. Remove external tool loads; press s to tare and explore.")
        phase = "tare"
        baseline = collect_tare(hold, stream)
        initial = hold()
        write_event(stream, {"event": "initial", **initial})
        quaternion = reference["sdk_end_orientation_xyzw"]
        previous = reference["sdk_end_position_m"]
        errors, angles, lateness, slide_errors = [], [], [], []
        for phase in ("exploration", "retract"):
            count = round((4 if phase == "exploration" else 2) * SAMPLE_HZ * time_scale)
            start = time.perf_counter()
            write_event(stream, {"event": "phase_start", "phase": phase, "perf_s": start})
            for index in range(1, count + 1):
                due = start + index * DT
                before = observe(robot, reader, cfg, "servo", reference, previous, baseline)
                if time.perf_counter() > due:
                    raise StateError("Control iteration overrun before send; no catch-up burst")
                protocol_t = 4 * index / count if phase == "exploration" else None
                if protocol_t is not None:
                    _, dy, dz = exploration_offset(protocol_t)
                else:
                    dy, dz = 0, -PRESS_DEPTH_M * (1 - index / count)
                target = [p + dy * s + dz * n for p, s, n in zip(
                    reference["sdk_end_position_m"], cfg["slide_direction_sdk"], cfg["table_normal_sdk"])]
                sent = time.perf_counter()
                robot.send(target, quaternion)
                time.sleep(max(0, due - time.perf_counter()))
                measured = observe(robot, reader, cfg, "servo", reference, target, baseline)
                sampled = time.perf_counter()
                late = sampled - due
                write_event(stream, {"event": "sample", "phase": phase, "index": index,
                                     "protocol_time_s": protocol_t, "due_perf_s": due,
                                     "send_perf_s": sent, "sample_perf_s": sampled, "lateness_s": late,
                                     "target_quaternion_xyzw": quaternion,
                                     "pre_send_force": before["force"], **measured})
                if late > MAX_LATENESS_S:
                    raise StateError("100 Hz deadline missed by over 5 ms; no catch-up")
                if phase == "exploration":
                    errors.append(measured["position_error_m"])
                    angles.append(measured["orientation_error_rad"])
                    lateness.append(late)
                    slide_errors.append(measured["slide_error_m"])
                previous = target
        rms = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
        write_event(stream, {"event": "motion_complete", "mode": "manual-start",
                             "acquisition_protocol": PROTOCOL, "exploration_samples": len(errors),
                             "force_monitoring": True, "force_limits_enforced": False,
                             "joint_position_limits_enforced": False,
                             "stationary_validated": False, "software_tare_applied": True,
                             "time_scale": time_scale, "nominal_protocol_match": time_scale == 1,
                             "press_speed_m_s": PRESS_SPEED_M_S, "press_duration_s": PRESS_DURATION_S,
                             "position_rms_m": rms, "position_rms_within_simulation_1mm": rms <= 0.001,
                             "max_orientation_error_rad": max(angles), "max_lateness_s": max(lateness),
                             "max_abs_slide_error_m": max(map(abs, slide_errors)),
                             "slide_error_over_5mm_samples": sum(abs(e) > 0.005 for e in slide_errors),
                             "tracking_error_policy": "record-only",
                             "return_position_error_m": measured["position_error_m"],
                             "return_orientation_error_rad": measured["orientation_error_rad"],
                             "return_within_tracking_limits": not (measured["position_tracking_exceeds_limit"]
                                                                     or measured["orientation_tracking_exceeds_limit"]),
                             "completion_semantics": "command sequence completed, not measured trajectory acceptance",
                             "encoder_ready": False})
        phase, index = "handoff", None
        write_event(stream, {"event": "phase_start", "phase": phase, "perf_s": time.perf_counter()})

        def returned_hold():
            row = observe(robot, reader, cfg, "servo", reference, reference["sdk_end_position_m"], baseline)
            robot.send(reference["sdk_end_position_m"], quaternion)
            return row

        keyboard.wait("IDLE", returned_hold,
                      "Return commands complete; actual return is NOT guaranteed. Servo holds the start target. "
                      "Arrange safe support, then type IDLE + Enter to release.")
        robot.idle()
        write_event(stream, {"event": "session_complete", "idle_confirmed": True})
    except BaseException as exc:
        stop_error = None
        if acquired and attempted:
            try:
                robot.abort()
            except BaseException as failure:
                stop_error = str(failure)
                print(f"STOP NOT CONFIRMED: {failure}; use physical emergency stop NOW", file=sys.stderr, flush=True)
        try:
            write_event(stream, {"event": "aborted", "mode": "manual-start", "error": str(exc),
                                 "context": {"phase": phase, "index": index},
                                 "software_stop_requested": acquired and attempted,
                                 "fault_observation": getattr(exc, "fault_observation", None),
                                 "stop_error": stop_error, "encoder_ready": False})
        except Exception:
            pass
        raise


def main(args):
    client = sensor = plot = None
    previous_signal = None
    try:
        cfg = load_setup(args.config)
        if args.action == "run":
            from scripts.force_sensor.kwr75_reader import Kwr75Reader

            print("MANUAL START: no start matching, stationary acceptance, project joint-position limits or force/torque limit stops. "
                  "Drag speed protection is SDK-internal only; project measured-speed stop resumes in servo. "
                  "Verify --no-return, full swept path, 1 mm gap, compression allowance, physical estop and support. "
                  "h=hold, s=tare/start; faults request software stop WITHOUT retract. Idle does not hold.", flush=True)
            if args.plot:
                from scripts.real_training.exploration_live_plot import ExplorationLivePlot

                plot = ExplorationLivePlot(args.output, args.plot_window, args.plot_hz)
                plot.start()
                print("Closing the plot does not stop robot motion.", flush=True)
            previous_signal = signal.signal(signal.SIGTERM, interrupt_on_signal)
        client = open_client(args.host, args.port)
        client._stub = DeadlineStub(client._stub)
        robot = HardwareRobot(client, cfg)
        state = robot.read("idle")
        check_state(state, cfg)
        if args.action == "check":
            print(json.dumps({"read_only": True, "mode": "manual-start", "state": state,
                              "start_matching_required": False, "force_checked": False,
                              "force_limits_enforced": False, "stationary_validated": False,
                              "joint_position_limits_enforced": False,
                              "drag_measured_joint_speed_stop_enforced": False,
                              "servo_measured_joint_speed_stop_enforced": True,
                              "workspace_bounds_enforced": False, "calibration_physically_verified": False}, indent=2))
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            write_event(stream, {"event": "session_start", "source_kind": "real", "sdk_version": SDK_VERSION,
                                 "mode": "manual-start", "acquisition_protocol": PROTOCOL,
                                 "config": cfg, "firmware": asdict(client.get_firmware_info()),
                                 "force_monitoring": True, "force_limits_enforced": False,
                                 "joint_position_limits_enforced": False,
                                 "drag_measured_joint_speed_stop_enforced": False,
                                 "servo_measured_joint_speed_stop_enforced": True,
                                 "stationary_validated": False, "workspace_bounds_enforced": False,
                                 "ft_frame": "sensor local at sensor origin; NOT simulation ft_frame",
                                 "encoder_ready": False})
            sensor = Kwr75Reader(port=cfg["sensor_port"])
            sensor.start()
            deadline = time.perf_counter() + 2
            while sensor.latest(net=False) is None and time.perf_counter() < deadline:
                time.sleep(DT)
            execute(robot, sensor, cfg, stream, args.time_scale)
        print(f"Command sequence complete; idle confirmed. Log: {args.output}")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted. Check physical support and emergency-stop state.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}. No automatic retry; check hardware safety state.", file=sys.stderr)
        return 1
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        try:
            if sensor is not None:
                sensor.stop()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if plot is not None:
                    plot.close()
