"""Attended runtime-origin deployment: drag, hold, tare, infer, hold, release."""

import copy
import json
import sys
import time

import numpy as np

from scripts.real_deploy.airbot_deploy import PolicyLoop
from scripts.real_deploy.wiping_frame import WipingFrame
from scripts.real_training.manual_exploration import Keyboard, ForceReader, observe, collect_tare
from scripts.robot_control.client import StateError
from scripts.robot_control.safety import write_event
from scripts.shared.real_preprocessing import finite_array

WORKFLOW = "runtime_start_h_z_s_g_v1"
DT = 0.01


def warm_policy(policy, embedding):
    """Initialize inference kernels before connecting, without live filter history."""
    history = np.zeros((1, 5, 6))
    for _ in range(3):
        xy = finite_array(policy.predict_xy(embedding), "warmup XY")
        height = finite_array(policy.predict_delta_h(embedding, history), "warmup height")
        if xy.shape != (1, 25, 2) or height.shape != (1, 1):
            raise ValueError("Unexpected policy warmup output shape")


def make_loop(policy, embedding, start, *, wiping_frame=None):
    frame = wiping_frame or WipingFrame()
    loop = PolicyLoop(policy, embedding, start, stationary_start=True, wiping_frame=frame)
    original = loop.xy.copy()
    translation = loop.initial[list(frame.tangent_axes)] - original[0] * frame.tangent_signs
    loop.xy = frame.translate_path(original, loop.initial)
    loop.segment_end = loop.initial.copy()
    path = {
        "original_xy_sdk_m": original.tolist(),
        "wiping_frame": frame.metadata(),
        "translated_tangent_sdk_m": loop.xy.tolist(),
        "tangent_path_sdk_m": [frame.endpoint(p, loop.initial[frame.normal_axis]).tolist() for p in loop.xy],
        "translation_sdk_m": translation.tolist(), "scale": 1.0,
        "initialization": "hold_pose_2s_then_policy_10s",
        "history_hold_s": loop.motion_start_tick * DT,
        "motion_duration_s": (loop.final_tick - loop.motion_start_tick) * DT,
    }
    if frame.mode == "horizontal":
        path["translated_xy_sdk_m"] = loop.xy.tolist()
    return loop, path


def infer(robot, sensor, reader, cfg, reference, baseline, loop, stream):
    sensor.flush_csv()
    stream.flush()
    start = time.perf_counter()
    for tick in range(loop.final_tick + 1):
        due = start + tick * DT
        holding = tick < loop.motion_start_tick
        timing = {}
        stage = "wake"
        stamp = due

        def mark(name):
            nonlocal stamp
            now = time.perf_counter()
            timing[name + "_ms"] = (now - stamp) * 1000
            stamp = now

        try:
            time.sleep(max(0, due - time.perf_counter()))
            mark("wake_lateness")
            stage = "recording_check"
            sensor.flush_csv()
            stream.flush()
            mark(stage)
            if time.perf_counter() - due > 0.005:
                raise StateError("Policy scheduling deadline missed; no catch-up")
            stage = "observation"
            row = observe(robot, reader, cfg, "servo", baseline=baseline, check_speed=False)
            mark(stage)
            stage = "causal_ft"
            item = sensor.latest_before(due, net=False)
            if item is None or not 0 <= due - item[0] <= 0.02:
                raise StateError("No fresh causal FT at policy time")
            raw = finite_array(item[1], "causal FT")
            if raw.shape != (6,):
                raise ValueError("Expected six FT channels")
            ft = raw - baseline
            mark(stage)
            stage = "policy"
            filtered = loop.push(tick, ft, row["state"]["sdk_end_position_m"])
            target = finite_array(loop.target(tick), "policy target").tolist()
            mark(stage)
            stage = "pre_send_check"
            sensor.flush_csv()
            stream.flush()
            mark(stage)
            timing["before_send_ms"] = (time.perf_counter() - due) * 1000
            if timing["before_send_ms"] > 9.0:
                raise StateError("Inference deadline exceeded before send")
            stage = "send"
            if holding:
                robot.send_joint(reference["joint_position_rad"])
            else:
                robot.send(target, reference["sdk_end_orientation_xyzw"])
            mark(stage)
            stage = "event_enqueue"
            write_event(stream, {"event": "sample", "phase": "policy", "tick": tick,
                                 "control_phase": "history_hold" if holding else "motion",
                                 "motion_tick": None if holding else tick - loop.motion_start_tick,
                                 "due_perf_s": due, "causal_sensor_receive_perf_s": item[0],
                                 "raw_ft": raw.tolist(), "tared_ft": ft.tolist(),
                                 "filtered_ft": filtered.tolist(), "target_sdk_m": target,
                                 "delta_h_m": loop.last_delta, "timing_ms": dict(timing), **row})
            mark(stage)
            if time.perf_counter() - due > DT:
                raise StateError("Policy cycle exceeded 10 ms")
        except BaseException as exc:
            timing["elapsed_ms"] = (time.perf_counter() - due) * 1000
            timing["unfinished_stage_ms"] = (time.perf_counter() - stamp) * 1000
            exc.control_timing = {"tick": tick, "stage": stage, "timing_ms": timing,
                                  "control_phase": "history_hold" if holding else "motion"}
            raise


def execute(robot, sensor, policy, embedding, cfg, stream, keyboard=None, *, on_policy_complete=None):
    frame = WipingFrame.from_settings(cfg)
    keyboard = keyboard or Keyboard()
    reader = ForceReader(sensor, cfg)
    acquired = False
    phase = "startup"
    try:
        robot.read("idle")
        reader.read()
        sensor.flush_csv()
        stream.flush()
        robot.acquire()
        acquired = True
        robot.switch("gravity_comp")
        phase = "drag"

        def drag():
            sensor.flush_csv()
            stream.flush()
            return observe(robot, reader, cfg, "gravity_comp", check_speed=False)

        prompt = "Drag near the wiping area, tool clear of surface. Press h to capture and hold."
        if frame.mode == "vertical":
            axis, angle = frame.tool_rotation_axis_angle
            prompt = (f"Vertical wiping: wall toward SDK {frame.wall_direction.upper()}. "
                      f"Orient tool by {angle:+d} deg about SDK {axis} from the table pose, facing the wall. "
                      "Keep tool clear of wall; press h to capture and hold this pose.")
        keyboard.wait("h", drag, prompt)
        reference = copy.deepcopy(drag()["state"])
        robot.switch("servo")
        robot.send_joint(reference["joint_position_rad"])
        baseline = None

        def hold():
            sensor.flush_csv()
            stream.flush()
            row = observe(robot, reader, cfg, "servo", baseline=baseline, check_speed=False)
            robot.send_joint(reference["joint_position_rad"])
            return row

        loop, path = make_loop(policy, embedding, reference["sdk_end_position_m"], wiping_frame=frame)
        write_event(stream, {"event": "reference_captured", "runtime_reference_pose": reference, **path})
        phase = "wait_tare"
        keyboard.wait("z", hold, "Holding start. Remove external tool loads; press z to tare.")
        phase = "tare"
        baseline = np.asarray(collect_tare(hold, stream))
        phase = "wait_inference"
        keyboard.wait("s", hold, "Tare complete. Press s to hold for 2 seconds, then run the 10-second policy.")
        phase = "policy"
        infer(robot, sensor, reader, cfg, reference, baseline, loop, stream)
        reference = copy.deepcopy(observe(robot, reader, cfg, "servo", baseline=baseline,
                                          check_speed=False)["state"])
        hold()
        phase = "final_hold"
        write_event(stream, {"event": "policy_complete", "final_hold_pose": reference})
        if on_policy_complete is not None:
            on_policy_complete()
        keyboard.wait("g", hold, "Policy complete; holding. Support the arm and press g for gravity compensation and exit.")
        phase = "gravity_handoff"
        robot.switch("gravity_comp")
        robot.read("gravity_comp")
        write_event(stream, {"event": "session_complete", "controller": "gravity_comp",
                             "automatic_return": False})
    except BaseException as exc:
        if acquired:
            try:
                robot.abort()
            except BaseException as stop_error:
                print(f"Emergency stop not confirmed: {stop_error}; use onsite hardware stop", file=sys.stderr)
        fault = {"event": "fault", "phase": phase, "error": str(exc),
                 **getattr(exc, "control_timing", {})}
        if "timing_ms" in fault:
            print("Control timing: " + json.dumps(fault), file=sys.stderr)
        try:
            write_event(stream, fault)
        except Exception as log_error:
            print(f"Fault log unavailable: {log_error}; fault={json.dumps(fault)}", file=sys.stderr)
        raise
