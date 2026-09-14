"""Offline replay and explicitly commissioned AIRBOT 5.2.2 policy execution."""

import argparse
import json
import select
import signal
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.real_training.config import load_config, native_profile, resolve
from scripts.real_training.data import PROTOCOL, load_prepared
from scripts.robot_control.adapter import DeadlineStub, Robot
from scripts.robot_control.airbot_initial_pose import validate_stationary
from scripts.robot_control.client import StateError, open_client
from scripts.robot_control.safety import (
    check_bounds,
    interrupt_on_signal,
    measured_joint_speed_limit,
    positive,
    quaternion_angle,
    read_force,
    write_event,
)
from scripts.shared.airbot_calibration import (
    Calibration,
    Frames,
    checked_received,
    rigid,
)
from scripts.shared.common import file_digest, write_json
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_preprocessing import finite_array, make_windows
from scripts.shared.sampling import previous_samples


class PolicyLoop:
    """100 Hz causal FT; five 0.4 s histories predict the NEXT waypoint."""

    def __init__(self, policy, embedding, initial_position):
        self.policy, self.embedding = policy, embedding
        self.xy = finite_array(policy.predict_xy(embedding)[0], "XY prediction")
        self.initial = finite_array(initial_position, "initial position")
        if self.xy.shape != (25, 2) or self.initial.shape != (3,):
            raise ValueError("Invalid motion trajectory or initial position")
        self.filter = policy.make_ft_filter(100)
        self.history = deque(maxlen=5)
        self.segment_start = self.initial.copy()
        self.segment_end = np.r_[self.xy[0], self.initial[2]]
        self.next_tick = 0
        self.last_delta = None

    def push(self, tick, raw_ft, measured_position):
        if tick != self.next_tick or not 0 <= tick <= 1000:
            raise ValueError(
                "Policy ticks must be contiguous 0..1000; no catch-up or skipped histories"
            )
        measured = finite_array(measured_position, "measured position")
        if measured.shape != (3,):
            raise ValueError("Expected measured position [3]")
        filtered = self.filter.push(raw_ft)
        self.last_delta = None
        if tick and tick % 40 == 0:
            self.history.append(filtered)
            k = tick // 40 - 1
            if k < 24:
                self.segment_start = self.segment_end.copy()
                height = self.initial[2]
                if len(self.history) == 5:
                    self.last_delta = float(
                        self.policy.predict_delta_h(self.embedding, np.array(self.history)[None])[
                            0, 0
                        ]
                    )
                    height = measured[2] + self.last_delta
                self.segment_end = np.r_[self.xy[k + 1], height]
        self.next_tick += 1
        return filtered

    def target(self, tick):
        if not 0 <= tick <= 1000:
            raise ValueError("Target outside the 10-second policy horizon")
        alpha = 1.0 if tick == 1000 else (tick % 40) / 40
        return self.segment_start + alpha * (self.segment_end - self.segment_start)


def deployment_blockers(policy, cfg):
    blockers = []
    meta = policy.metadata["data"]["metadata"]
    if policy.source_kind != "real":
        blockers.append("Synthetic policies cannot command hardware")
    if policy.metadata["training_contract"]["profile"] != "paper_downstream":
        blockers.append(
            "Native-frame offline model has no verified sensor/TCP calibration; calibrate and retrain"
        )
    if "exploration_outside_encoder_normalization_range_not_clipped" in policy.warnings:
        blockers.append(
            "Training exploration is outside the encoder range; resolve physical input mapping before deployment"
        )
    if "source_encoder_quality_warning" in policy.warnings:
        blockers.append(
            "Source encoder quality diagnostics failed; resolve/revalidate before deployment"
        )
    for flag in (
        "commissioning_verified",
        "physical_estop_verified",
        "server_no_return_verified",
        "tool_and_swept_path_verified",
        "training_frames_verified",
    ):
        if cfg.get(flag) is not True:
            blockers.append(f"On-site evidence required: {flag}")
    if cfg.get("policy_sha256") != file_digest(resolve(cfg["policy"])):
        blockers.append("Bind the exact reviewed policy_sha256 in the deployment configuration")
    if cfg.get("robot_sn") != meta["robot_id"] or cfg.get("sensor_port") != meta["sensor_id"]:
        blockers.append("Robot/sensor identity must match calibrated training metadata")
    if cfg.get("calibration_id") != meta["calibration_id"]:
        blockers.append("Calibration identity differs from training data")
    return blockers


def validate_setup(policy, cfg):
    blockers = deployment_blockers(policy, cfg)
    if blockers:
        raise ValueError("Deployment blocked: " + "; ".join(blockers))
    if cfg.get("schema_version") != 1 or cfg.get("expected_eef_type") != "NULL":
        raise ValueError("This deployment adapter supports AIRBOT 5.2.2 with NULL end effector")
    if cfg.get("mode", "force-guarded") != "force-guarded":
        raise ValueError("Policy deployment requires force-guarded mode")
    prepared = resolve(cfg["prepared_data"])
    if file_digest(prepared) != policy.metadata["bindings"]["prepared_sha256"]:
        raise ValueError(
            "Deployment preprocessing evidence differs from the policy's prepared data"
        )
    with h5py.File(prepared, "r") as h5:
        if not np.all(h5["ft_hz"][:] == 100):
            raise ValueError("This 100 Hz adapter requires training FT filtered on a 100 Hz grid")
    meta = policy.metadata["data"]["metadata"]
    calibration = Calibration(cfg["calibration_record"])
    frames = calibration.bind_deployment(cfg, meta)
    if not np.allclose(
        frames.ft, rigid(meta["sensor_to_ft_frame"], "training FT transform"), atol=1e-8
    ):
        raise ValueError("Deployment FT transform differs from training calibration")
    low = finite_array(cfg["joint_min_rad"], "joint")
    high = finite_array(cfg["joint_max_rad"], "joint")
    if low.shape != (6,) or high.shape != (6,) or np.any(low >= high):
        raise ValueError("Invalid approved joint limits")
    for key in (
        "max_force_n",
        "max_initial_force_n",
        "max_torque_nm",
        "max_cartesian_speed_m_s",
        "max_delta_h_m",
        "max_tracking_error_m",
        "max_start_error_m",
        "orientation_error_limit_rad",
    ):
        positive(cfg.get(key), key)
    positive(cfg["joint_speed_limit_rad_s"], "joint_speed_limit_rad_s", 0.5)
    measured_joint_speed_limit(cfg)
    currents = finite_array(cfg["joint_current_limits"], "joint_current_limits")
    if currents.shape != (6,) or np.any(currents <= 0) or np.any(currents > 20):
        raise ValueError("Invalid approved joint current limits")
    if cfg["max_initial_force_n"] > cfg["max_force_n"]:
        raise ValueError("Initial force limit exceeds running force limit")
    start = finite_array(cfg["initial_tcp_position_m"], "initial TCP")
    quat = finite_array(cfg["sdk_end_orientation_xyzw"], "SDK orientation")
    if (
        start.shape != (3,)
        or quat.shape != (4,)
        or not np.isclose(np.linalg.norm(quat), 1, atol=1e-6)
    ):
        raise ValueError("Set a measured initial TCP and unit xyzw orientation")
    return frames


def load_exploration(path, policy):
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_kind"].item()) != "real":
            raise ValueError("Deployment exploration must be real")
        meta = policy.metadata["data"]["metadata"]
        for key in (
            "robot_id",
            "sensor_id",
            "calibration_id",
            "ft_frame",
            "ft_reference_point",
            "compensation",
            "calibration_sha256",
        ):
            if str(data[key].item()) != meta[key]:
                raise ValueError(f"Deployment exploration {key} mismatch")
        ft = finite_array(data["ft"], "exploration FT")
        stamps = finite_array(data["time_s"], "exploration times")
        if (
            ft.shape != (400, 6)
            or stamps.shape != (400,)
            or not np.allclose(stamps, np.arange(1, 401) / 100)
        ):
            raise ValueError("Expected calibrated exploration [400,6] at 0.01..4.00 s")
        if json.loads(str(data["exploration_protocol"].item())) != PROTOCOL:
            raise ValueError("Deployment exploration protocol mismatch")
        _, held = checked_received(data["received_time_s"], data["received_sensor_ft"], 4.0)
        expected = Frames(meta["frame_calibration"]).wrenches(held)[1:]
        if not np.allclose(ft, expected, atol=1e-10, rtol=1e-10):
            raise ValueError(
                "Exploration values do not match calibrated original received observations"
            )
    normalized = policy.encoder.preprocessor.transform(ft[None])
    if np.any((normalized < 0) | (normalized > 0.9)):
        raise ValueError("New exploration is outside encoder normalization range")
    return policy.encode_exploration(ft[None])


def check_state(robot, sensor, cfg, frames, target=None, *, controller="servo", initial=False):
    robot.owned()
    force = read_force(sensor, cfg, initial=initial)
    state = robot.read(controller)
    check_bounds(state, cfg)
    if max(abs(v) for v in state["joint_velocity_rad_s"]) > measured_joint_speed_limit(cfg):
        raise StateError("Measured joint speed exceeded")
    if (
        quaternion_angle(state["sdk_end_orientation_xyzw"], cfg["sdk_end_orientation_xyzw"])
        > cfg["orientation_error_limit_rad"]
    ):
        raise StateError("Tool orientation exceeded approved tolerance")
    if (
        target is not None
        and np.linalg.norm(np.array(state["sdk_end_position_m"]) - target)
        > cfg["max_tracking_error_m"]
    ):
        raise StateError("Position tracking limit exceeded")
    return state, force


def guard_segment(loop, cfg):
    if np.linalg.norm(loop.segment_end - loop.segment_start) / 0.4 > cfg["max_cartesian_speed_m_s"]:
        raise StateError("Predicted segment exceeds approved Cartesian speed; no clipping")
    if loop.last_delta is not None and abs(loop.last_delta) > cfg["max_delta_h_m"]:
        raise StateError("Predicted vertical increment exceeds approved limit; no clipping")


def wait_for_policy_idle(check):
    print("Policy ended; servo remains active. Arrange safe support, then type IDLE.", flush=True)
    while True:
        check()
        if select.select([sys.stdin], [], [], 0.01)[0]:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("Terminal closed")
            if line.strip() == "IDLE":
                return


def execute(robot, sensor, policy, embedding, cfg, frames, stream, *, finish=wait_for_policy_idle):
    acquired = attempted = False
    try:
        robot.acquire()
        acquired = True
        stationary = []
        for _ in range(11):
            state, _ = check_state(robot, sensor, cfg, frames, controller="idle", initial=True)
            if (
                np.linalg.norm(frames.to_tcp(state) - cfg["initial_tcp_position_m"])
                > cfg["max_start_error_m"]
            ):
                raise StateError("Manually position at approved start; no automatic homing")
            stationary.append(state)
            time.sleep(0.05)
        validate_stationary(stationary)
        loop = PolicyLoop(policy, embedding, frames.to_tcp(state))
        guard_segment(loop, cfg)
        attempted = True
        robot.enter()
        start = time.perf_counter()
        target = np.array(state["sdk_end_position_m"])
        for tick in range(1001):
            due = start + tick / 100
            time.sleep(max(0, due - time.perf_counter()))
            if time.perf_counter() - due > 0.005:
                raise StateError("Missed 100 Hz deadline; no catch-up")
            state, force = check_state(robot, sensor, cfg, frames, target)
            item = sensor.latest_before(due, net=False)
            if item is None or not 0 <= due - item[0] <= 0.020:
                raise StateError("No fresh causal FT at the policy deadline")
            ft = frames.wrench(item[1])
            loop.push(tick, ft, frames.to_tcp(state))
            guard_segment(loop, cfg)
            target = frames.to_end(loop.target(tick), cfg["sdk_end_orientation_xyzw"])
            read_force(sensor, cfg)
            if time.perf_counter() - due > 0.009:
                raise StateError("Observation/inference deadline exceeded before motion command")
            robot.send(target.tolist(), cfg["sdk_end_orientation_xyzw"])
            write_event(
                stream,
                {
                    "event": "sample",
                    "tick": tick,
                    "due_perf_s": due,
                    "sensor_receive_perf_s": item[0],
                    "ft": ft.tolist(),
                    "target_sdk_m": target.tolist(),
                    "state": state,
                    "force_guard": force,
                    "delta_h_m": loop.last_delta,
                },
            )
        write_event(stream, {"event": "policy_complete", "hardware_performance_verified": False})
        print(
            "Policy horizon ended. No automatic retract. Arrange support before IDLE.", flush=True
        )
        finish(lambda: check_state(robot, sensor, cfg, frames, target))
        robot.idle()
        write_event(stream, {"event": "session_complete", "idle_confirmed": True})
    except BaseException as exc:
        stop_error = None
        if acquired and attempted:
            try:
                robot.abort()
            except BaseException as failure:
                stop_error = str(failure)
                print(f"STOP FAILED: {failure}; use physical emergency stop", file=sys.stderr)
        write_event(stream, {"event": "aborted", "error": str(exc), "stop_error": stop_error})
        raise


def replay(cfg, output):
    arrays, info = load_prepared(cfg)
    if (
        not native_profile(cfg)
        and info["metadata"].get("conversion") != "measured_frames_then_causal_100hz_hold_v1"
    ):
        raise ValueError("Calibrated replay requires the audited AIRBOT 100 Hz conversion")
    if not np.all(arrays["ft_hz"] == 100):
        raise ValueError("Replay requires a 100 Hz training filter grid")
    policy_path = resolve(cfg["output_dir"]) / "policy.pt"
    policy = OfflinePolicy(policy_path)
    if policy.metadata["bindings"]["prepared_sha256"] != info["sha256"]:
        raise ValueError("Policy and replay dataset differ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    targets, deltas, errors = [], [], []
    with h5py.File(resolve(cfg["raw_data"]), "r") as h5:
        for index, name in enumerate(info["demo_ids"]):
            group = h5[f"demonstrations/{name}"]
            start = group.attrs["start_time"]
            grid = np.arange(1001) / 100
            ft = previous_samples(group["ft_time"][:] - start, group["ft"][:], grid)
            key = "sdk_end_position" if native_profile(cfg) else "tcp_position"
            xyz = previous_samples(group["pose_time"][:] - start, group[key][:], grid)
            loop = PolicyLoop(policy, arrays["sponge"][index : index + 1], xyz[0])
            trajectory, prediction = [], []
            for tick in range(1001):
                loop.push(tick, ft[tick], xyz[tick])
                trajectory.append(loop.target(tick))
                if loop.last_delta is not None:
                    prediction.append(loop.last_delta)
            windows, _ = make_windows(
                arrays["ft"][index : index + 1], arrays["height"][index : index + 1]
            )
            expected = np.array(
                [
                    policy.predict_delta_h(arrays["sponge"][index : index + 1], w[None])[0, 0]
                    for w in windows[0]
                ]
            )
            np.testing.assert_allclose(prediction, expected, atol=1e-10, rtol=0)
            errors.append(float(np.max(np.abs(np.array(prediction) - expected))))
            targets.append(trajectory)
            deltas.append(prediction)
    np.savez_compressed(
        output / "replay.npz",
        target_position=targets,
        delta_h=deltas,
        source_kind="real",
        time_s=np.arange(1001) / 100,
    )
    result = {
        "scope": "recorded_FT_replay_not_closed_loop_or_hardware_test",
        "source_kind": "real",
        "hardware_ready": False,
        "policy_sha256": file_digest(policy_path),
        "position_frame": info["metadata"]["position_frame"],
        "profile": cfg["profile"],
        "calibration_sha256": info["metadata"].get("calibration_sha256"),
        "demos": 8,
        "ticks_per_demo": 1001,
        "feedback_predictions_per_demo": 20,
        "max_stream_vs_batch_error_m": max(errors),
        "warnings": policy.warnings,
    }
    write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "replay", "shadow", "run"), nargs="?", default="preflight"
    )
    parser.add_argument("--mode", choices=("calibrated", "fixed-setup"), default="calibrated")
    parser.add_argument("--config")
    parser.add_argument(
        "--training-config", default="configs/real_training/real_training_airbot_native.yaml"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args(argv)
    if args.mode == "fixed-setup":
        from scripts.real_deploy.fixed_setup import main as fixed_main

        return fixed_main(args)
    if args.action == "shadow":
        parser.error("shadow requires --mode fixed-setup")
    args.config = args.config or "configs/real_deploy/airbot_deployment.json"
    client = sensor = None
    previous_signal = None
    torch.set_num_threads(1)
    try:
        if args.action == "replay":
            if args.output is None:
                parser.error("replay requires a fresh --output directory")
            result = replay(load_config(args.training_config), args.output)
            print(json.dumps(result, indent=2))
            return 0
        cfg = json.loads(Path(args.config).read_text())
        policy_hash = file_digest(resolve(cfg["policy"]))
        policy = OfflinePolicy(resolve(cfg["policy"]))
        if file_digest(resolve(cfg["policy"])) != policy_hash:
            raise ValueError("Policy changed while being loaded")
        blockers = deployment_blockers(policy, cfg)
        if args.action == "preflight" and blockers:
            print(
                json.dumps(
                    {"hardware_connected": False, "executable": False, "blockers": blockers},
                    indent=2,
                )
            )
            return 2
        frames = validate_setup(policy, cfg)
        exploration_hash = file_digest(resolve(cfg["exploration"]))
        embedding = load_exploration(resolve(cfg["exploration"]), policy)
        if file_digest(resolve(cfg["exploration"])) != exploration_hash:
            raise ValueError("Task exploration changed while being loaded")
        loop = PolicyLoop(policy, embedding, cfg["initial_tcp_position_m"])
        guard_segment(loop, cfg)
        if args.action == "preflight":
            print(
                json.dumps(
                    {
                        "hardware_connected": False,
                        "offline_checks_passed": True,
                        "live_checks_still_required": True,
                    }
                )
            )
            return 0
        if not args.execute or not sys.stdin.isatty() or args.output is None:
            parser.error("run requires --execute, an attended terminal and a fresh --output JSONL")
        if args.output.exists():
            raise FileExistsError(args.output)
        print(
            "Physical emergency stop and safe support required. No automatic homing, retry or retract."
        )
        if input("Type DEPLOY to run the reviewed policy: ").strip() != "DEPLOY":
            return 0
        if (
            file_digest(resolve(cfg["policy"])) != policy_hash
            or file_digest(resolve(cfg["exploration"])) != exploration_hash
        ):
            raise ValueError("Reviewed policy or exploration changed before connection")
        Calibration(cfg["calibration_record"]).bind_deployment(
            cfg, policy.metadata["data"]["metadata"]
        )
        from scripts.force_sensor.kwr75_reader import Kwr75Reader

        previous_signal = signal.signal(signal.SIGTERM, interrupt_on_signal)
        client = open_client(args.host, args.port)
        client._stub = DeadlineStub(client._stub)
        sensor = Kwr75Reader(port=cfg["sensor_port"])
        sensor.start()
        deadline = time.perf_counter() + 2
        while sensor.latest(net=False) is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        robot = Robot(client, cfg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            write_event(
                stream,
                {
                    "event": "session_start",
                    "config": cfg,
                    "policy_sha256": file_digest(resolve(cfg["policy"])),
                    "exploration_sha256": file_digest(resolve(cfg["exploration"])),
                },
            )
            execute(robot, sensor, policy, embedding, cfg, frames, stream)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; verify physical support and emergency stop", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
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


if __name__ == "__main__":
    raise SystemExit(main())
