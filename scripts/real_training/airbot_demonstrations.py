"""Attended AIRBOT demonstrations: manual by default; explicit programmed mode available."""

import argparse
import hashlib
import json
import math
import os
import select
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from scripts.robot_control.adapter import DeadlineStub
from scripts.robot_control.airbot_initial_pose import validate_stationary
from scripts.robot_control.client import (
    StateError,
    health,
    open_client,
    snapshot,
    vector,
    wait_controller,
)
from scripts.robot_control.safety import positive, quaternion_angle, read_force
from scripts.shared.paths import ROOT


def record_only(cfg):
    # Retain the historical option name; it now selects only the force policy.
    mode = cfg.get("workspace_force_policy", "stop")
    if mode not in ("stop", "record-only"):
        raise ValueError("workspace_force_policy must be stop or record-only")
    return mode == "record-only"


def speed_record_only(cfg):
    policy = cfg.get("joint_speed_policy", "stop")
    if policy not in ("stop", "record-only"):
        raise ValueError("joint_speed_policy must be stop or record-only")
    return policy == "record-only"


def read_demo_force(sensor, cfg):
    if not record_only(cfg):
        return read_force(sensor, cfg)
    item = sensor.latest(net=False)
    if item is None:
        raise StateError("No force sensor data")
    stamp, raw = item
    age = time.perf_counter() - stamp
    if not math.isfinite(stamp) or not 0 <= age <= 0.020:
        raise StateError(f"Force sensor stale/invalid: age={age:.6f}s")
    raw = vector(raw, 6, "raw sensor wrench")
    corrected = vector([x - b for x, b in zip(raw, cfg["sensor_bias_si"])], 6, "corrected wrench")
    return {
        "sensor_receive_perf_s": stamp,
        "sensor_age_s": age,
        "raw_sensor_wrench_si": raw,
        "bias_corrected_sensor_wrench_si": corrected,
    }


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_config(cfg):
    for key, expected in {
        "schema_version": 1,
        "demonstrations": 8,
        "duration_s": 10,
        "sample_hz": 100,
        "training_hz": 2.5,
        "sponge_id": "normal",
    }.items():
        if cfg.get(key) != expected or isinstance(cfg.get(key), bool):
            raise ValueError(f"{key} must be {expected!r}")
    unbounded = record_only(cfg)
    if not unbounded and cfg.get("setup_confirmed") is not True:
        raise ValueError(
            "Complete measured setup fields and set setup_confirmed=true after onsite approval"
        )
    for key in (
        "setup_note",
        "surface_id",
        "exploration_id",
        "robot_sn",
        "sensor_port",
        "bias_note",
    ):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Missing {key}")
    low = vector(cfg["joint_min_rad"] or [], 6, "joint")
    high = vector(cfg["joint_max_rad"] or [], 6, "joint")
    if any(a >= b for a, b in zip(low, high)):
        raise ValueError("Invalid joint bounds")
    vector(cfg["sensor_bias_si"], 6, "sensor bias")
    if not unbounded:
        positive(cfg["max_force_n"], "max_force_n")
        positive(cfg["max_torque_nm"], "max_torque_nm")
    if not speed_record_only(cfg):
        positive(cfg["joint_speed_limit_rad_s"], "joint_speed_limit_rad_s", 0.5)


def observe(client, sensor, cfg):
    if not client.has_control():
        raise StateError("Control lost; no automatic reacquisition")
    state = snapshot(client, "gravity_comp")
    if any(
        not lo <= x <= hi
        for x, lo, hi in zip(state["joint_position_rad"], cfg["joint_min_rad"], cfg["joint_max_rad"])
    ):
        raise StateError("Outside approved bounds: joint_position_rad")
    if (
        not speed_record_only(cfg)
        and max(map(abs, state["joint_velocity_rad_s"])) > cfg["joint_speed_limit_rad_s"]
    ):
        raise StateError("Manual joint speed exceeded")
    return {
        "pose": state,
        "ft": read_demo_force(sensor, cfg),
        "observed_perf_s": time.perf_counter(),
    }


def command(client, sensor, cfg):
    while True:
        observe(client, sensor, cfg)
        sensor.flush_csv()
        if select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("Terminal closed")
            return line.strip().lower()


def check_reference(state, reference):
    delta = [p - r for p, r in zip(state["sdk_end_position_m"], reference["sdk_end_position_m"])]
    angle = quaternion_angle(
        state["sdk_end_orientation_xyzw"], reference["sdk_end_orientation_xyzw"]
    )
    return {
        "position_policy": "record-only",
        "position_delta_sdk_m": delta,
        "position_distance_m": math.dist(
            state["sdk_end_position_m"], reference["sdk_end_position_m"]
        ),
        "orientation_policy": "record-only",
        "orientation_angle_rad": angle,
    }


def stationary_start(client, sensor, cfg):
    samples = []
    for index in range(11):
        if index:
            time.sleep(0.05)
        samples.append(observe(client, sensor, cfg)["pose"])
        sensor.flush_csv()
    validate_stationary(samples)
    return samples[-1]


def quality(rows, start):
    pose = [r["pose"]["host_monotonic_s"] for r in rows]
    # Robot and sensor retain their own host-clock readings; do not fabricate timestamps.
    ft = [r["ft"]["sensor_receive_perf_s"] for r in rows]
    gaps = [b - a for a, b in zip(pose, pose[1:])]
    unique_ft = sorted(set(ft))
    ft_gaps = [b - a for a, b in zip(unique_ft, unique_ft[1:])]
    errors = []
    if (
        len(rows) < 2
        or rows[0]["observed_perf_s"] > start
        or rows[-1]["observed_perf_s"] < start + 10
    ):
        errors.append("segment coverage")
    if not gaps or min(gaps) <= 0 or max(gaps) > 0.020:
        errors.append("pose gap >20 ms or nonmonotonic")
    if not ft_gaps or max(ft_gaps) > 0.020 or any(b < a for a, b in zip(ft, ft[1:])):
        errors.append("FT receive gap >20 ms or clock reversal")
    if not ft or ft[0] > start or ft[-1] < start + 10:
        errors.append("FT does not cover complete segment")
    median = sorted(gaps)[len(gaps) // 2] if gaps else math.inf
    if not 0.009 <= median <= 0.011:
        errors.append("pose median period not 100 Hz +/-10%")
    if any(r["pose"]["read_duration_s"] > 0.020 for r in rows):
        errors.append("pose RPC read >20 ms")
    return {
        "passed": not errors,
        "errors": errors,
        "samples": len(rows),
        "max_pose_gap_s": max(gaps, default=0),
        "max_ft_receive_gap_s": max(ft_gaps, default=0),
        "median_pose_period_s": median if math.isfinite(median) else None,
    }


def record_episode(client, sensor, cfg, path):
    rows = []
    with path.open("x", encoding="utf-8") as stream:
        first = observe(client, sensor, cfg)
        start = first["observed_perf_s"]
        stream.write(json.dumps({"event": "start", "start_perf_s": start}) + "\n")
        row = first
        deadline = start
        while True:
            rows.append(row)
            stream.write(json.dumps({"event": "sample", **row}, allow_nan=False) + "\n")
            sensor.flush_csv()
            # Keep a short post-roll so the last received FT covers the 10s endpoint.
            if row["observed_perf_s"] >= start + cfg["duration_s"] + 0.020:
                break
            deadline += 1 / cfg["sample_hz"]
            time.sleep(max(0, deadline - time.perf_counter()))
            row = observe(client, sensor, cfg)
        report = quality(rows, start)
        stream.write(json.dumps({"event": "finished", "quality": report}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def accepted_records(folder):
    records = {}
    for path in folder.glob("demo_*.json"):
        entry = json.loads(path.read_text())
        if (
            entry.get("accepted") is not True
            or not entry.get("quality", {}).get("passed")
            or entry.get("training_ready") is not False
        ):
            raise ValueError(f"Invalid accepted record: {path}")
        raw = folder / entry["raw_file"]
        if raw.parent != folder or hashlib.sha256(raw.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Raw log integrity failure: {path}")
        records[path.name] = entry
    if (
        set(records) != {f"demo_{i:02d}.json" for i in range(1, len(records) + 1)}
        or len(records) > 8
    ):
        raise ValueError("Accepted demonstrations must be consecutive demo_01 through demo_08")
    return records


def session(client, sensor, idle, cfg, folder, reader=command, recorder=record_episode):
    count = len(accepted_records(folder))
    acquired = False
    try:
        health(client, "idle")
        if not client.acquire_control():
            raise StateError("Control acquisition refused")
        acquired = True
        health(client, "idle")
        if not client.enter_gravity_compensation_mode():
            raise StateError("Gravity compensation request failed")
        wait_controller(client, "gravity_comp")
        print(
            "Gravity compensation active. Support arm throughout; idle is NOT position hold.",
            flush=True,
        )
        while count < 8:
            print(
                f"{count}/8 accepted. Choose start pose, stop moving; s=record 10s, q=exit.",
                flush=True,
            )
            cmd = reader(client, sensor, cfg)
            if cmd == "q":
                return count
            if cmd != "s":
                continue
            try:
                initial = stationary_start(client, sensor, cfg)
                observe(client, sensor, cfg)
                reference_path = folder / "reference.json"
                if reference_path.exists():
                    reference = json.loads(reference_path.read_text())
                else:
                    write_json(reference_path, initial)
                    reference = initial
                deviation = check_reference(initial, reference)
            except ValueError as exc:
                print(f"Not started: {exc}", flush=True)
                continue
            raw = folder / f"attempt_{uuid.uuid4().hex}.jsonl"
            write_json(
                raw.with_suffix(".start.json"),
                {
                    "initial_pose": initial,
                    "reference_pose": reference,
                    "reference_start_deviation": deviation,
                },
            )
            print(
                f"Start position offset: {deviation['position_distance_m'] * 1000:.3f} mm (record-only).",
                flush=True,
            )
            print(
                f"Start orientation offset: {deviation['orientation_angle_rad']:.6f} rad (record-only).",
                flush=True,
            )
            print("RECORDING: wipe now, 10 seconds.", flush=True)
            report = recorder(client, sensor, cfg, raw)
            print(f"STOP wiping; keep supported. Quality: {report}", flush=True)
            if not report["passed"]:
                print("Rejected by timing checks. Raw attempt retained.")
                continue
            print(
                "a=accept only if wiping was error-free; q=exit; other=reject and retry.",
                flush=True,
            )
            cmd = reader(client, sensor, cfg)
            if cmd == "q":
                return count
            if cmd != "a":
                continue
            count += 1
            write_json(
                folder / f"demo_{count:02d}.json",
                {
                    "accepted": True,
                    "training_ready": False,
                    "source_kind": "real",
                    "quality": report,
                    "raw_file": raw.name,
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "initial_pose": initial,
                    "reference_start_deviation": deviation,
                },
            )
        return count
    finally:
        if acquired:
            print("Support arm NOW. Requesting idle, not hold or home.", flush=True)
            try:
                if not client.has_control() or not client.switch_controller(idle):
                    raise StateError("Idle request failed or control lost")
                wait_controller(client, "idle")
                print("Idle confirmed.", flush=True)
            except BaseException:
                print(
                    "WARNING: idle NOT confirmed; use onsite hardware safety procedure.",
                    file=sys.stderr,
                )
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "check", "run", "status"))
    parser.add_argument("--mode", choices=("manual", "programmed"), default="manual")
    parser.add_argument(
        "--config", type=Path
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args(argv)
    if args.mode == "programmed":
        from scripts.real_training.airbot_programmed_demonstrations import main

        return main(args)
    if args.action == "check":
        parser.error("check is available only with --mode programmed")
    args.config = args.config or ROOT / "configs/real_training/airbot_demonstrations.json"
    args.output = args.output or ROOT / "runs/real_training/session_001"
    client = sensor = None
    old_signal = None
    lock = None
    try:
        cfg = json.loads(args.config.read_text())
        if args.action == "status":
            if not (args.output / "session.json").is_file():
                raise ValueError("No recorded session at output path")
            records = accepted_records(args.output)
            print(
                json.dumps(
                    {
                        "accepted": len(records),
                        "required": 8,
                        "training_ready": False,
                        "hardware_connected": False,
                    },
                    indent=2,
                )
            )
            return 0 if len(records) == 8 else 1
        if args.action == "preview":
            try:
                validate_config(cfg)
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            print(
                json.dumps(
                    {
                        "hardware_connected": False,
                        "config": cfg,
                        "setup_error": error,
                        "training_ready": False,
                    },
                    indent=2,
                )
            )
            return 0 if error is None else 2
        if not args.execute or not sys.stdin.isatty():
            parser.error("run requires --execute and an attended terminal")
        validate_config(cfg)
        import fcntl

        from arm_sdk import Controller

        from scripts.force_sensor.kwr75_reader import Kwr75Reader

        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / ".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output / "session.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["config"] != cfg:
                raise ValueError("Config changed; use a new session directory")
        else:
            write_json(
                manifest,
                {
                    "config": cfg,
                    "training_ready": False,
                    "position_frame": "SDK configured end frame; sponge TCP uncalibrated",
                    "ft_frame": "sensor origin and axes; gravity retained",
                    "clock": "host monotonic (pose) and perf_counter (FT); no hardware timestamps",
                },
            )
        if len(accepted_records(args.output)) == 8:
            print("8/8 already accepted; no hardware connection made.")
            return 0
        print(
            "Verify --no-return server, supported load, emergency stop, fixed Normal sponge/sloped surface."
        )
        print(
            "Start position and orientation offsets are record-only; stationary-start checks remain active."
        )
        print("No project XYZ workspace boundary is enforced; verify the entire swept path on site.")
        if record_only(cfg):
            print(
                "WARNING: force/torque limits DISABLED; measurements recorded only."
            )
            print(
                "DRAG confirms onsite readiness for this session; no prior setup approval is assumed."
            )
        else:
            print(
                "Force limits only trigger idle; NOT safety-rated stop, load support or contact-force control."
            )
        if speed_record_only(cfg):
            print(
                "WARNING: joint speed stop DISABLED; actual velocities recorded without clipping."
            )
            print("Stationary-start checks remain active before each recording.")
        if input("Type DRAG to confirm onsite readiness: ").strip() != "DRAG":
            return 0

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        old_signal = signal.signal(signal.SIGTERM, interrupted)
        client = open_client(args.host, args.port)
        client._stub = DeadlineStub(client._stub)
        firmware = client.get_firmware_info()
        if (
            firmware is None
            or firmware.arm_sn != cfg["robot_sn"]
            or firmware.eef_type != cfg["expected_eef_type"]
        ):
            raise StateError("Robot serial number / end-effector mismatch")
        write_json(args.output / f"connection_{uuid.uuid4().hex}.json", asdict(firmware))
        sensor = Kwr75Reader(cfg["sensor_port"])
        sensor.start()
        sensor.start_csv(args.output / f"sensor_{uuid.uuid4().hex}.csv")
        time.sleep(0.1)
        read_demo_force(sensor, cfg)
        count = session(client, sensor, Controller.idle, cfg, args.output)
        print(f"Accepted {count}/8; raw logs only, training_ready=false.")
        return 0 if count == 8 else 1
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; support arm and check hardware state.", file=sys.stderr)
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
                if old_signal is not None:
                    signal.signal(signal.SIGTERM, old_signal)
                if lock is not None:
                    lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
