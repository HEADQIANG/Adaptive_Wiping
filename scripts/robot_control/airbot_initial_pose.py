"""AIRBOT arm-sdk 5.2.2: manual gravity-compensation teaching, never replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import tempfile
import termios
import time
import tty
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from scripts.shared.paths import ROOT

from scripts.robot_control.client import (
    StateError,
    health,
    open_client,
    snapshot,
    vector,
    wait_controller,
)

SDK_VERSION = "5.2.2"
SAMPLE_COUNT = 11
SAMPLE_INTERVAL_S = 0.05
MAX_JOINT_SPEED = 0.05
MAX_JOINT_SPAN = 0.01
MAX_POSITION_SPAN_M = 0.002
MAX_ORIENTATION_SPAN_RAD = 0.02
EXPLORATION_CONFIG = ROOT / "configs/real_training/airbot_exploration.json"


def update_exploration_config(pose_path, config_path=EXPLORATION_CONFIG):
    """Publish an existing taught pose without partially overwriting the config."""
    pose_path = Path(pose_path).resolve(strict=True)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        stored_path = str(pose_path.relative_to(ROOT))
    except ValueError:
        stored_path = str(pose_path)
    config["initial_pose_file"] = stored_path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=config_path.parent,
                                         prefix=".initial-pose-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(config, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(config_path.stat().st_mode & 0o777)
        os.replace(temporary, config_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Updated exploration config: {config_path}\ninitial_pose_file: {stored_path}", flush=True)


class NotStationary(ValueError):
    """An initial pose must be recorded while the arm is stationary."""


def validate_stationary(samples, *, max_joint_speed=MAX_JOINT_SPEED):
    if not math.isfinite(max_joint_speed) or max_joint_speed <= 0:
        raise ValueError("Stationary speed limit must be finite and positive")
    if len(samples) < SAMPLE_COUNT:
        raise NotStationary("Not enough stationary samples")
    peak_speeds = [max(abs(s["joint_velocity_rad_s"][axis]) for s in samples) for axis in range(6)]
    exceeded = [
        f"J{axis + 1} peak={speed:.9g} rad/s"
        for axis, speed in enumerate(peak_speeds)
        if speed > max_joint_speed
    ]
    if exceeded:
        raise NotStationary(
            f"Joint speed exceeds limit={max_joint_speed:g} rad/s: "
            + "; ".join(exceeded)
            + "; support and stop the arm"
        )
    for axis in range(6):
        values = [s["joint_position_rad"][axis] for s in samples]
        if max(values) - min(values) > MAX_JOINT_SPAN:
            raise NotStationary("Joint position changed by more than 0.01 rad")
    # Compare all pairs so a round trip during sampling cannot pass as stationary.
    for i, first in enumerate(samples):
        for second in samples[i + 1 :]:
            if (
                math.dist(first["sdk_end_position_m"], second["sdk_end_position_m"])
                > MAX_POSITION_SPAN_M
            ):
                raise NotStationary("End position changed by more than 2 mm")
            qa = first["sdk_end_orientation_xyzw"]
            qb = second["sdk_end_orientation_xyzw"]
            dot = abs(sum(a * b for a, b in zip(qa, qb)))
            norm = math.sqrt(sum(a * a for a in qa) * sum(b * b for b in qb))
            angle = 2 * math.acos(min(1.0, dot / norm))
            if angle > MAX_ORIENTATION_SPAN_RAD:
                raise NotStationary("End orientation changed by more than 0.02 rad")


def capture(client):
    samples = []
    for index in range(SAMPLE_COUNT):
        if index:
            time.sleep(SAMPLE_INTERVAL_S)
        if not client.has_control():
            raise StateError("Control lease lost; automatic reacquisition disabled")
        samples.append(snapshot(client, "gravity_comp"))
    validate_stationary(samples)
    return samples


def save_record(path, samples, metadata):
    validate_stationary(samples)
    record = {
        "schema_version": 1,
        "kind": "airbot_manual_initial_pose",
        "sdk_version": SDK_VERSION,
        "metadata": metadata,
        "coordinate_semantics": {
            "reference_frame": "SDK configured reference frame; verify server configuration",
            "end_frame": "SDK configured end frame; not a calibrated sponge TCP",
            "sponge_tcp_calibrated": False,
            "hardware_timestamp_available": False,
            "joint_and_pose_reads_atomic": False,
        },
        "initial_pose": samples[-1],
        "stationarity_check": {
            "sample_count": len(samples),
            "max_joint_speed_rad_s": MAX_JOINT_SPEED,
            "max_joint_span_rad": MAX_JOINT_SPAN,
            "max_position_span_m": MAX_POSITION_SPAN_M,
            "max_orientation_span_rad": MAX_ORIENTATION_SPAN_RAD,
        },
        "samples": samples,
    }
    payload = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Exclusive creation preserves every previously taught pose, including on retry.
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def capture_idle(client):
    """Observe an already idle arm without acquiring control or changing modes."""
    samples = []
    for index in range(SAMPLE_COUNT):
        if index:
            time.sleep(SAMPLE_INTERVAL_S)
        samples.append(snapshot(client, "idle"))
    validate_stationary(samples)
    return samples


def read_command(client):
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        # Keep signal handling (Ctrl+C), but read keys without Enter or echo.
        tty.setcbreak(fd, termios.TCSANOW)
        while True:
            if not client.has_control():
                raise StateError("Control lease lost; automatic reacquisition disabled")
            health(client, "gravity_comp")
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                key = os.read(fd, 1)
                if not key or key == b"\x04":
                    raise EOFError("Terminal closed")
                return key.decode("ascii", errors="ignore").lower()
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)


def teach(client, idle_controller, path, metadata, command_reader=read_command):
    acquired = False
    try:
        health(client, "idle")
        if not client.acquire_control():
            raise StateError("Control acquisition refused; no mode change requested")
        acquired = True
        health(client, "idle")
        if not client.enter_gravity_compensation_mode():
            raise StateError("Gravity compensation request failed")
        wait_controller(client, "gravity_comp")
        print("Gravity compensation active. Keep the arm supported throughout.", flush=True)
        print("Drag to the start pose; stop moving, then press S to save and exit (no Enter).")
        print("Support the arm BEFORE pressing S: saving requests idle, NOT position hold.")
        print("Ctrl+C cancels without saving and requests idle; keep the arm supported.")
        while True:
            command = command_reader(client).lower()
            if command == "s":
                try:
                    samples = capture(client)
                except NotStationary as exc:
                    print(f"Not saved: {exc}. Stop moving, support the arm and press S again.")
                    continue
                record = save_record(path, samples, metadata)
                print(f"Saved initial pose: {path}", flush=True)
                print(json.dumps(record["initial_pose"], indent=2))
                return True
            else:
                print("Press S to save and exit (no Enter), or Ctrl+C to cancel.")
    finally:
        if acquired:
            print(
                "Exiting: keep arm supported. Requesting idle; this is NOT a hold command.",
                flush=True,
            )
            try:
                if not client.has_control() or not client.switch_controller(idle_controller):
                    raise StateError("Idle request failed or control lease lost")
                wait_controller(client, "idle")
                print("Idle confirmed. No home, replay, or position target was sent.")
            except BaseException:
                print(
                    "WARNING: idle NOT confirmed. Use the hardware safety procedure; do not let go.",
                    file=sys.stderr,
                    flush=True,
                )
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "teach", "capture-idle"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--execute", action="store_true", help="Allow interactive gravity-compensation teaching"
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON file; existing files are never overwritten"
    )
    parser.add_argument("--label", default="wiping_start")
    parser.add_argument("--tool-note", default="unspecified; sponge TCP not calibrated")
    args = parser.parse_args(argv)
    if args.action == "capture-idle":
        if args.execute:
            parser.error("capture-idle is read-only; omit --execute")
        if args.output is None or args.output.exists():
            parser.error("capture-idle requires --output pointing to a new JSON file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "teach":
        if not args.execute:
            parser.error("teach requires --execute; no hardware connection made")
        if not sys.stdin.isatty():
            parser.error("teach requires an attended interactive terminal")
        if args.output is None:
            parser.error("teach requires --output pointing to a new JSON file")
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print("Confirm firmware/tool compatibility, emergency stop, and a clear workspace.")
        print("Server must use --no-return. Support arm BEFORE switching and THROUGHOUT use.")
        print(
            "Loss of power/control or idle may let the arm fall; software is not an emergency stop."
        )
        try:
            if (
                input("Type DRAG to allow gravity compensation (anything else cancels): ").strip()
                != "DRAG"
            ):
                print("Cancelled without connecting.")
                return 0
        except (EOFError, KeyboardInterrupt):
            return 130
    client = None
    try:
        client = open_client(args.host, args.port)
        firmware = client.get_firmware_info()
        if args.action == "inspect":
            print(
                json.dumps(
                    {
                        "sdk_version": SDK_VERSION,
                        "state": snapshot(client),
                        "firmware_cached": asdict(firmware) if firmware else None,
                    },
                    indent=2,
                )
            )
            return 0
        metadata = {
            "label": args.label,
            "tool_note": args.tool_note,
            "host": args.host,
            "port": args.port,
            "firmware_cached": asdict(firmware) if firmware else None,
        }
        if args.action == "capture-idle":
            if firmware is None:
                raise StateError("Firmware identity unavailable; no record saved")
            metadata["capture_method"] = "idle_read_only"
            record = save_record(args.output, capture_idle(client), metadata)
            print(f"Saved read-only idle pose: {args.output}")
            print(json.dumps(record["initial_pose"], indent=2))
            return 0
        from arm_sdk import Controller

        saved = teach(client, Controller.idle, args.output, metadata)
        if saved and args.label == "exploration_start":
            try:
                update_exploration_config(args.output)
            except Exception as exc:
                raise RuntimeError(
                    f"Pose saved at {args.output}, but exploration config update failed: {exc}. "
                    "Do not run exploration with the old initial_pose_file"
                ) from exc
        if not saved:
            print("Exited without recording an initial pose.")
        return 0 if saved else 1
    except (KeyboardInterrupt, EOFError):
        print("Interrupted. Check arm support and hardware safety state.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}. Check hardware safety state before proceeding.", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
