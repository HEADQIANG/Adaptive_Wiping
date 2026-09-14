"""Attended AIRBOT control; default operations use an offline fake robot."""

import argparse
import json
import os
import select
import signal
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.shared.paths import read_path, writable_path

COMMANDS = (
    "inspect",
    "gravity-comp",
    "zero-gravity",
    "hold",
    "keyboard",
    "move-joint",
    "move-pose",
    "capture-pose",
    "idle",
    "stop",
    "console",
)


@contextmanager
def terminal():
    fd = sys.stdin.fileno()
    before = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, before)


def key_available(timeout=0.0):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    data = os.read(sys.stdin.fileno(), 4096).decode("utf-8", errors="ignore")
    if not data or "\x04" in data:
        raise EOFError("Terminal closed")
    # Drain repeated input in one read; never queue a series of motions.
    return " " if " " in data else data[-1]


def save_pose(session, output):
    from .airbot_initial_pose import save_record

    path = writable_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = session.stationary()
    metadata = {
        "capture_method": "basic_control_stationary",
        "source_kind": "real" if session.robot.hardware else "synthetic",
        "robot_sn": session.cfg["robot_sn"],
        "label": "basic_control_pose",
    }
    if session.robot.hardware:
        firmware = session.robot.client.get_firmware_info()
        if firmware is None or firmware.arm_sn != session.cfg["robot_sn"]:
            raise RuntimeError("Firmware identity unavailable or changed; no pose saved")
        metadata["firmware_cached"] = asdict(firmware)
    record = save_record(path, samples, metadata)
    session.event("pose-saved", path=str(path))
    return record


def to_sdk_pose(position, quaternion, calibration):
    import numpy as np
    from scipy.spatial.transform import Rotation

    frame = calibration.frames
    sdk_r = frame.base[:3, :3].T @ Rotation.from_quat(quaternion).as_matrix() @ frame.tcp[:3, :3].T
    sdk_q = Rotation.from_matrix(sdk_r).as_quat()
    return frame.to_end(np.asarray(position), sdk_q).tolist(), sdk_q.tolist()


def interact(session, *, keyboard=False, output=None):
    pending = None
    with terminal():
        while not session.faulted:
            session.maintain()
            key = key_available(min(0.01, session.cfg["input_timeout_s"]))
            if not key:
                continue
            if key == " ":
                session.stop()
                return
            if pending:
                if key == "Y":
                    if pending == "quit":
                        session.finish(supported=True)
                        return
                    if pending == "idle":
                        session.finish(supported=True)
                        keyboard = False
                    elif pending == "keyboard":
                        session.hold()
                        keyboard = True
                    else:
                        session.switch(pending)
                        keyboard = False
                pending = None
                continue
            if key in ("q", "i", "g", "h"):
                pending = {"q": "quit", "i": "idle", "g": "gravity_comp", "h": "servo"}[key]
                print(
                    f"\nSupport arm and confirm {pending}: press Y; any other key cancels.",
                    flush=True,
                )
            elif key == "k":
                pending = "keyboard"
                print("\nSupport arm and confirm keyboard servo mode: press Y.", flush=True)
            elif key == "c":
                if output is None:
                    print("\nNo --output pose path configured.", flush=True)
                else:
                    save_pose(session, output)
            elif keyboard:
                # Only STOP is acted on during a point move; repeated jogs are dropped.
                session.jog(key, stop_requested=lambda: key_available() == " ")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Enable attended hardware mode after explicit confirmation",
    )
    p.add_argument("--config", default="configs/robot_control/basic_control.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--joints", type=float, nargs=6, metavar="RAD")
    p.add_argument("--position", type=float, nargs=3, metavar="M")
    p.add_argument("--quaternion", type=float, nargs=4, metavar="XYZW")
    p.add_argument("--frame", choices=("sdk", "tcp"), default="sdk")
    p.add_argument("--calibration", help="Verified calibration record, required for --frame tcp")
    p.add_argument("--output", type=Path, help="New stationary pose JSON (never overwritten)")
    p.add_argument(
        "--log", type=Path, help="New session JSONL (default: unique runs/robot_control path)"
    )
    p.add_argument(
        "--keys", help="Offline keyboard sequence; hardware uses attended keyboard input"
    )
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "inspect":
        if args.execute:
            p.error("inspect is read-only; omit --execute")
        from .adapter import DeadlineStub
        from .client import open_client, snapshot

        client = None
        try:
            client = open_client(args.host, args.port)
            client._stub = DeadlineStub(client._stub)
            firmware = client.get_firmware_info()
            print(
                json.dumps(
                    {
                        "state": snapshot(client),
                        "firmware_cached": asdict(firmware) if firmware else None,
                    },
                    indent=2,
                )
            )
            return 0
        except Exception as exc:
            print(f"Read-only inspection failed: {exc}", file=sys.stderr)
            return 2
        finally:
            if client is not None:
                client.close()
    from .motion import FakeRobot, MotionSession, fake_config, validate_config

    client = session = stream = None
    completed = False
    prior_signal = None
    try:
        cfg = json.loads(read_path(args.config).read_text()) if args.execute else fake_config()
        validate_config(cfg, hardware=args.execute)
        if args.command == "move-joint" and args.joints is None:
            p.error("move-joint requires --joints with six angles in radians")
        if args.command == "move-pose" and (args.position is None or args.quaternion is None):
            p.error("move-pose requires --position and --quaternion xyzw")
        if args.command == "capture-pose" and args.output is None:
            p.error("capture-pose requires --output")
        pose = (args.position, args.quaternion)
        if args.frame == "tcp":
            if args.command != "move-pose" or args.calibration is None:
                p.error("TCP frame is only valid for move-pose with --calibration")
            from scripts.shared.airbot_calibration import Calibration

            calibration = Calibration(args.calibration)
            if calibration.record["robot_id"] != cfg["robot_sn"]:
                raise ValueError("Calibration robot identity does not match this setup")
            pose = to_sdk_pose(*pose, calibration)
        if args.output:
            args.output = writable_path(args.output)
            if args.output.exists():
                raise FileExistsError(args.output)
        path = writable_path(
            args.log
            or f"runs/robot_control/session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')}.jsonl"
        )
        if path.exists():
            raise FileExistsError(path)
        if args.execute:
            if not sys.stdin.isatty():
                p.error("Hardware control requires an attended terminal")
            print("Support arm; verify physical emergency stop, payload and entire swept path.")
            print("No project XYZ workspace boundary is enforced.")
            print(
                "No collision planning. Idle/disconnect may let the arm fall. Server must use --no-return."
            )
            if input("Type CONTROL to permit this session: ").strip() != "CONTROL":
                print("Cancelled without connecting.")
                return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("x", encoding="utf-8")

        def emit(event):
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        if args.execute:
            from .adapter import DeadlineStub
            from .client import open_client
            from .hardware import HardwareRobot

            client = open_client(args.host, args.port)
            client._stub = DeadlineStub(client._stub)
            robot = HardwareRobot(client, cfg)
        else:
            robot = FakeRobot()
        session = MotionSession(robot, cfg, emit=emit)

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Signal interrupted control")

        prior_signal = signal.signal(signal.SIGTERM, interrupted)
        session.begin()
        command = args.command
        if command in ("gravity-comp", "zero-gravity"):
            session.switch("gravity_comp")
        elif command in ("hold", "keyboard"):
            session.hold()
        elif command == "move-joint":
            if args.execute:
                with terminal():
                    session.move(
                        "joint", args.joints, stop_requested=lambda: key_available() == " "
                    )
            else:
                session.move("joint", args.joints)
        elif command == "move-pose":
            if args.execute:
                with terminal():
                    session.move("pose", pose, stop_requested=lambda: key_available() == " ")
            else:
                session.move("pose", pose)
        elif command == "capture-pose":
            save_pose(session, args.output)
        elif command == "stop":
            session.stop()
        elif command == "idle":
            session.finish(supported=True)
        interactive = args.execute or (
            command in ("console", "keyboard") and sys.stdin.isatty() and args.keys is None
        )
        if interactive and not session.faulted and command != "idle":
            print(
                "Session active: q then Y exits with support; space requests latched software stop."
            )
            interact(session, keyboard=command == "keyboard", output=args.output)
        elif not args.execute:
            if command == "keyboard":
                for key in args.keys or "1+xr":
                    session.jog(key)
                    if session.faulted:
                        break
            elif command == "console":
                session.switch("gravity_comp")
                session.hold()
            session.finish(supported=True)
        completed = True
        print(
            json.dumps(
                {
                    "log": str(path),
                    "hardware_connected": args.execute,
                    "software_stop": session.faulted,
                    "mode": session.mode,
                }
            )
        )
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; use physical safety procedure and keep arm supported.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Control failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if (
            session is not None
            and session.acquired
            and not session.faulted
            and (not completed or session.mode != "idle")
        ):
            try:
                session.stop()
            except Exception as exc:
                print(
                    f"STOP NOT CONFIRMED: {exc}. Use physical emergency stop NOW.", file=sys.stderr
                )
        if prior_signal is not None:
            signal.signal(signal.SIGTERM, prior_signal)
        if client is not None:
            client.close()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
