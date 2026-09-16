"""Interactive, user-triggered raw FT/pose recording; no robot control commands."""

import argparse
import fcntl
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.force_sensor.tools.capture_unloaded_pose import assess, capture
from scripts.shared.paths import ROOT


def pose_files(folder):
    return sorted(
        (p for p in Path(folder).glob("pose_*.json") if re.fullmatch(r"pose_[0-9]+\.json", p.name)),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def next_path(folder):
    files = pose_files(folder)
    index = max((int(p.stem.split("_")[1]) for p in files), default=0) + 1
    return Path(folder) / f"pose_{index:03d}.json"


def session_status(folder):
    accepted, rejected, directions = [], [], []
    for path in pose_files(folder):
        try:
            record = json.loads(path.read_text())
            if (
                record.get("kind") != "raw_unloaded_pose_candidate"
                or record.get("source_kind") != "real"
                or record.get("complete") is not True
                or record.get("contact_free_operator_confirmed") is not True
                or record.get("manual_tare_performed") is not False
            ):
                raise ValueError("Incomplete, incompatible or unconfirmed record")
            if not assess(record["rows"], True)["eligible_unloaded_pose"]:
                raise ValueError("Stationarity/freshness check failed")
            rotation = Rotation.from_quat(
                [r["pose"]["sdk_end_orientation_xyzw"] for r in record["rows"]]
            ).mean()
            directions.append(rotation.inv().apply([0.0, 0.0, -1.0]))
            accepted.append(path.name)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"file": path.name, "reason": str(exc)})
    result = {"accepted": accepted, "rejected": rejected, "complete_calibration": False}
    if len(directions) >= 2:
        angles = np.rad2deg(np.arccos(np.clip(np.asarray(directions[:-1]) @ directions[-1], -1, 1)))
        result["last_gravity_direction_from_first_deg"] = float(angles[0])
        result["last_gravity_direction_nearest_previous_deg"] = float(angles.min())
    if len(directions) >= 4:
        singular = np.linalg.svd(
            np.column_stack([np.ones(len(directions)), directions]), compute_uv=False
        )
        result["direction_design_singular_values"] = singular.tolist()
        result["direction_design_full_rank"] = bool(singular[-1] > singular[0] * 1e-6)
    return result


def show_status(folder):
    status = session_status(folder)
    print(
        f"Accepted stationary poses: {len(status['accepted'])}; retained rejected files: {len(status['rejected'])}"
    )
    if "last_gravity_direction_nearest_previous_deg" in status:
        angle = status["last_gravity_direction_nearest_previous_deg"]
        print(
            f"Last pose: {status['last_gravity_direction_from_first_deg']:.1f} deg from first, "
            f"{angle:.1f} deg from nearest previous gravity direction."
        )
        if angle < 10:
            print(
                "Similar gravity direction: useful for repeat/drift checks, limited new orientation information."
            )
    if "direction_design_full_rank" in status and not status["direction_design_full_rank"]:
        print(
            "Orientation coverage is rank-deficient. Count alone does not establish calibration readiness."
        )
    print("Calibration is NOT yet fitted or validated.", flush=True)


def interactive(folder, *, read=input, recorder=capture):
    print("Keep the separate gravity-compensation teaching terminal OPEN and the arm supported.")
    print("This recorder NEVER switches modes, moves the robot, tares, or sends a robot stop.")
    print("Close other force-reader/display programs before recording.")
    print(
        "Keep mount/sponge/tare state unchanged. Plan 6-8 distinct SAFE orientations, then repeat the first."
    )
    print(
        "Do not force idle joints, joint limits or unsafe poses; no specific angle is prescribed."
    )
    show_status(folder)
    while True:
        print("\nBefore r: arm reliably supported and STATIONARY; sponge suspended;")
        print(
            "nobody touches/loads the sensor-side tool; onsite supervision and emergency stop available."
        )
        command = (
            read(
                "r + Enter = confirm these conditions and record 3s; s = status; q = quit recorder: "
            )
            .strip()
            .lower()
        )
        if command == "q":
            return
        if command == "s":
            show_status(folder)
            continue
        if command != "r":
            print("No capture. Enter r, s or q.")
            continue
        path = next_path(folder)
        print(f"Recording {path.name}; keep still and keep support.", flush=True)
        try:
            report = recorder(path, confirmed=True)
        except (OSError, ValueError, RuntimeError) as exc:
            print(
                f"Capture failed: {exc}. Any partial record is retained. No robot mode change requested."
            )
            continue
        if report["eligible_unloaded_pose"]:
            print(
                f"Saved: {path.name}; {report['observations']} observations; "
                f"mean force norm {report['raw_mean_force_norm_N']:.3f} N."
            )
        else:
            print(f"Saved but NOT accepted: {path.name}; {report['errors']}")
        show_status(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path
    )
    parser.add_argument(
        "--status", action="store_true", help="Read saved files only; never connect to devices"
    )
    args = parser.parse_args(argv)
    directory_supplied = args.directory is not None
    args.directory = args.directory or ROOT / "runs/force_sensor/unloaded_calibration"
    if args.status:
        show_status(args.directory)
        return 0
    if not sys.stdin.isatty():
        parser.error("Recording requires an attended interactive terminal")
    from scripts.shared.run_paths import new_output

    args.directory = new_output(
        args.directory, resume=directory_supplied and (args.directory / ".session.lock").is_file()
    )
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / ".session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another interactive recorder owns this directory")
        try:
            interactive(args.directory)
        except (KeyboardInterrupt, EOFError):
            print("\nRecorder interrupted; any partial capture was retained.")
        finally:
            print(
                "Recorder closed. Robot mode is UNCHANGED; gravity compensation may still be active."
            )
            print(
                "Keep supporting the arm. To end teaching, use q in the SEPARATE teaching terminal only after support is secure."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
