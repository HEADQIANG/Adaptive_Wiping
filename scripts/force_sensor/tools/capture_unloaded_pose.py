"""Bounded raw FT/SDK-pose observation, never acquires control or moves the arm."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.robot_control.airbot_initial_pose import validate_stationary
from scripts.robot_control.client import open_client, snapshot
from scripts.shared.common import file_digest
from scripts.shared.paths import ROOT

SENSOR = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AG0KL3D4-if00-port0"


def assess(rows, confirmed):
    if not rows:
        raise ValueError("No observations")
    errors = []
    try:
        validate_stationary([r["pose"] for r in rows])
    except ValueError as exc:
        errors.append(str(exc))
    stamps = np.array([r["sensor_receive_perf_s"] for r in rows])
    ft = np.array([r["raw_sensor_wrench_si"] for r in rows])
    if ft.shape != (len(rows), 6) or not np.isfinite(ft).all() or not np.isfinite(stamps).all():
        raise ValueError("Invalid raw force observations")
    ages = np.array([r["observation_perf_s"] - r["sensor_receive_perf_s"] for r in rows])
    if np.any(ages < 0) or np.max(ages) > 0.05:
        errors.append("Stale or future force timestamp")
    if len(set(r["pose"]["service_state"]["controller_state"] for r in rows)) != 1:
        errors.append("Robot controller changed during observation")
    _, distinct = np.unique(stamps, return_index=True)
    if len(distinct) < 20:
        errors.append("Fewer than 20 distinct received force observations")
    if np.ptp([r["observation_perf_s"] for r in rows]) < 1.9:
        errors.append("Less than 1.9 seconds of observations")
    values = ft[distinct]
    return {
        "stationary_and_fresh": not errors,
        "errors": errors,
        "contact_free_operator_confirmed": confirmed,
        "eligible_unloaded_pose": bool(confirmed and not errors),
        "complete_calibration": False,
        "hardware_ready": False,
        "observations": len(rows),
        "distinct_receive_times": len(distinct),
        "max_force_age_s": float(ages.max()),
        "raw_mean_si": values.mean(0).tolist(),
        "raw_std_si": values.std(0).tolist(),
        "raw_mean_force_norm_N": float(np.linalg.norm(values.mean(0)[:3])),
    }


def capture(path, *, confirmed=False):
    from scripts.force_sensor.kwr75_reader import Kwr75Reader

    path = Path(path)
    if path.exists():
        raise FileExistsError("Output exists; never overwrite a pose")
    # Do not steal an existing serial reader's stream or stop it on cleanup.
    busy = subprocess.run(["fuser", str(Path(SENSOR).resolve())], capture_output=True, text=True)
    if busy.returncode != 1:
        raise RuntimeError(
            "Serial port busy or ownership check failed; close the other reader explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": "raw_unloaded_pose_candidate",
        "source_kind": "real",
        "sensor_identity": "operator identified KWR52-TiS S/N 1F06303",
        "sensor_port": SENSOR,
        "contact_free_operator_confirmed": confirmed,
        "manual_tare_performed": False,
        "motion_commands_sent": False,
        "control_acquired": False,
        "complete": False,
        "frame": "sensor native FT and SDK end pose; absolute transform unverified",
        "units": ["N"] * 3 + ["N*m"] * 3,
        "timestamps": "host CLOCK_MONOTONIC, asynchronous reads, not device hardware timestamps",
        "source_hashes": {
            str(p): file_digest(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/robot_control/airbot_initial_pose.py",
                ROOT / "scripts/force_sensor/kwr75_reader.py",
            )
        },
        "rows": [],
    }
    client = reader = None
    with path.open("x", encoding="utf-8") as stream:
        try:
            client = open_client("127.0.0.1", 50051)
            first = snapshot(client)
            record["initial_robot_state"] = first
            if first["service_state"]["controller_state"] not in ("idle", "servo", "gravity_comp"):
                raise RuntimeError("Unexpected robot controller; no mode change will be attempted")
            reader = Kwr75Reader(SENSOR)
            reader.start()
            deadline = time.monotonic() + 2
            while reader.latest(net=False) is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("No force stream")
                time.sleep(0.01)
            started = time.monotonic()
            for i in range(61):
                remaining = started + i * 0.05 - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                pose = snapshot(client)
                value = reader.latest(net=False)
                if value is None:
                    raise RuntimeError("Force stream disappeared")
                stamp, ft = value
                observed = time.monotonic()
                record["rows"].append(
                    {
                        "pose": pose,
                        "sensor_receive_perf_s": stamp,
                        "observation_perf_s": observed,
                        "raw_sensor_wrench_si": ft.tolist(),
                    }
                )
                if observed - stamp > 0.05:
                    raise RuntimeError(
                        "Force stream stale; observation stopped, no robot stop command sent"
                    )
            record["assessment"] = assess(record["rows"], confirmed)
            record["complete"] = True
        except BaseException as exc:
            record["error"] = str(exc)
            raise
        finally:
            try:
                if reader is not None:
                    reader.stop()
            finally:
                try:
                    if client is not None:
                        client.close()
                finally:
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write("\n")
    return record["assessment"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contact-free-confirmed",
        action="store_true",
        help="Only after current explicit onsite confirmation: unchanged tool, suspended and untouched",
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.output, confirmed=args.contact_free_confirmed), indent=2))
