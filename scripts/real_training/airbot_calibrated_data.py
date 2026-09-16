"""Convert AIRBOT sensor wrenches without changing SDK poses; no robot I/O."""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from scripts.real_training.config import load_config, resolve
from scripts.real_training.data import (
    PROTOCOL,
    _load_raw,
    inspect,
    write_raw_log,
)
from scripts.real_training.import_airbot import assemble, exploration_episode
from scripts.shared.airbot_calibration import Calibration, checked_received
from scripts.shared.common import file_digest, write_json


def require_identity(calibration, robot_id, sensor_id):
    if (robot_id, sensor_id) != (calibration.record["robot_id"], calibration.record["sensor_id"]):
        raise ValueError("Calibration robot/sensor identity differs from the recordings")


def convert_episode(episode, calibration, *, demo):
    out = copy.deepcopy(episode)
    start = float(episode["attrs"]["start_time"])
    duration = 10.0 if demo else 4.0
    original = np.asarray(episode["ft"], dtype=np.float64)
    received_time = np.asarray(episode["ft_time"], dtype=np.float64)
    grid, held = checked_received(received_time - start, original, duration)
    out.update(
        received_ft_time=received_time.copy(),
        received_sensor_ft=original.copy(),
        received_calibrated_ft=calibration.frames.wrenches(original),
        ft_time=start + grid,
        ft=calibration.frames.wrenches(held),
    )
    out["attrs"].update(
        received_ft_hz=episode["attrs"]["ft_hz"],
        ft_hz=100.0,
        ft_rate_semantics="causal_processing_grid_not_independent_sensor_rate",
    )
    return out


def _publish(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream, Path(source).open("rb") as original:
        for chunk in iter(lambda: original.read(1024 * 1024), b""):
            stream.write(chunk)


def convert_training(cfg, calibration_path, exploration, session, *, audit_only=False):
    if cfg["profile"] != "airbot_sensor_calibrated_offline":
        raise ValueError("Sensor conversion requires the airbot_sensor_calibrated_offline profile")
    calibration = Calibration(calibration_path)
    target = resolve(cfg["raw_data"])
    report_path = target.with_suffix(".import.json")
    if not audit_only and (target.exists() or report_path.exists()):
        raise FileExistsError("Calibrated data/report already exists; choose a new destination")
    meta, exploration_episodes, demos = assemble(exploration, session)
    require_identity(calibration, meta["robot_id"], meta["sensor_id"])
    with tempfile.TemporaryDirectory(prefix="airbot-calibrated-import-") as folder:
        native = Path(folder) / "native.h5"
        write_raw_log(native, meta, exploration_episodes, demos, source_kind="real")
        native_cfg = {
            **cfg,
            "profile": "airbot_native_offline",
            "raw_data": str(native),
            "wrench": {
                "frame": "sensor local",
                "reference_point": "sensor origin",
                "compensation": "raw_sensor_load_retained",
            },
        }
        _load_raw(native_cfg)
        sampling = {
            "processing_grid_hz": 100,
            "method": "causal_latest_received_max_age_20ms",
            "independent_100hz_acquisition": False,
            "received_rates_hz": {
                name: episode["attrs"]["ft_hz"]
                for name, episode in {**exploration_episodes, **demos}.items()
            },
            "paper_difference": "Receive timing retained; holding values does not create sensor bandwidth",
        }
        # Preserve received data beside the derived grid; never relabel raw sensor
        # axes as calibrated axes or claim the hold creates independent samples.
        meta.update(
            calibration.metadata(),
            sampling=sampling,
            conversion="sensor_frames_sdk_pose_causal_100hz_hold_v2",
        )
        meta.pop("resampling", None)
        meta["source_hashes"].update(
            {str(calibration.path): calibration.sha256, **calibration.evidence}
        )
        converted = Path(folder) / "calibrated.h5"
        write_raw_log(
            converted,
            meta,
            {
                name: convert_episode(e, calibration, demo=False)
                for name, e in exploration_episodes.items()
            },
            {name: convert_episode(e, calibration, demo=True) for name, e in demos.items()},
            source_kind="real",
        )
        _, info = inspect({**cfg, "raw_data": str(converted)})
        report = {
            "source_kind": "real",
            "hardware_ready": False,
            "calibration_sha256": calibration.sha256,
            "sampling": sampling,
            "valid_windows_total": info["valid_windows_total"],
            "warnings": info["warnings"],
            "source_hashes": meta["source_hashes"],
            "exploration_out_of_range_per_channel": info["exploration_out_of_range_per_channel"],
            "raw_sha256": file_digest(converted),
        }
        for path, expected in meta["source_hashes"].items():
            if file_digest(path) != expected:
                raise ValueError("Source or calibration changed during conversion")
        if not audit_only:
            _publish(converted, target)
            write_json(report_path, report)
    return report


def convert_exploration(calibration_path, exploration, output, *, audit_only=False):
    calibration = Calibration(calibration_path)
    source_hash = file_digest(exploration)
    episode, collection, header = exploration_episode(exploration)
    require_identity(calibration, header["config"]["robot_sn"], header["config"]["sensor_port"])
    converted = convert_episode(episode, calibration, demo=False)
    target = Path(output)
    report_path = target.with_suffix(".import.json")
    if not audit_only and (target.exists() or report_path.exists()):
        raise FileExistsError("Deployment exploration/report already exists")
    report = {
        "source_kind": "real",
        "hardware_ready": False,
        "source_log_sha256": source_hash,
        "calibration_sha256": calibration.sha256,
        "collection": collection,
        "samples": 400,
        "sampling": "causal_100hz_hold_not_independent_100hz_acquisition",
    }
    calibration.assert_unchanged()
    if file_digest(exploration) != source_hash:
        raise ValueError("Exploration changed during conversion")
    if not audit_only:
        metadata = calibration.metadata()
        keys = (
            "robot_id",
            "sensor_id",
            "calibration_id",
            "ft_frame",
            "ft_reference_point",
            "compensation",
        )
        with tempfile.TemporaryDirectory(prefix="airbot-task-exploration-") as folder:
            staged = Path(folder) / "exploration.npz"
            np.savez_compressed(
                staged,
                source_kind="real",
                ft=converted["ft"][1:],
                time_s=np.arange(1, 401) / 100,
                received_time_s=episode["ft_time"] - episode["attrs"]["start_time"],
                received_sensor_ft=episode["ft"],
                calibration_sha256=calibration.sha256,
                source_log_sha256=source_hash,
                exploration_protocol=json.dumps(PROTOCOL, sort_keys=True),
                **{key: metadata[key] for key in keys},
            )
            _publish(staged, target)
        report["sha256"] = file_digest(target)
        write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "training", "exploration"), nargs="?", default="check"
    )
    parser.add_argument("--calibration", default="configs/robot_control/airbot_calibration.json")
    parser.add_argument(
        "--config", default="configs/real_training/real_training_airbot_calibrated.yaml"
    )
    parser.add_argument(
        "--exploration", default="archive/real_training/real_robot/exploration_ft_007.jsonl"
    )
    parser.add_argument(
        "--session",
        default="archive/real_training/raw_data/manual_demonstrations/session_record_only_002",
    )
    parser.add_argument(
        "--output", default="runs/real_training/calibrated_deployment_exploration.npz"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            record = Calibration(args.calibration)
            result = {
                "hardware_connected": False,
                "software_structure_valid": True,
                "physical_calibration_certified_by_software": False,
                "calibration_sha256": record.sha256,
            }
        elif args.action == "training":
            from scripts.shared.run_paths import new_run_config

            cfg = load_config(args.config)
            if not args.audit_only:
                cfg = new_run_config(cfg, include_raw=True)
            result = convert_training(
                cfg,
                args.calibration,
                args.exploration,
                args.session,
                audit_only=args.audit_only,
            )
        else:
            if not args.audit_only:
                from scripts.shared.run_paths import new_output

                args.output = new_output(args.output)
            result = convert_exploration(
                args.calibration, args.exploration, args.output, audit_only=args.audit_only
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
