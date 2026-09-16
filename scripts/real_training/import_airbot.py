"""Audited import of received AIRBOT data without inventing physical calibration."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.real_training.airbot_demonstrations import quality
from scripts.real_training.config import load_config, native_profile, resolve
from scripts.real_training.data import (
    FT_UNITS,
    PROTOCOL,
    inspect,
    write_raw_log,
)
from scripts.shared.common import CHANNELS, file_digest, write_json
from scripts.shared.real_preprocessing import finite_array


def events(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def received_stream(records):
    # Repeated latest() values are not independent sensor measurements.
    unique = {}
    for row in records:
        stamp = float(row["sensor_receive_perf_s"])
        wrench = finite_array(row["raw_sensor_wrench_si"], "raw wrench")
        if not np.isfinite(stamp) or wrench.shape != (6,):
            raise ValueError("Invalid received FT record")
        if stamp in unique and not np.array_equal(unique[stamp], wrench):
            raise ValueError("Conflicting FT values share a receive timestamp")
        unique[stamp] = wrench
    time = np.array(sorted(unique), dtype=np.float64)
    if len(time) < 2:
        raise ValueError("Insufficient unique FT observations")
    ft = np.array([unique[t] for t in time])
    return time, ft


def stream_report(time):
    dt = np.diff(time)
    return {
        "unique_received_samples": len(time),
        "median_received_hz": float(1 / np.median(dt)),
        "max_received_gap_s": float(dt.max()),
    }


def exploration_episode(path):
    rows = events(path)
    if not rows:
        raise ValueError("Empty exploration log")
    header = rows[0]
    if header.get("mode") == "manual-start":
        from scripts.real_training.manual_exploration_contract import validate

        validate(rows)
    done = [r for r in rows if r["event"] == "motion_complete"]
    samples = [r for r in rows if r["event"] == "sample" and r["phase"] == "exploration"]
    if (
        header.get("source_kind") != "real"
        or header.get("mode") not in ("force-guarded", "manual-start")
        or rows[-1] != {"event": "session_complete", "idle_confirmed": True}
        or len(done) != 1
        or done[0].get("nominal_protocol_match") is not True
        or done[0].get("time_scale") != 1
        or len(samples) != 400
        or any(r["event"] == "aborted" for r in rows)
    ):
        raise ValueError("Expected a completed real 4-second force-guarded exploration")
    if [s["index"] for s in samples] != list(range(1, 401)):
        raise ValueError("Missing or reordered exploration samples")
    # Unversioned historical completions used the original 10 mm/s, 2 s press.
    recorded_press = {"press_speed_m_s": 0.01, "press_duration_s": 2.0}
    if any(key in done[0] for key in recorded_press):
        recorded_press = {key: done[0].get(key) for key in recorded_press}
    if any(recorded_press[key] != PROTOCOL[key] for key in recorded_press):
        raise ValueError("Exploration press protocol is missing or differs from the current protocol; recollect data")
    due = np.array([r["due_perf_s"] for r in samples])
    start = due[0] - 0.01
    if not np.allclose(due - start, np.arange(1, 401) / 100, atol=1e-8, rtol=0):
        raise ValueError("Exploration scheduling does not match the paper protocol")
    forces = []
    for r in rows:
        if r["event"] in ("sample", "initial"):
            for key in ("pre_send_force", "force"):
                if r.get(key) is not None:
                    forces.append(r[key])
    time, ft = received_stream(forces)
    # A measured post-roll sample establishes coverage, never a padded endpoint.
    lo = max(0, np.searchsorted(time, start, side="right") - 1)
    hi = np.searchsorted(time, start + 4, side="left") + 1
    time, ft = time[lo:hi], ft[lo:hi]
    report = stream_report(time)
    episode = {
        "attrs": {
            "sponge_id": header.get("config", {}).get("sponge_id", "normal"),
            "complete": True,
            "start_time": start,
            "ft_hz": report["median_received_hz"],
        },
        "ft_time": time,
        "ft": ft,
    }
    return episode, {**report, "completion": done[0]}, header


def assemble(exploration, session, *, programmed_hold_last=False, subtract_recorded_baseline=False,
             manual_tared=False, confirm_same_setup=False):
    if manual_tared and (programmed_hold_last or not subtract_recorded_baseline or not confirm_same_setup):
        raise ValueError("Manual tared import requires baseline subtraction and same-setup confirmation, without padding")
    if confirm_same_setup and not manual_tared:
        raise ValueError("Same-setup confirmation is only supported with --manual-tared")
    if programmed_hold_last:
        from scripts.real_training.programmed_padding import assemble_padded

        return assemble_padded(exploration, session, subtract_recorded_baseline=subtract_recorded_baseline)
    if subtract_recorded_baseline and not manual_tared:
        raise ValueError("Recorded baseline conversion requires --programmed-hold-last")
    exploration, session = Path(exploration).resolve(), Path(session).resolve()
    session_path = session / "session.json"
    metadata = json.loads(session_path.read_text())
    program_protocols = ("airbot_programmed_v1", "airbot_programmed_fixed_depth_v2")
    if (metadata.get("demonstration_source") == "programmed"
            or metadata.get("protocol") in program_protocols
            or metadata.get("config", {}).get("protocol") in program_protocols):
        raise ValueError(
            "Variable-length programmed demonstrations are not supported by the fixed 10s/25-frame importer; "
            "use --programmed-hold-last for an explicitly labeled derived copy; originals must remain unchanged"
        )
    cfg = metadata["config"]
    if cfg.get("force_recording") == "software_tared" and not manual_tared:
        raise ValueError(
            "Software-tared manual demonstrations require a matching tared training input contract; "
            "this raw-load importer cannot silently discard the recorded tare"
        )
    hashes = {str(p): file_digest(p) for p in (exploration, session_path)}
    exp, exp_report, header = exploration_episode(exploration)
    if manual_tared:
        from scripts.real_training.manual_tared_import import validate_setup, validate_tare, apply_baselines

        validate_setup(metadata, header)
    elif header.get("mode") == "manual-start":
        raise ValueError("Manual exploration requires newly bound programmed demonstrations and explicit tared hold-last import")
    if cfg["robot_sn"] != header["config"]["robot_sn"]:
        raise ValueError("Exploration and demonstrations belong to different robots")
    if cfg["sensor_port"] != header["config"]["sensor_port"]:
        raise ValueError("Exploration and demonstration sensor identities differ")
    manifests = sorted(session.glob("demo_*.json"))
    if [p.name for p in manifests] != [f"demo_{i:02}.json" for i in range(1, 9)]:
        raise ValueError("Expected exactly demo_01 through demo_08")
    demos, reports, paths, biases = {}, {}, set(), {}
    for manifest in manifests:
        record = json.loads(manifest.read_text())
        path = (session / record["raw_file"]).resolve()
        if path.parent != session or path in paths:
            raise ValueError("Demonstration raw paths must be distinct files in the session")
        paths.add(path)
        if (
            record.get("source_kind") != "real"
            or record.get("accepted") is not True
            or record.get("quality", {}).get("passed") is not True
            or file_digest(path) != record["sha256"]
        ):
            raise ValueError(f"Rejected or changed demonstration: {manifest.name}")
        rows = events(path)
        if rows[0]["event"] != "start" or rows[-1]["event"] != "finished":
            raise ValueError("Incomplete demonstration")
        if manual_tared:
            biases[manifest.stem] = validate_tare(session, path, rows, record, hashes)
        start = rows[0]["start_perf_s"]
        samples = [r for r in rows if r["event"] == "sample"]
        checked = quality(samples, start)
        if not checked["passed"]:
            raise ValueError(f"Demonstration quality failed: {checked}")
        time, ft = received_stream([r["ft"] for r in samples])
        pose_time = np.array([r["pose"]["host_monotonic_s"] for r in samples])
        # CPython/Linux uses CLOCK_MONOTONIC for both clocks. Check the recorded
        # adjacent reads as well; do not infer or fit an offset from trajectories.
        lag = np.array([r["observed_perf_s"] for r in samples]) - pose_time
        if np.any(lag < 0) or np.any(lag > 0.020):
            raise ValueError("Pose/FT host clock or observation latency mismatch")
        report = stream_report(time)
        demos[manifest.stem] = {
            "attrs": {
                "sponge_id": cfg["sponge_id"],
                "complete": True,
                "start_time": start,
                "ft_hz": report["median_received_hz"],
                "pose_hz": 100.0,
                "surface_id": cfg["surface_id"],
                "exploration_id": cfg["exploration_id"],
            },
            "ft_time": time,
            "ft": ft,
            "pose_time": pose_time,
            "sdk_end_position": [r["pose"]["sdk_end_position_m"] for r in samples],
            "sdk_end_quaternion": [r["pose"]["sdk_end_orientation_xyzw"] for r in samples],
        }
        reports[manifest.stem] = {**report, "quality": checked, "raw_file": str(path)}
        hashes.update({str(manifest): file_digest(manifest), str(path): record["sha256"]})
    meta = import_metadata(cfg, hashes, exp_report, reports)
    if manual_tared:
        apply_baselines(meta, exp, demos, biases, events(exploration), cfg, exploration, session_path)
    for path, expected in hashes.items():
        if file_digest(path) != expected:
            raise ValueError("Source changed during import")
    return meta, {cfg["exploration_id"]: exp}, demos


def import_metadata(cfg, hashes, exp_report, reports):
    return {
        "robot_id": cfg["robot_sn"],
        "sensor_id": cfg["sensor_port"],
        "calibration_id": "unverified_not_a_calibration",
        "calibration_status": "unverified",
        "position_definition": "SDK configured end frame",
        "sensor_frame": "KWR75 sensor native axes",
        "position_frame": "SDK configured reference",
        "position_units": "m",
        "orientation_order": "xyzw",
        "clock": "shared_monotonic_seconds",
        "clock_evidence": "Linux CPython monotonic and perf_counter use CLOCK_MONOTONIC; adjacent reads checked",
        "channels": CHANNELS,
        "ft_units": FT_UNITS,
        "ft_frame": "sensor local",
        "ft_reference_point": "sensor origin",
        "compensation": "raw_sensor_load_retained",
        "resampling": "causal_latest_received_100hz_max_age_20ms",
        "exploration_protocol": PROTOCOL,
        "source_hashes": hashes,
        "collection_report": {"exploration": exp_report, "demonstrations": reports},
    }


def import_dataset(cfg, exploration, session, *, audit_only=False, programmed_hold_last=False,
                   subtract_recorded_baseline=False, manual_tared=False, confirm_same_setup=False):
    if not native_profile(cfg):
        raise ValueError(
            "This importer requires airbot_native_offline; it cannot certify calibration"
        )
    if subtract_recorded_baseline != (cfg["profile"] == "airbot_native_tared_offline"):
        raise ValueError("Tared profile requires explicit --subtract-recorded-baseline; raw profiles forbid it")
    meta, exp, demos = assemble(exploration, session, programmed_hold_last=programmed_hold_last,
                                subtract_recorded_baseline=subtract_recorded_baseline,
                                manual_tared=manual_tared, confirm_same_setup=confirm_same_setup)
    target = resolve(cfg["raw_data"])
    if not audit_only and target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(prefix="airbot-import-check-") as temp:
        test_path = Path(temp) / "raw.h5"
        write_raw_log(test_path, meta, exp, demos, source_kind="real")
        _, info = inspect({**cfg, "raw_data": str(test_path)})
        report = {
            "source_kind": "real",
            "hardware_ready": False,
            "profile": cfg["profile"],
            "derivation": meta.get("derivation"),
            "setup_confirmation": meta.get("setup_confirmation"),
            "source_hashes": meta["source_hashes"],
            "collection": meta["collection_report"],
            "warnings": info["warnings"],
            "valid_windows_total": info["valid_windows_total"],
            "encoder_sha256": info["encoder_sha256"],
            "exploration_out_of_range_per_channel": info["exploration_out_of_range_per_channel"],
        }
        if not audit_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(test_path.read_bytes())
            report["raw_sha256"] = file_digest(target)
            write_json(target.parent / "import_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/real_training/real_training_airbot_native.yaml"
    )
    parser.add_argument(
        "--exploration", default="archive/real_training/real_robot/exploration_ft_007.jsonl"
    )
    parser.add_argument(
        "--session",
        default="archive/real_training/raw_data/manual_demonstrations/session_record_only_002",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--programmed-hold-last", action="store_true",
                        help="Explicitly derive 10s programmed episodes by holding final wipe state; originals unchanged")
    parser.add_argument("--subtract-recorded-baseline", action="store_true",
                        help="Use recorded unloaded baselines for the new tared-native offline profile")
    parser.add_argument("--manual-tared", action="store_true",
                        help="Import measured 10s software-tared manual demonstrations without padding")
    parser.add_argument("--confirm-same-setup", action="store_true",
                        help="Confirm same sponge, tool/sensor mounting and table setup after collection; manual tared only")
    args = parser.parse_args(argv)
    from scripts.shared.run_paths import new_run_config

    cfg = load_config(args.config)
    if not args.audit_only:
        cfg = new_run_config(cfg, include_raw=True)
    result = import_dataset(
        cfg, args.exploration, args.session, audit_only=args.audit_only,
        programmed_hold_last=args.programmed_hold_last,
        subtract_recorded_baseline=args.subtract_recorded_baseline,
        manual_tared=args.manual_tared,
        confirm_same_setup=args.confirm_same_setup,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
