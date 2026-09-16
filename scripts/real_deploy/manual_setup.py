"""Manual-demonstration deployment candidate, audited separately from programmed runs."""

import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.real_deploy import fixed_setup as fixed
from scripts.real_deploy.manual_path import CandidatePolicy, TRANSFORM, plot_path
from scripts.real_training.config import output_lock, resolve, training_contract
from scripts.real_training.data import load_prepared
from scripts.real_training.import_airbot import events
from scripts.real_training.manual_exploration_contract import load_completed
from scripts.robot_control.adapter import DeadlineStub, Robot
from scripts.robot_control.client import open_client
from scripts.robot_control.safety import check_bounds, positive, quaternion_angle, write_event
from scripts.shared.common import file_digest, write_json
from scripts.shared.paths import recorded_source_hash
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_preprocessing import finite_array, make_windows
from scripts.shared.sampling import previous_samples

MODE = "manual_tared_original_v1"
CONFIRMATIONS = fixed.CONFIRMATIONS + (
    "training_input_alignment_confirmed", "encoder_quality_exception_confirmed",
)
CAPS = {"max_cartesian_speed_m_s": 0.05, "max_delta_h_m": 0.003,
        "max_tracking_error_m": 0.005, "orientation_error_limit_rad": 0.02,
        "max_force_n": 40.0, "max_initial_force_n": 32.0}


def load_setup(path, *, policy_path=None):
    path = resolve(path)
    bindings = {str(path): file_digest(path)}
    spec = json.loads(path.read_text())
    if spec.get("workflow") not in (None, "runtime_start_h_z_s_g_v1"):
        raise ValueError("Unknown manual deployment workflow")
    if (spec.get("schema_version") != 1 or spec.get("mode") != MODE
            or spec.get("path_transform") != TRANSFORM
            or spec.get("cartesian_speed_policy") != "record-only"
            or spec.get("torque_limit_policy") != "record-only"
            or spec.get("guards", {}).get("max_torque_nm") is not None
            or spec.get("initialization") != "nominal_10mm_during_first_2s"
            or spec.get("same_hardware_and_sponge_confirmed") is not True):
        raise ValueError("Expected explicitly scoped manual-tared candidate configuration")
    from scripts.real_deploy.policy_selection import bind_runtime_software, select_setup

    spec, training, selected_bindings = select_setup(spec, policy_path)
    bindings.update(selected_bindings)
    if training["profile"] != "airbot_native_tared_offline":
        raise ValueError("Manual deployment requires tared native training")
    arrays, info = load_prepared(training)
    meta = info["metadata"]
    if (meta.get("derivation") != "manual_recorded_baseline_10s_v1"
            or meta.get("demonstration_source") != "manual" or meta.get("contains_derived_samples") is not False
            or meta.get("compensation") != "recorded_unloaded_baseline_subtracted"
            or meta.get("position_frame") != "SDK configured reference"
            or meta.get("setup_confirmation", {}).get("same_setup_confirmed") is not True
            or not np.all(arrays["ft_hz"] == 100)):
        raise ValueError("Unexpected manual training input contract")
    policy_path = resolve(spec["policy"])
    if file_digest(policy_path) != spec["policy_sha256"]:
        raise ValueError("Policy hash differs from manual candidate configuration")
    policy = OfflinePolicy(policy_path)
    if (policy.source_kind != "real" or policy.hardware_ready
            or policy.metadata["data"] != info
            or policy.metadata["training_contract"] != training_contract(training)
            or policy.encoder.metadata.get("frame") != "ft_frame local with output Y/Z reversed"):
        raise ValueError("Policy provenance or input convention differs from training")
    for key, expected in {"prepared_sha256": info["sha256"], "raw_sha256": info["raw_sha256"],
                          "encoder_sha256": info["encoder_sha256"]}.items():
        if policy.metadata["bindings"].get(key) != expected:
            raise ValueError(f"Policy binding mismatch: {key}")
    session = resolve(spec["collection_session"]) / "session.json"
    exploration = resolve(spec["exploration"])
    for source, key in ((session, "collection_session_sha256"), (exploration, "exploration_sha256")):
        if file_digest(source) != spec[key] or recorded_source_hash(meta["source_hashes"], source) != spec[key]:
            raise ValueError("Manual collection/exploration binding mismatch")
    header, _ = load_completed(exploration)
    frozen = json.loads(session.read_text())["config"]
    cfg = dict(header["config"])
    for key in ("robot_sn", "expected_eef_type", "sensor_port", "sponge_id", "exploration_id"):
        if cfg[key] != frozen[key]:
            raise ValueError(f"Manual deployment setup mismatch: {key}")
    if (cfg["table_normal_sdk"] != [0, 0, 1] or cfg["slide_direction_sdk"] != [0, 1, 0]
            or cfg["expected_eef_type"] != "NULL" or cfg["sensor_bias_si"] != [0] * 6):
        raise ValueError("Manual candidate requires recorded +Z/+Y native installation")
    pose = next(r for r in events(exploration) if r["event"] == "reference_captured")["runtime_reference_pose"]
    for key, width in (("sdk_end_position_m", 3), ("sdk_end_orientation_xyzw", 4), ("joint_position_rad", 6)):
        if finite_array(pose[key], key).shape != (width,):
            raise ValueError("Invalid candidate reference pose")
    if abs(np.linalg.norm(pose["sdk_end_orientation_xyzw"]) - 1) > 0.001:
        raise ValueError("Candidate reference quaternion is not normalized")
    for key in ("joint_min_rad", "joint_max_rad"):
        cfg[key] = finite_array(frozen[key], key).tolist()
        if len(cfg[key]) != 6:
            raise ValueError("Expected six recorded joint bounds")
    if np.any(np.asarray(cfg["joint_min_rad"]) >= cfg["joint_max_rad"]):
        raise ValueError("Invalid recorded joint bounds")
    positive(cfg["joint_speed_limit_rad_s"], "joint speed", 0.4)
    positive(cfg["measured_joint_speed_stop_rad_s"], "measured joint speed", 1.2)
    if len(cfg["joint_current_limits"]) != 6:
        raise ValueError("Expected six current limits")
    for limit in cfg["joint_current_limits"]:
        positive(limit, "joint current", 8)
    for key, cap in CAPS.items():
        cfg[key] = positive(spec["guards"][key], key, cap)
    if cfg["max_initial_force_n"] > cfg["max_force_n"]:
        raise ValueError("Initial force cap must not exceed motion force cap")
    cfg.update(mode="force-guarded", max_depth_m=0.02, stationary_speed_limit_rad_s=0.1,
               tracking_error_policy="stop", orientation_error_policy="stop",
               deployment_mode=MODE, cartesian_speed_policy="record-only")
    cfg.update(torque_limit_policy="record-only", max_torque_nm=None)
    check_bounds(pose, cfg)
    normalized = policy.encoder.preprocessor.transform(arrays["exploration"])
    if np.any((normalized < 0) | (normalized > 0.9)):
        raise ValueError("Exploration outside encoder range")
    embedding = policy.encode_exploration(arrays["exploration"])
    np.testing.assert_array_equal(arrays["sponge"], np.repeat(embedding, 8, axis=0))
    bindings.update(meta["source_hashes"])
    bindings.update({str(policy_path): spec["policy_sha256"],
                     str(resolve(training["raw_data"])): info["raw_sha256"],
                     str(resolve(training["encoder"])): info["encoder_sha256"],
                     str(resolve(training["output_dir"]) / "prepared.h5"): info["sha256"]})
    bind_runtime_software(spec, policy, bindings)
    fixed.assert_unchanged(bindings)
    candidate = CandidatePolicy(policy, embedding, pose["sdk_end_position_m"], cfg["max_cartesian_speed_m_s"])
    # The task region encloses the unchanged absolute path and start/return, not the old program wipe.
    points = np.vstack((pose["sdk_end_position_m"][:2], candidate.xy))
    pose["deployment_mode"] = MODE
    pose["task_xy_bounds_sdk_m"] = [
        (points.min(axis=0) - cfg["max_tracking_error_m"]).tolist(),
        (points.max(axis=0) + cfg["max_tracking_error_m"]).tolist(),
    ]
    candidate.path_report["task_xy_bounds_sdk_m"] = pose["task_xy_bounds_sdk_m"]
    angles = [quaternion_angle(q, pose["sdk_end_orientation_xyzw"]) for q in arrays["orientation"].reshape(-1, 4)]
    candidate.path_report["demo_to_candidate_orientation_range_rad"] = [min(angles), max(angles)]
    candidate.path_report["initial_pose_source"] = "recorded_exploration_reference_not_new_onsite_acceptance"
    return spec, cfg, pose, policy, candidate, embedding, arrays, training, bindings


def replay(spec, cfg, pose, policy, candidate, embedding, arrays, training, bindings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    grid = np.arange(1001) / 100
    trajectories, delta_predictions, parity_errors, violations = [], [], [], []
    peak_force, peak_torque = 0.0, 0.0
    with h5py.File(resolve(training["raw_data"]), "r") as h5:
        for index, name in enumerate(sorted(h5["demonstrations"])):
            group = h5["demonstrations"][name]
            stamps = group["ft_time"][:] - group.attrs["start_time"]
            raw = previous_samples(stamps, group["ft_raw_before_baseline"][:], grid)
            peak_force = max(peak_force, float(np.linalg.norm(raw[:, :3], axis=1).max()))
            peak_torque = max(peak_torque, float(np.linalg.norm(raw[:, 3:], axis=1).max()))
            bias = group["recorded_unloaded_baseline"][:]
            loop = fixed.FixedSetupLoop(candidate, embedding, pose["sdk_end_position_m"])
            measured = np.asarray(pose["sdk_end_position_m"])
            targets, deltas = [], []
            for tick in range(1001):
                loop.push(tick, raw[tick] - bias, measured)
                try:
                    fixed.guard_loop(loop, cfg, pose)
                except fixed.StateError as exc:
                    violations.append({"kind": "path", "demo": name, "tick": tick, "error": str(exc)})
                if np.linalg.norm(raw[tick, :3]) > cfg["max_force_n"]:
                    violations.append({"kind": "recorded_force", "demo": name, "tick": tick,
                                       "error": "Recorded raw force exceeds deployment cap"})
                measured = loop.target(tick)
                targets.append(measured.copy())
                if loop.last_delta is not None:
                    deltas.append(loop.last_delta)
            windows, _ = make_windows(arrays["ft"][index:index + 1], arrays["height"][index:index + 1])
            expected = policy.predict_delta_h(np.repeat(embedding, 20, axis=0), windows.reshape(20, 5, 6))[:, 0]
            np.testing.assert_allclose(deltas, expected, atol=1e-8, rtol=0)
            parity_errors.append(float(np.max(np.abs(np.asarray(deltas) - expected))))
            trajectories.append(targets)
            delta_predictions.append(deltas)
    fixed.assert_unchanged(bindings)
    original = policy.predict_xy(embedding)[0]
    np.savez_compressed(output / "replay.npz", original_xy_sdk_m=original, candidate_xy_sdk_m=candidate.xy,
                        candidate_target_sdk_m=trajectories, delta_h_m=delta_predictions, time_s=grid,
                        hardware_ready=False)
    plot_path(output, original, candidate.xy, np.asarray(pose["sdk_end_position_m"]))
    orientation_supported = candidate.path_report["demo_to_candidate_orientation_range_rad"][0] <= 0.02
    if not orientation_supported:
        violations.append({"kind": "input_alignment", "error": "Candidate orientation differs from every sampled demonstration by more than 0.02 rad"})
    report = {"mode": MODE, "passed": not violations, "bindings": bindings,
              "path_limits_passed": not any(v["kind"] == "path" for v in violations),
              "recorded_force_guard_passed": not any(v["kind"] == "recorded_force" for v in violations),
              "candidate_orientation_supported_by_demo": orientation_supported,
              "recorded_peak_raw_force_n": peak_force, "recorded_peak_raw_torque_nm": peak_torque,
              "ideal_tracking_peak_cartesian_speed_m_s": float(np.linalg.norm(np.diff(np.asarray(trajectories), axis=1), axis=2).max() / fixed.DT),
              "scope": "recorded_ft_inference_parity_and_ideal_tracking_candidate_not_closed_loop",
              "original_xy_preserved_exactly": bool(np.array_equal(candidate.xy, original)),
              "cartesian_speed_stop_enforced": False, "contact_dynamics_simulated": False,
              "torque_limit_enforced": False,
              "feedback_predictions": 160, "max_stream_batch_error_m": max(parity_errors),
              "violations": violations, "path_review": candidate.path_report,
              "warnings": policy.warnings + ["original_xy_project_cartesian_speed_record_only_sdk_joint_limits_unchanged",
                                             "bootstrap_is_not_learned_from_manual_contact_demonstrations"],
              "hardware_ready": False}
    write_json(output / "report.json", report)
    return report


def blockers(spec, bindings, path_review=None):
    result = [f"Onsite/model review required: {k}" for k in CONFIRMATIONS if spec.get(k) is not True]
    if path_review is not None and path_review["demo_to_candidate_orientation_range_rad"][0] > 0.02:
        result.append("Candidate orientation is outside all sampled demonstration orientations; resolve start pose/input mapping, not confirmation flags")
    report_path = resolve(spec["replay_report"])
    if not report_path.is_file():
        result.append("Manual candidate replay report missing")
    else:
        report = json.loads(report_path.read_text())
        if report.get("mode") != MODE or report.get("passed") is not True or report.get("bindings") != bindings:
            result.append("Manual candidate replay failed or stale")
    return result


def main(args):
    client = sensor = None
    plot_job = None
    previous_signal = None
    exit_code = 2
    torch.set_num_threads(1)
    try:
        loaded = load_setup(args.config or "configs/real_deploy/airbot_manual_tared.json",
                            policy_path=args.policy)
        spec, cfg, pose, policy, candidate, embedding, arrays, training, bindings = loaded
        from scripts.real_deploy import manual_runtime
        from scripts.real_deploy.wiping_frame import WipingFrame
        runtime = spec.get("workflow") == manual_runtime.WORKFLOW
        frame = WipingFrame(args.wiping_mode, args.wall_direction)
        if frame.mode == "vertical" and not runtime:
            raise ValueError("Vertical wiping requires runtime_start_h_z_s_g_v1; legacy setup is unchanged")
        cfg.update(wiping_mode=frame.mode, wall_direction=frame.wall_direction)
        spec.update(wiping_mode=frame.mode, wall_direction=frame.wall_direction, wiping_frame=frame.metadata())
        if runtime and args.action in ("replay", "shadow"):
            raise ValueError("Runtime start is captured live with h; use preflight or run, not legacy replay/shadow")
        if args.action == "replay":
            if args.output is None:
                raise ValueError("replay requires a fresh --output directory")
            from scripts.shared.run_paths import new_output

            report = replay(*loaded, new_output(args.output))
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 2
        reasons = [] if runtime else blockers(spec, bindings, candidate.path_report)
        if args.action == "preflight":
            print(json.dumps({"mode": MODE, "offline_inputs_valid": True, "hardware_connected": False,
                              "run_enabled": not reasons, "blockers": reasons,
                              "path_review": None if runtime else candidate.path_report, "warnings": policy.warnings,
                              "workflow": spec.get("workflow", "legacy"),
                              "runtime_start_required": runtime,
                              "wiping_frame": frame.metadata(),
                              "policy": spec.get("policy"), "policy_sha256": spec.get("policy_sha256"),
                              "training_software_differences": spec.get("training_software_differences", []),
                              "automatic_startup_approach": False,
                              "torque_limit_enforced": False,
                              "tracking_error_policy": "record-only" if runtime else "stop",
                              "orientation_error_policy": "record-only" if runtime else "stop"}, indent=2))
            return 2 if reasons else 0
        # Both read-only connection and motion require a reviewed, current candidate.
        if reasons:
            raise ValueError("Manual deployment blocked: " + "; ".join(reasons))
        if not args.execute or not sys.stdin.isatty() or args.output is None:
            raise ValueError("shadow/run require --execute, attended terminal and fresh --output JSONL")
        from scripts.shared.run_paths import new_output
        from scripts.force_sensor.kwr75_reader import Kwr75Reader
        from scripts.real_deploy.recording import EventWriter

        if runtime:
            manual_runtime.warm_policy(policy, embedding)
            from scripts.real_deploy.plot_inference import AutomaticPlots, reference_from_training

            spec["plot_reference_fz"] = reference_from_training(spec, training, bindings)

        output = new_output(args.output)
        if runtime:
            plot_job = AutomaticPlots(output)
        csv = output.with_suffix(".sensor.csv")
        if output.exists() or csv.exists():
            raise FileExistsError("Manual deployment output already exists")
        fixed.assert_unchanged(bindings)
        if not runtime and blockers(spec, bindings, candidate.path_report):
            raise ValueError("Manual approvals/replay changed before connection")
        shadow = args.action == "shadow"
        if frame.mode == "vertical":
            plane = "".join("XYZ"[i] for i in sorted(frame.tangent_axes))
            print(f"Vertical wiping: wall toward SDK {frame.wall_direction.upper()}, {plane} path, "
                  f"{'XYZ'[frame.normal_axis]} feedback sign {frame.normal_sign:+d}; sensor-local FT channels unchanged.")
        print("Runtime start: h hold, z tare, s infer, g gravity compensation and exit." if runtime else (
              "No automatic startup move. Manually position at reviewed non-contact pose with support. "
              "Shadow never sends motion." if shadow else
              "No automatic startup move. At the reviewed pose, s starts tare then the candidate including 10mm press."))
        with output_lock(output.parent), (EventWriter(output) if runtime else output.open("x")) as stream:
            write_event(stream, {"event": "session_start", "mode": MODE, "shadow": shadow,
                                 "spec": spec, "bindings": bindings,
                                 "path_review": None if runtime else candidate.path_report,
                                 "warnings": policy.warnings, "hardware_ready": False})
            previous_signal = signal.signal(signal.SIGTERM, fixed.interrupt_on_signal)
            client = open_client(args.host, args.port)
            client._stub = DeadlineStub(client._stub)
            from scripts.robot_control.hardware import HardwareRobot
            robot = HardwareRobot(client, cfg) if runtime else Robot(client, cfg)
            sensor = Kwr75Reader(cfg["sensor_port"])
            sensor.start()
            sensor.start_csv(csv, background=runtime)
            time.sleep(0.1)
            if runtime:
                manual_runtime.execute(robot, sensor, policy, embedding, cfg, stream,
                                       on_policy_complete=plot_job.start)
            else:
                fixed.execute(robot, sensor, candidate, embedding, cfg, pose, stream, fixed.program.Terminal(),
                              shadow=shadow, manual_start=True)
        exit_code = 0
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; verify stop and support. No automatic recovery.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        try:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception as exc:
                    print(f"Sensor cleanup failed; CSV may be incomplete: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    if exit_code == 0:
                        exit_code = 2
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    print(f"Client cleanup failed; verify control release: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    if exit_code == 0:
                        exit_code = 2
    if plot_job is not None and not plot_job.finish() and exit_code == 0:
        exit_code = 2
    return exit_code
