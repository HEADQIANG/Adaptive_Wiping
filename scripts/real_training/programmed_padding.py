"""Explicit, audited last-state extension of accepted program demonstrations."""

import json
from pathlib import Path

import numpy as np

from scripts.real_training.airbot_programmed_demonstrations import (
    PROTOCOL, accepted_records, duration,
)
from scripts.real_training.import_airbot import (
    events, exploration_episode, import_metadata, received_stream,
)
from scripts.shared.common import file_digest
from scripts.shared.sampling import previous_samples

DERIVATION = "programmed_hold_last_10s_v1"
GRID = np.arange(1001, dtype=np.float64) / 100


def padded_episode(rows, record, cfg):
    start_event = next(r for r in rows if r["event"] == "episode_start")
    start = start_event["start_perf_s"]
    seconds = duration(record["condition"])
    samples = [r for r in rows if r.get("phase") in ("press", "slide") and r["event"] == "sample"]
    steps = round(seconds * 100)
    if [r["index"] for r in samples] != list(range(steps + 1)):
        raise ValueError("Missing or reordered effective wipe samples")
    pose_time = np.array([r["pose"]["host_monotonic_s"] - start for r in samples])
    position = np.array([r["pose"]["sdk_end_position_m"] for r in samples])
    quaternion = np.array([r["pose"]["sdk_end_orientation_xyzw"] for r in samples])
    ft_time, ft = received_stream([r["ft"] for r in samples])
    ft_time -= start
    if (np.any(np.diff(pose_time) <= 0) or np.max(np.diff(pose_time)) > 0.020 + 1e-9
            or np.max(np.diff(ft_time)) > 0.020 + 1e-9):
        raise ValueError("Gaps in measured wipe data cannot be filled by endpoint padding")
    measured_grid = GRID[:steps + 1]
    # Causal alignment is limited to the real segment. Only the explicit tail
    # may exceed the 20 ms sample-age gate, and it is marked as derived.
    measured_position = previous_samples(pose_time, position, measured_grid)
    measured_quaternion = previous_samples(pose_time, quaternion, measured_grid)
    measured_ft = previous_samples(ft_time, ft, measured_grid)
    count = len(GRID) - len(measured_grid)
    def extend(values, final):
        return np.concatenate((values, np.repeat(np.asarray(final)[None, :], count, axis=0)))
    padding = GRID > seconds + 1e-9
    pose_sources = previous_samples(pose_time, pose_time, measured_grid)
    ft_sources = previous_samples(ft_time, ft_time, measured_grid)
    attrs = {"sponge_id": cfg["sponge_id"], "surface_id": cfg["surface_id"],
             "exploration_id": cfg["exploration_id"], "complete": True, "start_time": 0.0,
             "ft_hz": 100.0, "pose_hz": 100.0, "condition": record["condition"],
             "initial_depth_m": record["initial_depth_m"], "measured_duration_s": seconds,
             "padding_duration_s": 10 - seconds, "derivation": DERIVATION,
             "demonstration_source": "programmed", "contains_derived_samples": True}
    episode = {"attrs": attrs, "ft_time": GRID, "pose_time": GRID,
               "ft": extend(measured_ft, samples[-1]["ft"]["raw_sensor_wrench_si"]),
               "sdk_end_position": extend(measured_position, position[-1]),
               "sdk_end_quaternion": extend(measured_quaternion, quaternion[-1]),
               "is_padding": padding.astype(float),
               "pose_source_time": np.r_[pose_sources, np.repeat(pose_time[-1], count)],
               "ft_source_time": np.r_[ft_sources, np.repeat(ft_time[-1], count)]}
    report = {"condition": record["condition"], "measured_duration_s": seconds,
              "padded_duration_s": 10 - seconds, "padded_samples": count,
              "measured_grid_samples": len(measured_grid), "total_grid_samples": len(GRID),
              "original_start_perf_s": start, "quality": record["quality"],
              "endpoint_pose_relative_s": float(pose_time[-1]),
              "endpoint_ft_relative_s": float(ft_time[-1]),
              "padding_endpoint": "last press/slide observation; excludes postroll and retraction"}
    return episode, report


def assemble_padded(exploration, session, *, subtract_recorded_baseline=False):
    exploration, session = Path(exploration).resolve(), Path(session).resolve()
    manifest = session / "session.json"
    hashes = {str(p): file_digest(p) for p in (manifest, exploration)}
    frozen = json.loads(manifest.read_text())
    if frozen.get("protocol") != PROTOCOL or frozen.get("demonstration_source") != "programmed":
        raise ValueError("Explicit hold-last import requires fixed-depth programmed protocol")
    cfg = frozen["config"]
    manifests = sorted(session.glob("demo_*.json"))
    hashes.update({str(p): file_digest(p) for p in manifests})
    records = accepted_records(session)
    if len(records) != 8:
        raise ValueError("Exactly eight accepted program demonstrations are required")
    exp, exp_report, header = exploration_episode(exploration)
    for key in ("robot_sn", "sensor_port", "initial_pose_sha256", "table_normal_sdk", "slide_direction_sdk"):
        if not cfg.get(key) or cfg[key] != header["config"].get(key):
            raise ValueError(f"Exploration/program setup mismatch: {key}")
    demos, reports = {}, {}
    for i, record in enumerate(records, 1):
        path = session / record["raw_file"]
        hashes[str(path)] = record["sha256"]
        name = f"demo_{i:02d}"
        demos[name], reports[name] = padded_episode(events(path), record, cfg)
        reports[name].update(raw_file=str(path), raw_sha256=record["sha256"])
    meta = import_metadata(cfg, hashes, exp_report, reports)
    meta.update(derivation=DERIVATION, demonstration_source="programmed",
                contains_derived_samples=True, padding_method="hold_last_effective_wipe_state",
                padding_semantics="synthetic tail from measured endpoint; not observed 10s behavior",
                force_feedback_enabled=False, paper_equivalent_collection=False)
    if subtract_recorded_baseline:
        exp_events = events(exploration)
        tares = [r for r in exp_events if r["event"] == "tare_complete"]
        if len(tares) != 1 or tares[0].get("distinct_samples", 0) < 40:
            raise ValueError("Exploration requires one recorded unloaded baseline")
        biases = {cfg["exploration_id"]: tares[0]["tare_bias_si"]}
        for i, record in enumerate(records, 1):
            name = f"demo_{i:02d}"
            tare_events = [r for r in events(session / record["raw_file"]) if r["event"] == "tare_complete"]
            if (len(tare_events) != 1 or tare_events[0]["tare_bias_si"] != record["baseline_si"]
                    or tare_events[0].get("distinct_samples", 0) < 40):
                raise ValueError("Accepted demonstration baseline differs from successful recorded tare")
            biases[name] = record["baseline_si"]
        for name, episode in [(cfg["exploration_id"], exp), *demos.items()]:
            bias = np.asarray(biases[name], dtype=float)
            if bias.shape != (6,) or not np.isfinite(bias).all():
                raise ValueError("Invalid recorded baseline")
            episode["ft_raw_before_baseline"] = episode["ft"].copy()
            episode["ft"] = episode["ft"] - bias
            episode["recorded_unloaded_baseline"] = bias
        meta.update(compensation="recorded_unloaded_baseline_subtracted", recorded_baselines_si=biases,
                    real_axis_processing="native sensor axes unchanged; no extra Y/Z reversal",
                    simulation_comparison_convention="real tared FT vs already Y/Z-adjusted simulation output; uncalibrated")
    if any(file_digest(path) != expected for path, expected in hashes.items()):
        raise ValueError("Source changed during padded import")
    return meta, {cfg["exploration_id"]: exp}, demos
