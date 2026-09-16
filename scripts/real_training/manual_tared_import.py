"""Audited manual tare conversion with explicitly post-collection setup confirmation."""

import json
import re

import numpy as np

from scripts.real_training.import_airbot import received_stream
from scripts.shared.common import file_digest
from scripts.shared.real_preprocessing import finite_array

DERIVATION = "manual_recorded_baseline_10s_v1"


def validate_setup(metadata, header):
    cfg = metadata["config"]
    if (metadata.get("force_recording") != "software_tared"
            or cfg.get("force_recording") != "software_tared"
            or metadata.get("primary_force_field") != "tared_sensor_wrench_si"
            or header.get("mode") != "manual-start"):
        raise ValueError("Manual tared import requires software-tared manual demonstrations and manual-start exploration")
    if (cfg.get("demonstrations"), cfg.get("duration_s"), cfg.get("sample_hz"), cfg.get("training_hz")) != (8, 10, 100, 2.5):
        raise ValueError("Expected eight measured 10s manual demonstrations at 100 Hz")
    for key in ("robot_sn", "expected_eef_type", "sensor_port", "sponge_id", "exploration_id"):
        if not cfg.get(key) or cfg[key] != header["config"].get(key):
            raise ValueError(f"Exploration/manual setup mismatch: {key}")
    for key in ("table_normal_sdk", "slide_direction_sdk"):
        if key in cfg and cfg[key] != header["config"].get(key):
            raise ValueError(f"Exploration/manual setup mismatch: {key}")


def _force_stream(rows, *, tare=None):
    forces = [r["ft"] for r in rows]
    stamps = finite_array([f["sensor_receive_perf_s"] for f in forces], "FT timestamps")
    ages = finite_array([f["sensor_age_s"] for f in forces], "FT ages")
    observed = finite_array([r["observed_perf_s"] for r in rows], "observation timestamps")
    pose_time = finite_array([r["pose"]["host_monotonic_s"] for r in rows], "pose timestamps")
    read_duration = finite_array([r["pose"]["read_duration_s"] for r in rows], "pose read duration")
    if (np.any(np.diff(stamps) < 0) or np.any(np.diff(observed) <= 0)
            or np.any(ages < 0) or np.any(ages > 0.020)
            or np.any(observed < stamps) or np.any(observed - stamps > 0.020)
            or np.any(observed < pose_time) or np.any(observed - pose_time > 0.020)
            or np.any(read_duration < 0) or np.any(read_duration > 0.020)):
        raise ValueError("Invalid manual force/pose timing or stale observation")
    time, raw = received_stream(forces)
    if np.any(np.diff(time) > 0.020):
        raise ValueError("Manual FT receive gap exceeds 20 ms")
    if tare is not None:
        for force in forces:
            net = finite_array(force["tared_sensor_wrench_si"], "tared wrench")
            if (force.get("tare_id") != tare["id"] or net.shape != (6,)
                    or not np.allclose(np.asarray(force["raw_sensor_wrench_si"]) - tare["raw_baseline_si"],
                                       net, atol=1e-9, rtol=0)):
                raise ValueError("Manual tared force differs from raw minus recorded baseline or tare id")
    return time, raw


def validate_tare(session, raw_path, rows, record, hashes):
    if ([r.get("event") for r in rows].count("start") != 1
            or [r.get("event") for r in rows].count("finished") != 1
            or any(r.get("event") not in ("start", "sample", "finished") for r in rows)
            or rows[0].get("force_recording") != "software_tared"
            or record.get("force_recording") != "software_tared"
            or rows[-1].get("quality", {}).get("passed") is not True):
        raise ValueError("Incomplete or incompatible manual tared episode")
    tare = rows[0]["tare"]
    if (not isinstance(tare, dict) or not re.fullmatch(r"[a-f0-9]{32}", str(tare.get("id", "")))
            or tare.get("method") != "mean_raw_sensor_wrench_noncontact_1s"
            or tare.get("noncontact_confirmed_by") != "operator_z"
            or tare.get("stationary_check") is not False or record.get("tare") != tare):
        raise ValueError("Missing or inconsistent operator-confirmed manual tare")
    bias = finite_array(tare["raw_baseline_si"], "manual baseline")
    if bias.shape != (6,):
        raise ValueError("Invalid manual baseline dimensions")
    sidecars = (session / f"tare_{tare['id']}.json", raw_path.with_suffix(".start.json"))
    loaded = []
    for path in sidecars:
        if path.resolve().parent != session:
            raise ValueError("Manual tare/start paths must remain inside the session")
        fingerprint = file_digest(path)
        if hashes.setdefault(str(path), fingerprint) != fingerprint:
            raise ValueError("Manual sidecar changed during import")
        loaded.append(json.loads(path.read_text()))
    baseline, start = loaded
    if {k: v for k, v in baseline.items() if k != "samples"} != tare or start.get("tare") != tare:
        raise ValueError("Manual baseline/start sidecar differs from accepted tare")
    time, raw = _force_stream(baseline["samples"])
    if (len(time) < 40 or tare.get("unique_samples") != len(time)
            or time[-1] - time[0] < 0.9
            or not np.isclose(tare["receive_span_s"], time[-1] - time[0], atol=1e-9, rtol=0)
            or not np.allclose(raw.mean(axis=0), bias, atol=1e-9, rtol=0)):
        raise ValueError("Manual baseline count, coverage or mean does not match observations")
    samples = [r for r in rows if r["event"] == "sample"]
    sample_time, _ = _force_stream(samples, tare=tare)
    if time[-1] >= sample_time[0]:
        raise ValueError("Manual baseline must precede demonstration observations")
    return bias.tolist()


def apply_baselines(meta, exp, demos, biases, exploration_rows, cfg, exploration, session_path):
    # exploration_episode already validates the manual-start baseline and tared values.
    tare = next(r for r in exploration_rows if r["event"] == "tare_complete")
    biases[cfg["exploration_id"]] = tare["tare_bias_si"]
    for name, episode in [(cfg["exploration_id"], exp), *demos.items()]:
        bias = np.asarray(biases[name], dtype=float)
        episode["ft_raw_before_baseline"] = episode["ft"].copy()
        episode["ft"] = episode["ft"] - bias
        episode["recorded_unloaded_baseline"] = bias
    meta.update(
        derivation=DERIVATION, demonstration_source="manual", contains_derived_samples=False,
        compensation="recorded_unloaded_baseline_subtracted", recorded_baselines_si=biases,
        real_axis_processing="native sensor axes unchanged; no extra Y/Z reversal",
        simulation_comparison_convention="real tared FT vs already Y/Z-adjusted simulation output; uncalibrated",
        exploration_stationary_validated=False, exploration_force_limits_enforced=False,
        setup_confirmation={
            "same_setup_confirmed": True,
            "confirmation_stage": "offline_import_after_collection",
            "confirmed_by": "operator_explicit_confirm_same_setup",
            "scope": ["sponge", "tool_and_sensor_mounting", "table_setup"],
            "bound_before_collection": False,
            "exploration_sha256": meta["source_hashes"][str(exploration)],
            "session_sha256": meta["source_hashes"][str(session_path)],
            "exploration_id": cfg["exploration_id"],
        },
    )
