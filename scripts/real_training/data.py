"""Versioned raw logs, causal sampling and immutable prepared-data provenance."""

import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.real_training.config import (
    native_profile,
    output_lock,
    preparation_contract,
    resolve,
    validate_config,
)
from scripts.shared.common import CHANNELS, digest, file_digest, write_json
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.exploration_protocol import PRESS_DURATION_S, PRESS_SPEED_M_S
from scripts.shared.real_preprocessing import (
    CausalFTFilter,
    finite_array,
    make_windows,
    out_of_range,
)
from scripts.shared.sampling import previous_samples

SCHEMA_VERSION = 1
FT_UNITS = ["N"] * 3 + ["N*m"] * 3
POLICY_TIME = np.arange(1, 26, dtype=np.float64) * 0.4
EXPLORATION_TIME = np.arange(1, 401, dtype=np.float64) / 100
PROTOCOL = {
    "press_speed_m_s": PRESS_SPEED_M_S,
    "press_duration_s": PRESS_DURATION_S,
    "lateral_speed_m_s": 0.05,
    "lateral_duration_each_s": 1.0,
}


def _completed(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def write_raw_log(path, metadata, explorations, demonstrations, *, source_kind):
    """Write a new file. Each episode is a dict with attrs and numeric arrays.

    This is a serialization helper, not evidence of calibration or hardware safety.
    The inspect command performs the full training-readiness validation.
    """
    if source_kind not in ("real", "synthetic"):
        raise ValueError("source_kind must explicitly be real or synthetic")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as h5:
        h5.attrs.update(
            schema_version=SCHEMA_VERSION,
            source_kind=source_kind,
            complete=False,
            metadata=json.dumps(metadata, allow_nan=False),
        )
        for category, episodes in (
            ("explorations", explorations),
            ("demonstrations", demonstrations),
        ):
            parent = h5.create_group(category)
            for episode_id, episode in episodes.items():
                if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
                    raise ValueError(
                        "Episode IDs must contain only letters, digits, underscores and hyphens"
                    )
                group = parent.create_group(episode_id)
                group.attrs.update(episode["attrs"])
                group.attrs["source_kind"] = source_kind
                for name, value in episode.items():
                    if name != "attrs":
                        group.create_dataset(name, data=np.asarray(value, dtype=np.float64))
        h5.attrs["complete"] = True


def _metadata(meta, cfg):
    expected = {
        "position_frame": "base_link",
        "position_units": "m",
        "orientation_order": "xyzw",
        "clock": "shared_monotonic_seconds",
        "channels": CHANNELS,
        "ft_units": FT_UNITS,
        "ft_frame": cfg["wrench"]["frame"],
        "ft_reference_point": cfg["wrench"]["reference_point"],
        "compensation": cfg["wrench"]["compensation"],
        "exploration_protocol": PROTOCOL,
    }
    if native_profile(cfg):
        expected.update(
            position_frame="SDK configured reference",
            calibration_status="unverified",
            resampling="causal_latest_received_100hz_max_age_20ms",
        )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Incompatible metadata.{key}; expected {value!r}")
    for key in ("robot_id", "sensor_id", "calibration_id", "tcp_definition", "sensor_frame"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"Missing metadata.{key}")
    if native_profile(cfg):
        if "sensor_to_ft_frame" in meta:
            raise ValueError(
                "Native logs must not claim an unmeasured sensor-to-simulation transform"
            )
        return
    transform = finite_array(meta.get("sensor_to_ft_frame"), "sensor_to_ft_frame")
    if (
        transform.shape != (4, 4)
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("sensor_to_ft_frame must be a rigid right-handed transform")


def _stream(group, time_key, data_key, width, start, duration, hz, cfg):
    time = finite_array(group[time_key][:], time_key)
    data = finite_array(group[data_key][:], data_key)
    if time.ndim != 1 or len(time) < 2 or data.shape != (len(time), width):
        raise ValueError(f"Invalid {group.name}/{data_key} stream dimensions")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError(f"{group.name}/{time_key} must be strictly increasing")
    minimum_hz = 50 if native_profile(cfg) and time_key == "ft_time" else 100
    if not np.isfinite(hz) or hz < minimum_hz:
        raise ValueError(f"Nominal sample rates must be at least {minimum_hz} Hz")
    if abs(np.median(dt) * hz - 1) > 0.1:
        raise ValueError(
            f"{group.name}/{time_key} differs from nominal rate by more than 10 percent"
        )
    relative = time - start
    if relative[0] > 1e-9 or relative[-1] < duration - 1e-9:
        raise ValueError(f"{group.name}/{time_key} does not cover [start,start+{duration}]")
    # Preserve the last sample at/before the segment start; do not borrow pre-roll state.
    first = max(0, np.searchsorted(relative, 1e-9, side="right") - 1)
    last = min(len(time), np.searchsorted(relative, duration, side="left") + 1)
    relative, data = relative[first:last], data[first:last]
    if np.any(np.diff(relative) > cfg["processing"]["max_gap_s"] + 1e-9):
        raise ValueError(f"{group.name}/{time_key} has a data gap exceeding 20 ms")
    if -relative[0] > cfg["processing"]["max_age_s"] + 1e-9:
        raise ValueError(f"{group.name}/{time_key} starts with a stale sample")
    return relative, data


def _load_raw(cfg, *, testing=False):
    path = resolve(cfg["raw_data"])
    raw_hash = file_digest(path)
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_version") != SCHEMA_VERSION or not _completed(
            h5.attrs.get("complete")
        ):
            raise ValueError("Raw log has an unsupported schema or is incomplete")
        source = h5.attrs.get("source_kind")
        if source != ("synthetic" if testing else "real"):
            raise ValueError(
                "Formal training requires real data; synthetic provenance is test-only"
            )
        meta = json.loads(h5.attrs["metadata"])
        _metadata(meta, cfg)
        if len(h5["explorations"]) != 1 or len(h5["demonstrations"]) != 8:
            raise ValueError("Expected exactly one Normal exploration and eight demonstrations")
        exp_id = next(iter(h5["explorations"]))
        arrays, ids, surfaces = {}, [], set()
        for category in ("explorations", "demonstrations"):
            episodes = []
            for episode_id, group in sorted(h5[category].items()):
                attrs = group.attrs
                if not _completed(attrs.get("complete")) or attrs.get("source_kind") != source:
                    raise ValueError(f"Incomplete or inconsistent episode: {episode_id}")
                if attrs.get("sponge_id") != "normal":
                    raise ValueError("Paper downstream data must use sponge_id=normal")
                if cfg["profile"] == "airbot_native_tared_offline":
                    bias = finite_array(group["recorded_unloaded_baseline"][:], "recorded baseline")
                    raw_ft = finite_array(group["ft_raw_before_baseline"][:], "preserved raw FT")
                    if (bias.shape != (6,) or raw_ft.shape != group["ft"].shape
                            or not np.array_equal(bias, meta["recorded_baselines_si"][episode_id])
                            or not np.array_equal(raw_ft - bias, group["ft"][:])):
                        raise ValueError("Tared data differs from preserved raw FT and recorded baseline")
                start = float(attrs["start_time"])
                if not np.isfinite(start):
                    raise ValueError("Invalid episode start_time")
                demo = category == "demonstrations"
                duration = 10.0 if demo else 4.0
                ft_hz = float(attrs["ft_hz"])
                time, ft = _stream(group, "ft_time", "ft", 6, start, duration, ft_hz, cfg)
                if not demo:
                    episodes.append(previous_samples(time, ft, EXPLORATION_TIME))
                    continue
                if attrs.get("exploration_id") != exp_id:
                    raise ValueError(f"Unknown exploration association for {episode_id}")
                surface = attrs.get("surface_id")
                if not isinstance(surface, str) or not surface.strip():
                    raise ValueError("Missing demonstration surface_id")
                surfaces.add(surface)
                pose_hz = float(attrs["pose_hz"])
                position_key = "sdk_end_position" if native_profile(cfg) else "tcp_position"
                quaternion_key = "sdk_end_quaternion" if native_profile(cfg) else "tcp_quaternion"
                pose_time, position = _stream(
                    group, "pose_time", position_key, 3, start, duration, pose_hz, cfg
                )
                _, quaternion = _stream(
                    group, "pose_time", quaternion_key, 4, start, duration, pose_hz, cfg
                )
                if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-3, rtol=0):
                    raise ValueError("TCP orientation must use unit xyzw quaternions")
                p = cfg["processing"]
                if native_profile(cfg):
                    grid = np.arange(1001, dtype=np.float64) / 100
                    ft = previous_samples(time, ft, grid)
                    time, ft_hz = grid, 100.0
                filtered = CausalFTFilter(ft_hz, p["filter_order"], p["filter_cutoff_hz"]).process(
                    ft
                )
                sampled_pose = previous_samples(pose_time, position, POLICY_TIME)
                initial_h = previous_samples(pose_time, position, [0.0])[0, 2]
                episodes.append(
                    {
                        "ft": previous_samples(time, filtered, POLICY_TIME),
                        "xy": sampled_pose[:, :2],
                        "height": sampled_pose[:, 2],
                        "initial_height": initial_h,
                        "delta_h": np.diff(np.r_[initial_h, sampled_pose[:, 2]]),
                        "orientation": previous_samples(pose_time, quaternion, POLICY_TIME),
                        "ft_hz": ft_hz,
                    }
                )
                if meta.get("derivation") == "programmed_hold_last_10s_v1":
                    grid = np.arange(1001, dtype=np.float64) / 100
                    measured_duration = float(attrs["measured_duration_s"])
                    mask = group["is_padding"][:]
                    expected = grid > measured_duration + 1e-9
                    if (measured_duration not in (5.6, 6.0, 6.4) or mask.shape != grid.shape
                            or not np.array_equal(mask, expected.astype(float))
                            or attrs.get("derivation") != meta["derivation"]):
                        raise ValueError("Invalid explicit hold-last padding mask")
                    for key in ("ft", position_key, quaternion_key):
                        values = group[key][:]
                        if values.shape[0] != 1001 or not np.all(values[expected] == values[expected][0]):
                            raise ValueError("Padded tail must hold its endpoint constant")
                    episodes[-1]["is_padding"] = (POLICY_TIME > measured_duration + 1e-9).astype(float)
                ids.append(episode_id)
            if demo:
                for key in episodes[0]:
                    arrays[key] = np.asarray([item[key] for item in episodes])
            else:
                arrays["exploration"] = np.asarray(episodes)
        if len(surfaces) != 1:
            raise ValueError("The eight training demonstrations must share one fixed surface_id")
    if file_digest(path) != raw_hash:
        raise RuntimeError("Raw log changed while being read")
    return arrays, {
        "raw_sha256": raw_hash,
        "source_kind": source,
        "metadata": meta,
        "demo_ids": ids,
        "exploration_id": exp_id,
        "surface_id": next(iter(surfaces)),
    }


def encoder_source(cfg):
    torch.set_num_threads(cfg["training"]["threads"])
    path = resolve(cfg["encoder"])
    fingerprint = file_digest(path)
    encoder = FrozenSpongeEncoder(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("source_kind") == "synthetic" and cfg["profile"] != "synthetic_test":
        raise ValueError("Synthetic encoder cannot enter formal training")
    meta = encoder.metadata
    expected_frame = "ft_frame local" if native_profile(cfg) else cfg["wrench"]["frame"]
    if cfg["profile"] == "airbot_native_tared_offline":
        expected_frame = "ft_frame local with output Y/Z reversed"
    if (
        meta.get("architecture") != "frame6to5_flat2000_posterior10_split5"
        or meta.get("channels") != CHANNELS
        or meta.get("units") != FT_UNITS
        or meta.get("frame")
        != expected_frame
    ):
        raise ValueError("Incompatible encoder architecture, channels, units or frame")
    prep = meta["preprocessing"]
    if (prep["sample_hz"], prep["order"], prep["cutoff_hz"]) != (100, 2, 10.0):
        raise ValueError("Expected the existing encoder's 100 Hz, order-2, 10 Hz preprocessing")
    minimum = finite_array(prep["minimum"], "encoder minimum")
    maximum = finite_array(prep["maximum"], "encoder maximum")
    if (
        minimum.shape != (6,)
        or maximum.shape != (6,)
        or np.any(maximum < minimum)
        or not np.array_equal(np.asarray(prep["constant"]), maximum - minimum < 1e-8)
    ):
        raise ValueError("Invalid encoder normalization statistics")
    if any(not torch.isfinite(x).all() for x in encoder.model.state_dict().values()):
        raise ValueError("Non-finite encoder weights")
    for key, weight in encoder.model.state_dict().items():
        if not torch.equal(weight, payload["encoder"][key]):
            raise RuntimeError("Encoder changed while being loaded")
    if file_digest(path) != fingerprint:
        raise RuntimeError("Encoder file changed while being read")
    return encoder, payload, fingerprint


def _has_quality_warning(value):
    if isinstance(value, dict):
        return any(
            (key in ("passed", "latent_utilization_pass", "reconstruction_pass") and item is False)
            or (key == "collapse_warning" and item is True)
            or _has_quality_warning(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_quality_warning(item) for item in value)
    return False


def inspect(cfg, *, testing=False):
    validate_config(cfg, testing=testing)
    missing = [
        f"{key}: {resolve(cfg[key])}"
        for key in ("raw_data", "encoder")
        if not resolve(cfg[key]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing training inputs: " + "; ".join(missing))
    arrays, info = _load_raw(cfg, testing=testing)
    encoder, payload, fingerprint = encoder_source(cfg)
    embedding = encoder.encode(arrays["exploration"])
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite exploration embedding")
    arrays["sponge"] = np.repeat(embedding, 8, axis=0)
    normalized = encoder.preprocessor.transform(arrays["exploration"])
    warnings = [
        "offline_only_no_hardware_validation",
        "one_sponge_does_not_identify_conditional_motion_generalization",
    ]
    if native_profile(cfg):
        warnings.extend(
            [
                "UNCALIBRATED_AIRBOT_NATIVE_FRAME_OFFLINE_EXPERIMENT",
                "sensor_to_simulation_frame_unknown_embedding_not_physically_validated",
                "received_ft_about_56hz_causal_hold_to_100hz_not_100hz_independent_measurements",
                "SDK_end_not_calibrated_sponge_TCP",
                ("recorded_unloaded_baseline_removed_not_full_gravity_or_frame_calibration"
                 if cfg["profile"] == "airbot_native_tared_offline"
                 else "raw_electronic_bias_and_gravity_retained"),
            ]
        )
    if info["metadata"].get("contains_derived_samples"):
        warnings.extend([
            "PROGRAMMED_HOLD_LAST_TAIL_IS_DERIVED_NOT_MEASURED",
            "10s_shape_compatibility_is_not_paper_equivalent_collection",
            "padded_tail_included_in_training_and_metrics_can_favor_zero_height_change",
            "fixed_depth_demonstrations_do_not_demonstrate_force_feedback_corrections",
        ])
    if _has_quality_warning(payload.get("evaluation", {})):
        warnings.append("source_encoder_quality_warning")
    if np.any((normalized < 0) | (normalized > 0.9)):
        warnings.append("exploration_outside_encoder_normalization_range_not_clipped")
    randomization = payload.get("config", {}).get("randomization", {})
    if payload.get("epoch") != 200 or randomization.get("friction") != [0.0, 3.5]:
        warnings.append("pretraining_differs_from_paper")
    if testing:
        warnings.append("SYNTHETIC_SOFTWARE_TEST_ONLY")
    windows, _ = make_windows(arrays["ft"], arrays["height"])
    info.update(
        {
            "schema_version": SCHEMA_VERSION,
            "encoder_sha256": fingerprint,
            "preparation_hash": digest(preparation_contract(cfg)),
            "preparation_contract": preparation_contract(cfg),
            "warnings": warnings,
            "encoder_epoch": payload.get("epoch"),
            "encoder_randomization": randomization,
            "encoder_evaluation": payload.get("evaluation", {}),
            "exploration_out_of_range_per_channel": out_of_range(normalized),
            "valid_windows_per_demo": windows.shape[1],
            "valid_windows_total": int(np.prod(windows.shape[:2])),
            "policy_time_s": POLICY_TIME.tolist(),
            "exploration_time_s": EXPLORATION_TIME.tolist(),
        }
    )
    return arrays, info


def prepare(cfg, *, testing=False):
    arrays, info = inspect(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    with output_lock(out):
        path = out / "prepared.h5"
        if path.exists() or (out / "prepared_integrity.json").exists():
            raise FileExistsError("Prepared data already exists; use a new output directory")
        temp = out / "prepared.h5.tmp"
        with h5py.File(temp, "w") as h5:
            h5.attrs["info"] = json.dumps(info, allow_nan=False)
            for key, value in arrays.items():
                h5.create_dataset(key, data=value)
        temp.replace(path)
        write_json(out / "prepared_integrity.json", {"sha256": file_digest(path), **info})
    return info


def load_prepared(cfg, *, testing=False):
    validate_config(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    path = out / "prepared.h5"
    missing = [
        str(p)
        for p in (
            resolve(cfg["raw_data"]),
            resolve(cfg["encoder"]),
            path,
            out / "prepared_integrity.json",
        )
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing inputs; run inspect and prepare first: " + "; ".join(missing)
        )
    info = json.loads((out / "prepared_integrity.json").read_text())
    if file_digest(path) != info["sha256"]:
        raise ValueError("Prepared data content hash mismatch")
    if file_digest(resolve(cfg["raw_data"])) != info["raw_sha256"]:
        raise ValueError("Raw data changed since preparation")
    if file_digest(resolve(cfg["encoder"])) != info["encoder_sha256"]:
        raise ValueError("Encoder changed since preparation")
    if digest(preparation_contract(cfg)) != info["preparation_hash"]:
        raise ValueError("Preparation configuration mismatch")
    if info["source_kind"] != ("synthetic" if testing else "real"):
        raise ValueError("Synthetic data cannot enter formal training")
    with h5py.File(path, "r") as h5:
        embedded = json.loads(h5.attrs["info"])
        if embedded != {key: value for key, value in info.items() if key != "sha256"}:
            raise ValueError("Prepared metadata and integrity record differ")
        arrays = {key: finite_array(value[:], key) for key, value in h5.items()}
    expected = {
        "exploration": (1, 400, 6),
        "sponge": (8, 5),
        "ft": (8, 25, 6),
        "xy": (8, 25, 2),
        "height": (8, 25),
        "delta_h": (8, 25),
        "initial_height": (8,),
        "orientation": (8, 25, 4),
        "ft_hz": (8,),
    }
    if info["metadata"].get("derivation") == "programmed_hold_last_10s_v1":
        expected["is_padding"] = (8, 25)
    if set(arrays) != set(expected) or any(
        arrays[key].shape != shape for key, shape in expected.items()
    ):
        raise ValueError("Prepared array dimensions do not match the paper contract")
    if "is_padding" in expected:
        reports = info["metadata"]["collection_report"]["demonstrations"]
        masks = np.array([POLICY_TIME > reports[name]["measured_duration_s"] + 1e-9
                          for name in info["demo_ids"]], dtype=float)
        if not np.array_equal(arrays["is_padding"], masks):
            raise ValueError("Prepared padding mask differs from provenance")
    return arrays, info
