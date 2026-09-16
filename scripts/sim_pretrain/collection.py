"""Physics gate and resumable collection with immutable parameter assignments."""

import json

import h5py
import numpy as np

from scripts.shared.common import (
    CHANNELS,
    digest,
    file_digest,
    output_dir,
    provenance,
    sample_parameters,
    write_json,
)
from scripts.shared.paths import writable_path
from scripts.shared.run_layout import sim_data_path
from scripts.sim_pretrain.acceptance import contact_motion_acceptance


def tracking_passes(metrics, cfg):
    s = cfg["simulation"]
    return (
        metrics["position_rms_m"] <= s["position_rms_limit"]
        and metrics["orientation_max_deg"] <= s["orientation_limit_degrees"]
        and metrics["contact_fraction"] == 0
    )


def sanity(cfg):
    from scripts.sim_pretrain.simulation import (
        PretrainingWipe,
        parameter_grid,
        rollout_metrics,
    )

    out = output_dir(cfg)
    traces = sim_data_path(out, "sanity_traces")
    traces.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": False,
        "config_hash": digest(cfg),
        "provenance": provenance(),
        "selected_gain": None,
        "tracking": [],
        "grid": [],
        "scope": "simulation only; contact-parameter reproduction, not calibrated material stiffness",
    }
    path = sim_data_path(out, "sanity.json")
    write_json(path, report)
    for gain in cfg["simulation"]["gain_candidates"]:
        env = None
        row = {"gain": gain, "passed": False}
        try:
            env = PretrainingWipe(cfg, gain)
            data = env.rollout(extra_height=0.08)
            np.savez_compressed(traces / f"free_gain_{gain}.npz", **data)
            row["metrics"] = rollout_metrics(data)
            row["passed"] = tracking_passes(row["metrics"], cfg)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                env.close()
        report["tracking"].append(row)
        print(json.dumps(row), flush=True)
        write_json(path, report)
        if row["passed"]:
            report["selected_gain"] = gain
            break
    if report["selected_gain"] is None:
        report["blocker"] = (
            "All configured gains failed free-space tracking; collection is prohibited."
        )
        write_json(path, report)
        return report

    for index, parameters in enumerate(parameter_grid(cfg)):
        env = None
        row = {"parameters": list(parameters), "passed": False}
        try:
            env = PretrainingWipe(cfg, report["selected_gain"], parameters)
            data = env.rollout()
            np.savez_compressed(traces / f"contact_{index:02d}.npz", **data)
            row["metrics"] = rollout_metrics(data)
            # Weak forces are valid at the soft/low-friction limits; require contact geometry, not a force threshold.
            row["contact_motion"] = contact_motion_acceptance(data)
            if not np.any(data["contact_count"] > 0):
                raise RuntimeError("No tool/table contact during exploration")
            row["phase_mean_local_wrench"] = [
                data["ft"][a:b].mean(axis=0).tolist() for a, b in ((0, 200), (200, 300), (300, 400))
            ]
            row["unloaded_wrench"] = env.unload(data)
            if not row["contact_motion"]["passed"]:
                raise RuntimeError(
                    f"Contact motion failed: {row['contact_motion']['failed_checks']}"
                )
            row["passed"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                env.close()
        report["grid"].append(row)
        print(json.dumps(row), flush=True)
        write_json(path, report)
    from scripts.sim_pretrain.diagnostics import sensor_load_check

    env = None
    try:
        env = PretrainingWipe(cfg, report["selected_gain"])
        report["sensor_check"] = sensor_load_check(env)
    except Exception as exc:
        report["sensor_check"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if env is not None:
            env.close()
    report["passed"] = all(r["passed"] for r in report["grid"]) and report["sensor_check"]["passed"]
    if not report["passed"]:
        report["blocker"] = "Contact scan or independent sensor load check failed."
    write_json(path, report)
    return report


def require_sanity(cfg):
    path = output_dir(cfg) / "sanity.json"
    if not path.exists():
        raise RuntimeError("Run sanity before collection")
    report = json.loads(path.read_text())
    if report["config_hash"] != digest(cfg) or report["provenance"] != provenance():
        raise RuntimeError(
            "Sanity report is stale: config, source or environment changed; rerun sanity"
        )
    if not report["passed"] or report["selected_gain"] is None:
        raise RuntimeError(report.get("blocker", "Physics sanity gate has not passed"))
    return report


FIELDS = {
    "ft": (400, 6),
    "ft_raw": (400, 6),
    "position": (400, 3),
    "target_position": (400, 3),
    "quat_xyzw": (400, 4),
    "sensor_quat_xyzw": (400, 4),
    "joints": (400, 6),
    "time": (400,),
    "contact_count": (400,),
    "saturated": (400, 6),
    "orientation_error": (400,),
    "initial_bias": (6,),
}


def validate_trajectory(data):
    for key, shape in FIELDS.items():
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid trajectory field {key}: {value.shape}")
    if not np.allclose(data["time"], np.arange(1, 401) / 100, rtol=0, atol=1e-8):
        raise ValueError("Invalid sampling timestamps")
    if not np.any(data["contact_count"] > 0):
        raise ValueError("Exploration never contacts table")


def collect(cfg, retry_failed=False):
    writable_path(output_dir(cfg))
    report = require_sanity(cfg)
    from scripts.sim_pretrain.simulation import (
        PretrainingWipe, FT_OUTPUT_FRAME, FT_OUTPUT_SIGNS, FT_PROCESSING,
    )

    out = output_dir(cfg)
    proof = {"config": cfg, "sanity": report}
    proof_hash = digest(proof)
    with h5py.File(sim_data_path(out, "dataset.h5"), "a") as h5:
        if "config_hash" not in h5.attrs:
            h5.attrs.update(
                config_hash=digest(cfg),
                provenance_hash=proof_hash,
                channels=json.dumps(CHANNELS),
                frame=FT_OUTPUT_FRAME,
                raw_frame="ft_frame local",
                sensor_quat_frame="raw ft_frame local",
                ft_output_signs=json.dumps(FT_OUTPUT_SIGNS),
                ft_processing=FT_PROCESSING,
                units=json.dumps(["N"] * 3 + ["N*m"] * 3),
                complete=False,
            )
            h5.attrs["provenance_json"] = json.dumps(proof)
        if h5.attrs.get("ft_processing") != FT_PROCESSING:
            raise RuntimeError("Dataset FT processing mismatch; use a new output directory")
        if h5.attrs["config_hash"] != digest(cfg) or h5.attrs["provenance_hash"] != proof_hash:
            raise RuntimeError("Dataset provenance mismatch; do not mix configurations")
        # Fix all assignments before starting the first simulation.
        for split_index, (split, size) in enumerate(cfg["dataset"].items()):
            if split in h5:
                continue
            group = h5.create_group(split)
            group.create_dataset("parameters", data=sample_parameters(cfg, split_index, size))
            group.create_dataset("valid", shape=(size,), dtype="bool")
            group.create_dataset("status", shape=(size,), dtype="i1")
            group.create_dataset("error", shape=(size,), dtype=h5py.string_dtype())
            group.attrs["seed_sequence"] = [cfg["simulation"]["seed"], split_index]
            for key, shape in FIELDS.items():
                group.create_dataset(
                    key,
                    shape=(size, *shape),
                    dtype="f8",
                    fillvalue=np.nan,
                    chunks=(1, *shape),
                    compression="gzip",
                    compression_opts=1,
                )
        h5.flush()
        for split, size in cfg["dataset"].items():
            group = h5[split]
            env = None
            try:
                for index in range(size):
                    status = int(group["status"][index])
                    if group["valid"][index] or (status == -1 and not retry_failed):
                        continue
                    group["status"][index] = 1
                    h5.flush()
                    try:
                        if env is None:
                            env = PretrainingWipe(cfg, report["selected_gain"])
                        env.set_parameters(group["parameters"][index])
                        data = env.rollout()
                        validate_trajectory(data)
                        acceptance = contact_motion_acceptance(data)
                        if not acceptance["passed"]:
                            raise RuntimeError(
                                f"Contact motion failed: {acceptance['failed_checks']}"
                            )
                        env.unload(data)
                        for key in FIELDS:
                            group[key][index] = data[key]
                        group["error"][index] = ""
                        h5.flush()
                        group["valid"][index] = True
                        group["status"][index] = 2
                    except Exception as exc:
                        group["status"][index] = -1
                        group["error"][index] = f"{type(exc).__name__}: {exc}"
                        if env is not None:
                            env.close()
                            env = None
                    h5.flush()
                    print(
                        f"{split} {index + 1}/{size}: status={int(group['status'][index])}",
                        flush=True,
                    )
            finally:
                if env is not None:
                    env.close()
        failures = [
            {
                "split": split,
                "index": i,
                "status": int(h5[split]["status"][i]),
                "parameters": h5[split]["parameters"][i].tolist(),
                "error": h5[split]["error"].asstr()[i],
            }
            for split, count in cfg["dataset"].items()
            for i in range(count)
            if not h5[split]["valid"][i]
        ]
        h5.attrs["complete"] = not failures
        write_json(out / "collection_failures.json", failures)
        result = {
            "complete": not failures,
            "failed_or_pending": len(failures),
            "valid_counts": {split: int(h5[split]["valid"][:].sum()) for split in cfg["dataset"]},
        }
        write_json(sim_data_path(out, "collection.json"), result)
    if result["complete"]:
        write_json(sim_data_path(out, "dataset_integrity.json"), {"sha256": file_digest(out / "dataset.h5")})
    return result
