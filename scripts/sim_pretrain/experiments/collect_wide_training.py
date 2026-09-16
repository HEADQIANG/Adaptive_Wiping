"""Explicitly ungated experiment: retain motion diagnostics, validate numerical data."""

import argparse
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

from scripts.shared.common import CHANNELS, digest, file_digest, load_config, provenance, write_json
from scripts.shared.paths import writable_path
from scripts.sim_pretrain.collection import FIELDS
from scripts.shared.run_layout import sim_data_path
from scripts.sim_pretrain.experiments.match_real_amplitude import TIMES, random_grid


def assignments(cfg):
    bounds = cfg["randomization"]
    result = {}
    for i, (split, count) in enumerate(cfg["dataset"].items()):
        result[split] = random_grid(count, cfg["simulation"]["seed"]+i,
                                   [bounds[k] for k in ("gain", "friction", "stiffness_direct", "width")])
    return result


def validate_numerical(data):
    for key, shape in FIELDS.items():
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid trajectory field {key}")
    if not np.allclose(data["time"], TIMES, rtol=0, atol=1e-8):
        raise ValueError("Invalid sampling timestamps")


def simulate(job):
    from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics
    from scripts.sim_pretrain.acceptance import contact_motion_acceptance

    cfg, split, index, parameters = job
    env = None
    try:
        env = PretrainingWipe(cfg, parameters[0], parameters[1:])
        data = env.rollout()
        validate_numerical(data)
        return split, index, data, dict(metrics=rollout_metrics(data),
                                       motion=contact_motion_acceptance(data)), None
    except Exception as exc:
        return split, index, None, None, f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()


def collect(cfg, out, proof):
    from scripts.sim_pretrain.simulation import FT_OUTPUT_FRAME, FT_OUTPUT_SIGNS, FT_PROCESSING

    plan = assignments(cfg)
    with h5py.File(sim_data_path(out, "dataset.h5"), "a") as h5:
        if "config_hash" not in h5.attrs:
            h5.attrs.update(config_hash=digest(cfg), provenance_json=json.dumps(proof), complete=False,
                            channels=json.dumps(CHANNELS), frame=FT_OUTPUT_FRAME,
                            ft_processing=FT_PROCESSING, ft_output_signs=json.dumps(FT_OUTPUT_SIGNS),
                            raw_frame="ft_frame local", sensor_quat_frame="raw ft_frame local",
                            units=json.dumps(["N"]*3+["N*m"]*3), motion_gate_disabled=True,
                            valid_semantics="complete_finite_trajectory_not_motion_acceptance")
        if h5.attrs["config_hash"] != digest(cfg) or json.loads(h5.attrs["provenance_json"]) != proof:
            raise ValueError("Dataset provenance mismatch")
        for split, rows in plan.items():
            if split not in h5:
                group = h5.create_group(split)
                group.create_dataset("parameters", data=np.asarray(rows)[:,1:])
                group.create_dataset("gain", data=np.asarray(rows)[:,0])
                group.create_dataset("valid", shape=(len(rows),), dtype="bool")
                group.create_dataset("diagnostics", shape=(len(rows),), dtype=h5py.string_dtype())
                group.create_dataset("error", shape=(len(rows),), dtype=h5py.string_dtype())
                for key, shape in FIELDS.items():
                    group.create_dataset(key, shape=(len(rows), *shape), dtype="f8", fillvalue=np.nan,
                                         chunks=(1, *shape), compression="gzip", compression_opts=1)
            np.testing.assert_array_equal(h5[split]["gain"][:], np.asarray(rows)[:,0])
            np.testing.assert_array_equal(h5[split]["parameters"][:], np.asarray(rows)[:,1:])
        jobs = [(cfg, split, i, row) for split, rows in plan.items()
                for i, row in enumerate(rows) if not h5[split]["valid"][i]]
        done = sum(int(h5[s]["valid"][:].sum()) for s in plan)
        failures = []
        total = sum(cfg["dataset"].values())
        h5.flush()
        with ProcessPoolExecutor(max_workers=cfg["collection"]["workers"],
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            # Bound outstanding results so one slow rollout cannot buffer the whole dataset.
            for start in range(0, len(jobs), 16):
                for split, index, data, diagnostics, error in pool.map(simulate, jobs[start:start+16]):
                    group = h5[split]
                    if error:
                        group["error"][index] = error
                        failures.append(dict(split=split, index=index, error=error))
                    else:
                        for key in FIELDS:
                            group[key][index] = data[key]
                        group["diagnostics"][index] = json.dumps(diagnostics)
                        group["error"][index] = ""
                        h5.flush()
                        group["valid"][index] = True
                        done += 1
                    h5.flush()
                    write_json(out / "progress.json", dict(stage="collection", complete=done, total=total,
                                                           failures=failures))
                    print(f"Collected {done}/{total}: {split}[{index}] error={error}", flush=True)
        complete = all(np.all(h5[s]["valid"][:]) for s in plan)
        diagnostics = [json.loads(v) for s in plan for v in h5[s]["diagnostics"].asstr()[:] if v]
        summary = dict(complete=bool(complete), total=done, motion_gate_disabled=True,
                       motion_passed=sum(r["motion"]["passed"] for r in diagnostics),
                       any_saturation=sum(r["metrics"]["saturation_fraction"]>0 for r in diagnostics),
                       valid_counts={s:int(h5[s]["valid"][:].sum()) for s in plan}, failures=failures)
        h5.attrs["complete"] = bool(complete)
        write_json(sim_data_path(out, "collection.json"), summary)
    if not complete:
        raise RuntimeError("Numerically invalid/missing trajectories; rerun to retry same assignments")
    write_json(sim_data_path(out, "dataset_integrity.json"), dict(sha256=file_digest(out / "dataset.h5")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_wide_2000.yaml")
    parser.add_argument("--resume", action="store_true", help="Continue the exact config output_dir, including legacy runs")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.get("collection", {}).get("motion_gate") is not False:
        raise ValueError("This experiment requires explicit collection.motion_gate: false")
    if cfg["collection"].get("sampling") != "latin_hypercube_mu_uniform_others_log":
        raise ValueError("Unsupported sampling policy")
    if not 1 <= cfg["collection"]["workers"] <= 4:
        raise ValueError("workers must be in [1,4]")
    from scripts.shared.run_paths import new_run_config

    cfg = new_run_config(cfg, resume=args.resume)
    out = writable_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".experiment.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = dict(config=cfg, provenance=provenance(), policy="user_authorized_no_motion_gate")
        manifest = out / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != proof:
                raise ValueError("Config/source changed; use a new output directory")
        else:
            if any(p.name != ".experiment.lock" for p in out.iterdir()):
                raise ValueError("Output is not a fresh experiment directory")
            write_json(manifest, proof)
            write_json(sim_data_path(out, "assignments.json"), assignments(cfg))
        collect(cfg, out, proof)
        from scripts.sim_pretrain.learning import train, evaluate

        if not (out / "vae_last.pt").exists():
            write_json(out / "progress.json", dict(stage="training", epochs=cfg["training"]["epochs"]))
            train(cfg, resume=True)
        evaluation = evaluate(cfg)
        write_json(out / "progress.json", dict(stage="complete", epochs=cfg["training"]["epochs"],
                                               evaluation=evaluation))
        print(f"Completed collection and training: {out}", flush=True)


if __name__ == "__main__":
    main()
