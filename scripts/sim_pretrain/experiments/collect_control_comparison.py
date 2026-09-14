"""Paired simulation collection with explicitly record-only contact acceptance."""

import argparse
import copy
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import (
    CHANNELS,
    digest,
    file_digest,
    load_config,
    provenance,
    sample_parameters,
    write_json,
)
from scripts.shared.paths import ROOT, read_path, writable_path
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import FIELDS, validate_trajectory
from scripts.sim_pretrain.experiments.contact_control_experiment import (
    CANDIDATES,
    ExperimentWipe,
)
from scripts.sim_pretrain.experiments.contact_control_experiment import (
    sources as controller_sources,
)
from scripts.sim_pretrain.simulation import contact_parameters

MODES = {"normal": None, "impedance": CANDIDATES["cart_6000"]}
EXTRA_FIELDS = {
    "contact_wrench": (400, 6),
    "normal_sum": (400,),
    "requested_torque": (400, 6),
    "applied_torque": (400, 6),
    "nominal_twist": (400, 6),
    "contact_count_loaded": (400,),
}
POLICY = "user_authorized_contact_acceptance_record_only"


class CollectionWipe(ExperimentWipe):
    def contacts(self):
        # Same checks at the same physics substeps, vectorized across contacts.
        contact = self.sim.data._data.contact
        pairs = contact.geom
        expected = np.any(pairs == self.table_id, axis=1) & np.any(
            np.isin(pairs, self.tool_ids), axis=1
        )
        friction, solref, solimp = contact_parameters(*self.parameters)
        if np.any(expected):
            if not np.allclose(contact.solref[expected], solref) or not np.allclose(
                contact.solimp[expected], solimp
            ):
                raise RuntimeError(
                    "Effective contact solver parameters differ from requested parameters"
                )
            if not np.all(np.isclose(contact.friction[expected, 0], max(friction[0], 1e-5))):
                raise RuntimeError("Effective sliding friction differs from requested friction")
        unexpected = pairs[(~expected) & (contact.dist < -1e-5)]
        if len(unexpected):
            names = [
                [self.sim.model.geom_id2name(a), self.sim.model.geom_id2name(b)]
                for a, b in unexpected[:3]
            ]
            raise RuntimeError(f"Non-tool contact: {names}")
        return int(expected.sum())


def experiment_provenance():
    return {
        "production": provenance(),
        "sources": {
            **controller_sources(),
            str(Path(__file__).relative_to(ROOT)): file_digest(__file__),
        },
        "paper_sha256": file_digest(ROOT / "docs/references/Adaptive_wiping.pdf"),
    }


def initialize(out, base_cfg):
    out = writable_path(out)
    out.mkdir(parents=True, exist_ok=True)
    assignments = {
        split: sample_parameters(base_cfg, index, count).tolist()
        for index, (split, count) in enumerate(base_cfg["dataset"].items())
    }
    manifest = {
        "schema_version": 1,
        "policy": POLICY,
        "base_config": base_cfg,
        "controllers": MODES,
        "assignments": assignments,
        "assignments_sha256": digest(assignments),
        "provenance": experiment_provenance(),
        "scope": "AIRBOT simulation comparison; motion failures retained by user request; not passed wiping acceptance",
        "paper_sections": ["III-A", "IV-C1", "IV-D"],
        "engineering_additions": [
            "AIRBOT and direct_wrist_v3 instead of UR5e",
            "IK+FF300 versus Cartesian impedance cart_6000",
            "100 validation and 100 test trajectories beyond 1000 training",
            "uniform independent parameter sampling and current contact mapping",
            "batch32, seed42, second-order 10Hz zero-phase filtering",
        ],
    }
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Comparison manifest/config/source changed; use a new directory")
    else:
        write_json(path, manifest)
    configs = {}
    for name, spec in MODES.items():
        folder = out / name
        folder.mkdir(exist_ok=True)
        cfg = copy.deepcopy(base_cfg)
        cfg["output_dir"] = str(folder)
        cfg["research_comparison"] = {
            "controller": name,
            "spec": spec,
            "policy": POLICY,
            "normal_gain": 300,
            "manifest_sha256": digest(manifest),
        }
        config_path = folder / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != cfg:
                raise ValueError("Existing per-mode configuration mismatch")
        else:
            write_json(config_path, cfg)
        configs[name] = cfg
    return manifest, configs


def initialize_dataset(h5, cfg, manifest):
    if "config_hash" in h5.attrs:
        if h5.attrs["config_hash"] != digest(cfg) or h5.attrs["manifest_hash"] != digest(manifest):
            raise ValueError("Dataset provenance mismatch")
        if h5.attrs.get("contact_acceptance_policy") != POLICY:
            raise ValueError("Dataset acceptance policy mismatch")
        for split, parameters in manifest["assignments"].items():
            np.testing.assert_array_equal(h5[split]["parameters"][:], parameters)
        return
    h5.attrs.update(
        config_hash=digest(cfg),
        manifest_hash=digest(manifest),
        complete=False,
        contact_acceptance_policy=POLICY,
        controller=cfg["research_comparison"]["controller"],
        valid_meaning="complete finite sampled data, contact parameter/collision checks and unloading; NOT contact-motion acceptance",
        channels=json.dumps(CHANNELS),
        frame="ft_frame local",
        units=json.dumps(["N"] * 3 + ["N*m"] * 3),
        provenance_json=json.dumps(manifest["provenance"]),
    )
    for split, count in cfg["dataset"].items():
        group = h5.create_group(split)
        group.create_dataset("parameters", data=manifest["assignments"][split])
        for name, dtype in (
            ("valid", "bool"),
            ("motion_passed", "bool"),
            ("unloaded", "bool"),
            ("status", "i1"),
            ("attempts", "i4"),
        ):
            group.create_dataset(name, shape=(count,), dtype=dtype)
        for name in ("error", "acceptance_json"):
            group.create_dataset(name, shape=(count,), dtype=h5py.string_dtype())
        for name, shape in {**FIELDS, **EXTRA_FIELDS, "unloaded_wrench": (6,)}.items():
            group.create_dataset(
                name,
                shape=(count, *shape),
                dtype="f8",
                fillvalue=np.nan,
                chunks=(1, *shape),
                compression="gzip",
                compression_opts=1,
            )
    h5.flush()


def summarize(h5, cfg, elapsed):
    splits = {}
    for split, count in cfg["dataset"].items():
        group = h5[split]
        valid = group["valid"][:]
        splits[split] = {
            "requested": count,
            "recorded": int(valid.sum()),
            "motion_passed": int(np.sum(group["motion_passed"][:] & valid)),
            "failed": int(np.sum(group["status"][:] == -1)),
            "pending": int(np.sum(~valid & (group["status"][:] != -1))),
        }
    return {
        "controller": cfg["research_comparison"]["controller"],
        "policy": POLICY,
        "complete": all(row["recorded"] == row["requested"] for row in splits.values()),
        "splits": splits,
        "current_invocation_elapsed_s": elapsed,
        "note": "Recorded counts are not passed-contact-acceptance counts",
    }


def collect_mode(
    cfg,
    manifest,
    record_only=False,
    max_trajectories=None,
    retry_failed=False,
    factory=CollectionWipe,
):
    writable_path(cfg["output_dir"])
    if not record_only:
        raise ValueError("Explicit record-only contact gate authorization is required")
    folder = read_path(cfg["output_dir"])
    path = folder / "dataset.h5"
    mode = cfg["research_comparison"]["controller"]
    started, attempted, env = time.monotonic(), 0, None
    with (folder / ".collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with h5py.File(path, "a") as h5:
            initialize_dataset(h5, cfg, manifest)
            try:
                for split, count in cfg["dataset"].items():
                    group = h5[split]
                    for index in range(count):
                        if group["valid"][index] or (
                            group["status"][index] == -1 and not retry_failed
                        ):
                            continue
                        if max_trajectories is not None and attempted >= max_trajectories:
                            break
                        group["status"][index] = 1
                        group["attempts"][index] += 1
                        h5.attrs["complete"] = False
                        h5.flush()
                        attempted += 1
                        try:
                            if env is None:
                                env = factory(cfg, tuple(group["parameters"][index]), MODES[mode])
                            env.set_parameters(group["parameters"][index])
                            data = env.rollout()
                            validate_trajectory(data)
                            for name, shape in EXTRA_FIELDS.items():
                                if data[name].shape != shape or not np.isfinite(data[name]).all():
                                    raise ValueError(f"Invalid recorded field: {name}")
                            np.testing.assert_array_equal(
                                data["parameters"], group["parameters"][index]
                            )
                            acceptance = contact_motion_acceptance(data)
                            # Preserve complete rollouts even if unloading subsequently fails.
                            for name in {**FIELDS, **EXTRA_FIELDS}:
                                group[name][index] = data[name]
                            group["motion_passed"][index] = acceptance["passed"]
                            group["acceptance_json"][index] = json.dumps(acceptance)
                            unloaded = np.asarray(env.unload(data))
                            if unloaded.shape != (6,) or not np.isfinite(unloaded).all():
                                raise ValueError("Invalid unloaded wrench")
                            group["unloaded_wrench"][index] = unloaded
                            group["unloaded"][index] = True
                            group["error"][index] = ""
                            h5.flush()
                            group["valid"][index] = True
                            group["status"][index] = 2
                        except Exception as exc:
                            group["status"][index] = -1
                            group["error"][index] = f"{type(exc).__name__}: {exc}"
                            failure = {
                                "split": split,
                                "index": index,
                                "parameters": group["parameters"][index].tolist(),
                                "attempt": int(group["attempts"][index]),
                                "error": group["error"].asstr()[index],
                            }
                            with (folder / "failure_history.jsonl").open("a") as stream:
                                stream.write(json.dumps(failure) + "\n")
                            print(mode, "FAILED", failure, flush=True)
                            if env is not None:
                                env.close()
                                env = None
                        h5.flush()
                        result = summarize(h5, cfg, time.monotonic() - started)
                        write_json(folder / "collection.json", result)
                        if attempted == 1 or attempted % 10 == 0:
                            print(
                                mode, f"{split} {index + 1}/{count}", result["splits"], flush=True
                            )
            finally:
                if env is not None:
                    env.close()
            result = summarize(h5, cfg, time.monotonic() - started)
            h5.attrs["complete"] = result["complete"]
            write_json(folder / "collection.json", result)
        if result["complete"]:
            write_json(
                folder / "dataset_integrity.json",
                {
                    "sha256": file_digest(path),
                    "config_hash": digest(cfg),
                    "manifest_hash": digest(manifest),
                    "policy": POLICY,
                },
            )
        print(mode, "COLLECTION FINISHED", result, flush=True)
        return result


def verify(out, manifest, configs):
    rows = {}
    for name, cfg in configs.items():
        folder = read_path(cfg["output_dir"])
        path = folder / "dataset.h5"
        if not path.exists():
            raise ValueError(f"Dataset missing: {name}")
        with h5py.File(path, "r") as h5:
            assert h5.attrs["config_hash"] == digest(cfg)
            assert h5.attrs["manifest_hash"] == digest(manifest)
            assert h5.attrs["contact_acceptance_policy"] == POLICY
            for split, count in cfg["dataset"].items():
                group = h5[split]
                np.testing.assert_array_equal(
                    group["parameters"][:], manifest["assignments"][split]
                )
                for index in np.flatnonzero(group["valid"][:]):
                    data = {key: group[key][index] for key in FIELDS}
                    validate_trajectory(data)
                    acceptance = contact_motion_acceptance(data)
                    assert acceptance == json.loads(group["acceptance_json"].asstr()[index])
                    assert acceptance["passed"] == bool(group["motion_passed"][index])
                    assert group["unloaded"][index] and group["status"][index] == 2
                    for key, shape in EXTRA_FIELDS.items():
                        assert (
                            group[key][index].shape == shape
                            and np.isfinite(group[key][index]).all()
                        )
            rows[name] = summarize(h5, cfg, 0)
            assert bool(h5.attrs["complete"]) == rows[name]["complete"]
        if rows[name]["complete"]:
            assert json.loads((folder / "dataset_integrity.json").read_text())[
                "sha256"
            ] == file_digest(path)
    result = {
        "verified": True,
        "complete": all(row["complete"] for row in rows.values()),
        "paired_parameter_assignments": True,
        "policy": POLICY,
        "modes": rows,
        "source_unchanged": experiment_provenance() == manifest["provenance"],
    }
    if not result["source_unchanged"]:
        raise ValueError("Source changed during collection")
    write_json(Path(out) / "collection_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["both", *MODES], default="both")
    parser.add_argument("--record-only-contact-gate", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        help="Attempt at most this many new trajectories per selected mode; resume later",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.record_only_contact_gate:
        parser.error("This research collection requires --record-only-contact-gate")
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("--max-trajectories must be positive")
    manifest, configs = initialize(args.output, load_config(args.config))
    if not args.verify_only:
        names = list(MODES) if args.mode == "both" else [args.mode]
        if args.workers == 2 and len(names) == 2:
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        collect_mode,
                        configs[name],
                        manifest,
                        True,
                        args.max_trajectories,
                        args.retry_failed,
                    )
                    for name in names
                ]
                for future in futures:
                    future.result()
        else:
            for name in names:
                collect_mode(
                    configs[name], manifest, True, args.max_trajectories, args.retry_failed
                )
    if args.mode == "both" or args.verify_only:
        result = verify(args.output, manifest, configs)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["complete"] else 2
    return (
        0
        if json.loads((Path(configs[args.mode]["output_dir"]) / "collection.json").read_text())[
            "complete"
        ]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
