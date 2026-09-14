"""Parallel normal rollouts feeding the unchanged single-writer collector."""

import argparse
import fcntl
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import file_digest, load_config, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    EXTRA_FIELDS,
    CollectionWipe,
    collect_mode,
    experiment_provenance,
    initialize,
)

_worker_cfg = None
_worker_env = None


def _close_worker():
    global _worker_env
    if _worker_env is not None:
        _worker_env.close()
        _worker_env = None


def _init_worker(cfg):
    global _worker_cfg
    _worker_cfg = cfg
    Finalize(None, _close_worker, exitpriority=10)


def _simulate(parameters):
    global _worker_env
    try:
        if _worker_env is None:
            _worker_env = CollectionWipe(_worker_cfg, parameters, None)
        _worker_env.set_parameters(parameters)
        data = _worker_env.rollout()
    except Exception:
        _close_worker()
        raise
    try:
        unloaded = _worker_env.unload(data)
    except Exception as exc:
        _close_worker()
        # The original writer must still preserve the full rollout before unload fails.
        return {"data": data, "unloaded": None, "unload_error": f"{type(exc).__name__}: {exc}"}
    return {"data": data, "unloaded": unloaded, "unload_error": None}


class RolloutProxy:
    def __init__(self, provider, parameters):
        self.provider = provider
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        self.parameters = tuple(float(value) for value in parameters)

    def rollout(self):
        self.result = self.provider.take(self.parameters)
        np.testing.assert_array_equal(self.result["data"]["parameters"], self.parameters)
        return self.result["data"]

    def unload(self, data):
        if self.result["unload_error"]:
            raise RuntimeError(self.result["unload_error"])
        return self.result["unloaded"]

    def close(self):
        pass


class OrderedRollouts:
    def __init__(self, cfg, manifest, workers, executor_factory=ProcessPoolExecutor):
        if not 1 <= workers <= 8:
            raise ValueError("workers must be between1 and8")
        self.parameters = [tuple(row) for rows in manifest["assignments"].values() for row in rows]
        self.indices = {row: index for index, row in enumerate(self.parameters)}
        if len(self.indices) != len(self.parameters):
            raise ValueError("Duplicate parameter assignment")
        self.window = 2 * workers
        self.jobs = {}
        self.pool = executor_factory(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(cfg,),
        )

    def take(self, parameters):
        index = self.indices[parameters]
        for old in [key for key in self.jobs if key < index]:
            self.jobs.pop(old).cancel()
        for pending in range(index, min(index + self.window, len(self.parameters))):
            if pending not in self.jobs:
                self.jobs[pending] = self.pool.submit(_simulate, self.parameters[pending])
        return self.jobs.pop(index).result()

    def factory(self, cfg, parameters, spec):
        if spec is not None:
            raise ValueError("Parallel runner supports normal mode only")
        return RolloutProxy(self, parameters)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def verify_pilot(out, cfg, manifest, workers, count):
    from scripts.shared.paths import read_path

    folder = read_path(cfg["output_dir"])
    with (folder / ".collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / "dataset.h5"
        before = file_digest(path)
        with h5py.File(path, "r") as h5:
            indices = np.flatnonzero(h5["train/valid"][:])[:count]
            if count < 1 or len(indices) != count:
                raise ValueError("Not enough committed training records for requested pilot")
            rows = [
                {
                    key: h5[f"train/{key}"][index]
                    for key in (*FIELDS, *EXTRA_FIELDS, "parameters", "unloaded_wrench")
                }
                for index in indices
            ]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            futures = [pool.submit(_simulate, tuple(row["parameters"])) for row in rows]
            for expected, future in zip(rows, futures):
                actual = future.result()
                if actual["unload_error"]:
                    raise RuntimeError(actual["unload_error"])
                for key in (*FIELDS, *EXTRA_FIELDS, "parameters"):
                    np.testing.assert_array_equal(expected[key], actual["data"][key])
                np.testing.assert_array_equal(expected["unloaded_wrench"], actual["unloaded"])
        assert before == file_digest(path)
        assert experiment_provenance() == manifest["provenance"]
        report = {
            "verified": True,
            "train_indices": indices.tolist(),
            "workers": workers,
            "all_recorded_fields_and_unloading_exact": True,
            "dataset_unchanged": True,
            "dataset_sha256": before,
            "parallel_source_sha256": file_digest(__file__),
        }
        write_json(Path(out) / "parallel_pilot_verification.json", report)
        return report


def collect_parallel(out, cfg, manifest, workers=6, max_trajectories=None):
    if max_trajectories is not None and max_trajectories < 1:
        raise ValueError("max_trajectories must be positive")
    source_hash = file_digest(__file__)
    path = Path(out) / "parallel_collection.json"
    audit = (
        json.loads(path.read_text())
        if path.exists()
        else {"source_sha256": source_hash, "invocations": []}
    )
    if audit["source_sha256"] != source_hash:
        raise ValueError("Parallel source changed; do not mix unverified implementations")
    entry = {
        "state": "running",
        "workers": workers,
        "max_trajectories": max_trajectories,
        "single_writer": "unchanged collect_mode",
        "batch_close_interval": 50,
    }
    audit["invocations"].append(entry)
    with (Path(out) / ".parallel.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(path, audit)
        provider = OrderedRollouts(cfg, manifest, workers)
        attempted = 0
        try:
            while True:
                count = 50 if max_trajectories is None else min(50, max_trajectories - attempted)
                result = collect_mode(cfg, manifest, True, count, factory=provider.factory)
                attempted += count
                failed = sum(row["failed"] for row in result["splits"].values())
                if (
                    result["complete"]
                    or failed
                    or (max_trajectories is not None and attempted >= max_trajectories)
                ):
                    break
            entry.update(
                state="completed" if result["complete"] else "incomplete", collection=result
            )
        except BaseException as exc:
            entry.update(state="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            provider.close()
            entry["sources_unchanged"] = (
                source_hash == file_digest(__file__)
                and experiment_provenance() == manifest["provenance"]
            )
            write_json(path, audit)
        if not entry["sources_unchanged"]:
            raise ValueError("Source changed during collection")
        return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper_mu1p2.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--record-only-contact-gate", action="store_true")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument(
        "--verify-pilot",
        type=int,
        help="Replay this many committed training rows; no dataset writes",
    )
    args = parser.parse_args()
    if not args.record_only_contact_gate:
        parser.error("Explicit --record-only-contact-gate authorization is required")
    manifest, configs = initialize(args.output, load_config(args.config))
    cfg = configs["normal"]
    if args.verify_pilot is not None:
        result = verify_pilot(args.output, cfg, manifest, args.workers, args.verify_pilot)
        code = 0
    else:
        result = collect_parallel(args.output, cfg, manifest, args.workers, args.max_trajectories)
        code = 0 if result["state"] == "completed" else 2
    print(json.dumps(result, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
