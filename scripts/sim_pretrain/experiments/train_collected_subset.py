"""Preserve collected trajectories and train a smaller, explicitly re-split dataset."""

import argparse
import fcntl
import json

import h5py
import numpy as np

from scripts.shared.common import digest, file_digest, load_config, provenance, write_json
from scripts.shared.paths import read_path, writable_path
from scripts.sim_pretrain.collection import FIELDS
from scripts.sim_pretrain.experiments.collect_wide_training import validate_numerical


def prepare(cfg, source, out):
    if cfg["collection"].get("motion_gate") is not False:
        raise ValueError("Expected explicit ungated experiment")
    if any((source / name).exists() for name in ("vae_best.pt", "vae_last.pt")):
        raise ValueError("Cannot re-split data already used for training")
    manifest = json.loads((source / "manifest.json").read_text())
    for key in ("simulation", "randomization", "filter"):
        if cfg[key] != manifest["config"][key]:
            raise ValueError(f"Source {key} mismatch")
    source_hash = file_digest(source / "dataset.h5")
    path = out / "dataset.h5"
    if path.exists():
        integrity = json.loads((out / "dataset_integrity.json").read_text())
        if file_digest(path) != integrity["sha256"]:
            raise ValueError("Subset hash mismatch")
        with h5py.File(path, "r") as h5:
            if (h5.attrs["config_hash"] != digest(cfg) or not h5.attrs["complete"]
                    or h5.attrs["source_sha256"] != source_hash):
                raise ValueError("Subset provenance mismatch")
        return
    if any(p.name != ".experiment.lock" for p in out.iterdir()):
        raise ValueError("Output is not fresh; refusing to overwrite partial artifacts")
    proof = dict(config=cfg, source_sha256=source_hash, source_manifest=manifest,
                 conversion_provenance=provenance(),
                 policy="first_complete_rows_then_seeded_disjoint_split_no_motion_filter")
    count = sum(cfg["dataset"].values())
    with h5py.File(source / "dataset.h5", "r") as original:
        if (original.attrs["config_hash"] != digest(manifest["config"])
                or json.loads(original.attrs["provenance_json"]) != manifest):
            raise ValueError("Original provenance mismatch")
        rows = [(split,int(i)) for split in manifest["config"]["dataset"]
                for i in np.flatnonzero(original[split]["valid"][:])]
        if len(rows) < count:
            raise ValueError(f"Need {count} valid rows; found {len(rows)}")
        selected = rows[:count]
        shuffled = np.random.default_rng(cfg["collection"]["split_seed"]).permutation(count)
        mappings, diagnostic_rows = {}, []
        temporary = out / "dataset.building.h5"
        with h5py.File(temporary, "x") as target:
            for key,value in original.attrs.items():
                target.attrs[key] = value
            target.attrs.update(config_hash=digest(cfg), provenance_json=json.dumps(proof),
                                source_sha256=source_hash, complete=False)
            offset = 0
            for split,size in cfg["dataset"].items():
                group = target.create_group(split)
                mapping = [selected[int(i)] for i in shuffled[offset:offset+size]]
                mappings[split] = mapping
                offset += size
                for key,shape in FIELDS.items():
                    group.create_dataset(key, shape=(size,*shape), dtype="f8", chunks=(1,*shape),
                                         compression="gzip", compression_opts=1)
                group.create_dataset("parameters",shape=(size,3),dtype="f8")
                group.create_dataset("gain",shape=(size,),dtype="f8")
                group.create_dataset("valid",data=np.ones(size,dtype=bool))
                for key in ("diagnostics", "error", "source_split"):
                    group.create_dataset(key,shape=(size,),dtype=h5py.string_dtype())
                group.create_dataset("source_index",shape=(size,),dtype="i8")
                for i,(source_split,index) in enumerate(mapping):
                    src = original[source_split]
                    data = {key:src[key][index] for key in FIELDS}
                    validate_numerical(data)
                    for key,value in data.items():
                        group[key][i] = value
                    for key in ("parameters", "gain", "diagnostics", "error"):
                        group[key][i] = src[key][index]
                    group["source_split"][i], group["source_index"][i] = source_split,index
                    diagnostic_rows.append(json.loads(src["diagnostics"].asstr()[index]))
            target.attrs["complete"] = True
        temporary.replace(path)
    if file_digest(source / "dataset.h5") != source_hash:
        raise ValueError("Source changed during conversion")
    write_json(out / "manifest.json",proof)
    write_json(out / "source_mapping.json",mappings)
    write_json(out / "dataset_integrity.json",dict(sha256=file_digest(path)))
    write_json(out / "collection.json",dict(complete=True,total=count,valid_counts=cfg["dataset"],
               source_valid_rows=len(rows),motion_gate_disabled=True,
               motion_passed=sum(r["motion"]["passed"] for r in diagnostic_rows),
               any_saturation=sum(r["metrics"]["saturation_fraction"]>0 for r in diagnostic_rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/sim_pretrain/pretrain_wide_1200.yaml")
    cfg = load_config(parser.parse_args(argv).config)
    source = read_path(cfg["collection"]["source"])
    out = writable_path(cfg["output_dir"])
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Output must not overlap source")
    out.mkdir(parents=True,exist_ok=True)
    with (source / ".experiment.lock").open("r") as source_lock, (out / ".experiment.lock").open("a+") as lock:
        fcntl.flock(source_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(cfg,source,out)
    from scripts.sim_pretrain.learning import train,evaluate
    with (out / ".experiment.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out / "vae_last.pt").exists():
            write_json(out / "progress.json",dict(stage="training",epochs=cfg["training"]["epochs"]))
            train(cfg,resume=True)
        result = evaluate(cfg)
        write_json(out / "progress.json",dict(stage="complete",epochs=cfg["training"]["epochs"],evaluation=result))
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
