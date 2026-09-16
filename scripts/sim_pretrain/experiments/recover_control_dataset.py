"""Rebuild an interrupted comparison HDF5 file, retaining a byte-for-byte backup."""

import argparse
import fcntl
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import file_digest, load_config, write_json
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import FIELDS, validate_trajectory
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    EXTRA_FIELDS,
    MODES,
    initialize,
    initialize_dataset,
    summarize,
)


def read_text(dataset, index):
    return dataset.asstr()[index]


def rebuild(source, destination, cfg, manifest):
    initialize_dataset(destination, cfg, manifest)
    initialize_dataset(source, cfg, manifest)
    if bool(source.attrs["complete"]):
        raise ValueError("Recovery is only for interrupted, incomplete datasets")
    if set(source) != set(destination):
        raise ValueError("Unexpected source groups")
    destination.attrs.update(dict(source.attrs))
    unreadable = []
    for split in cfg["dataset"]:
        old, new = source[split], destination[split]
        if set(old) != set(new):
            raise ValueError(f"Unexpected source fields: {split}")
        valid = old["valid"][:]
        for key in old:
            if key not in ("error", "acceptance_json"):
                values = old[key][:]
                new[key][:] = values
                np.testing.assert_array_equal(new[key][:], values)
                continue
            for index in range(len(valid)):
                try:
                    value = read_text(old[key], index)
                except OSError as exc:
                    if valid[index]:
                        raise ValueError(
                            f"Committed string is unreadable: {split}/{key}/{index}"
                        ) from exc
                    unreadable.append(
                        {
                            "split": split,
                            "index": index,
                            "field": key,
                            "read_error": str(exc),
                            "replacement": "",
                            "parameters": old["parameters"][index].tolist(),
                            "attempts": int(old["attempts"][index]),
                        }
                    )
                    value = ""
                new[key][index] = value
                assert read_text(new[key], index) == value
        for index in np.flatnonzero(valid):
            data = {key: new[key][index] for key in FIELDS}
            validate_trajectory(data)
            acceptance = contact_motion_acceptance(data)
            assert acceptance == json.loads(read_text(new["acceptance_json"], index))
            assert acceptance["passed"] == bool(new["motion_passed"][index])
            assert new["unloaded"][index] and new["status"][index] == 2
            for key, shape in {**EXTRA_FIELDS, "unloaded_wrench": (6,)}.items():
                value = new[key][index]
                assert value.shape == shape and np.isfinite(value).all()
    destination.flush()
    return unreadable


def recover(out, mode, base_cfg):
    manifest, configs = initialize(out, base_cfg)
    cfg = configs[mode]
    from scripts.shared.paths import read_path

    folder = read_path(cfg["output_dir"])
    from scripts.shared.run_layout import sim_data_path

    path = sim_data_path(folder, "dataset.h5")
    with (folder / ".collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery = Path(tempfile.mkdtemp(prefix="recovery-", dir=folder))
        backup, candidate = recovery / "original.h5", recovery / "rebuilt.h5"
        shutil.copy2(path, backup)
        original_hash = file_digest(backup)
        assert file_digest(path) == original_hash
        with h5py.File(backup, "r") as source, h5py.File(candidate, "w") as destination:
            unreadable = rebuild(source, destination, cfg, manifest)
            progress = summarize(destination, cfg, 0)
        report = {
            "mode": mode,
            "backup": str(backup),
            "backup_sha256": original_hash,
            "rebuilt_sha256": file_digest(candidate),
            "unreadable_uncommitted_strings": unreadable,
            "all_numeric_fields_unchanged": True,
            "committed_rows_verified": True,
            "collection": progress,
            "note": "Only unreadable uncommitted strings replaced; status, attempts and assignments unchanged",
        }
        write_json(recovery / "recovery.json", report)
        # The old file stays recoverable, and replacement happens only after all checks.
        assert file_digest(path) == original_hash
        candidate.replace(path)
        write_json(sim_data_path(folder, "collection.json"), progress)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--record-only-contact-gate", action="store_true")
    args = parser.parse_args()
    if not args.record_only_contact_gate:
        parser.error("Recovery requires --record-only-contact-gate for this research dataset")
    recover(args.output, args.mode, load_config(args.config))
