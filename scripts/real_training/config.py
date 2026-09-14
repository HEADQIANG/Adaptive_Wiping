"""Independent configuration and provenance for offline real-data learning."""

import copy
import fcntl
import json
import platform
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import yaml

from scripts.shared.common import ROOT, digest, file_digest
from scripts.shared.paths import read_path, writable_path


def native_profile(cfg):
    return cfg["profile"] in ("airbot_native_offline", "airbot_native_tared_offline")


def load_config(path):
    with read_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    return cfg


def resolve(path):
    return read_path(path)


def validate_config(cfg, *, testing=False):
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema_version")
    for key in ("raw_data", "encoder", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"Missing configuration path: {key}")
    out = resolve(cfg["output_dir"]).resolve()
    for key in ("raw_data", "encoder"):
        source = resolve(cfg[key]).resolve()
        if source == out or out in source.parents or source.parent == out:
            raise ValueError("Input files must be outside the output directory")
    if out in (ROOT.resolve(), Path("/")):
        raise ValueError("Use a dedicated output subdirectory")
    if cfg["profile"] not in (
        ("synthetic_test",) if testing else ("paper_downstream", "airbot_native_offline", "airbot_native_tared_offline")
    ):
        raise ValueError("Unknown real training profile; synthetic tests are separate")
    p = cfg["processing"]
    if p != {"max_age_s": 0.02, "max_gap_s": 0.02, "filter_order": 2, "filter_cutoff_hz": 1.0}:
        raise ValueError("This profile fixes the documented processing defaults")
    expected = cfg["wrench"]
    frame, origin = (
        ("sensor local", "sensor origin")
        if native_profile(cfg)
        else ("ft_frame local", "ft_frame origin")
    )
    if expected["frame"] != frame or expected["reference_point"] != origin:
        raise ValueError("Convert FT to the documented local frame and origin before importing")
    compensation = (
        "raw_sensor_load_retained" if native_profile(cfg) else "sensor_bias_only_gravity_retained"
    )
    if cfg["profile"] == "airbot_native_tared_offline":
        compensation = "recorded_unloaded_baseline_subtracted"
    if expected["compensation"] != compensation:
        raise ValueError(
            "The current encoder requires gravity-retaining FT, without per-episode zeroing"
        )
    t = cfg["training"]
    if t["device"] != "cpu" or t["threads"] != 1:
        raise ValueError("The reproducible v1 trainer supports CPU with one thread")
    if t["learning_rate"] != 0.001 or t["xy_batch_size"] != 8 or t["ft_batch_size"] != 32:
        raise ValueError("Unexpected optimizer or batch settings for this profile")
    for key in ("xy_epochs", "ft_epochs", "checkpoint_every", "seed"):
        if type(t[key]) is not int or t[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid training.{key}")
    if not testing and (t["xy_epochs"], t["ft_epochs"]) != (10000, 2000):
        raise ValueError("Formal training requires 10000 XY and 2000 FT epochs")


def preparation_contract(cfg):
    return {
        key: copy.deepcopy(cfg[key])
        for key in ("schema_version", "profile", "processing", "wrench")
    }


def training_contract(cfg):
    return {**preparation_contract(cfg), "training": copy.deepcopy(cfg["training"])}


def provenance():
    sources = list(Path(__file__).parent.rglob("*.py"))
    sources += list((ROOT / "scripts/shared").rglob("*.py"))
    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "numpy", "scipy", "h5py", "PyYAML", "matplotlib")
        },
        "sources": {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(sources)},
    }


@contextmanager
def output_lock(folder):
    folder = writable_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".real_training.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
