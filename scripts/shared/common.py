"""Configuration, provenance and artifact utilities."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

from scripts.shared.paths import (
    ROOT,
    ROBOSUITE_PACKAGE,
    project_source_files,
    read_path,
    writable_path,
)
from scripts.shared.assets import DISCOVERSE_INERTIA_SOURCE, asset_files

CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def load_config(path):
    with read_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    s = cfg["simulation"]
    if s.get("ft_frame_profile", "legacy") not in ("legacy", "kwr52_left_v1"):
        raise ValueError("ft_frame_profile must be legacy or kwr52_left_v1")
    if s.get("tool_mount", "legacy") not in (
        "legacy",
        "bridge_v1",
        "compact_v2",
        "direct_wrist_v3",
    ):
        raise ValueError("tool_mount must be legacy, bridge_v1, compact_v2 or direct_wrist_v3")
    if s.get("base_mount", "stand") not in ("stand", "tabletop"):
        raise ValueError("base_mount must be stand or tabletop")
    if s.get("base_mount", "stand") == "tabletop":
        base_xy = np.asarray(s.get("base_xy", []), dtype=float)
        if base_xy.shape != (2,) or not np.isfinite(base_xy).all():
            raise ValueError("Tabletop mounting requires finite base_xy of length two")
    if s.get("reference_mode", "actual") not in ("actual", "nominal"):
        raise ValueError("reference_mode must be actual or nominal")
    if s.get("inertia_profile", "legacy") not in ("legacy", "discoverse_standard"):
        raise ValueError("inertia_profile must be legacy or discoverse_standard")
    if s.get("tool_inertia_profile", "legacy") not in ("legacy", "uniform_box_v1"):
        raise ValueError("tool_inertia_profile must be legacy or uniform_box_v1")
    if not isinstance(s.get("target_velocity_feedforward", False), bool):
        raise ValueError("target_velocity_feedforward must be a boolean")
    ramp = s.get("exploration_ramp_s", 0.0)
    if (isinstance(ramp, bool) or not isinstance(ramp, (int, float))
            or not np.isfinite(ramp) or not 0 <= ramp <= 0.5):
        raise ValueError("exploration_ramp_s must be finite and in [0, 0.5]")
    if s["sample_hz"] != 100 or s["duration"] != 4.0:
        raise ValueError("The paper profile requires 100 Hz and 400 samples / 4 seconds")
    ratio = 1 / s["sample_hz"] / s["timestep"]
    if not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError("Sampling period must be an integer number of physics steps")
    if s["timestep"] != 0.002:
        raise ValueError("The validated robosuite profile uses a 0.002 s physics timestep")
    if not 0 < cfg["filter"]["cutoff_hz"] < s["sample_hz"] / 2:
        raise ValueError("Filter cutoff must be below Nyquist")
    for name, bounds in cfg["randomization"].items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid bounds for {name}")
        if bounds[0] < 0 or (name != "friction" and bounds[0] == 0):
            raise ValueError(f"Invalid lower bound for {name}")
    for name, size in cfg["dataset"].items():
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"Invalid split size: {name}")
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with read_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def output_dir(cfg):
    return read_path(cfg["output_dir"])


def write_json(path, value):
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def provenance():
    packages = {}
    for name in ("numpy", "scipy", "mujoco", "torch", "h5py", "PyYAML", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    sources = [
        *asset_files(),
        ROBOSUITE_PACKAGE / "models/robots/robot_model.py",
        ROBOSUITE_PACKAGE / "controllers/config/robots/default_airbot_play.json",
        ROBOSUITE_PACKAGE / "controllers/parts/generic/joint_pos.py",
        ROBOSUITE_PACKAGE / "utils/ik_utils.py",
        ROBOSUITE_PACKAGE / "environments/manipulation/wipe.py",
        ROBOSUITE_PACKAGE / "macros.py",
        *project_source_files(ROOT / "scripts/sim_pretrain", ROOT / "scripts/shared"),
    ]
    return {
        "python": platform.python_version(),
        "packages": packages,
        "sources": {str(p.relative_to(ROOT)): file_digest(p) for p in sources},
    }


def sample_parameters(cfg, split_index, count):
    rng = np.random.default_rng(np.random.SeedSequence([cfg["simulation"]["seed"], split_index]))
    bounds = cfg["randomization"]
    return np.column_stack(
        [rng.uniform(*bounds[key], size=count) for key in ("friction", "stiffness_direct", "width")]
    )
