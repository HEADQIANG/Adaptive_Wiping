"""Seven run categories and byte-preserving compatibility for the former layout."""

import os
import re
from pathlib import Path

CATEGORIES = (
    "force_sensor", "sim_data", "sim_training", "real_exploration",
    "real_demonstrations", "real_training", "real_deploy",
)
DATA_EXPERIMENT_PREFIXES = (
    "explore_once", "match_real", "contact_", "reversal_", "direct_wrist_closeup",
    "normal_friction_ft", "normal_stiffness_ft", "normal_width_ft",
    "normal_ft_visualization", "normal_mu1p2_ft_visualization",
)
PREFIXES = (
    ("runs/real_training/manual_demonstrations", "runs/real_demonstrations/manual"),
    ("runs/real_training/programmed_demonstrations", "runs/real_demonstrations/programmed"),
    ("runs/real_training/programmed_analysis_004", "runs/real_demonstrations/analysis/programmed_004"),
    ("runs/real_training/real_robot", "runs/real_exploration"),
    ("runs/robot_control", "runs/real_deploy/robot_control"),
)
SIM_DATA_FILES = ("dataset.h5", "dataset_integrity.json", "collection.json", "source_mapping.json")


def relocated_run_path(path, root):
    """Translate only project-owned legacy paths; never rewrite saved metadata."""
    path, root = Path(path), Path(root)
    try:
        key = path.relative_to(root).as_posix()
    except ValueError:
        return path
    for old, new in PREFIXES:
        if key == old or key.startswith(old + "/"):
            return root / (new + key[len(old):])
    if key == "runs/sim_pretrain" or key.startswith("runs/sim_pretrain/"):
        suffix = key[len("runs/sim_pretrain"):].lstrip("/")
        parts = suffix.split("/")
        experiment = next((p for p in parts if not re.fullmatch(r"\d{4}_\d{6}", p)), "")
        category = "sim_data" if experiment.startswith(DATA_EXPERIMENT_PREFIXES) else "sim_training"
        if suffix.startswith("pretrain_wide_1200_v1/") and parts[-1] in SIM_DATA_FILES:
            category = "sim_data"
        return root / "runs" / category / suffix
    return path


def sim_data_path(output, name):
    """Store collected data in sim_data, with a relative link for training readers.

    Custom paths outside runs/sim_training retain their existing behavior. Return
    the physical target, so atomic replace also preserves the training-side link.
    Existing regular files are never silently moved or overwritten by this helper.
    """
    from scripts.shared import paths

    output = paths.writable_path(output)
    try:
        suffix = output.relative_to(paths.RUNS.resolve() / "sim_training")
    except ValueError:
        return output / name
    target = paths.writable_path(paths.RUNS / "sim_data" / suffix / name)
    link = output / name
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Simulation data link points to another run: {link}")
    elif link.exists():
        raise FileExistsError(f"Unmigrated simulation data at {link}; use a new output directory")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, output))
    return target
