"""Explicit synthetic software fixtures. Never a real-data collection path."""

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.real_training.config import resolve
from scripts.real_training.data import (
    FT_UNITS,
    PROTOCOL,
    prepare,
    write_raw_log,
)
from scripts.real_training.training import (
    cross_validate,
    evaluate,
    export,
    train,
)
from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import CHANNELS, file_digest
from scripts.shared.model import SpongeVAE
from scripts.shared.policy import OfflinePolicy
from scripts.shared.preprocessing import Preprocessor


def synthetic_fixture(folder, cfg, *, use_existing_encoder=False):
    folder = Path(folder)
    cfg = copy.deepcopy(cfg)
    cfg["profile"] = "synthetic_test"
    cfg["raw_data"] = str(folder / "synthetic_raw.h5")
    cfg["output_dir"] = str(folder / "synthetic_training")
    cfg["training"].update(xy_epochs=2, ft_epochs=2, checkpoint_every=1)
    rng = np.random.default_rng(739)
    t_exp = np.arange(401) / 100
    exploration = np.column_stack([0.1 * np.sin(t_exp * (i + 1)) + 0.05 * i for i in range(6)])
    if not use_existing_encoder:
        torch.manual_seed(739)
        model = SpongeVAE()
        prep = Preprocessor().fit(rng.normal(size=(10, 400, 6)))
        cfg["encoder"] = str(folder / "synthetic_encoder.pt")
        save_checkpoint(
            cfg["encoder"],
            {
                "encoder": model.encoder.state_dict(),
                "preprocessing": prep.state(),
                "architecture": "frame6to5_flat2000_posterior10_split5",
                "epoch": 0,
                "frame": "ft_frame local",
                "channels": CHANNELS,
                "units": FT_UNITS,
                "source_kind": "synthetic",
                "evaluation": {"passed": False},
                "config": {"randomization": {"friction": [0.0, 1.2]}},
            },
        )
    else:
        cfg["encoder"] = str(resolve(cfg["encoder"]))
    metadata = {
        "robot_id": "synthetic_not_hardware",
        "sensor_id": "synthetic_sensor",
        "calibration_id": "synthetic_no_physical_calibration",
        "tcp_definition": "synthetic_tcp",
        "sensor_frame": "synthetic_sensor_axes",
        "sensor_to_ft_frame": np.eye(4).tolist(),
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
    explorations = {
        "normal_exp": {
            "attrs": {"sponge_id": "normal", "complete": True, "start_time": 20.0, "ft_hz": 100.0},
            "ft_time": 20.0 + t_exp,
            "ft": exploration,
        }
    }
    demonstrations = {}
    t = np.arange(1001) / 100
    for i in range(8):
        start = 100.0 + 20 * i
        phase = t + i * 0.08
        ft = np.column_stack([np.sin(phase * (j + 1) / 3) * 0.05 + i * 0.04 for j in range(6)])
        position = np.column_stack(
            [
                0.30 + 0.01 * np.sin(phase) + i * 0.002,
                0.02 * np.cos(phase),
                0.2 + 0.001 * np.sin(phase * 0.8) + i * 0.0002 * t,
            ]
        )
        demonstrations[f"demo_{i + 1:02d}"] = {
            "attrs": {
                "sponge_id": "normal",
                "complete": True,
                "start_time": start,
                "ft_hz": 100.0,
                "pose_hz": 100.0,
                "exploration_id": "normal_exp",
                "surface_id": "synthetic_training_slope",
            },
            "ft_time": start + t,
            "ft": ft,
            "pose_time": start + t,
            "tcp_position": position,
            "tcp_quaternion": np.tile([0, 0, 0, 1.0], (len(t), 1)),
        }
    write_raw_log(cfg["raw_data"], metadata, explorations, demonstrations, source_kind="synthetic")
    return cfg


def smoke_test(cfg, *, keep=False):
    def execute(folder):
        test_cfg = synthetic_fixture(folder, cfg, use_existing_encoder=True)
        source_hash = file_digest(test_cfg["encoder"])
        prepared = prepare(test_cfg, testing=True)
        cv = cross_validate(test_cfg, testing=True)
        trained = train(test_cfg, testing=True)
        evaluation = evaluate(test_cfg, testing=True)
        exported = export(test_cfg, testing=True)
        policy = OfflinePolicy(exported["policy"])
        assert policy.source_kind == "synthetic" and not policy.hardware_ready
        assert file_digest(test_cfg["encoder"]) == source_hash
        return {
            "software_pipeline_passed": True,
            "source_kind": "synthetic",
            "hardware_ready": False,
            "valid_windows": prepared["valid_windows_total"],
            "cross_validation_folds": len(cv["folds"]),
            "epochs": trained["epochs"],
            "roundtrip_verified": exported["roundtrip_verified"],
            "training_metrics_test_only": evaluation["metrics"],
            "artifacts": str(folder) if keep else "temporary fixtures removed after the check",
        }

    if keep:
        return execute(Path(tempfile.mkdtemp(prefix="adaptive-wiping-real-smoke-")))
    with tempfile.TemporaryDirectory(prefix="adaptive-wiping-real-smoke-") as folder:
        return execute(Path(folder))
