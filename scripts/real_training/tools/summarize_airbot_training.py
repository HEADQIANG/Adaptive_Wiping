"""Verify saved real-data artifacts and render a compact, explicitly offline report."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.real_deploy.airbot_deploy import deployment_blockers
from scripts.real_training.config import load_config, provenance, resolve
from scripts.real_training.data import load_prepared
from scripts.shared.common import file_digest, write_json
from scripts.shared.paths import ROOT
from scripts.shared.policy import OfflinePolicy


def summarize(cfg, deploy_config, destination):
    arrays, info = load_prepared(cfg)
    root = resolve(cfg["output_dir"])
    final = json.loads((root / "final/evaluation.json").read_text())
    cv = json.loads((root / "cross_validation/evaluation.json").read_text())
    policy = OfflinePolicy(root / "policy.pt")
    expected_epochs = {"xy": 10000, "feedback": 2000}
    if policy.metadata["epochs"] != expected_epochs or len(cv["folds"]) != 8:
        raise ValueError("Training or cross-validation is incomplete")
    source = torch.load(resolve(cfg["encoder"]), map_location="cpu", weights_only=True)
    payload = torch.load(root / "policy.pt", map_location="cpu", weights_only=True)
    for key, tensor in source["encoder"].items():
        if not torch.equal(tensor, payload["encoder"]["encoder"][key]):
            raise ValueError("Exported frozen encoder weights changed")
    checkpoints = [(root / "final/training.pt", final["checkpoint_sha256"])]
    for index, fold in enumerate(cv["folds"]):
        expected = info["demo_ids"][index]
        if (
            fold["held_out_demo_id"] != expected
            or expected in fold["train_demo_ids"]
            or len(fold["train_demo_ids"]) != 7
        ):
            raise ValueError("Invalid episode-level holdout")
        checkpoints.append(
            (root / f"cross_validation/fold_{index + 1:02}/training.pt", fold["checkpoint_sha256"])
        )
    for path, expected in checkpoints:
        if file_digest(path) != expected:
            raise ValueError(f"Changed checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["epochs"] != expected_epochs or saved["bindings"]["software"] != provenance():
            raise ValueError("Checkpoint epochs or source/software provenance differs")
    for path, expected in info["metadata"]["source_hashes"].items():
        if file_digest(path) != expected:
            raise ValueError("Original collection changed")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    blockers = deployment_blockers(policy, deploy_config)
    delta_rmse = cv["metrics"]["delta_h"]["rmse_m"]
    zero_rmse = cv["metrics"]["delta_h_zero_baseline"]["rmse_m"]
    result = {
        "source_kind": "real",
        "profile": cfg["profile"],
        "hardware_ready": False,
        "scope": "native_frame_offline_training_and_recorded_FT_validation_not_paper_hardware_replication",
        "epochs_per_model_pair": expected_epochs,
        "model_pairs_completed": 9,
        "demonstrations": 8,
        "feedback_windows": 160,
        "frozen_encoder_exact_match": True,
        "encoder_epoch": source["epoch"],
        "encoder_randomization": source["config"]["randomization"],
        "encoder_active_latent_dimensions": source["evaluation"][
            "active_latent_dimensions_at_1e-4"
        ],
        "original_sources_unchanged": True,
        "training_provenance_verified": True,
        "final_metrics": final["metrics"],
        "leave_one_demo_out_metrics": cv["metrics"],
        "height_validation_beats_zero_baseline": delta_rmse < zero_rmse,
        "height_validation_rmse_relative_to_zero": delta_rmse / zero_rmse,
        "deployment_blockers": blockers,
        "warnings": info["warnings"],
        "source_hashes": info["metadata"]["source_hashes"],
        "artifacts": {
            str(p.relative_to(root)): file_digest(p)
            for p in (
                root / "policy.pt",
                root / "prepared.h5",
                root / "final/training.pt",
                root / "final/evaluation.json",
                root / "cross_validation/evaluation.json",
            )
        },
    }
    write_json(destination / "summary.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#2874a6", "#bc3f47", "#697677")
    for trajectory in arrays["xy"]:
        axes[0, 0].plot(
            trajectory[:, 0] * 1000, trajectory[:, 1] * 1000, color="#bbbbbb", linewidth=1
        )
    pred = policy.predict_xy(arrays["sponge"])[0] * 1000
    axes[0, 0].plot(
        pred[:, 0], pred[:, 1], color=colors[0], linewidth=2, label="Model (same sponge input)"
    )
    axes[0, 0].set(
        title="Eight demonstrations and learned XY",
        xlabel="SDK end X (mm)",
        ylabel="SDK end Y (mm)",
    )
    axes[0, 0].legend(fontsize=9)
    categories = ["Training", "Held-out demos"]
    metrics = [final["metrics"], cv["metrics"]]
    for offset, (key, label, color) in enumerate(
        zip(
            ("delta_h", "delta_h_zero_baseline", "delta_h_training_mean_baseline"),
            ("Model", "Zero increment", "Train mean"),
            colors,
        )
    ):
        axes[0, 1].bar(
            np.arange(2) + (offset - 1) * 0.24,
            [m[key]["rmse_m"] * 1000 for m in metrics],
            width=0.24,
            label=label,
            color=color,
        )
    axes[0, 1].set(
        xticks=[0, 1],
        xticklabels=categories,
        ylabel="Height increment RMSE (mm)",
        title="Generalization does not beat zero baseline",
    )
    axes[0, 1].legend(fontsize=9)
    with np.load(root / "cross_validation/predictions.npz", allow_pickle=False) as p:
        actual = p["delta_h_target"].ravel() * 1000
        predicted = p["delta_h_prediction"].ravel() * 1000
    axes[1, 0].scatter(actual, predicted, s=18, alpha=0.7, color=colors[0])
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[1, 0].plot([low, high], [low, high], color=colors[2], linestyle="--")
    axes[1, 0].set(
        xlabel="Recorded next height increment (mm)",
        ylabel="Held-out prediction (mm)",
        title="160 held-out predictions, recorded FT",
    )
    x = np.arange(1, 9)
    axes[1, 1].plot(
        x,
        [f["metrics"]["delta_h"]["rmse_m"] * 1000 for f in cv["folds"]],
        "o-",
        label="Model",
        color=colors[0],
    )
    axes[1, 1].plot(
        x,
        [f["metrics"]["delta_h_zero_baseline"]["rmse_m"] * 1000 for f in cv["folds"]],
        "s--",
        label="Zero increment",
        color=colors[1],
    )
    axes[1, 1].set(
        xticks=x,
        xlabel="Held-out demonstration",
        ylabel="RMSE (mm)",
        title="Leave-one-demo-out folds",
    )
    axes[1, 1].legend(fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=0.15)
    fig.suptitle(
        "AIRBOT real data | Paper-sized networks | Uncalibrated offline experiment\n"
        "No closed-loop hardware validation; original timestamps and loads retained",
        fontsize=13,
    )
    fig.savefig(destination / "results.png", dpi=150)
    plt.close(fig)
    return {
        "output": str(destination),
        "height_validation_beats_zero": delta_rmse < zero_rmse,
        "original_sources_unchanged": True,
        "training_provenance_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/real_training/real_training_airbot_native.yaml"
    )
    parser.add_argument("--deployment-config", default="configs/real_deploy/airbot_deployment.json")
    parser.add_argument("--output", default="runs/real_training/report")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                load_config(args.config),
                json.loads(Path(args.deployment_config).read_text()),
                args.output,
            ),
            indent=2,
        )
    )
