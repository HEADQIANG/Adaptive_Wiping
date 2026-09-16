"""Offline, PCA-initialized full-trajectory VAE; preserves the paper baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import CHANNELS, file_digest, load_config, write_json
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.model import SpongeEncoder, vae_loss
from scripts.shared.paths import ROOT, project_source_files
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.vae_ablation import (
    diagnose,
    finite_step,
    seed_all,
    snapshot,
    verify_snapshot,
)
from scripts.sim_pretrain.learning import load_data

PROTOCOL = {
    "kind": "engineering_repair_not_paper_replication",
    "decoder": "unconstrained_linear_5_to_2400",
    "encoder": "unchanged_frame6to5_flat2000_posterior10_split5",
    "initialization": "train_only_channel_PCA5_then_trajectory_PCA5_and_decoder_least_squares",
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "beta": 0.0001,
    "warmup_epochs": 50,
    "seeds": [42, 43, 44],
    "selection": "minimum_validation_MSE_at_epochs_50_to_200",
    "reconstruction_fraction": 0.8,
    "ablation_increase": 0.1,
    "export_seed": 42,
    "required_passing_seeds": 2,
    "test_caveat": "same historical split; not a new blind benchmark",
    "hardware_ready": False,
}


class RepairedVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpongeEncoder()
        self.decoder = nn.Linear(5, 2400)

    def decode(self, z):
        return self.decoder(z).reshape(-1, 400, 6)

    def forward(self, x, sample=True):
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar


@torch.no_grad()
def initialize(model, training):
    """Fit every projection/offset on training rows, never validation or test."""
    x = np.asarray(training, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (400, 6) or len(x) < 6 or not np.isfinite(x).all():
        raise ValueError("Initialization needs at least six finite [400,6] training trajectories")
    frames = x.reshape(-1, 6)
    frame_mean = frames.mean(0)
    _, channel_axes = np.linalg.eigh(np.cov(frames, rowvar=False))
    projection = channel_axes[:, -5:][:, ::-1].copy()
    projected = ((x - frame_mean) @ projection).reshape(len(x), 2000)
    center = projected.mean(0)
    u, singular, vh = np.linalg.svd(projected - center, full_matrices=False)
    scale = singular[:5] / np.sqrt(len(x))
    if np.any(scale < 1e-8):
        raise ValueError("Training data do not support five nonconstant initialization dimensions")
    axes = vh[:5] / scale[:, None]
    codes = u[:, :5] * np.sqrt(len(x))
    target = x.reshape(len(x), 2400)
    template = target.mean(0)
    decoder = np.linalg.lstsq(codes, target - template, rcond=None)[0]
    copy = lambda parameter, value: parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))
    copy(model.encoder.frame.weight, projection.T)
    copy(model.encoder.frame.bias, -frame_mean @ projection)
    model.encoder.posterior.weight.zero_()
    model.encoder.posterior.bias.fill_(-6.0)
    copy(model.encoder.posterior.weight[:5], axes)
    copy(model.encoder.posterior.bias[:5], -axes @ center)
    copy(model.decoder.weight, decoder.T)
    copy(model.decoder.bias, template)
    # An affine 5D channel plane is a lower bound even before ReLU constraints.
    channel_singular = np.linalg.svd(frames - frame_mean, compute_uv=False)
    return {
        "affine_channel_rank5_mse_lower_bound": float(channel_singular[-1] ** 2 / frames.size),
        "initialized_mse": float(np.mean((codes @ decoder + template - target) ** 2)),
        "template_mse": float(np.mean((target - template) ** 2)),
        "initialization_rows": len(x),
        "fitting_split": "train",
    }


@torch.no_grad()
def mse(model, values):
    model.eval()
    return (
        sum(
            float((model(b, sample=False)[0] - b).square().sum())
            for b in torch.from_numpy(values).split(32)
        )
        / values.size
    )


def train_seed(train, validation, prep, folder, seed, *, epochs=200):
    seed_all(seed)
    model = RepairedVAE()
    initialization = initialize(model, train)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL["learning_rate"])
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "initialization.json", initialization)
    x = torch.from_numpy(train)
    for epoch in range(1, epochs + 1):
        model.train()
        beta = PROTOCOL["beta"] * min((epoch - 1) / 49.0, 1.0)
        total = np.zeros(3)
        for indices in torch.randperm(len(x)).split(PROTOCOL["batch_size"]):
            batch = x[indices]
            prediction, mu, logvar = model(batch, sample=True)
            losses = vae_loss(prediction, batch, mu, logvar, beta)
            finite_step(losses, model, optimizer)
            total += np.asarray([v.item() for v in losses]) * len(batch)
        val = mse(model, validation)
        if not np.isfinite(val):
            raise FloatingPointError("Non-finite validation MSE")
        record = {
            "epoch": epoch,
            "beta": beta,
            "train_loss_mse_kl": (total / len(x)).tolist(),
            "validation_mse": val,
        }
        history.append(record)
        payload = {
            "format": "full_trajectory_vae_repair_v1",
            "model": model.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "preprocessing": prep.state(),
            "protocol": PROTOCOL,
        }
        if epoch >= 50 and val < best:
            best = val
            save_checkpoint(folder / "best.pt", payload)
        if epoch == epochs:
            save_checkpoint(folder / "last.pt", payload)
        if epoch == 1 or epoch % 25 == 0:
            write_json(folder / "history.json", history)
            print(json.dumps({"seed": seed, **record}), flush=True)
    write_json(folder / "history.json", history)
    return initialization


def restore(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "full_trajectory_vae_repair_v1":
        raise ValueError("Unsupported repair checkpoint")
    model = RepairedVAE().eval()
    model.load_state_dict(saved["model"])
    return model, saved


def export_encoder(model, prep, cfg, data_hash, validation, test, epoch, path):
    if not validation["passed"] or not test["passed"]:
        raise ValueError("Refusing to export a failed encoder")
    variance = np.asarray(test["latent_variance"])
    evaluation = {
        **test,
        "test_mse": test["mse"],
        "test_kl": test["kl"],
        "shuffled_latent_mse": test["shuffled_latent_mse_mean"],
        "beats_mean_baseline": test["reconstruction_pass"],
        "collapse_warning": not test["latent_utilization_pass"],
        "active_latent_dimensions_at_1e-4": int(np.sum(variance > 1e-4)),
    }
    save_checkpoint(
        path,
        {
            "encoder": model.encoder.state_dict(),
            "preprocessing": prep.state(),
            "architecture": "frame6to5_flat2000_posterior10_split5",
            "epoch": epoch,
            "config": cfg,
            "data_provenance_hash": data_hash,
            "evaluation": evaluation,
            "channels": CHANNELS,
            "frame": "ft_frame local",
            "units": ["N"] * 3 + ["N*m"] * 3,
            "repair_protocol": PROTOCOL,
            "validation": validation,
            "hardware_ready": False,
        },
    )


def run(config, output):
    cfg = load_config(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", PROTOCOL)
    from scripts.shared.paths import read_path

    baseline_dir = read_path(cfg["output_dir"])
    protected = snapshot(
        [
            *baseline_dir.glob("*.pt"),
            baseline_dir / "dataset.h5",
            Path(config),
            ROOT / "configs/robot_control/airbot_calibration.json",
            ROOT / "archive/real_training/raw_data/real_training/airbot_native_20260910/raw.h5",
            *ROOT.glob("archive/sim_pretrain/normal_vae_ablation_v1/**/*.pt"),
            *ROOT.glob("archive/sim_pretrain/normal_decoder_activation_v1/**/*.pt"),
        ]
    )
    sources = snapshot(
        [
            Path(__file__),
            Path(__file__).with_name("vae_ablation.py"),
            *project_source_files(ROOT / "scripts/sim_pretrain", ROOT / "scripts/shared"),
        ]
    )
    write_json(out / "manifest.json", {"protected": protected, "sources": sources, "config": cfg})
    write_json(out / "status.json", {"state": "running", "hardware_ready": False})
    try:
        raw, data_hash = load_data(cfg)
        prep = Preprocessor(**cfg["filter"], sample_hz=100).fit(raw["train"])
        train, validation = prep.transform(raw["train"]), prep.transform(raw["validation"])
        template = train.mean(0)
        validations, initializations, epochs = {}, {}, {}
        for seed in PROTOCOL["seeds"]:
            folder = out / f"seed_{seed}"
            initializations[str(seed)] = train_seed(train, validation, prep, folder, seed)
            model, saved = restore(folder / "best.pt")
            validations[str(seed)] = diagnose(model, validation, template, prep)
            epochs[str(seed)] = saved["epoch"]
            write_json(folder / "validation.json", validations[str(seed)])
        eligible = (
            validations["42"]["passed"] and sum(v["passed"] for v in validations.values()) >= 2
        )
        write_json(
            out / "selection.json",
            {
                "validation_only": True,
                "export_seed": 42,
                "eligible": eligible,
                "epochs": epochs,
                "validation": validations,
            },
        )
        tests = {}
        if eligible:
            values = prep.transform(raw["test"])
            for seed in PROTOCOL["seeds"]:
                model, _ = restore(out / f"seed_{seed}" / "best.pt")
                tests[str(seed)] = diagnose(model, values, template, prep)
                write_json(out / f"seed_{seed}" / "test.json", tests[str(seed)])
        stable = bool(
            eligible and tests["42"]["passed"] and sum(v["passed"] for v in tests.values()) >= 2
        )
        audit = {"protected": verify_snapshot(protected), "sources": verify_snapshot(sources)}
        if not all(v["unchanged"] for v in audit.values()):
            raise RuntimeError("Protected files or experiment sources changed during training")
        if stable:
            model, saved = restore(out / "seed_42/best.pt")
            export_encoder(
                model,
                prep,
                cfg,
                data_hash,
                validations["42"],
                tests["42"],
                saved["epoch"],
                out / "encoder.pt",
            )
            frozen = FrozenSpongeEncoder(out / "encoder.pt")
            with torch.no_grad():
                expected = model.encoder(torch.from_numpy(prep.transform(raw["test"][:7])))[
                    0
                ].numpy()
            if not np.array_equal(frozen.encode(raw["test"][:7]), expected):
                raise RuntimeError("Frozen export predictions differ")
        report = {
            "outcome": "simulation_repair_passed" if stable else "repair_failed",
            "hardware_ready": False,
            "protocol": PROTOCOL,
            "initialization": initializations,
            "validation": validations,
            "test": tests,
            "audit": audit,
            "limitations": [
                "Real sensor distribution and coordinate alignment remain unverified",
                "One real exploration cannot validate cross-material adaptation",
                "Decoder and KL differ from the paper baseline",
            ],
        }
        write_json(out / "report.json", report)
        lines = [
            "# Full-trajectory VAE repair",
            "",
            report["outcome"],
            "",
            "Engineering variant, not paper replication. Hardware ready: false.",
            "",
            "| Seed | Epoch | Validation MSE | Test MSE | Test template MSE | Pass |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for seed in PROTOCOL["seeds"]:
            key = str(seed)
            test = tests.get(key, {})
            lines.append(
                f"| {seed} | {epochs[key]} | {validations[key]['mse']:.8g} | "
                f"{test.get('mse')} | {test.get('baseline_mse')} | {test.get('passed', False)} |"
            )
        lines += ["", *report["limitations"], "", "Original files and sources verified unchanged."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            out / "status.json",
            {"state": "completed", "outcome": report["outcome"], "hardware_ready": False},
        )
        return stable
    except Exception as exc:
        write_json(
            out / "status.json", {"state": "failed", "error": str(exc), "hardware_ready": False}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default="archive/sim_pretrain/two_control_pretraining_v2/normal/config.json",
    )
    parser.add_argument("--output", default="runs/sim_training/sponge_vae_repair_v1")
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    return 0 if run(args.dataset_config, args.output) else 2


if __name__ == "__main__":
    raise SystemExit(main())
