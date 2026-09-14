"""Training, held-out diagnostics, and a standalone frozen encoder interface."""

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import (
    CHANNELS,
    digest,
    file_digest,
    output_dir,
    provenance,
    write_json,
)
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.model import SpongeEncoder, SpongeVAE, vae_loss
from scripts.shared.paths import writable_path
from scripts.shared.preprocessing import Preprocessor


def load_data(cfg):
    path = output_dir(cfg) / "dataset.h5"
    integrity = json.loads((output_dir(cfg) / "dataset_integrity.json").read_text())
    actual_hash = file_digest(path)
    if actual_hash != integrity["sha256"]:
        raise ValueError("Dataset content hash mismatch")
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("config_hash") != digest(cfg):
            raise ValueError(
                "Dataset/config mismatch; use the original config or a different output directory"
            )
        if not h5.attrs.get("complete", False):
            raise ValueError("Dataset is incomplete; run collect and inspect failures first")
        result = {}
        for split, count in cfg["dataset"].items():
            group = h5[split]
            if group["ft"].shape != (count, 400, 6) or not np.all(group["valid"][:]):
                raise ValueError(f"Invalid split: {split}")
            result[split] = group["ft"][:]
        return result, actual_hash


def configure_torch(cfg):
    t = cfg["training"]
    torch.set_num_threads(t["threads"])
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.use_deterministic_algorithms(True)
    return torch.device(t["device"])


def training_snapshot(out, epoch, history, model, prep, validation, device):
    """Periodic validation plots; held-out test data is used only after training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / f"epoch_{epoch:04d}"
    folder.mkdir(exist_ok=True)
    write_json(folder / "history.json", history)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(prep.transform(validation[:1])).to(device)
        recon, _, _ = model(x, sample=False)
    target = prep.filtered(validation[:1])[0]
    prediction = prep.inverse(recon.cpu().numpy())[0]
    write_json(folder / "validation.json", dict(epoch=epoch, metrics=history[-1]["validation"],
               plotted_sample_channel_rmse=np.sqrt(np.mean((prediction-target)**2, axis=0)).tolist()))
    fig, axes = plt.subplots(3, 2, figsize=(11,8), sharex=True, layout="constrained")
    for ax, j in zip(axes.flat, (0,3,1,4,2,5)):
        ax.plot(np.arange(1,401)/100, target[:,j], label="Filtered validation")
        ax.plot(np.arange(1,401)/100, prediction[:,j], label="VAE reconstruction")
        ax.set_ylabel(f"{CHANNELS[j]} ({'N' if j<3 else 'N m'})")
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.savefig(folder / "reconstruction.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12,3), layout="constrained")
    for ax, metric in zip(axes, ("loss", "mse", "kl")):
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [r[split][metric] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.savefig(folder / "training.png", dpi=120)
    plt.close(fig)


def train(cfg, resume=False):
    out = writable_path(output_dir(cfg))
    if (out / "vae_last.pt").exists():
        raise FileExistsError("vae_last.pt exists; choose a new output directory to retrain")
    raw, data_provenance = load_data(cfg)
    with h5py.File(out / "dataset.h5", "r") as source:
        source_frame = source.attrs.get("frame", "ft_frame local")
    device = configure_torch(cfg)
    prep = Preprocessor(**cfg["filter"], sample_hz=cfg["simulation"]["sample_hz"]).fit(raw["train"])
    arrays = {key: prep.transform(value) for key, value in raw.items()}
    loaders = {
        key: DataLoader(
            TensorDataset(torch.from_numpy(value)),
            batch_size=cfg["training"]["batch_size"],
            shuffle=key == "train",
            num_workers=0,
        )
        for key, value in arrays.items()
        if key != "test"
    }
    model = SpongeVAE().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    history, best = [], float("inf")
    interval = cfg["training"].get("checkpoint_every", 0)
    if not isinstance(interval, int) or interval < 0:
        raise ValueError("checkpoint_every must be a nonnegative integer")
    start_epoch = 1
    if resume and (out / "vae_resume.pt").exists():
        state = torch.load(out / "vae_resume.pt", map_location=device, weights_only=True)
        if state["config_hash"] != digest(cfg) or state["data_provenance_hash"] != data_provenance:
            raise ValueError("Resume checkpoint provenance mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        history, best = state["history"], state["best"]
        start_epoch = state["epoch"]+1
    source_provenance = provenance()
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        record = {"epoch": epoch}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            total = np.zeros(3)
            size = 0
            with torch.set_grad_enabled(training):
                for (batch,) in loader:
                    batch = batch.to(device)
                    reconstruction, mu, logvar = model(batch, sample=training)
                    losses = vae_loss(reconstruction, batch, mu, logvar, cfg["training"]["beta"])
                    if not all(torch.isfinite(loss).item() for loss in losses):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        losses[0].backward()
                        if any(
                            p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in model.parameters()
                        ):
                            raise RuntimeError("Non-finite gradients")
                        optimizer.step()
                    total += np.array([loss.item() for loss in losses]) * len(batch)
                    size += len(batch)
            record[split] = dict(zip(("loss", "mse", "kl"), (total / size).tolist()))
        history.append(record)
        payload = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "preprocessing": prep.state(),
            "config": cfg,
            "config_hash": digest(cfg),
            "epoch": epoch,
            "data_provenance_hash": data_provenance,
            "training_provenance": source_provenance,
            "channels": CHANNELS,
            "frame": source_frame,
            "units": ["N"] * 3 + ["N*m"] * 3,
            "train_mean_trajectory": torch.from_numpy(arrays["train"].mean(axis=0)),
        }
        if record["validation"]["loss"] < best:
            best = record["validation"]["loss"]
            save_checkpoint(out / "vae_best.pt", payload)
        write_json(out / "history.json", history)
        if interval and (epoch % interval == 0 or epoch == cfg["training"]["epochs"]):
            training_snapshot(out, epoch, history, model, prep, raw["validation"], device)
            snapshot = dict(payload, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                            history=history, best=best)
            save_checkpoint(out / f"epoch_{epoch:04d}" / "vae.pt", snapshot)
            save_checkpoint(out / "vae_resume.pt", snapshot)
        if epoch == cfg["training"]["epochs"]:
            save_checkpoint(out / "vae_last.pt", payload)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record), flush=True)
    write_json(out / "preprocessing.json", prep.state())
    return history[-1]


def load_checkpoint(cfg):
    checkpoint = torch.load(output_dir(cfg) / "vae_last.pt", map_location="cpu", weights_only=True)
    if checkpoint["config_hash"] != digest(cfg):
        raise ValueError("Checkpoint/config mismatch")
    model = SpongeVAE()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model, Preprocessor.from_state(checkpoint["preprocessing"])


def evaluate(cfg, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = writable_path(output if output is not None else output_dir(cfg))
    if output is not None:
        out.mkdir(parents=True, exist_ok=False)
    configure_torch(cfg)
    raw, data_hash = load_data(cfg)
    checkpoint, model, prep = load_checkpoint(cfg)
    if checkpoint["data_provenance_hash"] != data_hash:
        raise ValueError("Checkpoint was trained on different dataset content")
    x = torch.from_numpy(prep.transform(raw["test"]))
    with torch.no_grad():
        recon, mu, logvar = model(x, sample=False)
        _, mse, kl = vae_loss(recon, x, mu, logvar)
        fixed = model.decode(mu.mean(dim=0, keepdim=True).expand_as(mu))
        permutation = torch.randperm(len(mu))
        shuffled = model.decode(mu[permutation])
    target_physical = prep.filtered(raw["test"])
    prediction = prep.inverse(recon.numpy())
    mean_baseline = checkpoint["train_mean_trajectory"].numpy()
    latent_variance = mu.numpy().var(axis=0)
    report = {
        "checkpoint_sha256": file_digest(output_dir(cfg) / "vae_last.pt"),
        "dataset_sha256": data_hash,
        "epoch": checkpoint["epoch"],
        "test_mse": mse.item(),
        "test_kl": kl.item(),
        "baseline_mse": float(np.mean((x.numpy() - mean_baseline) ** 2)),
        "fixed_latent_mse": float((fixed - x).square().mean()),
        "shuffled_latent_mse": float((shuffled - x).square().mean()),
        "channel_rmse_physical": np.sqrt(
            np.mean((prediction - target_physical) ** 2, axis=(0, 1))
        ).tolist(),
        "latent_variance": latent_variance.tolist(),
        "active_latent_dimensions_at_1e-4": int(np.sum(latent_variance > 1e-4)),
        "normalization_out_of_range_fraction": float(np.mean((x.numpy() < 0) | (x.numpy() > 0.9))),
        "channels": CHANNELS,
    }
    report["beats_mean_baseline"] = report["test_mse"] < report["baseline_mse"]
    report["collapse_warning"] = bool(
        np.max(latent_variance) < 1e-4 or report["shuffled_latent_mse"] <= report["test_mse"] * 1.01
    )
    write_json(out / "evaluation.json", report)
    np.savez_compressed(out / "test_embeddings.npz", mu=mu.numpy(), logvar=logvar.numpy())
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    time = np.arange(1, 401) / 100
    for i, ax in enumerate(axes.flat):
        ax.plot(time, target_physical[0, :, i], label="Filtered simulation")
        ax.plot(time, prediction[0, :, i], label="VAE mean reconstruction")
        ax.set_ylabel(f"{CHANNELS[i]} ({'N' if i < 3 else 'N m'})")
        ax.set_xlabel("Time (s)")
    axes.flat[0].legend()
    fig.savefig(out / "reconstruction.png", dpi=150)
    plt.close(fig)
    history = json.loads((output_dir(cfg) / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for split in ("train", "validation"):
        for ax, metric in zip(axes, ("mse", "kl")):
            ax.plot([r["epoch"] for r in history], [r[split][metric] for r in history], label=split)
            ax.set(xlabel="Epoch", ylabel=metric.upper())
            ax.legend()
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)
    return report


def export(cfg):
    checkpoint, model, prep = load_checkpoint(cfg)
    evaluation_path = output_dir(cfg) / "evaluation.json"
    if not evaluation_path.exists():
        raise RuntimeError("Run evaluate before export")
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation["checkpoint_sha256"] != file_digest(output_dir(cfg) / "vae_last.pt"):
        raise ValueError("Evaluation is stale; rerun evaluate for this checkpoint")
    payload = {
        "encoder": model.encoder.state_dict(),
        "preprocessing": prep.state(),
        "channels": CHANNELS,
        "frame": checkpoint["frame"],
        "units": checkpoint["units"],
        "architecture": "frame6to5_flat2000_posterior10_split5",
        "epoch": checkpoint["epoch"],
        "config": cfg,
        "data_provenance_hash": checkpoint["data_provenance_hash"],
        "evaluation": evaluation,
    }
    path = output_dir(cfg) / "encoder.pt"
    save_checkpoint(path, payload)
    return {"encoder": str(path), "evaluation": payload["evaluation"]}
