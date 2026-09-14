"""Bounded normal-mode representation experiments; never edits frozen training code."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import (
    CHANNELS,
    digest,
    file_digest,
    load_config,
    output_dir,
    provenance,
    write_json,
)
from scripts.shared.model import SpongeEncoder, SpongeVAE, vae_loss
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.learning import load_checkpoint, load_data

ARCHITECTURES = ("original_linear", "nonlinear_gelu")
SPLITS = ("train", "validation", "test")
PHASES = {"press": (0, 200), "forward": (200, 300), "reverse": (300, 400)}
TEST_INDICES = list(range(0, 100, 10))
SHUFFLE_SEEDS = list(range(4200, 4220))
CANDIDATES = [
    {"name": "lr_control", "beta": 0.06, "warmup": False, "dropout": 0.1},
    {"name": "beta_small", "beta": 0.001, "warmup": False, "dropout": 0.1},
    {"name": "warm_small", "beta": 0.001, "warmup": True, "dropout": 0.1},
    {"name": "warm_medium", "beta": 0.01, "warmup": True, "dropout": 0.1},
    {"name": "warm_original", "beta": 0.06, "warmup": True, "dropout": 0.1},
    {"name": "warm_no_dropout", "beta": 0.001, "warmup": True, "dropout": 0.0},
]
PROTOCOL = {
    "version": 1,
    "purpose": "engineering experiment, not original paper configuration",
    "latent_dim": 5,
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.001,
    "seed": 42,
    "repeat_seeds": [43, 44],
    "small_size": 32,
    "small_steps": 2000,
    "small_absolute_mse": 0.001,
    "small_baseline_fraction": 0.1,
    "reconstruction_fraction": 0.8,
    "ablation_increase": 0.1,
    "warmup_epochs": [1, 50],
    "selection_epochs": [50, 200],
    "shuffle_seeds": SHUFFLE_SEEDS,
    "test_plot_indices": TEST_INDICES,
    "phases_frame_slices": PHASES,
    "candidates": CANDIDATES,
    "fixed_code": "mean mu of the evaluated split; diagnostic only",
    "device": "cpu",
    "threads": 1,
    "probe": {
        "ridge_alphas": [1e-4, 1e-2, 1.0, 100.0],
        "hidden": [64, 32],
        "epochs": 400,
        "patience": 50,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "seed": 42,
        "bootstrap_repeats": 1000,
        "bootstrap_seed": 42,
    },
}


class NonlinearEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 5))
        self.posterior = nn.Sequential(nn.Linear(2000, 64), nn.GELU(), nn.Linear(64, 10))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (400, 6):
            raise ValueError("Expected [batch,400,6]")
        return self.posterior(self.frame(x).flatten(1)).chunk(2, dim=-1)


def make_encoder(architecture):
    if architecture == ARCHITECTURES[0]:
        return SpongeEncoder()
    if architecture == ARCHITECTURES[1]:
        return NonlinearEncoder()
    raise ValueError(f"Unsupported encoder architecture: {architecture}")


class AblationVAE(SpongeVAE):
    def __init__(self, architecture="original_linear", dropout=0.1):
        super().__init__()
        if architecture != ARCHITECTURES[0]:
            self.encoder = make_encoder(architecture)
        self.decode_hidden[2] = nn.Dropout(dropout)
        self.architecture = architecture


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def beta_at(epoch, target, warmup):
    if epoch < 1:
        raise ValueError("Epoch numbers start at 1")
    return float(target * min((epoch - 1) / 49.0, 1.0) if warmup else target)


def reserve_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(paths):
    return {str(Path(p).resolve()): file_digest(p) for p in sorted(paths, key=str)}


def verify_snapshot(before, *, historical=False):
    from scripts.shared.paths import read_path

    paths = {
        p: read_path(p, historical=True, source_sha256=expected) if historical else Path(p)
        for p, expected in before.items()
    }
    changed = [
        p
        for p, expected in before.items()
        if not paths[p].is_file() or file_digest(paths[p]) != expected
    ]
    return {
        "unchanged": not changed,
        "changed_files": changed,
        "file_count": len(before),
        "scope": "historical_evidence" if historical else "current_files",
    }


def split_audit(raw, parameters):
    """Reject exact cross-split duplicates without filtering failed contact samples."""
    seen_parameters, seen_trajectories = {}, {}
    counts = {}
    import hashlib

    for split in SPLITS:
        x, y = raw[split], parameters[split]
        if x.shape[1:] != (400, 6) or y.shape != (len(x), 3):
            raise ValueError(f"Bad split shapes: {split}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite split: {split}")
        for trajectory, parameter in zip(x, y):
            for value, seen in ((parameter, seen_parameters), (trajectory, seen_trajectories)):
                key = hashlib.sha256(
                    np.ascontiguousarray(value, dtype=np.float64).tobytes()
                ).hexdigest()
                if key in seen and seen[key] != split:
                    raise ValueError(f"Cross-split duplicate: {seen[key]} / {split}")
                seen[key] = split
        counts[split] = len(x)
    return {
        "counts": counts,
        "cross_split_duplicates": False,
        "removed_samples": 0,
        "fitting_split": "train",
        "selection_split": "validation",
        "final_reporting_split": "test",
    }


@torch.no_grad()
def infer(model, values):
    model.eval()
    outputs = [model(batch, sample=False) for batch in torch.from_numpy(values).split(32)]
    return tuple(torch.cat([item[i] for item in outputs]) for i in range(3))


@torch.no_grad()
def decode_batches(model, codes):
    return torch.cat([model.decode(batch) for batch in codes.split(32)])


def finite_step(losses, model, optimizer=None):
    if not all(torch.isfinite(loss).all().item() for loss in losses):
        raise FloatingPointError("Non-finite loss")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Non-finite updated parameter")


def reconstruction_pass(mse, baseline):
    return bool(baseline > 0 and mse <= 0.8 * baseline)


def diagnostic_arrays(target, prediction, mu, logvar, fixed, shuffled, mean_template, prep):
    mse = float(np.mean((prediction - target) ** 2))
    baseline = float(np.mean((mean_template - target) ** 2))
    fixed_mse = float(np.mean((fixed - target) ** 2))
    shuffled_mses = [float(np.mean((value - target) ** 2)) for value in shuffled]
    shuffled_mean = float(np.mean(shuffled_mses))
    physical_error = prep.inverse(prediction) - prep.inverse(target)
    ratio = lambda value: float(value / mse - 1) if mse > 0 else None
    use_pass = bool(
        fixed_mse >= 1.1 * mse
        and shuffled_mean >= 1.1 * mse
        and fixed_mse > mse
        and shuffled_mean > mse
    )
    result = {
        "mse": mse,
        "baseline_mse": baseline,
        "reconstruction_improvement": 1 - mse / baseline if baseline > 0 else None,
        "fixed_latent_mse": fixed_mse,
        "fixed_relative_increase": ratio(fixed_mse),
        "shuffled_latent_mses": shuffled_mses,
        "shuffle_seeds": SHUFFLE_SEEDS,
        "shuffled_latent_mse_mean": shuffled_mean,
        "shuffled_relative_increase": ratio(shuffled_mean),
        "latent_variance": np.var(mu, axis=0).tolist(),
        "kl": float(0.5 * np.mean(np.sum(mu**2 + np.exp(logvar) - 1 - logvar, axis=-1))),
        "channel_rmse_physical": np.sqrt(np.mean(physical_error**2, axis=(0, 1))).tolist(),
        "channels": CHANNELS,
        "units": ["N"] * 3 + ["N m"] * 3,
        "phases": {},
        "reconstruction_pass": reconstruction_pass(mse, baseline),
        "latent_utilization_pass": use_pass,
    }
    for phase, (start, stop) in PHASES.items():
        result["phases"][phase] = {
            "mse": float(np.mean((prediction[:, start:stop] - target[:, start:stop]) ** 2)),
            "baseline_mse": float(
                np.mean((mean_template[start:stop] - target[:, start:stop]) ** 2)
            ),
            "channel_rmse_physical": np.sqrt(
                np.mean(physical_error[:, start:stop] ** 2, axis=(0, 1))
            ).tolist(),
        }
    result["passed"] = result["reconstruction_pass"] and use_pass
    return result


@torch.no_grad()
def diagnose(model, values, mean_template, prep):
    prediction, mu, logvar = infer(model, values)
    fixed = decode_batches(model, mu.mean(0, keepdim=True).expand_as(mu)).numpy()
    shuffled = [
        decode_batches(model, mu[np.random.default_rng(seed).permutation(len(mu))]).numpy()
        for seed in SHUFFLE_SEEDS
    ]
    return diagnostic_arrays(
        values, prediction.numpy(), mu.numpy(), logvar.numpy(), fixed, shuffled, mean_template, prep
    )


def checkpoint_payload(model, prep, spec, epoch, mean_template, metadata):
    return {
        "format": "normal_vae_ablation_v1",
        "architecture": model.architecture,
        "model": model.state_dict(),
        "preprocessing": prep.state(),
        "spec": spec,
        "epoch": epoch,
        "train_mean_trajectory": torch.from_numpy(mean_template),
        "provenance": metadata,
    }


def load_experiment_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "normal_vae_ablation_v1":
        raise ValueError("Unsupported experiment checkpoint")
    model = AblationVAE(payload["architecture"], payload["spec"]["dropout"])
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def export_encoder(model, prep, path, metadata):
    save_checkpoint(
        path,
        {
            "format": "ablation_frozen_encoder_v1",
            "architecture": model.architecture,
            "encoder": model.encoder.state_dict(),
            "preprocessing": prep.state(),
            "interface": "raw local FT [B,400,6] -> deterministic mu [B,5]",
            "supervised": False,
            "metadata": metadata,
        },
    )


class FrozenAblationEncoder:
    """Independent architecture-aware raw-FT interface; original loader is unchanged."""

    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != "ablation_frozen_encoder_v1":
            raise ValueError("Unsupported encoder format")
        self.preprocessor = Preprocessor.from_state(payload["preprocessing"])
        self.model = make_encoder(payload["architecture"]).eval().requires_grad_(False)
        self.model.load_state_dict(payload["encoder"])
        self.metadata = {k: v for k, v in payload.items() if k != "encoder"}

    @torch.no_grad()
    def encode(self, raw_ft):
        values = self.preprocessor.transform(raw_ft)
        if not len(values):
            return np.empty((0, 5), dtype=np.float32)
        return torch.cat([self.model(x)[0] for x in torch.from_numpy(values).split(32)]).numpy()


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(history, out):
    if not history:
        return
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("mse", "kl", "weighted_kl", "loss")):
        for split in ("train", "validation"):
            rows = [r for r in history if split in r]
            if rows:
                ax.plot([r["epoch"] for r in rows], [r[split][metric] for r in rows], label=split)
        ax.set(xlabel="Step" if history[0].get("small") else "Epoch", ylabel=metric)
        ax.legend()
    fig.savefig(out / "training.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.plot([r["epoch"] for r in history], [r["beta"] for r in history])
    ax.set(xlabel="Step / epoch", ylabel="Beta")
    fig.savefig(out / "beta.png", dpi=140)
    plt.close(fig)


def fit_run(
    out,
    architecture,
    spec,
    training,
    validation,
    prep,
    mean_template,
    metadata,
    *,
    small=False,
    epochs=200,
    small_steps=2000,
):
    """Training accepts FT only. Labels and test trajectories cannot enter this API."""
    out = reserve_output(out)
    seed_all(spec["seed"])
    model = AblationVAE(architecture, spec["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(training)),
        batch_size=32,
        shuffle=not small,
        generator=torch.Generator().manual_seed(spec["seed"]),
    )
    history, best, started = [], float("inf"), time.monotonic()
    status = {"state": "running", "architecture": architecture, "spec": spec, "small": small}
    write_json(out / "status.json", status)
    write_json(out / "spec.json", spec)
    small_baseline = float(np.mean((training - training.mean(0)) ** 2)) if small else None
    small_threshold = min(0.001, 0.1 * small_baseline) if small else None
    try:
        for epoch in range(1, (small_steps if small else epochs) + 1):
            beta = beta_at(epoch, spec["beta"], spec["warmup"])
            row = {"epoch": epoch, "beta": beta, "small": small}
            model.train()
            totals = np.zeros(3)
            for (batch,) in train_loader:
                prediction, mu, logvar = model(batch, sample=not spec["deterministic"])
                losses = vae_loss(prediction, batch, mu, logvar, beta)
                finite_step(losses, model, optimizer)
                totals += np.array([v.item() for v in losses]) * len(batch)
            row["train"] = dict(zip(("loss", "mse", "kl"), (totals / len(training)).tolist()))
            row["train"]["weighted_kl"] = beta * row["train"]["kl"]
            if small:
                with torch.no_grad():
                    model.eval()
                    prediction, mu, logvar = model(torch.from_numpy(training), sample=False)
                    losses = vae_loss(prediction, torch.from_numpy(training), mu, logvar, 0.0)
                    finite_step(losses, model)
                row["post_step_mse"] = losses[1].item()
                eligible = True
                score = row["post_step_mse"]
            else:
                with torch.no_grad():
                    prediction, mu, logvar = infer(model, validation)
                    losses = vae_loss(prediction, torch.from_numpy(validation), mu, logvar, beta)
                    finite_step(losses, model)
                row["validation"] = dict(zip(("loss", "mse", "kl"), [v.item() for v in losses]))
                row["validation"]["weighted_kl"] = beta * row["validation"]["kl"]
                eligible = spec["deterministic"] or epoch >= 50
                score = row["validation"]["mse"]
            history.append(row)
            if eligible and score < best:
                best = score
                save_checkpoint(
                    out / "best.pt",
                    checkpoint_payload(model, prep, spec, epoch, mean_template, metadata),
                )
            if not small or epoch % 20 == 0:
                write_json(out / "history.json", history)
                write_json(out / "status.json", {**status, "epoch": epoch, "score": score})
            if epoch == 1 or epoch % (200 if small else 25) == 0:
                print(
                    json.dumps({"run": out.name, "epoch": epoch, "mse": score, "beta": beta}),
                    flush=True,
                )
            if small and score < small_threshold:
                break
        save_checkpoint(
            out / "last.pt", checkpoint_payload(model, prep, spec, epoch, mean_template, metadata)
        )
        status.update(state="completed", epoch=epoch, seconds=time.monotonic() - started)
        if small:
            status.update(
                passed=best < small_threshold,
                train_mse=best,
                subset_baseline_mse=small_baseline,
                strict_threshold=small_threshold,
            )
            if not status["passed"]:
                status["failure_reason"] = (
                    "Small AE did not meet both strict reconstruction thresholds"
                )
        else:
            chosen, payload = load_experiment_model(out / "best.pt")
            result = diagnose(chosen, validation, mean_template, prep)
            write_json(out / "validation.json", result)
            status.update(
                best_epoch=payload["epoch"],
                validation=result,
                passed=result["reconstruction_pass"] if spec["deterministic"] else result["passed"],
            )
            if not status["passed"]:
                status["failure_reason"] = (
                    "Validation reconstruction and/or latent utilization threshold not met"
                )
    except FloatingPointError as error:
        status.update(
            state="failed",
            passed=False,
            failure_reason=str(error),
            epoch=epoch,
            seconds=time.monotonic() - started,
        )
    except BaseException as error:
        status.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            passed=False,
            failure_reason=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )
        raise
    finally:
        write_json(out / "history.json", history)
        write_json(out / "status.json", status)
        plot_history(history, out)
    return status


def run_spec(candidate=None, seed=42):
    return {
        "name": "deterministic_ae",
        "beta": 0.0,
        "warmup": False,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "seed": seed,
        "deterministic": candidate is None,
        **(candidate or {}),
    }


def choose_candidate(rows):
    eligible = [r for r in rows if r.get("passed") and r.get("state") == "completed"]
    return min(eligible, key=lambda r: (r["validation"]["mse"], r["order"])) if eligible else None


def plot_diagnostics(model, values, prep, out, label):
    prediction, mu, logvar = infer(model, values)
    prediction, mu, logvar = prediction.numpy(), mu.numpy(), logvar.numpy()
    np.savez_compressed(out / "test_embeddings.npz", mu=mu, logvar=logvar)
    plt = plotting()
    plot_dir = out / "reconstructions"
    plot_dir.mkdir(exist_ok=True)
    target, estimate = prep.inverse(values), prep.inverse(prediction)
    t = np.arange(1, 401) / 100.0
    for index in TEST_INDICES:
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
        for channel, ax in enumerate(axes.flat):
            ax.plot(t, target[index, :, channel], label="Filtered FT")
            ax.plot(t, estimate[index, :, channel], label="Mean reconstruction")
            for boundary in (2, 3):
                ax.axvline(boundary, color="gray", linewidth=0.7, linestyle=":")
            ax.set(
                ylabel=f"{CHANNELS[channel]} ({'N' if channel < 3 else 'N m'})", xlabel="Time (s)"
            )
        axes.flat[0].legend()
        fig.suptitle(f"{label}: test index {index}")
        fig.savefig(plot_dir / f"test_{index:03d}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(np.arange(5), mu.var(0))
    axes[0].set(xlabel="Latent dimension", ylabel="Var(mu)")
    axes[1].bar(np.arange(5), np.exp(0.5 * logvar).mean(0))
    axes[1].set(xlabel="Latent dimension", ylabel="Mean posterior sigma")
    report = json.loads((out / "test.json").read_text())
    axes[2].bar(
        ["mu", "fixed", "shuffled", "template"],
        [
            report[k]
            for k in ("mse", "fixed_latent_mse", "shuffled_latent_mse_mean", "baseline_mse")
        ],
    )
    axes[2].set(ylabel="Test MSE")
    fig.suptitle(label)
    fig.savefig(out / "latent_diagnostics.png", dpi=140)
    plt.close(fig)


def write_report(out, report):
    write_json(out / "report.json", report)
    lines = [
        "# Normal-mode VAE ablation",
        "",
        "Engineering diagnostics, not paper-reported results.",
        "",
        f"Outcome: {report['outcome']}",
        "",
        "All 1000/100/100 trajectories were retained. No collection, controller or material changes.",
        "AE results are optimization diagnostics, not repaired VAE results.",
        "",
        "## Validation comparisons",
        "",
        "| Run | State | MSE | Template MSE | Fixed increase | Shuffle increase | Pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["runs"]:
        d = row.get("validation")
        if d:
            lines.append(
                f"| {row['run']} | {row['state']} | {d['mse']:.8g} | {d['baseline_mse']:.8g} | "
                f"{d['fixed_relative_increase']:.4g} | {d['shuffled_relative_increase']:.4g} | {row['passed']} |"
            )
        else:
            lines.append(
                f"| {row['run']} | {row['state']} | {row.get('train_mse', '')} | "
                f"{row.get('subset_baseline_mse', '')} | small AE | train only | {row.get('passed', False)} |"
            )
    lines += [
        "",
        "## Locked selection and test",
        "",
        f"Selected configuration: {report.get('selection')}",
        f"Stable across seeds: {report.get('stable', False)} (at least 2 of 3 test passes).",
        "",
        "| Seed/model | Epoch | MSE | Template MSE | Reconstruction | Latent use | Pass |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, d in report.get("test_results", {}).items():
        lines.append(
            f"| {name} | {d.get('epoch', 200)} | {d['mse']:.8g} | {d['baseline_mse']:.8g} | "
            f"{d['reconstruction_pass']} | {d['latent_utilization_pass']} | {d['passed']} |"
        )
    lines += [
        "",
        "## Physical and phase errors",
        "",
        "Filtered FT is the reconstruction target. Force RMSE is N; torque RMSE is N m.",
        "",
        "| Model / phase | MSE | Fx | Fy | Fz | Tx | Ty | Tz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in report.get("test_results", {}).items():
        for phase, error in [("all", d), *d["phases"].items()]:
            channels = " | ".join(f"{v:.6g}" for v in error["channel_rmse_physical"])
            lines.append(f"| {name} / {phase} | {error['mse']:.6g} | {channels} |")
    lines += [
        "",
        "## Property probes",
        "",
        "See probes/report.json and probes/report.md for each frozen input and predictor.",
        "Labels never participate in VAE optimization. stiffness_direct is not calibrated physical N/m.",
        "Bootstrap intervals describe held-out sample uncertainty conditional on the fitted model, not seed uncertainty.",
        "A failed probe does not prove that absolutely no information exists.",
        "",
        "## Failures and bounds",
        "",
    ]
    for row in report["runs"]:
        if row.get("failure_reason"):
            lines.append(f"- {row['run']}: {row['failure_reason']}")
    lines += [
        "",
        "Only one conditional nonlinear encoder fallback is permitted. Test scores never select configurations.",
        f"Original-file verification: {report.get('integrity')}",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def execute(dataset_config, output):
    cfg = load_config(dataset_config)
    if cfg.get("research_comparison", {}).get("controller") != "normal":
        raise ValueError("This protocol accepts only the normal-mode dataset")
    if cfg["dataset"] != dict(train=1000, validation=100, test=100):
        raise ValueError("Expected frozen 1000/100/100 dataset")
    if cfg["training"] != dict(
        epochs=200, batch_size=32, learning_rate=0.0001, beta=0.06, seed=42, device="cpu", threads=1
    ):
        raise ValueError("Original training configuration does not match the frozen baseline")
    source_dir = output_dir(cfg)
    frozen_provenance = provenance()
    original_paths = {ROOT / name for name in frozen_provenance["sources"]}
    original_paths.update(p for p in source_dir.iterdir() if p.is_file())
    original_paths.add(Path(dataset_config).resolve())
    before = snapshot(original_paths)
    out = reserve_output(output)
    own_sources = [
        Path(__file__),
        Path(__file__).with_name("vae_property_probe.py"),
    ]
    own_hashes = snapshot(own_sources)
    write_json(out / "protocol.json", PROTOCOL)
    write_json(out / "original_hashes.json", before)
    write_json(out / "source_hashes.json", own_hashes)
    write_json(out / "status.json", {"state": "running"})
    report = {"outcome": "running", "runs": [], "stable": False, "test_results": {}}
    try:
        seed_all(42)
        raw, data_hash = load_data(cfg)
        baseline_payload, baseline_model, prep = load_checkpoint(cfg)
        if (
            baseline_payload["epoch"] != 200
            or baseline_payload["data_provenance_hash"] != data_hash
        ):
            raise ValueError("Baseline checkpoint epoch or dataset mismatch")
        fitted = Preprocessor(**cfg["filter"], sample_hz=100).fit(raw["train"])
        if fitted.state() != prep.state():
            raise ValueError("Original preprocessing does not match training-only refit")
        with h5py.File(source_dir / "dataset.h5", "r") as h5:
            labels = {s: h5[s]["parameters"][:] for s in SPLITS}
        audit = split_audit(raw, labels)
        arrays = {s: prep.transform(raw[s]) for s in SPLITS}
        template = arrays["train"].mean(0)
        np.testing.assert_array_equal(template, baseline_payload["train_mean_trajectory"].numpy())
        indices = np.random.default_rng(42).choice(len(arrays["train"]), 32, replace=False)
        metadata = {
            "dataset_sha256": data_hash,
            "dataset_config_sha256": file_digest(dataset_config),
            "protocol_sha256": digest(PROTOCOL),
            "source_hashes": own_hashes,
            "packages": frozen_provenance["packages"],
            "python": frozen_provenance["python"],
        }
        write_json(
            out / "manifest.json",
            {
                **metadata,
                "split_audit": audit,
                "small_train_indices": indices.tolist(),
                "original_provenance": frozen_provenance,
            },
        )
        write_json(out / "preprocessing.json", prep.state())
        baseline_out = reserve_output(out / "baseline")
        baseline_val = diagnose(baseline_model, arrays["validation"], template, prep)
        write_json(baseline_out / "validation.json", baseline_val)
        original_history = json.loads((source_dir / "history.json").read_text())
        for row in original_history:
            row["beta"] = 0.06
            for split in ("train", "validation"):
                row[split]["weighted_kl"] = row[split]["kl"] * 0.06
        write_json(baseline_out / "history.json", original_history)
        plot_history(original_history, baseline_out)
        baseline_row = {
            "run": "baseline_reused",
            "state": "reused",
            "validation": baseline_val,
            "passed": baseline_val["passed"],
            "checkpoint": str(source_dir / "vae_last.pt"),
        }
        write_json(baseline_out / "status.json", baseline_row)
        report["runs"].append(baseline_row)
        selection = None
        for architecture in ARCHITECTURES:
            for small in (True, False):
                name = f"{architecture}_{'small' if small else 'full'}_ae"
                result = fit_run(
                    out / name,
                    architecture,
                    run_spec(),
                    arrays["train"][indices] if small else arrays["train"],
                    None if small else arrays["validation"],
                    prep,
                    template,
                    metadata,
                    small=small,
                )
                report["runs"].append({"run": name, **result})
                write_json(out / "progress.json", report)
                if not result["passed"]:
                    break
            if not result["passed"]:
                continue
            sweep = []
            for order, candidate in enumerate(CANDIDATES):
                name = f"{architecture}_{candidate['name']}_seed42"
                result = fit_run(
                    out / name,
                    architecture,
                    run_spec(candidate),
                    arrays["train"],
                    arrays["validation"],
                    prep,
                    template,
                    metadata,
                )
                row = {"run": name, "order": order, **result}
                sweep.append(row)
                report["runs"].append(row)
                write_json(out / "progress.json", report)
            selection = choose_candidate(sweep)
            if selection:
                break
        # Persist the configuration choice before any test diagnostics or supervised probes.
        report["selection"] = (
            {
                "run": selection["run"],
                "architecture": selection["architecture"],
                "spec": selection["spec"],
                "criterion": "validation MSE, table order breaks ties",
            }
            if selection
            else None
        )
        write_json(out / "selection.json", report["selection"])
        selected_runs = [selection] if selection else []
        if selection:
            for seed in (43, 44):
                spec = {**selection["spec"], "seed": seed}
                name = f"{selection['architecture']}_{spec['name']}_seed{seed}"
                result = fit_run(
                    out / name,
                    selection["architecture"],
                    spec,
                    arrays["train"],
                    arrays["validation"],
                    prep,
                    template,
                    metadata,
                )
                row = {"run": name, **result}
                report["runs"].append(row)
                selected_runs.append(row)
                write_json(out / "progress.json", report)
        baseline_test = diagnose(baseline_model, arrays["test"], template, prep)
        baseline_test["epoch"] = 200
        report["test_results"]["baseline"] = baseline_test
        write_json(baseline_out / "test.json", baseline_test)
        plot_diagnostics(
            baseline_model, arrays["test"], prep, baseline_out, "Original 200-epoch baseline"
        )
        new_model = None
        for row in selected_runs:
            if row["state"] != "completed":
                continue
            run_out = out / row["run"]
            model, payload = load_experiment_model(run_out / "best.pt")
            test = diagnose(model, arrays["test"], template, prep)
            test["epoch"] = payload["epoch"]
            report["test_results"][str(row["spec"]["seed"])] = test
            write_json(run_out / "test.json", test)
            export_encoder(
                model,
                prep,
                run_out / "encoder.pt",
                {**metadata, "validation": row["validation"], "test": test, "spec": row["spec"]},
            )
            loaded = FrozenAblationEncoder(run_out / "encoder.pt")
            np.testing.assert_allclose(
                loaded.encode(raw["test"]), infer(model, arrays["test"])[1].numpy(), atol=1e-7
            )
            plot_diagnostics(model, arrays["test"], prep, run_out, row["run"])
            if row["spec"]["seed"] == 42:
                new_model = model
        report["stable"] = (
            sum(d["passed"] for s, d in report["test_results"].items() if s != "baseline") >= 2
        )
        report["outcome"] = (
            "stable_improvement"
            if report["stable"]
            else "selected_but_not_stable"
            if selection
            else "bounded_experiments_failed"
        )
        from scripts.sim_pretrain.experiments.vae_property_probe import (
            run_probes,
        )

        report["probes"] = run_probes(
            out / "probes", arrays, labels, baseline_model, new_model, metadata
        )
        report["integrity"] = verify_snapshot(before)
        report["experiment_source_integrity"] = verify_snapshot(own_hashes)
        if (
            not report["integrity"]["unchanged"]
            or not report["experiment_source_integrity"]["unchanged"]
        ):
            raise RuntimeError("Frozen original or experiment source changed during execution")
        write_report(out, report)
        write_json(out / "status.json", {"state": "completed", "outcome": report["outcome"]})
        print(
            json.dumps(
                {"output": str(out), "outcome": report["outcome"], "selection": report["selection"]}
            ),
            flush=True,
        )
        return report
    except BaseException as error:
        report["outcome"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "execution_failed"
        )
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["integrity"] = verify_snapshot(before)
        write_report(out, report)
        write_json(
            out / "status.json",
            {"state": report["outcome"], "failure_reason": report["failure_reason"]},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.dataset_config, args.output)
    return 0 if report["stable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
