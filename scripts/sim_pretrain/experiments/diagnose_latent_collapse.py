"""Offline attribution of a saved VAE collapse; never replaces a production model."""

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from scripts.shared.common import CHANNELS, file_digest, write_json
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.paths import read_path
from scripts.shared.preprocessing import Preprocessor
from scripts.shared.run_paths import new_output
from scripts.sim_pretrain.experiments.repair_sponge_vae import RepairedVAE, initialize
from scripts.sim_pretrain.experiments.vae_property_probe import (
    Standardizer, fit_feature_probes, fit_pca5, ridge_coefficients, transform_pca,
)

SPLITS = ("train", "validation", "test")
ALPHAS = (0.0, 0.0001, 0.01, 1.0, 100.0, 1000.0)
PROTOCOL = {
    "purpose": "offline component diagnosis, not a replacement/export",
    "seeds": [42, 43, 44],
    "shared_decoder_epochs": 400,
    "shared_decoder_learning_rate": 0.001,
    "shared_decoder_batch_size": 32,
    "shared_decoder_dropout": 0.0,
    "paired_vae_epochs": 200,
    "paired_vae_learning_rate": 0.0001,
    "paired_vae_batch_size": 32,
    "paired_vae_betas": [0.0, 0.0001, 0.06],
    "paired_vae_initialization": "identical train-only PCA initialization and full linear decoder",
    "selection": "validation MSE only; every predeclared seed is reported",
    "ridge_alphas": list(ALPHAS),
    "shuffle_seeds": list(range(4200, 4220)),
    "bootstrap_repeats": 1000,
    "test_caveat": "existing historical split, not a fresh blind benchmark",
    "hardware_ready": False,
}


def mse(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float64) - b) ** 2))


def readout(features, targets):
    """Only train/validation are consumed for fitting and selection."""
    scaler = Standardizer().fit(features["train"])
    x = {s: scaler.transform(features[s]) for s in ("train", "validation")}
    center = np.asarray(targets["train"], dtype=np.float64).mean(0)
    y = targets["train"] - center
    coefficients = [np.linalg.lstsq(x["train"], y, rcond=None)[0]]
    coefficients += ridge_coefficients(x["train"], y, ALPHAS[1:])
    scores = [mse(x["validation"] @ w + center, targets["validation"]) for w in coefficients]
    selected = int(np.argmin(scores))
    return scaler, center, coefficients[selected], {
        "alpha": ALPHAS[selected], "validation_mse": scores[selected],
        "candidates": dict(zip(map(str, ALPHAS), scores)), "fit_split": "train",
    }


def noise_optimal_decoder(mu, logvar, targets):
    """Exact expected-MSE optimum of an affine decoder under q(z|x)."""
    z = np.asarray(mu, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    z_mean, y_mean = z.mean(0), y.mean(0)
    z, y = z - z_mean, y - y_mean
    noise = np.exp(np.asarray(logvar, dtype=np.float64)).sum(0)
    weight = np.linalg.solve(z.T @ z + np.diag(noise), z.T @ y)
    return z_mean, y_mean, weight


def metric_row(prediction, target, template, prep, *, seed=42):
    pred, target = np.asarray(prediction), np.asarray(target)
    errors = np.mean((pred - target) ** 2, axis=(1, 2))
    base = np.mean((target - template) ** 2, axis=(1, 2))
    indices = np.random.default_rng(seed).integers(0, len(target), (1000, len(target)))
    ci = np.quantile((base - errors)[indices].mean(1), [0.025, 0.975])
    physical_error = prep.inverse(pred) - prep.inverse(target)
    return {
        "mse": float(errors.mean()), "baseline_mse": float(base.mean()),
        "improvement_over_template": float(1 - errors.mean() / base.mean()),
        "paired_mse_improvement_ci95": ci.tolist(),
        "channel_rmse_physical": np.sqrt(np.mean(physical_error ** 2, axis=(0, 1))).tolist(),
    }


def fit_neural(model, features, targets, folder, *, seed, epochs, lr, batch_size, beta=None):
    """Same initialization / batches / sampling seeds within each paired comparison."""
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.as_tensor(features["train"], dtype=torch.float32)
    y = torch.as_tensor(targets["train"], dtype=torch.float32)
    vx = torch.as_tensor(features["validation"], dtype=torch.float32)
    vy = torch.as_tensor(targets["validation"], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=False)
    history, best, best_state = [], float("inf"), None
    for epoch in range(epochs + 1):
        train_mse, train_kl = 0.0, 0.0
        if epoch:
            model.train()
            for idx in torch.randperm(len(x), generator=generator).split(batch_size):
                if beta is None:
                    prediction = model(x[idx])
                    recon_loss = (prediction - y[idx]).square().mean()
                    kl = torch.zeros(())
                    loss = recon_loss
                else:
                    prediction, mu, logvar = model(x[idx], sample=True)
                    loss, recon_loss, kl = vae_loss(prediction, y[idx], mu, logvar, beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {folder}, epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                train_mse += float(recon_loss.detach()) * len(idx) / len(x)
                train_kl += float(kl.detach()) * len(idx) / len(x)
        model.eval()
        with torch.no_grad():
            if beta is None:
                prediction = model(vx)
                validation_kl, variance = None, None
            else:
                prediction, mu, logvar = model(vx, sample=False)
                validation_kl = float(vae_loss(prediction, vy, mu, logvar, beta)[2])
                variance = mu.var(0, unbiased=False).tolist()
            score = float((prediction - vy).square().mean())
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation error")
        row = dict(epoch=epoch, train_mse=train_mse if epoch else None,
                   train_kl=train_kl if epoch else None, validation_mse=score,
                   validation_kl=validation_kl, latent_variance=variance)
        history.append(row)
        if score < best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 50 == 0 or epoch == epochs:
            write_json(folder / "history.json", history)
            print(json.dumps(dict(run=folder.name, **row)), flush=True)
    torch.save(dict(model=model.state_dict(), epoch=epochs), folder / "last.pt")
    torch.save(dict(model=best_state, epoch=best_epoch), folder / "best.pt")
    selection = dict(seed=seed, best_epoch=best_epoch, validation_mse=best,
                     last_validation_mse=score, beta=beta, epochs=epochs,
                     selection_split="validation", initial_validation_mse=history[0]["validation_mse"])
    write_json(folder / "selection.json", selection)
    last_model = copy.deepcopy(model).eval()
    model.load_state_dict(best_state)
    return model.eval(), last_model, selection


class SharedDecoder(nn.Module):
    """Original decoder architecture, deterministic inputs and no dropout."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(5, 2000), nn.ReLU())
        self.frame = nn.Linear(5, 6)

    def forward(self, z):
        return self.frame(self.hidden(z).reshape(-1, 400, 5))


def plot_results(out, targets, predictions, prep, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["original", "frozen_mu_linear", "frozen_frame_pca5", "pca5_linear"]
    names += [n for n in predictions if n.startswith("shared_")]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(np.arange(len(names)), [metrics[n]["mse"] for n in names])
    ax.axhline(metrics["original"]["baseline_mse"], color="black", linestyle="--", label="Train mean template")
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("Test normalized MSE (lower is better)")
    ax.legend()
    fig.savefig(out / "reconstruction_errors.png", dpi=150)
    plt.close(fig)
    for index in [0, 30, 60, 90]:
        if index >= len(targets):
            continue
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained", sharex=True)
        t = np.arange(1, 401) / 100
        for j, ax in enumerate(axes.flat):
            ax.plot(t, prep.inverse(targets[index])[:, j], color="black", label="Target")
            for name in ["original", "frozen_mu_linear", "pca5_linear"]:
                ax.plot(t, prep.inverse(predictions[name][index])[:, j], label=name, alpha=.85)
            ax.set_ylabel(f"{CHANNELS[j]} ({'N' if j < 3 else 'N m'})")
            ax.set_xlabel("Time (s)")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Fixed test row {index}")
        fig.savefig(out / f"reconstruction_{index:03d}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for path in sorted((out / "paired_vae").glob("seed_42_*/history.json")):
        history = json.loads(path.read_text())
        label = path.parent.name
        axes[0].plot([r["epoch"] for r in history], [r["validation_mse"] for r in history], label=label)
        axes[1].plot([r["epoch"] for r in history], [r["validation_kl"] for r in history], label=label)
    for ax, label in zip(axes, ["Validation mean-decoding MSE", "Validation KL"]):
        ax.set(xlabel="Epoch", ylabel=label, yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(out / "paired_beta.png", dpi=150)
    plt.close(fig)


def run(source, output):
    source = read_path(source).resolve()
    out = Path(output).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Diagnostic output must not overlap the source")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protected = [*source.glob("*.pt"), source / "dataset.h5", source / "evaluation.json",
                 source / "preprocessing.json", source / "history.json"]
    sources = [Path(__file__), Path(__file__).parents[2] / "shared/model.py",
               Path(__file__).parents[2] / "shared/preprocessing.py",
               Path(__file__).with_name("vae_property_probe.py"),
               Path(__file__).with_name("repair_sponge_vae.py")]
    hashes = {str(p): file_digest(p) for p in protected + sources}
    write_json(out / "manifest.json", dict(source=str(source), sha256=hashes,
               protocol=PROTOCOL, torch=torch.__version__, numpy=np.__version__))
    write_json(out / "status.json", dict(state="running", stage="representation_readouts"))
    saved = torch.load(source / "vae_last.pt", map_location="cpu", weights_only=True)
    prior = json.loads((source / "evaluation.json").read_text())
    if (hashes[str(source / "dataset.h5")] != saved["data_provenance_hash"]
            or prior["checkpoint_sha256"] != hashes[str(source / "vae_last.pt")]):
        raise ValueError("Source dataset or evaluation provenance mismatch")
    original = SpongeVAE().eval().requires_grad_(False)
    original.load_state_dict(saved["model"])
    original_state = copy.deepcopy(original.state_dict())
    prep = Preprocessor.from_state(saved["preprocessing"])
    with h5py.File(source / "dataset.h5", "r") as f:
        if not f.attrs["complete"] or any(not f[s]["valid"][:].all() for s in SPLITS):
            raise ValueError("Source dataset is incomplete")
        raw = {s: f[s]["ft"][:] for s in SPLITS}
        labels = {s: f[s]["parameters"][:] for s in SPLITS}
    arrays = {s: prep.transform(raw[s]) for s in SPLITS}
    flat = {s: x.reshape(len(x), -1).astype(np.float64) for s, x in arrays.items()}
    template = arrays["train"].mean(0)
    features, logvars, frames = {}, {}, {}
    with torch.no_grad():
        for s in SPLITS:
            mu, lv = original.encoder(torch.from_numpy(arrays[s]))
            features[s], logvars[s] = mu.numpy(), lv.numpy()
            frames[s] = original.encoder.frame(torch.from_numpy(arrays[s])).flatten(1).numpy()
    pca = fit_pca5(flat["train"])
    frame_pca = fit_pca5(frames["train"])
    representation = {
        "frozen_mu_linear": features,
        "frozen_mu_logvar_linear": {s: np.column_stack([features[s], logvars[s]]) for s in SPLITS},
        "frozen_frame_pca5": {s: transform_pca(frames[s], frame_pca) for s in SPLITS},
        "pca5_linear": {s: transform_pca(flat[s], pca) for s in SPLITS},
    }
    np.savez_compressed(out / "pca5.npz", **pca)
    np.savez_compressed(out / "frame_pca5.npz", **frame_pca)
    predictions, selections, decode_functions = {}, {}, {}
    with torch.no_grad():
        predictions["original"] = original(torch.from_numpy(arrays["test"]), sample=False)[0].numpy()
    for name, values in representation.items():
        scaler, center, weight, selection = readout(values, flat)
        selections[name] = selection
        np.savez_compressed(out / f"{name}.npz", feature_mean=scaler.mean,
                            feature_scale=scaler.scale, target_mean=center, weight=weight)
        decode_functions[name] = (lambda z, sc=scaler, c=center, w=weight:
                                   (sc.transform(z) @ w + c).reshape(-1, 400, 6))
    write_json(out / "linear_selection.json", selections)
    for name in representation:
        predictions[name] = decode_functions[name](representation[name]["test"])
    metrics = {name: metric_row(pred, arrays["test"], template, prep)
               for name, pred in predictions.items()}
    for name, decoder in decode_functions.items():
        z = representation[name]["test"]
        metrics[name]["fixed_latent_mse"] = mse(decoder(np.broadcast_to(z.mean(0), z.shape)), arrays["test"])
        metrics[name]["shuffled_latent_mse_mean"] = float(np.mean([
            mse(decoder(z[np.random.default_rng(seed).permutation(len(z))]), arrays["test"])
            for seed in PROTOCOL["shuffle_seeds"]]))
    latent = dict(train_mu_variance=features["train"].var(0).tolist(),
                  train_posterior_noise_variance=np.exp(logvars["train"]).mean(0).tolist(),
                  signal_to_noise_ratio=(features["train"].var(0) / np.exp(logvars["train"]).mean(0)).tolist(),
                  train_logvar_variance=logvars["train"].var(0).tolist(),
                  train_mu_singular_values=np.linalg.svd(features["train"] - features["train"].mean(0), compute_uv=False).tolist())
    zm, ym, nw = noise_optimal_decoder(features["train"], logvars["train"], flat["train"])
    noisy_mean = (features["test"] - zm) @ nw + ym
    noise_penalty = float(np.mean(np.exp(logvars["test"]) @ np.sum(nw ** 2, axis=1)) / flat["test"].shape[1])
    latent["optimal_sampled_linear_expected_test_mse"] = mse(noisy_mean, flat["test"]) + noise_penalty
    latent["optimal_sampled_linear_mean_decode_test_mse"] = mse(noisy_mean, flat["test"])
    write_json(out / "latent_diagnostics.json", latent)
    write_json(out / "linear_results.json", metrics)
    print(json.dumps(dict(stage="linear_complete", metrics=metrics, latent=latent)), flush=True)

    # Refit exactly the original decoder architecture on standardized frozen codes.
    for name in ("frozen_mu_linear", "pca5_linear"):
        sc = Standardizer().fit(representation[name]["train"])
        standardized = {s: sc.transform(representation[name][s]).astype(np.float32) for s in SPLITS}
        write_json(out / f"shared_{name}_scaler.json", sc.state())
        for seed in PROTOCOL["seeds"]:
            key = f"shared_{name}_seed_{seed}"
            write_json(out / "status.json", dict(state="running", stage=key))
            torch.manual_seed(seed)
            model, _, selection = fit_neural(SharedDecoder(), standardized, arrays, out / key,
                seed=seed, epochs=PROTOCOL["shared_decoder_epochs"],
                lr=PROTOCOL["shared_decoder_learning_rate"], batch_size=32)
            selections[key] = selection
            with torch.no_grad():
                predictions[key] = model(torch.from_numpy(standardized["test"])).numpy()
            metrics[key] = metric_row(predictions[key], arrays["test"], template, prep)
            write_json(out / "reconstruction_results.json", metrics)

    # Conditional causal test: hold informative initialization/architecture fixed; vary beta only.
    torch.manual_seed(42)
    initial = RepairedVAE()
    initialization = initialize(initial, arrays["train"])
    torch.save(initial.state_dict(), out / "paired_initial.pt")
    paired = {}
    for seed in PROTOCOL["seeds"]:
        for beta in PROTOCOL["paired_vae_betas"]:
            key = f"seed_{seed}_beta_{beta:g}"
            write_json(out / "status.json", dict(state="running", stage=key))
            best, last, selection = fit_neural(initial, arrays, arrays, out / "paired_vae" / key,
                seed=seed, epochs=PROTOCOL["paired_vae_epochs"],
                lr=PROTOCOL["paired_vae_learning_rate"], batch_size=32, beta=beta)
            paired[key] = dict(selection=selection)
            for which, model in [("best", best), ("last", last)]:
                with torch.no_grad():
                    prediction, mu, lv = model(torch.from_numpy(arrays["test"]), sample=False)
                    row = metric_row(prediction.numpy(), arrays["test"], template, prep)
                    row["kl"] = float(vae_loss(prediction, torch.from_numpy(arrays["test"]), mu, lv, beta)[2])
                    row["latent_variance"] = mu.var(0, unbiased=False).tolist()
                    row["posterior_noise_variance"] = lv.exp().mean(0).tolist()
                    row["fixed_latent_mse"] = mse(model.decode(mu.mean(0).expand_as(mu)).numpy(), arrays["test"])
                    row["shuffled_latent_mse_mean"] = float(np.mean([
                        mse(model.decode(mu[np.random.default_rng(s).permutation(len(mu))]).numpy(), arrays["test"])
                        for s in PROTOCOL["shuffle_seeds"]]))
                paired[key][which] = row
            write_json(out / "paired_results.json", paired)

    properties = {}
    for name in ("frozen_mu_linear", "pca5_linear"):
        write_json(out / "status.json", dict(state="running", stage=f"property_{name}"))
        properties[name], _ = fit_feature_probes(name, representation[name], labels,
            out / "properties" / name, {"checkpoint_sha256": prior["checkpoint_sha256"]})
    if any(not torch.equal(original_state[k], original.state_dict()[k]) for k in original_state):
        raise RuntimeError("Frozen original parameters changed")
    unchanged = {p: file_digest(p) == h for p, h in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("Protected source or artifact changed during experiment")
    report = dict(protocol=PROTOCOL, source=str(source), original_epoch=saved["epoch"],
        reconstruction=metrics, selection=selections, latent=latent,
        paired_initialization=initialization, paired_vae=paired, properties=properties,
        audit=dict(all_protected_hashes_unchanged=True, original_encoder_frozen=True),
        limitations=["Same historical test split; not a fresh blind benchmark.",
            "Frozen mean readout and stochastic VAE reconstruction are different questions.",
            "Shared decoder refits change conditioning, optimization and dropout; not a pure architecture attribution.",
            "Beta pairing isolates beta only within PCA-initialized full-linear-decoder architecture.",
            "Failed property probes do not prove absence of all nonlinear information.",
            "Material parameter labels are simulator parameters, not calibrated real material properties."])
    write_json(out / "report.json", report)
    np.savez_compressed(out / "test_reconstructions.npz", target=arrays["test"], **predictions)
    plot_results(out, arrays["test"], predictions, prep, metrics)
    lines = ["# Current VAE component diagnosis", "", f"Source: `{source}`", "",
        "Normalized MSE; physical RMSE and paired bootstrap intervals are in report.json.", "",
        "| Variant | Test MSE | Improvement over mean |", "|---|---:|---:|"]
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['mse']:.9g} | {row['improvement_over_template']:.2%} |")
    lines += ["", "## Paired beta test (last epoch)", "",
        "| Run | Test MSE | KL | Latent variance max |", "|---|---:|---:|---:|"]
    for name, row in paired.items():
        r = row["last"]
        lines.append(f"| {name} | {r['mse']:.9g} | {r['kl']:.6g} | {max(r['latent_variance']):.6g} |")
    lines += ["", "## Frozen property readouts", "",
        "| Input | Selected predictor | Friction R2 | Stiffness R2 | Width R2 |", "|---|---|---:|---:|---:|"]
    for name, info in properties.items():
        chosen = info["selected"]
        r2 = [chosen["test"][k]["r2"] for k in ("friction", "stiffness_direct", "width")]
        lines.append(f"| {name} | {chosen['name']} | {r2[0]:.4f} | {r2[1]:.4f} | {r2[2]:.4f} |")
    lines += ["", "All protected hashes are unchanged. The source encoder stayed frozen.", ""]
    lines += [f"- {s}" for s in report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "status.json", dict(state="complete", report=str(out / "report.json")))
    print(f"Completed diagnosis: {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/sim_training/pretrain_wide_1200_v1")
    parser.add_argument("--output", default="runs/sim_training/latent_collapse_diagnosis")
    args = parser.parse_args()
    out = new_output(args.output)
    try:
        run(args.source, out)
    except Exception as exc:
        if out.exists() and (out / "status.json").exists():
            write_json(out / "failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
