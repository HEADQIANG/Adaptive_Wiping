"""Joint original-architecture VAE retraining with train-only initialization and KL warmup."""

import argparse
import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import CHANNELS, digest, file_digest, write_json
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.paths import read_path, writable_path
from scripts.shared.preprocessing import Preprocessor
from scripts.shared.run_paths import new_output
from scripts.sim_pretrain.experiments.repair_sponge_vae import RepairedVAE, initialize
from scripts.sim_pretrain.experiments.vae_ablation import diagnose, finite_step, infer, seed_all
from scripts.sim_pretrain.experiments.vae_property_probe import (
    fit_feature_probes, fit_pca5, transform_pca,
)

FORMAT = "wide_joint_vae_retrain_v1"


def validate_config(cfg):
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported retraining schema")
    t = cfg["training"]
    for name in ("epochs", "deterministic_warmup_epochs", "kl_ramp_epochs",
                 "batch_size", "checkpoint_every"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(f"Invalid {name}")
    if t["epochs"] <= t["deterministic_warmup_epochs"] + t["kl_ramp_epochs"]:
        raise ValueError("Require training after KL reaches its final value")
    if (not t["beta_candidates"] or len(set(t["beta_candidates"])) != len(t["beta_candidates"])
            or not all(np.isfinite(v) and v > 0 for v in t["beta_candidates"])):
        raise ValueError("Require unique positive finite beta candidates")
    if not 0 < t["final_learning_rate"] <= t["learning_rate"] < 1:
        raise ValueError("Invalid learning-rate schedule")
    if t["dropout"] != 0 or t["device"] != "cpu" or t["threads"] != 1:
        raise ValueError("This protocol requires no dropout and single-thread CPU")
    seeds = [t["selection_seed"], *t["repeat_seeds"]]
    if len(set(seeds)) != len(seeds) or any(type(v) is not int or v < 0 for v in seeds):
        raise ValueError("Require distinct nonnegative seeds")
    a = cfg["acceptance"]
    if not 0 < a["reconstruction_fraction"] < 1 or a["latent_ablation_ratio"] <= 1:
        raise ValueError("Invalid acceptance thresholds")
    if not 1 <= a["required_passing_seeds"] <= len(seeds):
        raise ValueError("Invalid seed acceptance count")


def schedule(epoch, target_beta, training):
    if epoch < 1 or epoch > training["epochs"]:
        raise ValueError("Epoch outside the training budget")
    warm = training["deterministic_warmup_epochs"]
    sample = epoch > warm
    beta = target_beta * min(max(epoch - warm, 0) / training["kl_ramp_epochs"], 1)
    progress = max(epoch - warm, 0) / (training["epochs"] - warm)
    lo, hi = training["final_learning_rate"], training["learning_rate"]
    lr = lo + (hi - lo) * (1 + math.cos(math.pi * progress)) / 2
    return float(beta), sample, lr


@torch.no_grad()
def initialize_original(training):
    """Fit a five-dimensional reconstruction inside the existing ReLU decoder."""
    linear = RepairedVAE()
    info = initialize(linear, training)
    model = SpongeVAE()
    model.decode_hidden[2].p = 0.0
    model.encoder.load_state_dict(linear.encoder.state_dict())
    x = np.asarray(training, dtype=np.float64)
    center = x.mean((0, 1))
    projection = model.encoder.frame.weight.numpy().T.astype(np.float64)
    projected = ((x - center) @ projection).reshape(len(x), 2000)
    codes = model.encoder(torch.from_numpy(x.astype(np.float32)))[0].numpy().astype(np.float64)
    design = np.column_stack([codes, np.ones(len(codes))])
    coefficient = np.linalg.lstsq(design, projected, rcond=None)[0]
    values = (design @ coefficient).reshape(-1, 400, 5)
    # A per-channel shift works with the existing time-shared final affine layer.
    shift = np.maximum(-values.min((0, 1)), 0) + 0.01
    model.decode_hidden[0].weight.copy_(torch.tensor(coefficient[:5].T, dtype=torch.float32))
    model.decode_hidden[0].bias.copy_(torch.tensor(coefficient[5] + np.tile(shift, 400), dtype=torch.float32))
    model.decode_frame.weight.copy_(torch.tensor(projection, dtype=torch.float32))
    model.decode_frame.bias.copy_(torch.tensor(center - shift @ projection.T, dtype=torch.float32))
    model.eval()
    error = float((model(torch.from_numpy(training), sample=False)[0] - torch.from_numpy(training)).square().mean())
    return model, {**info, "original_architecture_initialized_mse": error,
                   "relu_shift": shift.tolist(), "fit_split": "train"}


def load_model(path):
    saved = torch.load(read_path(path), map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Not a joint wide-data retraining checkpoint")
    model = SpongeVAE()
    model.decode_hidden[2].p = saved["training"]["dropout"]
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def passed(metrics, acceptance):
    return bool(metrics["mse"] <= acceptance["reconstruction_fraction"] * metrics["baseline_mse"]
        and metrics["fixed_latent_mse"] >= acceptance["latent_ablation_ratio"] * metrics["mse"]
        and metrics["shuffled_latent_mse_mean"] >= acceptance["latent_ablation_ratio"] * metrics["mse"])


def train_joint(initial, arrays, prep, folder, cfg, beta, seed, provenance):
    seed_all(seed)
    t = cfg["training"]
    folder.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(initial).requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    x, vx = (torch.from_numpy(arrays[s]) for s in ("train", "validation"))
    loader_rng = torch.Generator().manual_seed(seed)
    initial_weights = copy.deepcopy(model.state_dict())
    template = arrays["train"].mean(0)
    history, best_score, best_payload = [], float("inf"), None
    selection_start = t["deterministic_warmup_epochs"] + t["kl_ramp_epochs"]
    for epoch in range(1, t["epochs"] + 1):
        weight, sample, lr = schedule(epoch, beta, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = np.zeros(3)
        for idx in torch.randperm(len(x), generator=loader_rng).split(t["batch_size"]):
            reconstruction, mu, logvar = model(x[idx], sample=sample)
            losses = vae_loss(reconstruction, x[idx], mu, logvar, weight)
            finite_step(losses, model, optimizer)
            total += np.array([float(v.detach()) for v in losses]) * len(idx) / len(x)
        model.eval()
        with torch.no_grad():
            reconstruction, mu, logvar = model(vx, sample=False)
            validation = float((reconstruction - vx).square().mean())
            kl = float(vae_loss(reconstruction, vx, mu, logvar, weight)[2])
        if not np.isfinite(validation) or not np.isfinite(kl):
            raise FloatingPointError("Nonfinite validation metrics")
        history.append(dict(epoch=epoch, beta=weight, sample=sample, learning_rate=lr,
            train_loss=float(total[0]), train_mse=float(total[1]), train_kl=float(total[2]),
            validation_mse=validation, validation_kl=kl,
            validation_latent_variance=mu.var(0, unbiased=False).tolist()))
        payload = dict(format=FORMAT, model=copy.deepcopy(model.state_dict()), epoch=epoch,
            training=t, target_beta=beta, seed=seed, preprocessing=prep.state(),
            train_mean_trajectory=torch.from_numpy(template), **provenance)
        if epoch >= selection_start and validation < best_score:
            best_score, best_payload = validation, payload
            save_checkpoint(folder / "best.pt", payload)
        if epoch % t["checkpoint_every"] == 0 or epoch == t["epochs"]:
            save_checkpoint(folder / f"epoch_{epoch:04d}.pt", payload)
            write_json(folder / "history.json", history)
            print(json.dumps(dict(run=folder.name, **history[-1])), flush=True)
    save_checkpoint(folder / "last.pt", payload)
    model.load_state_dict(best_payload["model"])
    changes = {
        part: float(torch.sqrt(sum((model.state_dict()[key] - value).square().sum()
            for key, value in initial_weights.items() if key.startswith(prefix))))
        for part, prefix in [("encoder", "encoder."), ("decoder", "decode_")]
    }
    if any(value == 0 for value in changes.values()):
        raise RuntimeError("Both encoder and decoder must have been trained")
    validation = diagnose(model, arrays["validation"], template, prep)
    validation["passed"] = passed(validation, cfg["acceptance"])
    result = dict(beta=beta, seed=seed, best_epoch=best_payload["epoch"], validation=validation,
                  selection_start_epoch=selection_start, parameter_update_l2=changes,
                  checkpoint=str(folder / "best.pt"))
    write_json(folder / "validation.json", result)
    return result


def export_encoder(model, saved, validation, test, path):
    if not validation["passed"] or not test["passed"]:
        raise ValueError("Cannot export an unaccepted encoder")
    evaluation = dict(test, test_mse=test["mse"], test_kl=test["kl"],
        beats_mean_baseline=test["mse"] < test["baseline_mse"], collapse_warning=False,
        shuffled_latent_mse=test["shuffled_latent_mse_mean"])
    evaluation["active_latent_dimensions_at_1e-4"] = int(
        np.sum(np.asarray(test["latent_variance"]) > 1e-4))
    save_checkpoint(path, dict(encoder=model.encoder.state_dict(), preprocessing=saved["preprocessing"],
        architecture="frame6to5_flat2000_posterior10_split5", channels=saved["channels"],
        frame=saved["frame"], units=saved["units"], ft_processing=saved["ft_processing"],
        epoch=saved["epoch"], data_provenance_hash=saved["data_provenance_hash"],
        evaluation=evaluation, config=saved["retraining_config"], training_format=FORMAT,
        source_run=saved["source_run"], hardware_ready=False))


def figures(out, histories, target, prediction, original, prep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for name, history in histories.items():
        for ax, key in zip(axes, ["validation_mse", "validation_kl", "beta"]):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=name)
            ax.set(xlabel="Epoch", ylabel=key)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)
    for idx in [0, 30, 60, 90]:
        if idx >= len(target):
            continue
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained", sharex=True)
        for channel, ax in enumerate(axes.flat):
            for values, label in [(target, "Target"), (original, "Old VAE"), (prediction, "Retrained VAE")]:
                ax.plot(np.arange(1, 401) / 100, prep.inverse(values[idx])[:, channel], label=label)
            ax.set(ylabel=f"{CHANNELS[channel]} ({'N' if channel < 3 else 'N m'})", xlabel="Time (s)")
        axes.flat[0].legend()
        fig.suptitle(f"Fixed test row {idx}")
        fig.savefig(out / f"reconstruction_{idx:03d}.png", dpi=140)
        plt.close(fig)


def run(cfg, output, config_path):
    validate_config(cfg)
    source, out = read_path(cfg["source_run"]).resolve(), writable_path(output)
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Output overlaps source")
    out.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(out)
    write_json(out / "run_config.json", cfg)
    write_json(out / "status.json", dict(stage="loading", state="running"))
    seed_all(cfg["training"]["selection_seed"])
    protected = [*source.glob("*.pt"), source / "dataset.h5", source / "evaluation.json",
        source / "preprocessing.json", source / "history.json", read_path(config_path), Path(__file__),
        Path(__file__).parents[2] / "shared/model.py",
        Path(__file__).parents[2] / "shared/preprocessing.py",
        Path(__file__).with_name("repair_sponge_vae.py"), Path(__file__).with_name("vae_ablation.py"),
        Path(__file__).with_name("vae_property_probe.py")]
    hashes = {str(p.resolve()): file_digest(p) for p in protected}
    write_json(out / "manifest.json", dict(config=cfg, protected_sha256=hashes,
        selection="validation only after final beta is reached; export selection_seed",
        test_caveat="historical test split, not a new blind benchmark"))
    old = torch.load(source / "vae_last.pt", map_location="cpu", weights_only=True)
    data_hash = hashes[str((source / "dataset.h5").resolve())]
    if data_hash != old["data_provenance_hash"]:
        raise ValueError("Source checkpoint/data mismatch")
    prep = Preprocessor.from_state(old["preprocessing"])
    with h5py.File(source / "dataset.h5", "r") as f:
        if not f.attrs["complete"] or f.attrs["config_hash"] != old["config_hash"]:
            raise ValueError("Dataset incomplete or config mismatch")
        for s, count in old["config"]["dataset"].items():
            if f[s]["ft"].shape != (count, 400, 6) or not f[s]["valid"][:].all():
                raise ValueError(f"Invalid split {s}")
        raw_train = f["train"]["ft"][:]
        arrays = {"train": prep.transform(raw_train), "validation": prep.transform(f["validation"]["ft"][:])}
        processing = str(f.attrs["ft_processing"])
    fitted = Preprocessor(prep.order, prep.cutoff_hz, prep.sample_hz).fit(raw_train)
    if fitted.state() != prep.state():
        raise ValueError("Source normalization is not reproducible from train split")
    initial, init_info = initialize_original(arrays["train"])
    template = arrays["train"].mean(0)
    init_info["validation"] = diagnose(initial, arrays["validation"], template, prep)
    write_json(out / "initialization.json", init_info)
    provenance = dict(source_run=str(source), data_provenance_hash=data_hash,
        frame=old["frame"], channels=old["channels"], units=old["units"], ft_processing=processing,
        retraining_config=cfg, retraining_config_hash=digest(cfg))
    t = cfg["training"]
    candidates = []
    for beta in t["beta_candidates"]:
        folder = out / f"candidate_beta_{beta:g}"
        write_json(out / "status.json", dict(stage=folder.name, state="running"))
        candidates.append(train_joint(initial, arrays, prep, folder, cfg, beta, t["selection_seed"], provenance))
    eligible = [r for r in candidates if r["validation"]["passed"]]
    if not eligible:
        write_json(out / "selection.json", dict(candidates=candidates, selected=None))
        raise RuntimeError("No candidate passed validation")
    chosen = min(eligible, key=lambda r: r["validation"]["mse"])
    write_json(out / "selection.json", dict(candidates=candidates, selected=chosen, test_used=False))
    seeds = {str(t["selection_seed"]): chosen}
    for seed in t["repeat_seeds"]:
        folder = out / f"repeat_seed_{seed}"
        write_json(out / "status.json", dict(stage=folder.name, state="running"))
        seeds[str(seed)] = train_joint(initial, arrays, prep, folder, cfg, chosen["beta"], seed, provenance)
    write_json(out / "locked_selections.json", seeds)
    if sum(r["validation"]["passed"] for r in seeds.values()) < cfg["acceptance"]["required_passing_seeds"]:
        raise RuntimeError("Selected training failed multi-seed validation")
    # Test data is first transformed/scored after all hyperparameters and epochs are locked.
    with h5py.File(source / "dataset.h5", "r") as f:
        raw_test = f["test"]["ft"][:]
        arrays["test"] = prep.transform(raw_test)
        labels = {s: f[s]["parameters"][:] for s in arrays}
    tests = {}
    for seed, result in seeds.items():
        model, saved = load_model(result["checkpoint"])
        test = diagnose(model, arrays["test"], template, prep)
        test["passed"] = passed(test, cfg["acceptance"])
        tests[seed] = test
        write_json(Path(result["checkpoint"]).parent / "test.json", test)
    if sum(r["passed"] for r in tests.values()) < cfg["acceptance"]["required_passing_seeds"]:
        raise RuntimeError("Selected training failed historical test acceptance")
    selected, saved = load_model(chosen["checkpoint"])
    selected_test = tests[str(t["selection_seed"])]
    save_checkpoint(out / "vae_best.pt", saved)
    last = torch.load(Path(chosen["checkpoint"]).parent / "last.pt", map_location="cpu", weights_only=True)
    save_checkpoint(out / "vae_last.pt", last)
    export_encoder(selected, saved, chosen["validation"], selected_test, out / "encoder.pt")
    frozen = FrozenSpongeEncoder(out / "encoder.pt")
    with torch.no_grad():
        expected = selected.encoder(torch.from_numpy(arrays["test"]))[0].numpy()
    np.testing.assert_array_equal(frozen.encode(raw_test), expected)
    baseline = SpongeVAE().eval()
    baseline.load_state_dict(old["model"])
    baseline_metrics = diagnose(baseline, arrays["test"], template, prep)
    models = {"original": baseline, "retrained": selected}
    features = {}
    for name, model in models.items():
        model.eval().requires_grad_(False)
        features[name] = {s: infer(model, x)[1].numpy() for s, x in arrays.items()}
    flat = {s: x.reshape(len(x), -1) for s, x in arrays.items()}
    pca = fit_pca5(flat["train"])
    features["pca5"] = {s: transform_pca(x, pca) for s, x in flat.items()}
    np.savez_compressed(out / "pca5.npz", **pca)
    properties = {}
    for name, values in features.items():
        write_json(out / "status.json", dict(stage=f"properties_{name}", state="running"))
        properties[name], _ = fit_feature_probes(name, values, labels, out / "properties" / name,
            {"source_dataset_sha256": data_hash, "training_selection_locked": True})
    unchanged = all(file_digest(p) == h for p, h in hashes.items())
    if not unchanged:
        raise RuntimeError("Protected input or source changed during training")
    prediction = infer(selected, arrays["test"])[0].numpy()
    old_prediction = infer(baseline, arrays["test"])[0].numpy()
    np.savez_compressed(out / "test_reconstructions.npz", target=arrays["test"],
        original=old_prediction, retrained=prediction, mu=features["retrained"]["test"])
    histories = {Path(r["checkpoint"]).parent.name: json.loads((Path(r["checkpoint"]).parent / "history.json").read_text()) for r in candidates}
    figures(out, histories, arrays["test"], prediction, old_prediction, prep)
    report = dict(outcome="joint_retraining_passed", source=str(source), selected=chosen,
        test=tests, original_test=baseline_metrics, initialization=init_info, properties=properties,
        original_mse_improvement=1 - selected_test["mse"] / baseline_metrics["mse"],
        active_latent_dimensions=int(np.sum(np.asarray(selected_test["latent_variance"]) > 1e-4)),
        all_protected_hashes_unchanged=True, frozen_export_exact=True, hardware_ready=False,
        artifacts=dict(encoder=str(out / "encoder.pt"), vae=str(out / "vae_best.pt")),
        limitations=["Historical split, not a new blind benchmark.",
            "Motion-quality limitations of the source dataset remain.",
            "Existing downstream policies require new preparation/training for a changed encoder.",
            "A trained VAE need not beat deterministic PCA initialization; both scores are reported."])
    write_json(out / "report.json", report)
    lines = ["# Joint VAE retraining", "", f"Selected beta: {chosen['beta']}; export seed: {t['selection_seed']}.",
        "", "| Seed | Selected epoch | Test MSE | Fixed code MSE | Shuffled code MSE | Passed |",
        "|---|---:|---:|---:|---:|---|"]
    for seed, r in tests.items():
        lines.append(f"| {seed} | {seeds[seed]['best_epoch']} | {r['mse']:.9g} | {r['fixed_latent_mse']:.9g} | {r['shuffled_latent_mse_mean']:.9g} | {r['passed']} |")
    lines += ["", f"Old test MSE: {baseline_metrics['mse']:.9g}; improvement: {report['original_mse_improvement']:.2%}.",
        f"Active latent dimensions: {report['active_latent_dimensions']}/5.", "",
        "| Property input | Friction R2 | Stiffness R2 | Width R2 |", "|---|---:|---:|---:|"]
    for name, p in properties.items():
        scores = p["selected"]["test"]
        lines.append(f"| {name} | {scores['friction']['r2']:.4f} | {scores['stiffness_direct']['r2']:.4f} | {scores['width']['r2']:.4f} |")
    lines += ["", "Original files unchanged; frozen export exactly matches trained encoder.", "", *report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "status.json", dict(state="complete", outcome=report["outcome"]))
    print(json.dumps(dict(output=str(out), beta=chosen["beta"], epoch=chosen["best_epoch"], test=selected_test)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/retrain_wide_1200.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(read_path(args.config).read_text())
    validate_config(cfg)
    out = new_output(cfg["output_dir"])
    try:
        run(cfg, out, args.config)
    except Exception as exc:
        if (out / "status.json").exists():
            write_json(out / "status.json", dict(state="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
