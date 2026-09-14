"""Paired ReLU/LeakyReLU diagnostic with gated normal-mode AE/VAE continuation."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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
    write_json,
)
from scripts.shared.model import vae_loss
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.vae_ablation import (
    CANDIDATES,
    AblationVAE,
    FrozenAblationEncoder,
    beta_at,
    choose_candidate,
    diagnose,
    export_encoder,
    finite_step,
    infer,
    load_experiment_model,
    plot_diagnostics,
    plot_history,
    plotting,
    reserve_output,
    run_spec,
    seed_all,
    snapshot,
    verify_snapshot,
)
from scripts.sim_pretrain.learning import load_checkpoint, load_data

PROTOCOL = {
    "version": 1,
    "purpose": "engineering activation experiment, not paper configuration",
    "architecture": "original_linear",
    "activations": ["relu", "leaky_relu"],
    "negative_slope": 0.01,
    "shared_initialization_seed": 42,
    "small_size": 32,
    "small_steps": 2000,
    "small_absolute_mse": 0.001,
    "small_template_fraction": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "small_deterministic": True,
    "small_beta": 0.0,
    "small_dropout": 0.0,
    "reproduction_rtol": 1e-6,
    "reproduction_atol": 1e-8,
    "epochs": 200,
    "validation_template_fraction": 0.8,
    "vae_candidates": CANDIDATES,
    "vae_selection_epochs": [50, 200],
    "vae_warmup_epochs": [1, 50],
    "repeat_seeds": [43, 44],
    "latent_use_increase": 0.1,
    "diagnostic_steps": "initial, step1, every100 steps, and last step",
    "fallback": "none; failed LeakyReLU small check stops, no linear decoder experiment",
    "test_policy": "score only after validation selection and seed repeats are locked",
    "property_policy": "existing frozen-probe results remain independent; no supervised fitting",
}


class ActivationVAE(AblationVAE):
    def __init__(self, activation, dropout=0.0):
        super().__init__("original_linear", dropout)
        if activation not in ("relu", "leaky_relu"):
            raise ValueError("Expected relu or leaky_relu")
        self.activation = activation
        if activation == "leaky_relu":
            self.decode_hidden[1] = nn.LeakyReLU(negative_slope=0.01)


def state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def initial_state(seed=42):
    seed_all(seed)
    return {k: v.detach().clone() for k, v in ActivationVAE("relu").state_dict().items()}


def payload(model, prep, spec, epoch, template, provenance):
    return {
        "format": "decoder_activation_v1",
        "architecture": "original_linear",
        "decoder_activation": model.activation,
        "negative_slope": 0.01 if model.activation == "leaky_relu" else 0.0,
        "model": model.state_dict(),
        "preprocessing": prep.state(),
        "spec": spec,
        "epoch": epoch,
        "train_mean_trajectory": torch.from_numpy(template),
        "provenance": provenance,
    }


def load_activation_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "decoder_activation_v1" or saved["architecture"] != "original_linear":
        raise ValueError("Unsupported activation checkpoint")
    expected_slope = 0.01 if saved["decoder_activation"] == "leaky_relu" else 0.0
    if saved["negative_slope"] != expected_slope:
        raise ValueError("Unexpected negative slope")
    model = ActivationVAE(saved["decoder_activation"], saved["spec"]["dropout"])
    model.load_state_dict(saved["model"])
    return model.eval(), saved


@torch.no_grad()
def activation_diagnostic(model, values, step):
    model.eval()
    x = torch.from_numpy(values)
    prediction, mu, _ = model(x, sample=False)
    pre = model.decode_hidden[0](mu)
    post = model.decode_hidden[1](pre)
    return {
        "step": step,
        "mse": float((prediction - x).square().mean()),
        "nonpositive_preactivation_fraction": float((pre <= 0).float().mean()),
        "never_positive_on_diagnostic_rows_fraction": float((pre <= 0).all(0).float().mean()),
        "exact_zero_activation_fraction": float((post == 0).float().mean()),
        "prediction_variance": float(prediction.var(0, unbiased=False).mean()),
        "mean_trajectory_fit_mse": float((prediction.mean(0) - x.mean(0)).square().mean()),
        "latent_variance": mu.var(0, unbiased=False).tolist(),
        "gradient_norms_last_training_batch": {
            name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None
        },
        "note": "Never-positive is measured on these rows, not proof of globally inactive units",
    }


def train_activation(
    output,
    activation,
    spec,
    training,
    validation,
    prep,
    template,
    provenance,
    *,
    initial=None,
    small=False,
    small_steps=2000,
    epochs=200,
):
    """FT-only trainer: same loss/order as v1, with explicit activation and initial weights."""
    out = reserve_output(output)
    seed_all(spec["seed"])
    model = ActivationVAE(activation, spec["dropout"])
    if initial is not None:
        model.load_state_dict(initial)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(training)),
        batch_size=32,
        shuffle=not small,
        generator=torch.Generator().manual_seed(spec["seed"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    status = {
        "state": "running",
        "activation": activation,
        "architecture": "original_linear",
        "spec": spec,
        "small": small,
        "initial_state_sha256": state_hash(model.state_dict()),
    }
    write_json(out / "status.json", status)
    write_json(out / "spec.json", {**spec, "activation": activation})
    baseline = float(np.mean((training - training.mean(0)) ** 2)) if small else None
    threshold = min(0.001, 0.1 * baseline) if small else None
    history, diagnostics, best = [], [], float("inf")
    started, epoch = time.monotonic(), 0
    try:
        diagnostics.append(activation_diagnostic(model, training[:32], 0))
        for epoch in range(1, (small_steps if small else epochs) + 1):
            beta = beta_at(epoch, spec["beta"], spec["warmup"])
            row = {"epoch": epoch, "beta": beta, "small": small}
            model.train()
            totals = np.zeros(3)
            for (batch,) in loader:
                prediction, mu, logvar = model(batch, sample=not spec["deterministic"])
                losses = vae_loss(prediction, batch, mu, logvar, beta)
                finite_step(losses, model, optimizer)
                totals += np.array([value.item() for value in losses]) * len(batch)
            row["train"] = dict(zip(("loss", "mse", "kl"), (totals / len(training)).tolist()))
            row["train"]["weighted_kl"] = beta * row["train"]["kl"]
            with torch.no_grad():
                if small:
                    model.eval()
                    prediction, mu, logvar = model(torch.from_numpy(training), sample=False)
                    losses = vae_loss(prediction, torch.from_numpy(training), mu, logvar, 0.0)
                    finite_step(losses, model)
                    score = row["post_step_mse"] = losses[1].item()
                else:
                    prediction, mu, logvar = infer(model, validation)
                    losses = vae_loss(prediction, torch.from_numpy(validation), mu, logvar, beta)
                    finite_step(losses, model)
                    row["validation"] = dict(
                        zip(("loss", "mse", "kl"), [value.item() for value in losses])
                    )
                    row["validation"]["weighted_kl"] = beta * row["validation"]["kl"]
                    score = row["validation"]["mse"]
            history.append(row)
            if (small or spec["deterministic"] or epoch >= 50) and score < best:
                best = score
                save_checkpoint(
                    out / "best.pt", payload(model, prep, spec, epoch, template, provenance)
                )
            if epoch == 1 or epoch % 100 == 0:
                diagnostics.append(activation_diagnostic(model, training[:32], epoch))
                write_json(out / "activation_diagnostics.json", diagnostics)
            if not small or epoch % 20 == 0:
                write_json(out / "history.json", history)
                write_json(out / "status.json", {**status, "epoch": epoch, "score": score})
            if epoch == 1 or epoch % (200 if small else 25) == 0:
                print(
                    json.dumps({"run": out.name, "epoch": epoch, "mse": score, "beta": beta}),
                    flush=True,
                )
            if small and score < threshold:
                break
        save_checkpoint(out / "last.pt", payload(model, prep, spec, epoch, template, provenance))
        if diagnostics[-1]["step"] != epoch:
            diagnostics.append(activation_diagnostic(model, training[:32], epoch))
        status.update(
            state="completed",
            epoch=epoch,
            seconds=time.monotonic() - started,
            final_state_sha256=state_hash(model.state_dict()),
        )
        if small:
            status.update(
                passed=best < threshold,
                train_mse=best,
                subset_baseline_mse=baseline,
                strict_threshold=threshold,
            )
        else:
            selected, saved = load_activation_model(out / "best.pt")
            validation_report = diagnose(selected, validation, template, prep)
            write_json(out / "validation.json", validation_report)
            status.update(
                best_epoch=saved["epoch"],
                validation=validation_report,
                passed=validation_report["reconstruction_pass"]
                if spec["deterministic"]
                else validation_report["passed"],
            )
        if not status["passed"]:
            status["failure_reason"] = (
                "Small AE threshold not met" if small else "Validation gate not met"
            )
    except FloatingPointError as error:
        status.update(state="failed", passed=False, failure_reason=str(error), epoch=epoch)
    except BaseException as error:
        status.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            passed=False,
            failure_reason=f"{type(error).__name__}: {error}",
            epoch=epoch,
        )
        raise
    finally:
        write_json(out / "status.json", status)
        write_json(out / "history.json", history)
        write_json(out / "activation_diagnostics.json", diagnostics)
        plot_history(history, out)
    return status


def compare_reproduction(current, previous):
    current, previous = Path(current), Path(previous)
    a = json.loads((current / "history.json").read_text())
    b = json.loads((previous / "history.json").read_text())
    if len(a) != len(b) or not a:
        return {"passed": False, "reason": "History length mismatch", "lengths": [len(a), len(b)]}
    # Include KL and pre-update train losses, not just the final post-update score.
    flatten = lambda rows: np.array(
        [
            [
                r["epoch"],
                r["beta"],
                r["post_step_mse"],
                *[r["train"][k] for k in ("loss", "mse", "kl", "weighted_kl")],
            ]
            for r in rows
        ]
    )
    x, y = flatten(a), flatten(b)
    old, _ = load_experiment_model(previous / "last.pt")
    new, _ = load_activation_model(current / "last.pt")
    old_state, new_state = old.state_dict(), new.state_dict()
    tolerance = dict(rtol=PROTOCOL["reproduction_rtol"], atol=PROTOCOL["reproduction_atol"])
    history_match = bool(np.allclose(x, y, **tolerance))
    weights_match = old_state.keys() == new_state.keys() and all(
        torch.allclose(old_state[k], new_state[k], **tolerance) for k in old_state
    )
    return {
        "passed": history_match and weights_match,
        "steps": len(a),
        "history_match": history_match,
        "weights_match": weights_match,
        "history_bitwise_equal": bool(np.array_equal(x, y)),
        "weights_bitwise_equal": state_hash(old_state) == state_hash(new_state),
        "history_max_abs_difference": float(np.max(np.abs(x - y))),
        "weights_max_abs_difference": max(
            float((old_state[k] - new_state[k]).abs().max()) for k in old_state
        ),
        **tolerance,
    }


def next_stage(reproduction, leaky_small=None, full_ae=None):
    if not reproduction.get("passed"):
        return "stop_reproduction_mismatch"
    if leaky_small is None:
        return "leaky_small"
    if leaky_small.get("state") != "completed" or not leaky_small.get("passed"):
        return "stop_small_ae_failed"
    if full_ae is None:
        return "full_ae"
    if full_ae.get("state") != "completed" or not full_ae.get("passed"):
        return "stop_full_ae_failed"
    return "vae_sweep"


def plot_pair(out, values, prep, indices):
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    estimates = {}
    for name in ("relu_small_ae", "leaky_small_ae"):
        folder = out / name
        if not (folder / "last.pt").exists():
            continue
        history = json.loads((folder / "history.json").read_text())
        diagnostics = json.loads((folder / "activation_diagnostics.json").read_text())
        axes[0].semilogy(
            [r["epoch"] for r in history], [r["post_step_mse"] for r in history], label=name
        )
        axes[1].plot(
            [r["step"] for r in diagnostics],
            [r["never_positive_on_diagnostic_rows_fraction"] for r in diagnostics],
            label=name,
        )
        axes[2].semilogy(
            [r["step"] for r in diagnostics],
            [r["mean_trajectory_fit_mse"] for r in diagnostics],
            label=name,
        )
        model, _ = load_activation_model(folder / "best.pt")
        estimates[name] = prep.inverse(infer(model, values)[0].numpy())
    threshold = min(0.001, 0.1 * float(np.mean((values - values.mean(0)) ** 2)))
    axes[0].axhline(threshold, color="black", linestyle="--", label="Small AE threshold")
    for ax, label in zip(
        axes,
        ("Post-step training MSE", "Never-positive fraction on fixed32", "Mean-trajectory fit MSE"),
    ):
        ax.set(xlabel="Step", ylabel=label)
        ax.legend(fontsize=8)
    fig.savefig(out / "activation_comparison.png", dpi=140)
    plt.close(fig)
    target = prep.inverse(values)
    for local_index in (0, 10, 20, 30):
        if local_index >= len(values):
            continue
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
        for c, ax in enumerate(axes.flat):
            t = np.arange(1, 401) / 100.0
            ax.plot(t, target[local_index, :, c], label="Filtered train FT")
            for name, estimate in estimates.items():
                ax.plot(t, estimate[local_index, :, c], label=name, alpha=0.8)
            ax.set(xlabel="Time (s)", ylabel=f"{CHANNELS[c]} ({'N' if c < 3 else 'N m'})")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Training-only paired reconstruction: dataset row {indices[local_index]}")
        fig.savefig(out / f"train_reconstruction_{indices[local_index]:03d}.png", dpi=140)
        plt.close(fig)


def continue_after_pair(out, small_status, arrays, prep, template, metadata, report):
    """Validation-only continuation; no architecture fallback and no test input."""
    if next_stage(report["reproduction"], small_status) != "full_ae":
        return None
    full = train_activation(
        out / "leaky_full_ae",
        "leaky_relu",
        run_spec(),
        arrays["train"],
        arrays["validation"],
        prep,
        template,
        metadata,
    )
    report["runs"].append({"run": "leaky_full_ae", **full})
    write_json(out / "progress.json", report)
    if next_stage(report["reproduction"], small_status, full) != "vae_sweep":
        report["outcome"] = "full_ae_failed"
        return None
    candidates = []
    for order, candidate in enumerate(CANDIDATES):
        name = f"leaky_{candidate['name']}_seed42"
        result = train_activation(
            out / name,
            "leaky_relu",
            run_spec(candidate),
            arrays["train"],
            arrays["validation"],
            prep,
            template,
            metadata,
        )
        row = {"run": name, "order": order, **result}
        candidates.append(row)
        report["runs"].append(row)
        write_json(out / "progress.json", report)
    selection = choose_candidate(candidates)
    if selection is None:
        report["outcome"] = "vae_gates_failed"
    return selection


def summarize(out, report):
    write_json(out / "report.json", report)
    lines = [
        "# Decoder activation experiment",
        "",
        "Engineering experiment, not paper configuration.",
        "",
        f"Outcome: {report['outcome']}",
        "",
        f"ReLU reproduction: {report.get('reproduction')}",
        "",
        "| Run | State | Best small-train / validation MSE | Passed |",
        "|---|---|---:|---|",
    ]
    for row in report["runs"]:
        mse = row.get("train_mse", row.get("validation", {}).get("mse", "unavailable"))
        lines.append(f"| {row['run']} | {row['state']} | {mse} | {row.get('passed', False)} |")
    lines += [
        "",
        f"Selected VAE: {report.get('selection')}",
        f"Test scored: {report.get('test_scored', False)}; stable: {report.get('stable', False)}.",
        "",
        "The small AE threshold is strictly below both 1e-3 and 10% of the fixed subset template MSE.",
        "Only LeakyReLU(.01) differs between A and B; all initial parameter tensors are shared.",
        "No linear decoder, nonlinear encoder, longer budget, new data or supervised training is enabled.",
        "Never-positive activation on 32 rows is not proof of globally inactive neurons.",
        "Small AE improvements do not by themselves establish repaired or stable VAE behavior.",
        "",
        "## Conditional test results",
        "",
    ]
    for name, metrics in report.get("test_results", {}).items():
        lines.append(
            f"- {name}: MSE={metrics['mse']:.8g}, baseline={metrics['baseline_mse']:.8g}, passed={metrics['passed']}"
        )
    lines += [
        "",
        "Test data are not used for any gate or configuration choice. Existing test results were already",
        "seen in earlier work; any final test report is not a new blind benchmark.",
        "",
        f"Protected original and previous experiment files: {report.get('integrity')}",
        f"Experiment sources: {report.get('source_integrity')}",
        "",
    ]
    if report.get("failure_reason"):
        lines += [f"Failure: {report['failure_reason']}", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def execute(dataset_config, previous_output, output):
    from scripts.shared.paths import ARCHIVE, read_path

    previous_output = Path(previous_output).resolve()
    cfg = load_config(dataset_config)
    destination = Path(output).resolve()
    data_directory = read_path(cfg["output_dir"])
    if not data_directory.is_absolute():
        data_directory = ROOT / data_directory
    if destination.is_relative_to(previous_output) or destination.is_relative_to(
        data_directory.resolve()
    ):
        raise ValueError(
            "Output must be outside previous experiment and frozen dataset directories"
        )
    if cfg.get("research_comparison", {}).get("controller") != "normal":
        raise ValueError("Normal-mode data required")
    if cfg["dataset"] != dict(train=1000, validation=100, test=100):
        raise ValueError("Frozen 1000/100/100 data required")
    previous_manifest = json.loads((previous_output / "manifest.json").read_text())
    previous_status = json.loads((previous_output / "status.json").read_text())
    if previous_status != {"state": "completed", "outcome": "bounded_experiments_failed"}:
        raise ValueError("Expected the completed previous bounded experiment")
    if file_digest(dataset_config) != previous_manifest["dataset_config_sha256"]:
        raise ValueError("Use the same frozen dataset configuration")
    protected = {
        **json.loads((previous_output / "original_hashes.json").read_text()),
        **json.loads((previous_output / "source_hashes.json").read_text()),
    }
    if previous_output.is_relative_to(ARCHIVE):
        protected = {
            str(read_path(path, historical=True, source_sha256=expected)): expected
            for path, expected in protected.items()
        }
    if not verify_snapshot(protected)["unchanged"]:
        raise ValueError("Original files or previous experiment sources changed")
    protected.update(snapshot(p for p in previous_output.rglob("*") if p.is_file()))
    out = reserve_output(output)
    own_sources = snapshot(
        [
            Path(__file__),
            Path(__file__).with_name("vae_ablation.py"),
        ]
    )
    write_json(out / "protocol.json", PROTOCOL)
    write_json(out / "original_hashes.json", protected)
    write_json(out / "source_hashes.json", own_sources)
    write_json(out / "status.json", {"state": "running"})
    report = {
        "outcome": "running",
        "runs": [],
        "selection": None,
        "test_scored": False,
        "stable": False,
    }
    try:
        seed_all(42)
        raw, data_sha = load_data(cfg)
        if data_sha != previous_manifest["dataset_sha256"]:
            raise ValueError("Different dataset content")
        baseline, _, prep = load_checkpoint(cfg)
        if baseline["data_provenance_hash"] != data_sha or baseline["epoch"] != 200:
            raise ValueError("Original baseline mismatch")
        fitted = Preprocessor(**cfg["filter"], sample_hz=100).fit(raw["train"])
        if fitted.state() != prep.state():
            raise ValueError("Training-only preprocessing mismatch")
        arrays = {s: prep.transform(raw[s]) for s in ("train", "validation")}
        indices = np.asarray(previous_manifest["small_train_indices"], dtype=int)
        expected = np.random.default_rng(42).choice(len(arrays["train"]), 32, replace=False)
        np.testing.assert_array_equal(indices, expected)
        template = arrays["train"].mean(0)
        np.testing.assert_array_equal(template, baseline["train_mean_trajectory"].numpy())
        state = initial_state()
        save_checkpoint(
            out / "shared_initialization.pt",
            {"seed": 42, "model": state, "state_sha256": state_hash(state)},
        )
        metadata = {
            "dataset_sha256": data_sha,
            "dataset_config_sha256": file_digest(dataset_config),
            "protocol_sha256": digest(PROTOCOL),
            "source_hashes": own_sources,
            "previous_manifest_sha256": file_digest(previous_output / "manifest.json"),
        }
        write_json(
            out / "manifest.json",
            {
                **metadata,
                "small_train_indices": indices.tolist(),
                "shared_initial_state_sha256": state_hash(state),
                "previous_output": str(previous_output),
                "split_counts": {s: len(x) for s, x in raw.items()},
                "preprocessing_fit_split": "train",
                "removed_samples": 0,
            },
        )
        write_json(out / "preprocessing.json", prep.state())
        small = arrays["train"][indices]
        control = train_activation(
            out / "relu_small_ae",
            "relu",
            run_spec(),
            small,
            None,
            prep,
            template,
            metadata,
            initial=state,
            small=True,
        )
        report["runs"].append({"run": "relu_small_ae", **control})
        report["reproduction"] = (
            compare_reproduction(
                out / "relu_small_ae", previous_output / "original_linear_small_ae"
            )
            if control["state"] == "completed"
            else {"passed": False, "reason": "ReLU run failed"}
        )
        write_json(out / "reproduction.json", report["reproduction"])
        write_json(out / "progress.json", report)
        selection = None
        if next_stage(report["reproduction"]) != "leaky_small":
            report["outcome"] = "reproduction_mismatch"
        else:
            leaky = train_activation(
                out / "leaky_small_ae",
                "leaky_relu",
                run_spec(),
                small,
                None,
                prep,
                template,
                metadata,
                initial=state,
                small=True,
            )
            report["runs"].append({"run": "leaky_small_ae", **leaky})
            if control["initial_state_sha256"] != leaky["initial_state_sha256"]:
                raise RuntimeError("Paired initial weights differ")
            report["outcome"] = "small_ae_failed"
            write_json(out / "progress.json", report)
            selection = continue_after_pair(out, leaky, arrays, prep, template, metadata, report)
        plot_pair(out, small, prep, indices)
        report["selection"] = (
            {"run": selection["run"], "spec": selection["spec"], "activation": "leaky_relu"}
            if selection
            else None
        )
        write_json(out / "selection.json", report["selection"])
        if selection:
            confirmed = [selection]
            for seed in (43, 44):
                spec = {**selection["spec"], "seed": seed}
                name = f"leaky_{spec['name']}_seed{seed}"
                result = train_activation(
                    out / name,
                    "leaky_relu",
                    spec,
                    arrays["train"],
                    arrays["validation"],
                    prep,
                    template,
                    metadata,
                )
                row = {"run": name, **result}
                report["runs"].append(row)
                confirmed.append(row)
                write_json(out / "progress.json", report)
            values = prep.transform(raw["test"])
            report["test_scored"], report["test_results"] = True, {}
            for row in confirmed:
                if row["state"] != "completed":
                    continue
                folder = out / row["run"]
                model, saved = load_activation_model(folder / "best.pt")
                metrics = diagnose(model, values, template, prep)
                metrics["epoch"] = saved["epoch"]
                report["test_results"][str(row["spec"]["seed"])] = metrics
                write_json(folder / "test.json", metrics)
                export_encoder(
                    model,
                    prep,
                    folder / "encoder.pt",
                    {
                        **metadata,
                        "decoder_activation": "leaky_relu",
                        "spec": row["spec"],
                        "test": metrics,
                    },
                )
                loaded = FrozenAblationEncoder(folder / "encoder.pt")
                np.testing.assert_allclose(
                    loaded.encode(raw["test"]), infer(model, values)[1].numpy(), atol=1e-7
                )
                plot_diagnostics(model, values, prep, folder, row["run"])
            report["stable"] = sum(v["passed"] for v in report["test_results"].values()) >= 2
            report["outcome"] = (
                "stable_improvement" if report["stable"] else "selected_but_not_stable"
            )
        report["integrity"] = verify_snapshot(protected)
        report["source_integrity"] = verify_snapshot(own_sources)
        if not report["integrity"]["unchanged"] or not report["source_integrity"]["unchanged"]:
            raise RuntimeError("Protected files or experiment sources changed")
        summarize(out, report)
        write_json(out / "status.json", {"state": "completed", "outcome": report["outcome"]})
        print(
            json.dumps(
                {
                    "output": str(out),
                    "outcome": report["outcome"],
                    "test_scored": report["test_scored"],
                }
            ),
            flush=True,
        )
        return report
    except BaseException as error:
        report.update(
            outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "execution_failed",
            failure_reason=f"{type(error).__name__}: {error}",
            integrity=verify_snapshot(protected),
        )
        summarize(out, report)
        write_json(
            out / "status.json",
            {"state": report["outcome"], "failure_reason": report["failure_reason"]},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument(
        "--previous-output", type=Path, default=ROOT / "archive/sim_pretrain/normal_vae_ablation_v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.dataset_config, args.previous_output, args.output)
    return 0 if report["stable"] else 3 if report["outcome"] == "reproduction_mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
