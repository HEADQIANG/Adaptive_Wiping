"""Supervised readouts of frozen FT representations, separate from VAE training."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import write_json

PROPERTIES = ("friction", "stiffness_direct", "width")
RIDGE_ALPHAS = (1e-4, 1e-2, 1.0, 100.0)


class Standardizer:
    def fit(self, training):
        values = np.asarray(training, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected nonempty finite feature matrix")
        self.mean = values.mean(0)
        self.scale = values.std(0)
        # Retain small but nonzero mu variance; do not use a collapse threshold here.
        self.scale = np.where(np.ptp(values, axis=0) == 0, 1.0, self.scale)
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit_split": "train"}

    @classmethod
    def from_state(cls, state):
        instance = cls()
        instance.mean, instance.scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        return instance


def fit_pca5(training):
    """Exact train-centered PCA via the smaller sample Gram matrix."""
    values = np.asarray(training, dtype=np.float64)
    center = values.mean(0)
    x = values - center
    if min(x.shape) < 5:
        raise ValueError("PCA5 requires at least five samples/features")
    if x.shape[0] < x.shape[1]:
        eigenvalues, vectors = linalg.eigh(x @ x.T, subset_by_index=[len(x) - 5, len(x) - 1])
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        tolerance = np.finfo(float).eps * max(x.shape) * max(float(eigenvalues[0]), 1.0)
        components = np.zeros((5, x.shape[1]))
        valid = eigenvalues > tolerance
        components[valid] = (x.T @ vectors[:, valid] / np.sqrt(eigenvalues[valid])).T
    else:
        _, singular, components = linalg.svd(x, full_matrices=False)
        components = components[:5]
        eigenvalues = singular[:5] ** 2
    return {"mean": center, "components": components, "eigenvalues": eigenvalues}


def transform_pca(values, state):
    return (np.asarray(values, dtype=np.float64) - state["mean"]) @ state["components"].T


def ridge_coefficients(x, y, alphas=RIDGE_ALPHAS):
    """Unnormalized squared-error + alpha*||W||^2, intercept supplied by scaling."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dual = x.shape[1] > len(x)
    gram = x @ x.T if dual else x.T @ x
    values, vectors = linalg.eigh(gram)
    projected = vectors.T @ (y if dual else x.T @ y)
    left = x.T @ vectors if dual else vectors
    return [left @ (projected / (np.maximum(values, 0)[:, None] + alpha)) for alpha in alphas]


def property_metrics(target, prediction, train_mean, *, repeats=1000, seed=42):
    target, prediction = (
        np.asarray(target, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
    )
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Expected matching [N,3] property arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite property prediction")
    error2 = (prediction - target) ** 2
    baseline2 = (np.asarray(train_mean) - target) ** 2
    indices = np.random.default_rng(seed).integers(0, len(target), size=(repeats, len(target)))
    improvements = np.sqrt(baseline2[indices].mean(1)) - np.sqrt(error2[indices].mean(1))
    interval = np.quantile(improvements, [0.025, 0.975], axis=0)
    variance = ((target - target.mean(0)) ** 2).mean(0)
    report = {}
    for i, name in enumerate(PROPERTIES):
        r2 = float(1 - error2[:, i].mean() / variance[i]) if variance[i] > 0 else None
        report[name] = {
            "rmse": float(np.sqrt(error2[:, i].mean())),
            "mae": float(np.abs(prediction[:, i] - target[:, i]).mean()),
            "r2": r2,
            "constant_rmse": float(np.sqrt(baseline2[:, i].mean())),
            "rmse_improvement": float(
                np.sqrt(baseline2[:, i].mean()) - np.sqrt(error2[:, i].mean())
            ),
            "paired_rmse_improvement_ci95": interval[:, i].tolist(),
            "predictive_information_detected": bool(
                r2 is not None and r2 > 0 and interval[0, i] > 0
            ),
        }
    return report


def make_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
    )


def fit_mlp(x, y, output, metadata, *, epochs=400, patience=50):
    from scripts.sim_pretrain.experiments.vae_ablation import (
        finite_step,
        seed_all,
    )

    seed_all(42)
    model = make_mlp(x["train"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x["train"], dtype=torch.float32),
            torch.tensor(y["train"], dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    vx, vy = (
        torch.tensor(x["validation"], dtype=torch.float32),
        torch.tensor(y["validation"], dtype=torch.float32),
    )
    best, stale, history, best_state = float("inf"), 0, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for features, labels in loader:
            loss = (model(features) - labels).square().mean()
            finite_step([loss], model, optimizer)
            total += loss.item() * len(features)
        with torch.no_grad():
            model.eval()
            validation = (model(vx) - vy).square().mean()
            finite_step([validation], model)
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / len(x["train"]),
                "validation_mse": validation.item(),
            }
        )
        if validation.item() < best:
            best, stale = validation.item(), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            save_checkpoint(
                output / "mlp_best.pt",
                {
                    **metadata,
                    "kind": "mlp",
                    "model": best_state,
                    "input_dim": x["train"].shape[1],
                    "epoch": epoch,
                },
            )
        else:
            stale += 1
        write_json(output / "mlp_history.json", history)
        if epoch == 1 or epoch % 50 == 0:
            print(
                json.dumps(
                    {"probe": output.name, "epoch": epoch, "validation_mse": validation.item()}
                ),
                flush=True,
            )
        if stale >= patience:
            break
    save_checkpoint(
        output / "mlp_last.pt",
        {
            **metadata,
            "kind": "mlp",
            "model": model.state_dict(),
            "input_dim": x["train"].shape[1],
            "epoch": epoch,
        },
    )
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, {
        "name": "mlp",
        "kind": "mlp",
        "validation_standardized_mse": best,
        "best_epoch": best_epoch,
        "epochs": epoch,
        "state": "completed",
    }


class FrozenPropertyPredictor:
    """Takes features, not raw FT. See metadata for the required frozen input."""

    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != "supervised_property_probe_v1":
            raise ValueError("Unsupported supervised predictor")
        self.metadata = {k: v for k, v in payload.items() if k not in ("model", "coefficient")}
        self.input_scaler = Standardizer.from_state(payload["input_standardizer"])
        self.label_scaler = Standardizer.from_state(payload["label_standardizer"])
        self.kind = payload["kind"]
        if self.kind == "mlp":
            self.model = make_mlp(payload["input_dim"]).eval().requires_grad_(False)
            self.model.load_state_dict(payload["model"])
        elif self.kind == "ridge":
            self.coefficient = payload["coefficient"].numpy()
        else:
            raise ValueError("Unknown predictor kind")

    @torch.no_grad()
    def predict(self, features):
        x = self.input_scaler.transform(features)
        if self.kind == "ridge":
            result = x @ self.coefficient
        else:
            result = self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        return self.label_scaler.inverse(result)


def fit_feature_probes(name, features, labels, output, provenance):
    from scripts.sim_pretrain.experiments.vae_ablation import (
        plotting,
        reserve_output,
    )

    out = reserve_output(output)
    input_scaler = Standardizer().fit(features["train"])
    label_scaler = Standardizer().fit(labels["train"])
    x = {s: input_scaler.transform(features[s]) for s in ("train", "validation")}
    y = {s: label_scaler.transform(labels[s]) for s in ("train", "validation")}
    metadata = {
        "format": "supervised_property_probe_v1",
        "supervised": True,
        "feature_input": name,
        "input_standardizer": input_scaler.state(),
        "label_standardizer": label_scaler.state(),
        "properties": list(PROPERTIES),
        "purpose": "engineering frozen-representation readout only",
        "stiffness_units": "MuJoCo stiffness_direct, not calibrated N/m",
        "provenance": provenance,
    }
    write_json(out / "status.json", {"state": "running"})
    rows = []
    for alpha, coefficient in zip(RIDGE_ALPHAS, ridge_coefficients(x["train"], y["train"])):
        val = float(np.mean((x["validation"] @ coefficient - y["validation"]) ** 2))
        candidate = f"ridge_{alpha:g}"
        if not np.isfinite(coefficient).all() or not np.isfinite(val):
            rows.append(
                {"name": candidate, "state": "failed", "failure_reason": "Non-finite ridge fit"}
            )
            continue
        save_checkpoint(
            out / f"{candidate}.pt",
            {
                **metadata,
                "kind": "ridge",
                "alpha": alpha,
                "coefficient": torch.from_numpy(coefficient),
            },
        )
        rows.append(
            {
                "name": candidate,
                "kind": "ridge",
                "alpha": alpha,
                "state": "completed",
                "validation_standardized_mse": val,
            }
        )
    try:
        _, mlp_row = fit_mlp(x, y, out, metadata)
        rows.append(mlp_row)
    except FloatingPointError as error:
        rows.append({"name": "mlp", "state": "failed", "failure_reason": str(error)})
    eligible = [r for r in rows if r["state"] == "completed"]
    selected = min(eligible, key=lambda r: r["validation_standardized_mse"]) if eligible else None
    write_json(
        out / "selection.json",
        {"selected": selected, "candidates": rows, "selection_split": "validation"},
    )
    predictions = {}
    # Only after predictor selection is persisted may any held-out predictions be scored.
    for row in eligible:
        filename = "mlp_best.pt" if row["kind"] == "mlp" else f"{row['name']}.pt"
        predictor = FrozenPropertyPredictor(out / filename)
        prediction = predictor.predict(features["test"])
        row["test"] = property_metrics(labels["test"], prediction, label_scaler.mean)
        predictions[row["name"]] = prediction
    if selected:
        source = "mlp_best.pt" if selected["kind"] == "mlp" else f"{selected['name']}.pt"
        payload = torch.load(out / source, map_location="cpu", weights_only=True)
        save_checkpoint(out / "selected_predictor.pt", {**payload, "selection": selected})
        loaded = FrozenPropertyPredictor(out / "selected_predictor.pt")
        np.testing.assert_allclose(
            loaded.predict(features["test"]), predictions[selected["name"]], atol=1e-7
        )
    np.savez_compressed(out / "test_predictions.npz", target=labels["test"], **predictions)
    history_path = out / "mlp_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plt = plotting()
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for split in ("train", "validation"):
            ax.plot(
                [r["epoch"] for r in history], [r[f"{split}_mse"] for r in history], label=split
            )
        ax.set(xlabel="Epoch", ylabel="Mean standardized property MSE", title=name)
        ax.legend()
        fig.savefig(out / "mlp_training.png", dpi=140)
        plt.close(fig)
    result = {
        "input": name,
        "state": "completed" if selected else "failed",
        "selected": selected,
        "candidates": rows,
        "input_dim": features["train"].shape[1],
    }
    write_json(out / "report.json", result)
    write_json(out / "status.json", {"state": result["state"]})
    return result, predictions[selected["name"]] if selected else None


def run_probes(output, arrays, labels, original_model, new_model, provenance):
    from scripts.sim_pretrain.experiments.vae_ablation import (
        infer,
        plotting,
        reserve_output,
    )

    out = reserve_output(output)
    original_model.eval().requires_grad_(False)
    original_model.zero_grad(set_to_none=True)
    if new_model is not None:
        new_model.eval().requires_grad_(False)
        new_model.zero_grad(set_to_none=True)
    flattened = {s: x.reshape(len(x), -1) for s, x in arrays.items()}
    pca = fit_pca5(flattened["train"])
    np.savez_compressed(out / "pca5.npz", **pca)
    features = {"original_mu": {s: infer(original_model, x)[1].numpy() for s, x in arrays.items()}}
    if new_model is not None:
        features["new_mu"] = {s: infer(new_model, x)[1].numpy() for s, x in arrays.items()}
    features["pca5"] = {s: transform_pca(x, pca) for s, x in flattened.items()}
    features["full_ft"] = flattened
    means = labels["train"].mean(0)
    constant = np.broadcast_to(means, labels["test"].shape)
    results = {
        "purpose": "supervised frozen-FT engineering probes, not VAE loss",
        "label_order": list(PROPERTIES),
        "constant_train_means": means.tolist(),
        "constant_test": property_metrics(labels["test"], constant, means),
        "new_encoder_available": new_model is not None,
        "inputs": {},
        "bootstrap": {
            "repeats": 1000,
            "seed": 42,
            "interval": "paired RMSE improvement, percentile95",
        },
    }
    predictions = {}
    for name, values in features.items():
        results["inputs"][name], predictions[name] = fit_feature_probes(
            name, values, labels, out / name, provenance
        )
        write_json(out / "progress.json", results)
    for model in (original_model, new_model):
        if model is not None and any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise RuntimeError("Encoder was not frozen during property probing")
    results["encoders_frozen_without_gradients"] = True
    plt = plotting()
    fig, axes = plt.subplots(
        len(features), 3, figsize=(12, 3.2 * len(features)), constrained_layout=True, squeeze=False
    )
    for i, name in enumerate(features):
        for j, prop in enumerate(PROPERTIES):
            ax = axes[i, j]
            prediction = predictions[name]
            if prediction is None:
                ax.set_title(f"{name}: predictor failed")
                continue
            target = labels["test"][:, j]
            ax.scatter(target, prediction[:, j], s=13, alpha=0.65)
            low, high = (
                min(target.min(), prediction[:, j].min()),
                max(target.max(), prediction[:, j].max()),
            )
            ax.plot([low, high], [low, high], color="gray", linestyle="--")
            ax.set(xlabel=f"Actual {prop}", ylabel="Prediction", title=f"{name}: {prop}")
    fig.savefig(out / "property_scatter.png", dpi=140)
    plt.close(fig)
    lines = [
        "# Frozen property probes",
        "",
        "Predictor choice uses validation standardized MSE only.",
        "",
        "stiffness_direct is a MuJoCo parameter, not calibrated N/m. RMSE improvement is constant minus predictor.",
        "Intervals are paired bootstrap1000 percentile95, conditional on the fitted predictor.",
        "",
        "| Input | Predictor | Property | RMSE | MAE | R2 | RMSE improvement CI95 | Detected |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for name, info in results["inputs"].items():
        chosen = info["selected"]
        if not chosen:
            continue
        for prop, d in chosen["test"].items():
            lines.append(
                f"| {name} | {chosen['name']} | {prop} | {d['rmse']:.6g} | {d['mae']:.6g} | "
                f"{d['r2']:.6g} | {d['paired_rmse_improvement_ci95']} | {d['predictive_information_detected']} |"
            )
    lines += [
        "",
        "All ridge/MLP candidate validation and test metrics are retained in each input's report.json.",
        "A failed readout does not prove complete absence of property information.",
        "No new-mu comparison is fabricated when bounded VAE experiments fail.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "report.json", results)
    return results
