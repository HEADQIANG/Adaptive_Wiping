"""Independent supervised branches, episode-level validation and exact resume."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.real_training.config import (
    output_lock,
    provenance,
    resolve,
    training_contract,
)
from scripts.real_training.data import encoder_source, load_prepared
from scripts.shared.checkpoints import save_checkpoint
from scripts.shared.common import digest, file_digest, write_json
from scripts.shared.policy import OfflinePolicy
from scripts.shared.real_models import HeightFeedback, XYDecoder
from scripts.shared.real_preprocessing import (
    ChannelScaler,
    make_windows,
    out_of_range,
)


def _seed(cfg):
    torch.set_num_threads(cfg["training"]["threads"])
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": {
            "generator": state[0],
            "keys": torch.from_numpy(state[1].astype(np.int64)),
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        },
    }


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state(
        (
            value["generator"],
            value["keys"].numpy().astype(np.uint32),
            value["position"],
            value["has_gauss"],
            value["cached_gaussian"],
        )
    )


def fit_scalers(arrays, indices):
    if not indices or len(set(indices)) != len(indices) or any(i not in range(8) for i in indices):
        raise ValueError("Invalid training demonstration indices")
    return {
        "xy": ChannelScaler().fit(arrays["xy"][indices]),
        "ft": ChannelScaler().fit(arrays["ft"][indices]),
        "delta_h": ChannelScaler().fit(arrays["delta_h"][indices, :, None]),
    }


def _tensors(arrays, indices, scalers):
    windows, target = make_windows(arrays["ft"][indices], arrays["height"][indices])
    sponge = arrays["sponge"][indices].astype(np.float32)
    return {
        "xy": (
            torch.from_numpy(sponge),
            torch.from_numpy(scalers["xy"].transform(arrays["xy"][indices])),
        ),
        "feedback": (
            torch.from_numpy(np.repeat(sponge, 20, axis=0)),
            torch.from_numpy(scalers["ft"].transform(windows).reshape(-1, 5, 6)),
            torch.from_numpy(scalers["delta_h"].transform(target).reshape(-1, 1)),
        ),
    }


def _bindings(cfg, info, indices, run_kind):
    return {
        "training_config_hash": digest(training_contract(cfg)),
        "prepared_sha256": info["sha256"],
        "raw_sha256": info["raw_sha256"],
        "encoder_sha256": info["encoder_sha256"],
        "source_kind": info["source_kind"],
        "train_indices": list(indices),
        "run_kind": run_kind,
        "software": provenance(),
    }


def _assert_frozen(encoder, original, cfg, fingerprint):
    if encoder.model.training or any(
        p.requires_grad or p.grad is not None for p in encoder.model.parameters()
    ):
        raise RuntimeError("Frozen encoder became trainable")
    if any(
        not torch.equal(value, original["encoder"][key])
        for key, value in encoder.model.state_dict().items()
    ):
        raise RuntimeError("Frozen encoder weights changed")
    if encoder.preprocessor.state() != original["preprocessing"]:
        raise RuntimeError("Frozen encoder preprocessing changed")
    if file_digest(resolve(cfg["encoder"])) != fingerprint:
        raise RuntimeError("Source encoder file changed during training")


def _complete(checkpoint, cfg):
    return checkpoint["epochs"] == {
        "xy": cfg["training"]["xy_epochs"],
        "feedback": cfg["training"]["ft_epochs"],
    }


def _validate_saved_state(saved, cfg):
    if saved.get("schema_version") != 1:
        raise ValueError("Unsupported training checkpoint schema")
    epochs = saved["epochs"]
    limits = {"xy": cfg["training"]["xy_epochs"], "feedback": cfg["training"]["ft_epochs"]}
    if set(epochs) != set(limits) or any(
        type(epochs[k]) is not int or not 0 <= epochs[k] <= limits[k] for k in limits
    ):
        raise ValueError("Invalid saved epoch progress")
    if epochs["feedback"] and epochs["xy"] != limits["xy"]:
        raise ValueError("Feedback progress precedes completion of the XY branch")
    expected = [
        (branch, epoch) for branch in ("xy", "feedback") for epoch in range(1, epochs[branch] + 1)
    ]
    history = saved["history"]
    if [(r["branch"], r["epoch"]) for r in history] != expected or any(
        not np.isfinite(r["mse"]) for r in history
    ):
        raise ValueError("Saved history is incomplete or non-finite")

    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(item) for item in value)
        return True

    if not finite_tensors({key: saved[key] for key in ("xy", "feedback", "optimizers")}):
        raise ValueError("Saved model or optimizer contains non-finite tensors")


def _run(cfg, arrays, info, indices, folder, *, resume, run_kind, stop_after=None):
    folder = Path(folder)
    binding = _bindings(cfg, info, indices, run_kind)
    checkpoint_path = folder / "training.pt"
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError("No checkpoint to resume")
    if not resume and any(
        (folder / name).exists() for name in ("training.pt", "run.json", "history.json")
    ):
        raise FileExistsError("Training artifacts exist; use --resume or a new output directory")
    encoder, encoder_payload, fingerprint = encoder_source(cfg)
    if fingerprint != info["encoder_sha256"]:
        raise ValueError("Encoder no longer matches the prepared data")
    _assert_frozen(encoder, encoder_payload, cfg, fingerprint)
    actual_sponge = encoder.encode(arrays["exploration"])
    if not np.array_equal(arrays["sponge"], np.repeat(actual_sponge, 8, axis=0)):
        raise ValueError("Cached embeddings do not match the frozen encoder")
    _seed(cfg)
    models = {"xy": XYDecoder(), "feedback": HeightFeedback()}
    optimizers = {
        key: torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        for key, model in models.items()
    }
    scalers = fit_scalers(arrays, indices)
    epochs, history = {"xy": 0, "feedback": 0}, []
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_saved_state(saved, cfg)
        if saved["bindings"] != binding:
            raise ValueError(
                "Resume rejected: data, encoder, config, split or software provenance changed"
            )
        if saved["scalers"] != {key: value.state() for key, value in scalers.items()}:
            raise ValueError("Resume scaler mismatch")
        for key, model in models.items():
            model.load_state_dict(saved[key])
            optimizers[key].load_state_dict(saved["optimizers"][key])
        epochs, history = saved["epochs"], saved["history"]
        _restore_rng(saved["rng"])
    tensors = _tensors(arrays, indices, scalers)
    _, height_target = make_windows(arrays["ft"][indices], arrays["height"][indices])
    baseline = {
        "xy": torch.from_numpy(arrays["xy"][indices].mean(axis=0)),
        "delta_h": float(height_target.mean()),
    }
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "run.json", {"bindings": binding, "config": cfg, "info": info})

    def save():
        _assert_frozen(encoder, encoder_payload, cfg, fingerprint)
        checkpoint = {
            "schema_version": 1,
            "bindings": binding,
            "config": cfg,
            "info": info,
            "xy": models["xy"].state_dict(),
            "feedback": models["feedback"].state_dict(),
            "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
            "epochs": dict(epochs),
            "history": list(history),
            "rng": _rng_state(),
            "scalers": {key: value.state() for key, value in scalers.items()},
            "encoder": encoder_payload,
            "baseline": baseline,
        }
        save_checkpoint(checkpoint_path, checkpoint)
        write_json(folder / "history.json", history)
        write_json(
            folder / "status.json",
            {
                "state": "completed" if _complete(checkpoint, cfg) else "incomplete",
                "epochs": epochs,
                "source_kind": info["source_kind"],
                "hardware_ready": False,
            },
        )
        return checkpoint

    saved = save()
    steps = 0
    for branch, target_epochs, batch_size in (
        ("xy", cfg["training"]["xy_epochs"], cfg["training"]["xy_batch_size"]),
        ("feedback", cfg["training"]["ft_epochs"], cfg["training"]["ft_batch_size"]),
    ):
        model, optimizer = models[branch], optimizers[branch]
        batch_tensors = tensors[branch]
        model.train()
        for epoch in range(epochs[branch] + 1, target_epochs + 1):
            order = torch.randperm(len(batch_tensors[0]))
            total = 0.0
            for offset in range(0, len(order), batch_size):
                batch = [tensor[order[offset : offset + batch_size]] for tensor in batch_tensors]
                loss = (model(*batch[:-1]) - batch[-1]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite {branch} loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise RuntimeError("Non-finite or missing gradients")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError("Non-finite updated parameters")
                total += loss.item() * len(batch[0])
            epochs[branch] = epoch
            history.append({"branch": branch, "epoch": epoch, "mse": total / len(order)})
            steps += 1
            checkpoint_due = (
                epoch % cfg["training"]["checkpoint_every"] == 0 or epoch == target_epochs
            )
            if checkpoint_due or (stop_after is not None and steps >= stop_after):
                saved = save()
            if epoch == 1 or checkpoint_due:
                print(json.dumps({"run": run_kind, **history[-1]}), flush=True)
            if stop_after is not None and steps >= stop_after:
                return saved
    return save()


def train(cfg, *, resume=False, testing=False, _stop_after=None):
    if _stop_after is not None and not testing:
        raise ValueError("Short interrupted runs are test-only")
    arrays, info = load_prepared(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    with output_lock(out):
        checkpoint = _run(
            cfg,
            arrays,
            info,
            list(range(8)),
            out / "final",
            resume=resume,
            run_kind="full_eight_demonstrations",
            stop_after=_stop_after,
        )
    return {
        "completed": _complete(checkpoint, cfg),
        "epochs": checkpoint["epochs"],
        "source_kind": info["source_kind"],
        "warnings": info["warnings"],
        "hardware_ready": False,
    }


def _policy_payload(checkpoint):
    info = checkpoint["info"]
    return {
        "schema_version": 1,
        "artifact_kind": "offline_wiping_policy",
        "hardware_ready": False,
        "source_kind": info["source_kind"],
        "warnings": info["warnings"],
        "encoder": checkpoint["encoder"],
        "xy": checkpoint["xy"],
        "feedback": checkpoint["feedback"],
        "scalers": checkpoint["scalers"],
        "processing": checkpoint["config"]["processing"],
        "metadata": {
            "data": info,
            "bindings": checkpoint["bindings"],
            "epochs": checkpoint["epochs"],
            "training_contract": training_contract(checkpoint["config"]),
            "history": "FT[k-4:k+1] -> h[k+1]-h[k], k=4..23",
            "xy_units": "m",
            "delta_h_units": "m",
            "policy_period_s": 0.4,
        },
    }


def _metrics(prediction, target):
    error = np.asarray(prediction) - np.asarray(target)
    return {"rmse_m": float(np.sqrt(np.mean(error**2))), "mae_m": float(np.mean(np.abs(error)))}


def _predictions(checkpoint, arrays, indices):
    policy = OfflinePolicy.from_payload(_policy_payload(checkpoint))
    windows, targets = make_windows(arrays["ft"][indices], arrays["height"][indices])
    sponge = arrays["sponge"][indices]
    return {
        "xy_prediction": policy.predict_xy(sponge),
        "xy_target": arrays["xy"][indices],
        "xy_baseline": np.broadcast_to(
            checkpoint["baseline"]["xy"].numpy(), (len(indices), 25, 2)
        ).copy(),
        "delta_h_prediction": policy.predict_delta_h(
            np.repeat(sponge, 20, axis=0), windows.reshape(-1, 5, 6)
        ).reshape(len(indices), 20, 1),
        "delta_h_target": targets,
        "delta_h_mean_baseline": np.full_like(targets, checkpoint["baseline"]["delta_h"]),
    }


def _report(predictions):
    return {
        "xy": _metrics(predictions["xy_prediction"], predictions["xy_target"]),
        "xy_mean_trajectory_baseline": _metrics(
            predictions["xy_baseline"], predictions["xy_target"]
        ),
        "delta_h": _metrics(predictions["delta_h_prediction"], predictions["delta_h_target"]),
        "delta_h_zero_baseline": _metrics(
            np.zeros_like(predictions["delta_h_target"]), predictions["delta_h_target"]
        ),
        "delta_h_training_mean_baseline": _metrics(
            predictions["delta_h_mean_baseline"], predictions["delta_h_target"]
        ),
    }


def _range_report(checkpoint, arrays, indices):
    scalers = {key: ChannelScaler.from_state(value) for key, value in checkpoint["scalers"].items()}
    return {
        "xy": out_of_range(scalers["xy"].transform(arrays["xy"][indices])),
        "ft": out_of_range(scalers["ft"].transform(arrays["ft"][indices])),
        "delta_h": out_of_range(scalers["delta_h"].transform(arrays["delta_h"][indices, :, None])),
    }


def _plot(folder, predictions, history, source_kind):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = (
        "SYNTHETIC SOFTWARE TEST"
        if source_kind == "synthetic"
        else "Offline demonstration prediction"
    )
    count = len(predictions["xy_target"])
    fig, axes = plt.subplots(
        count, 2, figsize=(10, max(3, count * 2.5)), squeeze=False, constrained_layout=True
    )
    for row in range(count):
        ax = axes[row, 0]
        for key, name in (
            ("xy_target", "Demonstration"),
            ("xy_prediction", "Prediction"),
            ("xy_baseline", "Train mean"),
        ):
            values = predictions[key][row]
            ax.plot(values[:, 0], values[:, 1], label=name)
        ax.set(xlabel="Base X (m)", ylabel="Base Y (m)", title=f"Episode {row + 1}")
        ax = axes[row, 1]
        for key, name in (
            ("delta_h_target", "Demonstration"),
            ("delta_h_prediction", "Prediction"),
            ("delta_h_mean_baseline", "Train mean"),
        ):
            ax.plot(np.arange(6, 26) * 0.4, predictions[key][row, :, 0], label=name)
        ax.axhline(0, color="gray", linestyle=":", label="Zero")
        ax.set(xlabel="Next sample time (s)", ylabel="Delta height (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(label + "; recorded FT, not closed-loop execution", fontsize=11)
    fig.savefig(Path(folder) / "predictions.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, branch in zip(axes, ("xy", "feedback")):
        records = [record for record in history if record["branch"] == branch]
        ax.plot([record["epoch"] for record in records], [record["mse"] for record in records])
        ax.set(title=branch, xlabel="Epoch", ylabel="Normalized training MSE")
    fig.suptitle(label, fontsize=11)
    fig.savefig(Path(folder) / "training.png", dpi=130)
    plt.close(fig)


def _load_final(cfg, info):
    path = resolve(cfg["output_dir"]) / "final" / "training.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_saved_state(checkpoint, cfg)
    if checkpoint["bindings"] != _bindings(cfg, info, list(range(8)), "full_eight_demonstrations"):
        raise ValueError(
            "Final checkpoint provenance does not match current inputs/config/software"
        )
    if not _complete(checkpoint, cfg):
        raise RuntimeError("Training is incomplete; resume before evaluating or exporting")
    return checkpoint, path


def evaluate(cfg, *, testing=False):
    arrays, info = load_prepared(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    with output_lock(out):
        checkpoint, path = _load_final(cfg, info)
        predictions = _predictions(checkpoint, arrays, list(range(8)))
        report = {
            "scope": "training_set_reconstruction_with_recorded_ft_not_closed_loop",
            "source_kind": info["source_kind"],
            "hardware_ready": False,
            "checkpoint_sha256": file_digest(path),
            "bindings": checkpoint["bindings"],
            "metrics": _report(predictions),
            "warnings": info["warnings"],
            "out_of_range_per_channel": _range_report(checkpoint, arrays, list(range(8))),
            "demo_ids": info["demo_ids"],
        }
        np.savez_compressed(
            out / "final" / "predictions.npz",
            **predictions,
            demo_ids=np.array(info["demo_ids"]),
            source_kind=info["source_kind"],
        )
        _plot(out / "final", predictions, checkpoint["history"], info["source_kind"])
        write_json(out / "final" / "evaluation.json", report)
    return report


def cross_validate(cfg, *, resume=False, testing=False):
    arrays, info = load_prepared(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    root = out / "cross_validation"
    with output_lock(out):
        if root.exists() and not resume:
            raise FileExistsError("Cross-validation exists; use --resume or a new output directory")
        if resume and not root.is_dir():
            raise FileNotFoundError("No cross-validation run to resume")
        root.mkdir(exist_ok=True)
        folds, predictions = [], []
        for held_out in range(8):
            indices = [i for i in range(8) if i != held_out]
            folder = root / f"fold_{held_out + 1:02d}"
            has_checkpoint = (folder / "training.pt").is_file()
            checkpoint = _run(
                cfg,
                arrays,
                info,
                indices,
                folder,
                resume=resume and has_checkpoint,
                run_kind=f"leave_one_out_{held_out}",
            )
            pred = _predictions(checkpoint, arrays, [held_out])
            fold = {
                "train_demo_ids": [info["demo_ids"][i] for i in indices],
                "held_out_demo_id": info["demo_ids"][held_out],
                "metrics": _report(pred),
                "scalers": checkpoint["scalers"],
                "source_kind": info["source_kind"],
                "out_of_range_per_channel": _range_report(checkpoint, arrays, [held_out]),
                "bindings": checkpoint["bindings"],
                "checkpoint_sha256": file_digest(folder / "training.pt"),
            }
            write_json(folder / "evaluation.json", fold)
            np.savez_compressed(folder / "predictions.npz", **pred, source_kind=info["source_kind"])
            _plot(folder, pred, checkpoint["history"], info["source_kind"])
            folds.append(fold)
            predictions.append(pred)
            write_json(root / "progress.json", {"completed_folds": len(folds), "folds": folds})
        joined = {
            key: np.concatenate([pred[key] for pred in predictions], axis=0)
            for key in predictions[0]
        }
        report = {
            "scope": "leave_one_demonstration_out_prediction_with_recorded_ft_not_closed_loop",
            "source_kind": info["source_kind"],
            "hardware_ready": False,
            "folds": folds,
            "metrics": _report(joined),
            "warnings": info["warnings"],
            "prepared_sha256": info["sha256"],
            "training_config_hash": digest(training_contract(cfg)),
        }
        np.savez_compressed(
            root / "predictions.npz",
            **joined,
            demo_ids=np.array(info["demo_ids"]),
            source_kind=info["source_kind"],
        )
        write_json(root / "evaluation.json", report)
    return report


def export(cfg, *, testing=False):
    arrays, info = load_prepared(cfg, testing=testing)
    out = resolve(cfg["output_dir"])
    with output_lock(out):
        checkpoint, checkpoint_path = _load_final(cfg, info)
        report_path = out / "final" / "evaluation.json"
        if not report_path.is_file():
            raise RuntimeError("Run evaluate before export")
        report = json.loads(report_path.read_text())
        if (
            report["checkpoint_sha256"] != file_digest(checkpoint_path)
            or report["bindings"] != checkpoint["bindings"]
        ):
            raise ValueError("Evaluation is stale; rerun evaluate")
        path = out / "policy.pt"
        if path.exists():
            raise FileExistsError(
                "Policy export exists; do not overwrite previously exported artifacts"
            )
        payload = _policy_payload(checkpoint)
        payload["evaluation"] = report
        temporary = out / "policy.pt.pending"
        save_checkpoint(temporary, payload)
        before = OfflinePolicy.from_payload(copy.deepcopy(payload))
        after = OfflinePolicy(temporary)
        for policy in (before, after):
            if policy.hardware_ready:
                raise RuntimeError("Offline policy cannot be hardware-ready")
        z_before = before.encode_exploration(arrays["exploration"])
        z_after = after.encode_exploration(arrays["exploration"])
        np.testing.assert_array_equal(z_before, z_after)
        np.testing.assert_array_equal(before.predict_xy(z_before), after.predict_xy(z_after))
        windows, _ = make_windows(arrays["ft"], arrays["height"])
        z = np.repeat(z_after, 160, axis=0)
        ft = windows.reshape(-1, 5, 6)
        np.testing.assert_array_equal(before.predict_delta_h(z, ft), after.predict_delta_h(z, ft))
        temporary.replace(path)
        result = {
            "policy": str(path),
            "sha256": file_digest(path),
            "roundtrip_verified": True,
            "source_kind": info["source_kind"],
            "hardware_ready": False,
            "warnings": info["warnings"],
        }
        write_json(out / "export.json", result)
    return result
