"""Continue the original VAE, recovering missing Adam/RNG state by exact replay."""

import argparse
import fcntl
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
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
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor
from scripts.sim_pretrain.experiments.vae_ablation import (
    diagnose,
    infer,
    plot_diagnostics,
    plot_history,
    plotting,
    snapshot,
    verify_snapshot,
)
from scripts.sim_pretrain.learning import configure_torch, load_data

FORMAT = "pretraining_continuation_v1"


def reserve_output(path, protected_folders):
    path = Path(path).resolve()
    for folder in protected_folders:
        folder = Path(folder).resolve()
        if path.is_relative_to(folder) or folder.is_relative_to(path):
            raise ValueError("Output overlaps a protected input directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def rng_state():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "name": state[0],
            "keys": state[1].tolist(),
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        },
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["name"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["position"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )


def initialize_training(cfg, arrays):
    # Preserve the original loader/model construction order and random stream.
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
    if set(loaders) != {"train", "validation"}:
        raise ValueError("Expected train and validation loaders only")
    model = SpongeVAE()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    return model, optimizer, loaders


def run_epoch(model, optimizer, loaders, beta, epoch):
    record = {"epoch": epoch}
    for split, loader in loaders.items():
        training = split == "train"
        model.train(training)
        total, size = np.zeros(3), 0
        with torch.set_grad_enabled(training):
            for (batch,) in loader:
                reconstruction, mu, logvar = model(batch, sample=training)
                losses = vae_loss(reconstruction, batch, mu, logvar, beta)
                if not all(torch.isfinite(loss).item() for loss in losses):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    losses[0].backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise FloatingPointError(f"Non-finite parameter at epoch {epoch}")
                total += np.array([loss.item() for loss in losses]) * len(batch)
                size += len(batch)
        record[split] = dict(zip(("loss", "mse", "kl"), (total / size).tolist()))
    return record


def require_exact_replay(model, history, parent, expected_history):
    if history != expected_history:
        raise ValueError("Replay history differs; continuation is forbidden")
    actual = model.state_dict()
    if actual.keys() != parent["model"].keys() or any(
        not torch.equal(value, parent["model"][key]) for key, value in actual.items()
    ):
        raise ValueError("Replay weights differ; continuation is forbidden")


def payload(model, optimizer, parent, epoch, run):
    return {
        **parent,
        "format": FORMAT,
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "rng_state": rng_state(),
        "continuation": run,
        "training_provenance": provenance(),
    }


def load_continued_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Not a continuation checkpoint")
    model = SpongeVAE()
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def verify_frozen_export(out, model, raw, prep):
    frozen = FrozenSpongeEncoder(out / "encoder.pt")
    if frozen.preprocessor.state() != prep.state():
        raise ValueError("Export preprocessing mismatch")
    for key, value in model.encoder.state_dict().items():
        if not torch.equal(value, frozen.model.state_dict()[key]):
            raise ValueError("Export encoder weights mismatch")
    assert all(not p.requires_grad and p.grad is None for p in frozen.model.parameters())
    codes = frozen.encode(raw)
    assert codes.shape == (len(raw), 5) and np.isfinite(codes).all()
    np.testing.assert_array_equal(codes, frozen.encode(raw))
    with torch.no_grad():
        reference, _ = model.encoder(torch.from_numpy(prep.transform(raw)))
    np.testing.assert_allclose(codes, reference.numpy(), rtol=1e-6, atol=1e-8)
    # Match diagnostic batch sizes before comparing float32 reductions.
    batched = np.concatenate([frozen.encode(raw[i : i + 32]) for i in range(0, len(raw), 32)])
    with np.load(out / "test_embeddings.npz") as data:
        np.testing.assert_allclose(batched, data["mu"], rtol=1e-6, atol=1e-8)
        saved_difference = float(np.abs(batched - data["mu"]).max())
    return {
        "frozen_export_verified": True,
        "export_weights_and_preprocessing_exact": True,
        "export_max_abs_difference_same_batch": saved_difference,
        "export_full_batch_reference_max_abs_difference": float(
            np.abs(codes - reference.numpy()).max()
        ),
        "export_full_vs_diagnostic_batch_max_abs_difference": float(np.abs(codes - batched).max()),
    }


def evaluate_and_export(out, model, parent, raw, arrays, prep, history, run):
    model.eval()
    previous = SpongeVAE().eval()
    previous.load_state_dict(parent["model"])
    template = parent["train_mean_trajectory"].numpy()
    report = {
        "scope": "Fixed requested final epoch; no test-based model selection",
        "from_epoch": parent["epoch"],
        "epoch": history[-1]["epoch"],
        "dataset_sha256": parent["data_provenance_hash"],
        "checkpoint_sha256": file_digest(out / "vae_last.pt"),
        "splits": {},
    }
    for split in ("validation", "test"):
        before = diagnose(previous, arrays[split], template, prep)
        after = diagnose(model, arrays[split], template, prep)
        report["splits"][split] = {
            "before": before,
            "after": after,
            "mse_reduction_fraction": 1 - after["mse"] / before["mse"],
        }
    write_json(out / "evaluation.json", report)
    write_json(out / "test.json", report["splits"]["test"]["after"])
    beta = parent["config"]["training"]["beta"]
    plot_history(
        [
            {
                **row,
                "beta": beta,
                **{
                    split: {**row[split], "weighted_kl": beta * row[split]["kl"]}
                    for split in ("train", "validation")
                },
            }
            for row in history
        ],
        out,
    )
    plot_diagnostics(
        model, arrays["test"], prep, out, f"Continued original VAE, epoch {report['epoch']}"
    )

    plt = plotting()
    before, _, _ = infer(previous, arrays["test"][:1])
    after, _, _ = infer(model, arrays["test"][:1])
    target = prep.filtered(raw["test"][:1])[0]
    estimates = [prep.inverse(value.numpy())[0] for value in (before, after)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    times = np.arange(1, 401) / 100
    for index, ax in enumerate(axes.flat):
        ax.plot(times, target[:, index], color="#222222", label="Filtered FT")
        for estimate, epoch, color in zip(
            estimates, (parent["epoch"], report["epoch"]), ("#c65a28", "#2476a8")
        ):
            ax.plot(times, estimate[:, index], color=color, alpha=0.85, label=f"Epoch {epoch}")
        for boundary in (2, 3):
            ax.axvline(boundary, color="gray", linewidth=0.6, linestyle=":")
        ax.set(xlabel="Time (s)", ylabel=f"{CHANNELS[index]} ({'N' if index < 3 else 'N m'})")
    axes.flat[0].legend()
    fig.suptitle("Test index 0: same data, preprocessing and VAE architecture")
    fig.savefig(out / "reconstruction_comparison.png", dpi=150)
    plt.close(fig)

    save_checkpoint(
        out / "encoder.pt",
        {
            "encoder": model.encoder.state_dict(),
            "preprocessing": prep.state(),
            "architecture": "frame6to5_flat2000_posterior10_split5",
            "epoch": report["epoch"],
            "config": parent["config"],
            "data_provenance_hash": parent["data_provenance_hash"],
            "channels": CHANNELS,
            "frame": parent["frame"],
            "units": parent["units"],
            "evaluation": report,
            "continuation": run,
            "supervised": False,
        },
    )
    report.update(verify_frozen_export(out, model, raw["test"], prep))
    write_json(out / "evaluation.json", report)
    return report


def continue_training(dataset_config, output, additional_epochs=200, checkpoint=None):
    if additional_epochs < 1:
        raise ValueError("additional_epochs must be positive")
    cfg = load_config(dataset_config)
    if cfg["training"]["device"] != "cpu":
        raise ValueError("Exact legacy replay currently supports CPU only")
    folder = output_dir(cfg).resolve()
    path = Path(checkpoint).resolve() if checkpoint else folder / "vae_last.pt"
    parent = torch.load(path, map_location="cpu", weights_only=True)
    if parent["config_hash"] != digest(cfg) or parent["config"] != cfg:
        raise ValueError("Checkpoint/config mismatch")
    if parent["training_provenance"] != provenance():
        raise ValueError("Training source or dependency provenance changed")
    history = json.loads((path.parent / "history.json").read_text())
    start = parent["epoch"]
    if [row["epoch"] for row in history] != list(range(1, start + 1)):
        raise ValueError("Parent history does not end at checkpoint epoch")
    native = parent.get("format") == FORMAT
    if native and not all(key in parent for key in ("optimizer", "rng_state")):
        raise ValueError("Continuation checkpoint is missing optimizer/RNG state")
    if not native and start != cfg["training"]["epochs"]:
        raise ValueError("Legacy recovery requires the original last checkpoint")
    protected_folders = {folder, path.parent}
    if (folder.parent / "manifest.json").exists():
        protected_folders.add(folder.parent)
    files = {p for base in protected_folders for p in base.rglob("*") if p.is_file()}
    files.update(
        (
            Path(dataset_config).resolve(),
            Path(__file__),
            ROOT / "scripts/sim_pretrain/experiments/vae_ablation.py",
        )
    )
    protected = snapshot(files)
    raw, data_hash = load_data(cfg)
    if data_hash != parent["data_provenance_hash"]:
        raise ValueError("Checkpoint/dataset mismatch")
    out = reserve_output(output, protected_folders)
    run = {
        "format": FORMAT,
        "dataset_config": str(Path(dataset_config).resolve()),
        "dataset_sha256": data_hash,
        "parent_checkpoint": str(path),
        "parent_checkpoint_sha256": file_digest(path),
        "source_epoch": start,
        "additional_epochs": additional_epochs,
        "target_epoch": start + additional_epochs,
        "output_dir": str(out),
        "training": cfg["training"],
        "state_recovery": "saved_optimizer_rng" if native else "verified_exact_replay",
        "source_sha256": file_digest(__file__),
        "diagnostics_source_sha256": file_digest(
            ROOT / "scripts/sim_pretrain/experiments/vae_ablation.py"
        ),
        "note": "Original dataset config is immutable; target_epoch specifies the extended training budget",
    }
    status = {
        "state": "initializing",
        "last_completed_epoch": start,
        "target_epoch": run["target_epoch"],
    }
    write_json(out / "run_config.json", run)
    write_json(out / "protected_hashes.json", protected)
    try:
        configure_torch(cfg)
        prep = Preprocessor(**cfg["filter"], sample_hz=cfg["simulation"]["sample_hz"]).fit(
            raw["train"]
        )
        if prep.state() != parent["preprocessing"]:
            raise ValueError("Training-only preprocessing differs from checkpoint")
        arrays = {key: prep.transform(value) for key, value in raw.items()}
        np.testing.assert_array_equal(
            arrays["train"].mean(0), parent["train_mean_trajectory"].numpy()
        )
        model, optimizer, loaders = initialize_training(cfg, arrays)
        if native:
            model.load_state_dict(parent["model"])
            optimizer.load_state_dict(parent["optimizer"])
            restore_rng(parent["rng_state"])
            recovery = {"method": run["state_recovery"], "epoch": start, "restored": True}
        else:
            status["state"] = "recovering_optimizer_rng"
            write_json(out / "status.json", status)
            replay = []
            for epoch in range(1, start + 1):
                row = run_epoch(model, optimizer, loaders, cfg["training"]["beta"], epoch)
                replay.append(row)
                if row != history[epoch - 1]:
                    raise ValueError(f"Replay history mismatch at epoch {epoch}; not continuing")
                if epoch == 1 or epoch % 10 == 0:
                    print(json.dumps({"phase": "replay", **row}), flush=True)
                    status["replayed_epochs"] = epoch
                    write_json(out / "status.json", status)
            require_exact_replay(model, replay, parent, history)
            recovery = {
                "method": run["state_recovery"],
                "epoch": start,
                "weights_exact": True,
                "all_history_exact": True,
                "optimizer_reset": False,
            }
        save_checkpoint(out / "vae_recovered.pt", payload(model, optimizer, parent, start, run))
        write_json(out / "recovery_verification.json", recovery)
        status.update(state="training", recovery_verified=True)
        write_json(out / "history.json", history)
        best, best_epoch = float("inf"), None
        for epoch in range(start + 1, run["target_epoch"] + 1):
            row = run_epoch(model, optimizer, loaders, cfg["training"]["beta"], epoch)
            history.append(row)
            saved = payload(model, optimizer, parent, epoch, run)
            if row["validation"]["loss"] < best:
                best, best_epoch = row["validation"]["loss"], epoch
                save_checkpoint(out / "vae_best.pt", saved)
            save_checkpoint(out / "vae_last.pt", saved)
            write_json(out / "history.json", history)
            status.update(last_completed_epoch=epoch, best_added_epoch=best_epoch)
            write_json(out / "status.json", status)
            if epoch == start + 1 or epoch % 10 == 0:
                print(json.dumps({"phase": "continuation", **row}), flush=True)
        write_json(out / "preprocessing.json", prep.state())
        status["state"] = "evaluating"
        write_json(out / "status.json", status)
        report = evaluate_and_export(out, model, parent, raw, arrays, prep, history, run)
        status.update(
            state="completed",
            frozen_export_verified=report["frozen_export_verified"],
            test_quality_passed=report["splits"]["test"]["after"]["passed"],
        )
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        check = verify_snapshot(protected)
        status.update(
            protected_inputs=check,
            original_provenance_unchanged=parent["training_provenance"] == provenance(),
        )
        if not check["unchanged"] or not status["original_provenance_unchanged"]:
            status.update(
                state="failed", integrity_error="Protected inputs or original provenance changed"
            )
        write_json(out / "status.json", status)
    if status["state"] != "completed":
        raise RuntimeError("Continuation integrity verification failed")
    print(json.dumps(status, indent=2), flush=True)
    return status


def finalize_export(output):
    """Recheck an already exported final model; never train or rewrite weights."""
    out = Path(output).resolve()
    with (out / ".finalize.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((out / "status.json").read_text())
        if status["state"] != "failed" or status["last_completed_epoch"] != status["target_epoch"]:
            raise ValueError(
                "Export recovery requires failed postprocessing after the requested final epoch"
            )
        run = json.loads((out / "run_config.json").read_text())
        cfg = load_config(run["dataset_config"])
        protected = json.loads((out / "protected_hashes.json").read_text())
        training_script_sha = protected.pop(str(Path(__file__).resolve()))
        check = verify_snapshot(protected)
        if not check["unchanged"]:
            raise ValueError("Protected inputs changed; export recovery refused")
        configure_torch(cfg)
        raw, data_hash = load_data(cfg)
        model, saved = load_continued_model(out / "vae_last.pt")
        if (
            saved["epoch"] != status["target_epoch"]
            or saved["config_hash"] != digest(cfg)
            or saved["data_provenance_hash"] != data_hash
            or saved["training_provenance"] != provenance()
        ):
            raise ValueError("Final checkpoint provenance mismatch")
        history = json.loads((out / "history.json").read_text())
        if [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
            raise ValueError("Final checkpoint/history mismatch")
        report = json.loads((out / "evaluation.json").read_text())
        if (
            report["checkpoint_sha256"] != file_digest(out / "vae_last.pt")
            or report["dataset_sha256"] != data_hash
            or report["epoch"] != saved["epoch"]
        ):
            raise ValueError("Stale evaluation; refusing metadata-only export recovery")
        artifacts = snapshot(
            [
                out / name
                for name in (
                    "vae_last.pt",
                    "vae_best.pt",
                    "vae_recovered.pt",
                    "encoder.pt",
                    "history.json",
                    "preprocessing.json",
                    "run_config.json",
                    "test_embeddings.npz",
                )
            ]
        )
        audit = {
            "previous_status": status,
            "training_script_sha256": training_script_sha,
            "postprocessing_script_sha256": file_digest(__file__),
            "training_artifacts_before": artifacts,
            "scope": "Export verification and report/status metadata only; no training or weight writes",
        }
        proof = verify_frozen_export(
            out, model, raw["test"], Preprocessor.from_state(saved["preprocessing"])
        )
        audit.update(proof=proof, training_artifacts_after=verify_snapshot(artifacts))
        assert audit["training_artifacts_after"]["unchanged"]
        write_json(out / "postprocessing_recovery.json", audit)
        report.update(proof)
        write_json(out / "evaluation.json", report)
        status = {
            key: value for key, value in status.items() if key not in ("error", "integrity_error")
        }
        status.update(
            state="completed",
            frozen_export_verified=True,
            protected_inputs=check,
            original_provenance_unchanged=True,
            postprocessing_recovery="postprocessing_recovery.json",
            test_quality_passed=report["splits"]["test"]["after"]["passed"],
        )
        write_json(out / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--additional-epochs", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verify-export-only", action="store_true")
    args = parser.parse_args()
    if args.verify_export_only:
        finalize_export(args.output)
    else:
        if args.dataset_config is None:
            parser.error("--dataset-config is required for training")
        from scripts.shared.run_paths import new_output

        args.output = new_output(args.output)
        continue_training(args.dataset_config, args.output, args.additional_epochs, args.checkpoint)


if __name__ == "__main__":
    main()
