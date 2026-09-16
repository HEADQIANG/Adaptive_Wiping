"""Paired beta diagnosis with the exact original VAE architecture and dropout."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.shared.common import file_digest, write_json
from scripts.shared.model import SpongeVAE, vae_loss
from scripts.shared.paths import read_path
from scripts.shared.preprocessing import Preprocessor
from scripts.shared.run_paths import new_output
from scripts.sim_pretrain.experiments.diagnose_latent_collapse import fit_neural, metric_row, mse
from scripts.sim_pretrain.experiments.repair_sponge_vae import RepairedVAE
from scripts.sim_pretrain.experiments.vae_property_probe import Standardizer, transform_pca


def informative_original(encoder_state, decoder_state, train_codes, train_pca_codes):
    """Transfer train-fitted useful codes/decoder into the exact original architecture."""
    model = SpongeVAE()
    model.encoder.load_state_dict(encoder_state)
    # Input-coordinate conversion is fit on training rows only. No shape/activation changes.
    design = np.column_stack([train_codes, np.ones(len(train_codes))])
    mapping = np.linalg.lstsq(design, train_pca_codes, rcond=None)[0]
    with torch.no_grad():
        w = decoder_state["hidden.0.weight"]
        model.decode_hidden[0].weight.copy_(w @ torch.tensor(mapping[:5].T, dtype=w.dtype))
        model.decode_hidden[0].bias.copy_(decoder_state["hidden.0.bias"] +
                                        w @ torch.tensor(mapping[5], dtype=w.dtype))
        model.decode_frame.weight.copy_(decoder_state["frame.weight"])
        model.decode_frame.bias.copy_(decoder_state["frame.bias"])
    return model, float(np.mean((design @ mapping - train_pca_codes) ** 2))


def run(diagnosis, output):
    diagnosis = read_path(diagnosis).resolve()
    source_report = json.loads((diagnosis / "report.json").read_text())
    source = read_path(source_report["source"]).resolve()
    out = Path(output).resolve()
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in [source, diagnosis]):
        raise ValueError("Output must not overlap diagnostic or production input")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds, betas = [42, 43, 44], [0.0, 0.0001, 0.06]
    protected = [source / "dataset.h5", source / "vae_last.pt", source / "encoder.pt",
        diagnosis / "report.json", diagnosis / "paired_initial.pt", diagnosis / "pca5.npz",
        diagnosis / "shared_pca5_linear_scaler.json", Path(__file__),
        Path(__file__).with_name("diagnose_latent_collapse.py"),
        Path(__file__).parents[2] / "shared/model.py"]
    protected += [diagnosis / f"shared_pca5_linear_seed_{seed}" / "best.pt" for seed in seeds]
    hashes = {str(p): file_digest(p) for p in protected}
    protocol = dict(architecture="unchanged SpongeVAE including ReLU/shared frame projection/Dropout(0.1)",
        epochs=200, learning_rate=.0001, batch_size=32, seeds=seeds, betas=betas,
        initialization="train-only PCA encoder plus train-fitted same-architecture decoder",
        pairing="same initial weights, batches and posterior/dropout RNG within each seed",
        selection="validation MSE only, initial epoch0 eligible; report last epoch separately",
        limitation="conditional effect from informative initialization; not a cold-start or hardware validation")
    write_json(out / "manifest.json", dict(sha256=hashes, protocol=protocol, input_diagnosis=str(diagnosis)))
    saved = torch.load(source / "vae_last.pt", map_location="cpu", weights_only=True)
    if hashes[str(source / "dataset.h5")] != saved["data_provenance_hash"]:
        raise ValueError("Source data hash mismatch")
    prep = Preprocessor.from_state(saved["preprocessing"])
    with h5py.File(source / "dataset.h5", "r") as f:
        arrays = {s: prep.transform(f[s]["ft"][:]) for s in ("train", "validation", "test")}
    template = arrays["train"].mean(0)
    pca = dict(np.load(diagnosis / "pca5.npz", allow_pickle=False))
    scaler = Standardizer.from_state(json.loads((diagnosis / "shared_pca5_linear_scaler.json").read_text()))
    pca_codes = scaler.transform(transform_pca(arrays["train"].reshape(len(arrays["train"]), -1), pca))
    initialized = RepairedVAE().eval()
    initialized.load_state_dict(torch.load(diagnosis / "paired_initial.pt", map_location="cpu", weights_only=True))
    with torch.no_grad():
        codes = initialized.encoder(torch.from_numpy(arrays["train"]))[0].numpy()
    results, transfers = {}, {}
    for seed in seeds:
        decoder = torch.load(diagnosis / f"shared_pca5_linear_seed_{seed}" / "best.pt",
                             map_location="cpu", weights_only=True)["model"]
        initial, transfer_error = informative_original(initialized.encoder.state_dict(), decoder, codes, pca_codes)
        torch.save(initial.state_dict(), out / f"initial_seed_{seed}.pt")
        transfers[str(seed)] = dict(train_coordinate_transfer_mse=transfer_error,
                                   dropout=initial.decode_hidden[2].p)
        for beta in betas:
            key = f"seed_{seed}_beta_{beta:g}"
            write_json(out / "status.json", dict(state="running", stage=key))
            best, last, selection = fit_neural(initial, arrays, arrays, out / key,
                seed=seed, epochs=200, lr=.0001, batch_size=32, beta=beta)
            results[key] = dict(selection=selection)
            for which, model in [("best", best), ("last", last)]:
                with torch.no_grad():
                    x = torch.from_numpy(arrays["test"])
                    prediction, mu, lv = model(x, sample=False)
                    row = metric_row(prediction.numpy(), arrays["test"], template, prep)
                    row["kl"] = float(vae_loss(prediction, x, mu, lv, beta)[2])
                    row["latent_variance"] = mu.var(0, unbiased=False).tolist()
                    row["posterior_noise_variance"] = lv.exp().mean(0).tolist()
                    row["fixed_latent_mse"] = mse(model.decode(mu.mean(0).expand_as(mu)).numpy(), arrays["test"])
                    row["shuffled_latent_mse_mean"] = float(np.mean([
                        mse(model.decode(mu[np.random.default_rng(s).permutation(len(mu))]).numpy(), arrays["test"])
                        for s in range(4200, 4220)]))
                results[key][which] = row
            write_json(out / "results.json", results)
    if any(file_digest(p) != h for p, h in hashes.items()):
        raise RuntimeError("Protected file changed")
    for seed in seeds:
        scores = [results[f"seed_{seed}_beta_{b:g}"]["selection"]["initial_validation_mse"] for b in betas]
        if len(set(scores)) != 1:
            raise RuntimeError("Paired initial predictions differ")
    report = dict(protocol=protocol, transfers=transfers, results=results,
                  all_protected_hashes_unchanged=True, paired_initial_predictions_identical=True)
    write_json(out / "report.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for seed, ax in zip(seeds, axes):
        for beta in betas:
            h = json.loads((out / f"seed_{seed}_beta_{beta:g}" / "history.json").read_text())
            ax.plot([r["epoch"] for r in h], [r["validation_mse"] for r in h], label=f"beta={beta:g}")
        ax.set(xlabel="Epoch", ylabel="Validation MSE", yscale="log", title=f"Original VAE, seed {seed}")
        ax.legend()
    fig.savefig(out / "original_beta_comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# Exact original architecture beta pairing", "",
        "Original SpongeVAE with Dropout(0.1); identical informative initialization within each seed.",
        "Fixed 200 epochs; this table uses the last checkpoint, not the validation-selected initial model.", "",
        "| Run | Last test MSE | Last KL | Fixed-code MSE | Shuffled-code MSE |", "|---|---:|---:|---:|---:|"]
    for name, value in results.items():
        r = value["last"]
        lines.append(f"| {name} | {r['mse']:.9g} | {r['kl']:.6g} | {r['fixed_latent_mse']:.9g} | {r['shuffled_latent_mse_mean']:.9g} |")
    lines += ["", "All input hashes unchanged. Paired initial predictions are identical.",
              "No production model/export/configuration was replaced.", "", protocol["limitation"], ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "status.json", dict(state="complete"))
    print(f"Completed original-architecture diagnosis: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", required=True)
    parser.add_argument("--output", default="runs/sim_training/original_vae_beta_diagnosis")
    args = parser.parse_args()
    run(args.diagnosis, new_output(args.output))


if __name__ == "__main__":
    main()
