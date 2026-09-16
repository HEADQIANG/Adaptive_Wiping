"""Visualize existing training FT only, without model inference or data changes."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.shared.common import (
    CHANNELS,
    digest,
    file_digest,
    load_config,
    output_dir,
    write_json,
)
from scripts.shared.paths import ROOT
from scripts.shared.preprocessing import Preprocessor

PANEL_CHANNELS = (0, 3, 1, 4, 2, 5)
PHASE_TEXT = "Press: 0-2 s   |   Forward: 2-3 s   |   Reverse: 3-4 s"


def checked_indices(indices, count):
    values = np.asarray(indices)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty list of integer training indices")
    if len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= count):
        raise ValueError(f"Training indices must be unique and in [0,{count - 1}]")
    return values.astype(int)


def load_training(config_path):
    cfg = load_config(config_path)
    folder = output_dir(cfg)
    paths = [
        Path(config_path).resolve(),
        folder / "dataset.h5",
        folder / "dataset_integrity.json",
        folder / "preprocessing.json",
    ]
    hashes = {str(path): file_digest(path) for path in paths}
    integrity = json.loads(paths[2].read_text())
    if hashes[str(paths[1])] != integrity["sha256"]:
        raise ValueError("Dataset integrity mismatch")
    with h5py.File(paths[1], "r") as h5:
        if not h5.attrs.get("complete", False) or h5.attrs.get("config_hash") != digest(cfg):
            raise ValueError("Incomplete dataset or mismatched configuration")
        group = h5["train"]
        raw = group["ft"][:]
        times = group["time"][:]
        parameters = group["parameters"][:]
        if raw.shape != (cfg["dataset"]["train"], 400, 6) or not np.all(group["valid"][:]):
            raise ValueError("Invalid training split")
        motion = group["motion_passed"][:].astype(bool)
    if times.shape != raw.shape[:2] or parameters.shape != (len(raw), 3):
        raise ValueError("Invalid training time/parameter shapes")
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(times).all()
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("Non-finite training values")
    np.testing.assert_allclose(
        times, np.broadcast_to(np.arange(1, 401) / 100.0, times.shape), rtol=0, atol=1e-10
    )
    prep = Preprocessor.from_state(json.loads(paths[3].read_text()))
    fitted = Preprocessor(**cfg["filter"], sample_hz=cfg["simulation"]["sample_hz"]).fit(raw)
    if fitted.state() != prep.state():
        raise ValueError("Saved preprocessing differs from training-only fit")
    return {
        "config": cfg,
        "folder": folder,
        "hashes": hashes,
        "raw": raw,
        "time": times[0],
        "filtered": prep.filtered(raw),
        "normalized": prep.transform(raw),
        "parameters": parameters,
        "motion_passed": motion,
        "preprocessing": prep.state(),
        "dataset_sha256": hashes[str(paths[1])],
    }


def quantile_band(values):
    return np.quantile(values, [0.1, 0.5, 0.9], axis=0)


def make_figure(title, normalized=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for channel, ax in zip(PANEL_CHANNELS, axes.flat):
        ax.set(
            xlabel="Time (s)",
            xlim=(0, 4),
            ylabel=(
                f"{CHANNELS[channel]} (normalized)"
                if normalized
                else f"{CHANNELS[channel]} ({'N' if channel < 3 else 'N m'})"
            ),
        )
        for boundary in (2, 3):
            ax.axvline(boundary, color="#899097", linestyle=":", linewidth=1)
        ax.grid(alpha=0.18)
        if normalized:
            ax.set_ylim(-0.03, 0.93)
    fig.suptitle(f"{title}\n{PHASE_TEXT}", fontsize=13)
    return plt, fig, axes


def plot_sample(data, index, output, normalized=False):
    mu, stiffness, width = data["parameters"][index]
    mode = data["config"].get("research_comparison", {}).get("controller", "training")
    title = (
        f"{mode}: training row {index} | mu={mu:.3f}, stiffness_direct={stiffness:.3f}, width={width:.4f}\n"
        + (
            "Actual normalized network input"
            if normalized
            else "Recorded raw FT and training-filtered FT (local frame)"
        )
    )
    plt, fig, axes = make_figure(title, normalized)
    for channel, ax in zip(PANEL_CHANNELS, axes.flat):
        if normalized:
            ax.plot(
                data["time"],
                data["normalized"][index, :, channel],
                color="#16836d",
                linewidth=1.6,
                label="Filtered + training-set normalization",
            )
        else:
            ax.plot(
                data["time"],
                data["raw"][index, :, channel],
                color="#9299a1",
                linewidth=0.9,
                alpha=0.75,
                label="Raw recorded FT",
            )
            ax.plot(
                data["time"],
                data["filtered"][index, :, channel],
                color="#216fa8",
                linewidth=1.5,
                label="10 Hz filtered FT",
            )
    axes[0, 0].legend(fontsize=9)
    filename = f"{'normalized_' if normalized else ''}sample_{index:03d}.png"
    fig.savefig(output / filename, dpi=150)
    plt.close(fig)
    return filename


def plot_overview(data, indices, output, *, subset=False, normalized=False):
    values = data["normalized" if normalized else "filtered"][indices]
    bands, mean = quantile_band(values), values.mean(0)
    mode = data["config"].get("research_comparison", {}).get("controller", "training")
    title = (
        f"{mode}: {'fixed AE subset' if subset else 'complete training split'}, N={len(indices)}\n"
        + ("Normalized input" if normalized else "Filtered local FT")
        + ": mean, median and cross-trajectory 10-90% band"
    )
    plt, fig, axes = make_figure(title, normalized)
    for channel, ax in zip(PANEL_CHANNELS, axes.flat):
        ax.fill_between(
            data["time"],
            bands[0, :, channel],
            bands[2, :, channel],
            color="#216fa8",
            alpha=0.18,
            label="10-90% of trajectories",
        )
        ax.plot(data["time"], bands[1, :, channel], color="#216fa8", linewidth=1.6, label="Median")
        ax.plot(
            data["time"],
            mean[:, channel],
            color="#c45a3a",
            linestyle="--",
            linewidth=1.1,
            label="Mean",
        )
    axes[0, 0].legend(fontsize=9)
    filename = (
        "subset_overview.png"
        if subset
        else "normalized_overview.png"
        if normalized
        else "training_overview.png"
    )
    fig.savefig(output / filename, dpi=150)
    plt.close(fig)
    return filename, bands


def visualize(config_path, output, indices=(0, 83), subset_manifest=None):
    cfg = load_config(config_path)
    out = Path(output).resolve()
    if out.is_relative_to(output_dir(cfg).resolve()):
        raise ValueError("Output must be outside the frozen dataset directory")
    if subset_manifest and out.is_relative_to(Path(subset_manifest).resolve().parent):
        raise ValueError("Output must be outside the subset experiment directory")
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    data = load_training(config_path)
    selected = checked_indices(indices, len(data["raw"]))
    subset = None
    if subset_manifest:
        manifest = json.loads(Path(subset_manifest).read_text())
        if manifest["dataset_sha256"] != data["dataset_sha256"]:
            raise ValueError("Subset manifest belongs to another dataset")
        if manifest["dataset_config_sha256"] != file_digest(config_path):
            raise ValueError("Subset manifest/config mismatch")
        subset = checked_indices(manifest["small_train_indices"], len(data["raw"]))
        data["hashes"][str(Path(subset_manifest).resolve())] = file_digest(subset_manifest)
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "existing training FT only; no inference, training or recollection",
        "state": "running",
        "channels": CHANNELS,
        "panel_order": [CHANNELS[c] for c in PANEL_CHANNELS],
        "frame": "ft_frame local",
        "units": ["N"] * 3 + ["N m"] * 3,
        "training_count": len(data["raw"]),
        "selected_indices": selected.tolist(),
        "subset_indices": subset.tolist() if subset is not None else None,
        "motion_pass_count": int(data["motion_passed"].sum()),
        "removed_samples": 0,
        "band_definition": "pointwise quantiles across trajectories, not a confidence interval",
        "raw_signal_note": "Includes tool weight; not bias-subtracted contact force",
        "stiffness_note": "stiffness_direct is not calibrated physical N/m",
        "preprocessing": data["preprocessing"],
        "source_sha256": file_digest(__file__),
        "input_hashes": data["hashes"],
        "figures": [],
    }
    write_json(out / "summary.json", summary)
    try:
        plot_data = {
            "time": data["time"],
            "indices": selected,
            "parameters": data["parameters"][selected],
            **{key: data[key][selected] for key in ("raw", "filtered", "normalized")},
        }
        for normalized in (False, True):
            filename, bands = plot_overview(
                data, np.arange(len(data["raw"])), out, normalized=normalized
            )
            summary["figures"].append(filename)
            plot_data["normalized_band" if normalized else "filtered_band"] = bands
        if subset is not None:
            filename, bands = plot_overview(data, subset, out, subset=True)
            summary["figures"].append(filename)
            plot_data.update(subset_indices=subset, subset_filtered_band=bands)
        for index in selected:
            for normalized in (False, True):
                summary["figures"].append(plot_sample(data, int(index), out, normalized))
        np.savez_compressed(out / "plotted_data.npz", **plot_data)
        for name in ("raw", "filtered", "normalized"):
            summary[f"{name}_channel_min"] = data[name].min(axis=(0, 1)).tolist()
            summary[f"{name}_channel_max"] = data[name].max(axis=(0, 1)).tolist()
        changed = [path for path, sha in data["hashes"].items() if file_digest(path) != sha]
        summary.update(state="completed", inputs_unchanged=not changed, changed_files=changed)
        if changed:
            raise RuntimeError("Input files changed during visualization")
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "output": str(out),
                "training_count": summary["training_count"],
                "figures": summary["figures"],
                "inputs_unchanged": summary["inputs_unchanged"],
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 83])
    parser.add_argument("--subset-manifest", type=Path)
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    visualize(args.dataset_config, args.output, args.indices, args.subset_manifest)


if __name__ == "__main__":
    main()
