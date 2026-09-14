"""Controlled contact reference/FF ablation; never collects a training dataset."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from scripts.shared.common import (
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
from scripts.sim_pretrain.collection import tracking_passes
from scripts.sim_pretrain.experiments.diagnose_contact import contact_result
from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics


class ReferenceTraceWipe(PretrainingWipe):
    def prepare(self, extra_height=0.0):
        self.reference_rows = []
        self.record_reference = False
        start = super().prepare(extra_height)
        self.record_reference = True
        return start

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if not getattr(self, "record_reference", False):
            return
        controller = self.robot.part_controllers["right"]
        model, data = self.sim.model._model, self.sim.data._data
        contact = contact_result(
            model,
            data,
            self.tool_ids,
            self.table_id,
            data.site_xpos[self.site_id],
            controller.qvel_index,
        )
        self.reference_rows.append(
            {
                "q_goal": controller.goal_qpos.copy(),
                "goal_velocity": getattr(controller, "goal_velocity", np.zeros(6)).copy(),
                "world_contact_wrench": contact["wrench"],
                "normal_sum": contact["normal_sum"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    report = {
        "config": cfg,
        "provenance": provenance(),
        "script_sha256": file_digest(__file__),
        "gain": 300,
        "scope": "engineering ablation, not collection authorization",
        "cases": [],
    }
    cases = [
        ("free", (1.75, 500.25, 0.16), 0.08),
        ("mu0_k1000", (0, 1000, 0.02), 0),
        ("mu3p5_k1000", (3.5, 1000, 0.02), 0),
        ("mu3p5_k0p5", (3.5, 0.5, 0.02), 0),
    ]
    for mode, ff in (("actual", True), ("actual", False), ("nominal", True), ("nominal", False)):
        variant = f"{mode}_{'ff' if ff else 'noff'}"
        local = copy.deepcopy(cfg)
        local["simulation"].update(reference_mode=mode, target_velocity_feedforward=ff)
        for name, parameters, height in cases:
            row = {
                "variant": variant,
                "case": name,
                "parameters": list(parameters),
                "passed": False,
            }
            env = None
            print(f"Starting {variant}/{name}", flush=True)
            try:
                env = ReferenceTraceWipe(local, 300, parameters)
                data = env.rollout(extra_height=height)
                trace = {
                    key: np.asarray([r[key] for r in env.reference_rows])
                    for key in env.reference_rows[0]
                }
                np.savez_compressed(out / f"{variant}_{name}.npz", **data, **trace)
                row["metrics"] = rollout_metrics(data)
                if name == "free":
                    row["passed"] = bool(tracking_passes(row["metrics"], local))
                else:
                    row["contact_motion"] = contact_motion_acceptance(data)
                    row["normal_load_slide_mean_n"] = float(trace["normal_sum"][200:].mean())
                    row["normal_load_peak_n"] = float(trace["normal_sum"].max())
                    row["joint_error_max_rad"] = float(
                        np.abs(trace["q_goal"] - data["joints"]).max()
                    )
                    env.record_reference = False
                    row["unloaded_wrench"] = env.unload(data)
                    row["unloaded"] = True
                    row["passed"] = row["contact_motion"]["passed"] and row["unloaded"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None:
                    if env.reference_rows:
                        np.savez_compressed(
                            out / f"{variant}_{name}_reference.npz",
                            **{
                                key: np.asarray([r[key] for r in env.reference_rows])
                                for key in env.reference_rows[0]
                            },
                        )
                    env.close()
            report["cases"].append(row)
            write_json(out / "report.json", report)
            print(f"Finished {variant}/{name}: {row}", flush=True)
    report["eligible_variants"] = [
        variant
        for variant in ("actual_ff", "actual_noff", "nominal_ff", "nominal_noff")
        if all(row["passed"] for row in report["cases"] if row["variant"] == variant)
    ]
    report["source_unchanged"] = report["provenance"] == provenance()
    report["nominal_reference_contact_independence"] = {}
    for variant in ("nominal_ff", "nominal_noff"):
        paths = [out / f"{variant}_{name}.npz" for name, _, _ in cases[1:]]
        if all(path.exists() for path in paths):
            with np.load(paths[0]) as first:
                reference = {key: first[key].copy() for key in ("q_goal", "goal_velocity")}
            errors = {}
            for key, values in reference.items():
                peak = 0.0
                for path in paths[1:]:
                    with np.load(path) as other:
                        peak = max(peak, float(np.abs(values - other[key]).max()))
                errors[f"{key}_max_difference"] = peak
            report["nominal_reference_contact_independence"][variant] = errors
    write_json(out / "report.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for col, (name, _, _) in enumerate(cases[1:]):
        for variant in ("actual_ff", "actual_noff", "nominal_ff", "nominal_noff"):
            path = out / f"{variant}_{name}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                axes[0, col].plot(
                    data["time"],
                    1000 * (data["position"][:, 1] - data["target_position"][199, 1]),
                    label=variant,
                )
                axes[1, col].plot(data["time"], np.rad2deg(data["orientation_error"]))
                axes[2, col].plot(data["time"], data["normal_sum"])
        axes[0, col].set_title(name)
        axes[0, col].plot([0, 2, 3, 4], [0, 0, 50, 0], "k--", label="target")
        axes[1, col].axhline(2, color="black", linestyle="--")
        axes[2, col].set_xlabel("Time (s)")
    for row, label in enumerate(
        ("Y displacement (mm)", "Orientation error (deg)", "Normal contact load (N)")
    ):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
