# UR5e Contact Comparison

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This isolated simulation comparison uses the vendored robosuite UR5e. It does not
modify the AIRBOT config, production controllers, source XML or collection gate.
The implementation is in `scripts/sim_pretrain/experiments/test_ur5e_contact.py`, outside the production
package, so existing AIRBOT source provenance is not invalidated by this script.

## Comparison Definition

- Retain UR5e's native rigid-body inertias, damping .001, frictionloss .01,
  and robosuite's automatically supplied armature [5,2.5,1.666667,1.25,1,.833333].
  These are robosuite model settings, not a calibrated hardware identification.
- Retain native motor limits: first three joints +/-150 Nm, wrists +/-28 Nm.
  Do not apply AIRBOT's +/-10/5 Nm limits or DISCOVERSE inertial overrides.
- Use UR5e's default RethinkMount and Wipe base placement, the original
  WipingGripper (30 g), and no AIRBOT-specific compact mounting hardware.
- Keep the same table .5 x .8 m at [.15,0,.9], but center exploration at [.15,0]
  for the UR workspace. The actual compiled base position is recorded per case.
- Keep 1 mm initial gap, 20 mm target press in 2 s, +/-50 mm sliding in 1 s each,
  .002 s physics, 100 Hz sampling, 400 samples, and the same parameter mapping.
- Apply the unchanged contact-motion gate in `docs/sim_pretrain/contact_acceptance.md`.

Two controller variants distinguish robot and controller changes:

1. `ik_ff`: the same actual-state IK, 2 rad/s limit, mass-scaled joint PD and
   target-difference FF as AIRBOT. Select the first free-space passing gain from
   [100,300,1000], then run all 27 contact cases.
2. `osc`: UR5e's stock OSC_POSE at its native gain150 and damping. Only its input
   representation changes from relative/base to absolute/world so it receives
   the same pose targets. No velocity FF or gain tuning is added. If free-space
   fails, run three representative contacts as diagnostics only.

If no gain passes free-space for IK, three diagnostic contacts still run at
gain300. Such diagnostics cannot pass overall acceptance. Sensor validation and
unloading are checked; all failures remain recorded. This is not a controlled
robot-only causal comparison: mounts, inertias, armature, limits and workspace
differ. OSC also changes the controller. Never mix these FT data with AIRBOT data.

## Run

From the project root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.test_ur5e_contact --output runs/sim_training/ur5e_contact_v1
python -m unittest discover -s tests -t . -p 'test_ur5e_comparison.py' -v
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if activation is unavailable.
`--mode ik_ff` or `--mode osc` selects one variant. The output directory must be
new. Outputs include incremental report.json, per-mode effective config.json,
compiled dynamics and limits, source/model asset hashes, and completed rollout
NPZ files. The script finishing is not an acceptance pass; inspect each variant's
`passed`, `contact_scope`, free-space results and failed checks in report.json.
The per-mode config.json files are experiment evidence, not inputs to the
AIRBOT-only production `scripts.sim_pretrain.pretrain` CLI. Reproduce with this script
and its source `--config`, not by feeding a UR profile to the AIRBOT collector.
No formal collection, VAE training or real hardware control is performed.

A supplemental equal-gain diagnostic uses UR5e IK+FF gain300, matching AIRBOT's
selected gain. This does not replace the primary first-passing gain100 scan.
Run the following in a Python session from the project root with the same thread
environment settings; use a new output directory:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
from test_ur5e_contact import comparison_config, run_case, experiment_sources
from scripts.shared.common import load_config, write_json, provenance
out = Path("runs/sim_training/ur5e_matched_gain300_repeat")
out.mkdir(exist_ok=False)
cfg = comparison_config(load_config("configs/sim_pretrain/pretrain_paper.yaml"), "ik_ff")
cfg["output_dir"] = str(out)
report = dict(scope="matched gain300 diagnostic only", config=cfg,
              provenance=provenance(), experiment_sources=experiment_sources(), cases=[])
cases = [("free", (1.75, 500.25, .16), .08), ("mu0_k1000", (0, 1000, .02), 0),
         ("mu3p5_k1000", (3.5, 1000, .02), 0), ("mu3p5_k0p5", (3.5, .5, .02), 0)]
for name, parameters, height in cases:
    row = run_case(cfg, "ik_ff", 300, parameters, height, out / (name + ".npz"))
    row["name"] = name
    report["cases"].append(row)
    write_json(out / "report.json", report)
report["source_unchanged"] = (report["provenance"] == provenance()
                              and report["experiment_sources"] == experiment_sources())
write_json(out / "report.json", report)
```

After both experiments finish, plot the three representative contacts:

```bash
python -m scripts.sim_pretrain.experiments.plot_ur5e_contact --ur runs/sim_training/ur5e_contact_v1 --matched archive/sim_pretrain/ur5e_matched_gain300_v1 --output runs/sim_training/ur5e_contact_comparison.png
```

The graph compares motion, not identical installations; labels include each
controller and gain. It does not reinterpret raw FT as world normal force.

## Results (2026-09-08)

Primary results: `archive/sim_pretrain/ur5e_contact_v1/report.json`.
Matched gain300 diagnostics: `archive/sim_pretrain/ur5e_matched_gain300_v1/report.json`.
Plot: `archive/sim_pretrain/ur5e_contact_comparison.png` (visually checked).

| Variant | Free-space RMS mm | Free-space max pose deg | Contact scope | Contact passes |
| --- | ---: | ---: | --- | ---: |
| UR5e IK+FF gain100, first passing gain | .921002 | .008584 | full27 | 0/27 |
| UR5e IK+FF gain300, matched-gain diagnostic | .424010 | .002454 | representative3 | 0/3 |
| UR5e stock OSC gain150 | 5.274229 | .367660 | representative3, free-space failed | 0/3 |

All 36 rollouts completed: 3 free-space and 33 contact. All 33 contact trials
unloaded. Neither model/controller initialization nor non-tool collisions caused
these failures. Both primary variants passed the independent sensor load check.
All rollouts had zero recorded actuator saturation. Compiled UR base world
position was [-.41,0,.912], tool mass .03 kg, and native joint limits/dynamics
matched the audited model. All 32 unit tests passed, including native UR
parameters, both controller modes, reset, absolute OSC targets and sensor signs.

The following representative cases all use width=.02. Negative reverse travel
means the net motion continued in the forward direction during the return phase.
Every row below fails the fixed motion gate.

| Mode / gain | mu / k | Forward mm | Reverse mm | Max Y error mm | Max X error mm | Max pose deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IK+FF / 100 | 0 / 1000 | 51.308 | 51.551 | 5.111 | 13.088 | 2.909 |
| IK+FF / 100 | 3.5 / 1000 | .594 | .013 | 49.425 | .640 | 6.238 |
| IK+FF / 100 | 3.5 / .5 | 2.217 | -2.909 | 47.774 | 1.318 | 6.252 |
| IK+FF / 300 | 0 / 1000 | 51.326 | 51.570 | 4.051 | 12.353 | 2.752 |
| IK+FF / 300 | 3.5 / 1000 | .728 | -.179 | 49.291 | .667 | 6.144 |
| IK+FF / 300 | 3.5 / .5 | 3.386 | -5.803 | 46.611 | 3.270 | 6.203 |
| OSC / 150 | 0 / 1000 | 42.633 | 34.693 | 11.207 | 13.106 | .273 |
| OSC / 150 | 3.5 / 1000 | .186 | -.191 | 49.887 | .619 | .055 |
| OSC / 150 | 3.5 / .5 | 3.399 | -7.156 | 46.744 | 4.154 | .190 |

For context, current AIRBOT compact gain300 at mu3.5/k1000/width.02 travels
.667/.401 mm with 9.400 degrees peak pose error. UR IK+FF reduces the peak tilt
to about 6 degrees without restoring sliding. Stock UR OSC keeps pose error
below .06 degrees in that case, yet also barely translates. Thus preventing
tool tilt alone is not sufficient for the specified high-friction exploration.
Low-friction UR IK+FF follows Y but acquires 12-13 mm X error; a visually correct
Y stroke alone is not a motion-gate pass.

These observations do not prove an intrinsic UR limitation or uniquely identify
the failure mechanism. Native UR armature, inertia, limits, attachment and
workspace differ from AIRBOT. The stock OSC comparison additionally has no
nominal velocity feedforward, and its free-space lag already fails the 1 mm gate.
No additional gain tuning, armature override, friction reduction or threshold
relaxation was used. Changing only to UR5e did not resolve the current protocol's
contact failure; directional contact feedback and normal loading remain the
next design question.

`archive/sim_pretrain/ur5e_contact_v1/verification.json` records post-run finite-array checks,
recomputed acceptance for all 33 contact NPZ files, source/model hash matching
and the still-valid AIRBOT provenance. No AIRBOT config, source model, production
controller or existing report was changed. No collection or training was run.
