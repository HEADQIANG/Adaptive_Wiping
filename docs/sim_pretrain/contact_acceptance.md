# Contact Motion Acceptance And Reference Ablation

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

These are engineering acceptance criteria, not thresholds quoted from the paper.
They are fixed before the reference ablation; a failed experiment must not tune
them to pass. No formal collection is authorized by this experiment.

## Gate Version 1

- Use the 400 post-integration samples at 0.01 through 4.00 seconds.
- Forward travel: Y(3)-Y(2), 45 to 55 mm inclusive.
- Reverse travel: Y(3)-Y(4), 45 to 55 mm inclusive.
- Return displacement: abs(Y(4)-Y(2)) <= 5 mm.
- Maximum absolute Y target error over 2.00 through 4.00 s <= 5 mm.
- Maximum absolute X target error over the same interval <= 3 mm.
- Maximum orientation error over the full rollout <= 2 degrees.
- Each sliding phase (2.01..3.00, 3.01..4.00) has tool/table contact in >=95%
  of samples, with no missing-contact run longer than 50 ms at 100 Hz.
- At most 1% of command intervals contain actuator saturation (any axis/physical
  substep). This is a sampled interval criterion, not an exact duration estimate.
- Existing finite-data, clock, effective-parameter, unexpected-collision and
  post-rollout unloading checks remain mandatory in sanity.

Ten percent travel tolerance and a 5 mm lateral error budget allow small transient
tracking errors but reject near-static high-friction trajectories. The 2 degree
pose bound is slightly looser than the free-space 1 degree bound while excluding
the previously observed 10 degree edge-pivoting. Z displacement and world normal
load are reported, not thresholded as XYZ free-space error or a minimum force:
normal compliance is expected and low-stiffness signals may legitimately be weak.
Geometric contact is not proof of a nonzero load. These checks are a minimum
motion-quality gate, not calibrated material or hardware safety limits.

Both the 27-point sanity scan and each collected trajectory now apply this gate
and the same post-rollout unloading check.
Collection failures keep their assigned parameters. Production source changes
invalidate old sanity provenance; never edit old reports to bypass that check.

## Experimental Reference

The production default remains the original actual-state IK (`reference_mode`
omitted or `actual`). The experiment also runs `reference_mode: nominal`:

1. Keep the existing actual-state preparation and settle checks.
2. At exploration start, initialize one private MuJoCo data object from actual q.
3. Run the existing damped IK on this nominal kinematic state, integrating from
   the preceding nominal q, with unchanged 2 rad/s limit and joint ranges.
4. Feed successive nominal q differences to the existing velocity FF. Joint PD
   uses nominal q minus actual q and nominal velocity minus actual velocity.
   Contact dynamics never update the nominal reference after initialization.
5. Abort if any nominal/actual joint deviation exceeds 0.35 rad, or nominal FK
   misses the commanded pose by more than 3 mm / 2 degrees. This bounds reference
   accumulation by stopping the simulation trial, not by changing torque limits
   or silently continuing a failed trajectory. It is not a true anti-windup
   recovery controller and must not be used for hardware control.
6. Continue unloading from the current nominal pose. Starting that interpolation
   from the displaced actual pose would introduce a reference jump; this was
   detected by the feasibility guard in the first experiment and corrected.

The four-way comparison is actual/nominal reference crossed with FF on/off,
all at gain300. Each variant runs free-space and three prior contact cases.
FF off also applies during preparation; this is explicitly a full-controller
ablation. No gains, inertia, contact ranges, torque limits or speeds are tuned.

## Run

From the project root, with the verified `clean` environment:

```bash
conda activate clean
python -m unittest discover -s tests -t . -p 'test_*.py' -v
python -m scripts.sim_pretrain.experiments.validate_contact_reference --config configs/sim_pretrain/pretrain_paper.yaml --output runs/sim_data/contact_reference_v2
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if activation is unavailable.
The output directory must not exist. Outputs include per-case rollout/reference
NPZ, incremental report.json with config/provenance and failures, and comparison.png.
Use this script for nominal-reference experiments; the older contact-chain
diagnostic reconstructs actual-state IK by design and is not a nominal solver
diagnostic. The additional `verification.json` in the completed v2 directory
records the post-run finite-array, baseline, FF-equation and hash checks.
Aborted cases retain available reference samples; absence of a completed rollout
is not success. World contact loads are read after command integration/forward;
they are not the raw local FT sensor values.
The v1 directory is preserved as the initial experiment with the unloading
reference discontinuity; use v2 for the corrected complete comparison.

A variant is eligible for further validation only if free-space and all three
contacts including unloading pass. Only then run a fresh full sanity scan in a
new output directory with an explicitly selected config. Passing these three
cases is not permission to collect. Formal collection and training are excluded.

To exercise the new gate with the unchanged production controller, use a fresh
directory and preserve the effective config (this runs the full 27-point scan,
not the nominal-reference candidate):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -c 'import os; from pathlib import Path; from scripts.shared.common import load_config, write_json; from scripts.sim_pretrain.collection import sanity; cfg=load_config("configs/sim_pretrain/pretrain_paper.yaml"); cfg["output_dir"]="runs/sim_data/contact_acceptance_sanity_repeat"; out=Path(cfg["output_dir"]); out.mkdir(exist_ok=False); write_json(out / "config.json", cfg); write_json(out / "runtime.json", {k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}); report=sanity(cfg); raise SystemExit(0 if report["passed"] else 2)'
```

Exit 2 is the expected failed physical acceptance, not a dependency failure.
The initial `archive/sim_pretrain/contact_acceptance_sanity_v1` run was interrupted by SIGTERM
(exit143) after 9/27 grid entries, before the sensor check. It is not a completed
sanity result. The rerun uses output `archive/sim_pretrain/contact_acceptance_sanity_v2`, with
the numerical-library single-thread settings recorded in `runtime.json`.

## Completed Reference Experiment (2026-09-08)

Authoritative artifacts: `archive/sim_pretrain/contact_reference_v2/report.json`, 16 rollout
NPZ files (four free-space and twelve contact), per-case reference traces, and
`comparison.png`. All sixteen rollouts completed without guards firing and all
twelve contacts unloaded. No variant passed all four cases. No new nominal
27-point scan or formal collection was initiated because its high-friction
representative cases already failed.

Free-space RMS (mm): actual/FF 0.447193, actual/no-FF 3.983825,
nominal/FF 0.426558, nominal/no-FF 3.697693. Both FF variants pass the unchanged
1 mm / 1 degree free-space gate; neither no-FF variant passes it.

All contacts below have width .02. Travel is signed phase endpoint displacement,
not full-rollout range or accumulated path length. Negative reverse travel means
the tool moved in the wrong net direction during the return interval.

| Reference / FF | mu / k | Forward mm | Reverse mm | Max Y error mm | Max pose deg | Motion gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| actual / on | 0 / 1000 | 49.649 | 49.000 | 5.667 | .870 | fail |
| actual / off | 0 / 1000 | 32.877 | 15.828 | 17.113 | .655 | fail |
| nominal / on | 0 / 1000 | 50.345 | 50.566 | 2.374 | .601 | pass |
| nominal / off | 0 / 1000 | 44.558 | 38.985 | 5.755 | .599 | fail |
| actual / on | 3.5 / 1000 | .687 | .451 | 49.312 | 9.643 | fail |
| actual / off | 3.5 / 1000 | .285 | -.025 | 49.714 | 6.593 | fail |
| nominal / on | 3.5 / 1000 | .800 | .157 | 49.198 | 10.083 | fail |
| nominal / off | 3.5 / 1000 | .590 | -.033 | 49.408 | 9.329 | fail |
| actual / on | 3.5 / .5 | -.235 | .403 | 50.231 | 10.055 | fail |
| actual / off | 3.5 / .5 | -.245 | .005 | 50.242 | 7.076 | fail |
| nominal / on | 3.5 / .5 | -.130 | -.910 | 50.126 | 10.439 | fail |
| nominal / off | 3.5 / .5 | -.173 | -.703 | 50.169 | 9.754 | fail |

In each nominal variant, q_goal and FF velocity arrays have exactly zero maximum
difference across the three contact parameter sets. A unit test also perturbs
live q independently and confirms unchanged nominal goals/velocities with no
live state mutation. The intended reference decoupling is implemented, but that
does not establish sufficient contact-control authority.
The nominal/FF velocity equals the goal difference at all measured transitions
exactly, with peak .121952 rad/s across its four rollouts, below the unchanged
2 rad/s bound. No FF clipping is needed for these nominal trajectories.

For mu3.5/k1000, mean sliding normal load rises from 1.602 N (actual/FF) to
7.268 N (nominal/FF), while maximum post-integration joint target error rises
from .02191 to .17887 rad. For the low-friction case mean normal load rises from
2.229 to 6.606 N. Thus removing the small actual-state-relative reference also
increases pressing load; it does not independently improve lateral force and
orientation regulation. All sixteen trajectories have zero recorded saturation.
This is evidence against reference decoupling alone being sufficient, not proof
that a specific replacement controller or gain will work.

The three actual/FF position and raw-FT trajectories match the previous
uninstrumented `runs/sim_data/contact_diagnosis/*_baseline.npz` exactly (maximum
difference zero). The v2 production provenance was unchanged throughout the
experiment. The comparison figure was visually checked. All 25 tests passed,
including nominal independence/guards, acceptance boundaries, and end-to-end
mocked sanity/collection rejection of static contact.

Offline re-scoring of the old 27 traces is recorded separately in
`archive/sim_pretrain/contact_gate_recheck_v1/report.json` with input hashes: only the three
mu0/k.5 widths pass; 24 fail. This is not a new candidate physics scan.

The subsequent full original-controller sanity rerun completed in
`archive/sim_pretrain/contact_acceptance_sanity_v2`: gain100 failed free-space, gain300
passed, all 27 contact cases ran, 3 passed the motion gate and 24 failed, and
the independent static sensor check passed. Overall `passed=false`, exit code 2.
All 27 position/raw-FT trajectories match the old traces exactly, despite the
single-thread runtime settings. `verification.json` records the comparisons,
current provenance match and `require_sanity` rejection. No old report was
overwritten or manually invalidated. The production default output still holds
its old report, which is rejected as stale; the fresh output is rejected for
physical contact-motion failure.

## Next Decision

Keep the actual-state reference as production default and keep collection
blocked. Do not lower friction, raise torque limits, restore artificial armature,
or change the acceptance thresholds to make these results pass. The next
controlled experiment should address directional contact impedance: lateral and
orientation feedback must be evaluated separately from normal pressing load,
while preserving the nominal exploration trajectory. A Cartesian pose-impedance
comparison is a candidate, not an already verified fix. No force-target/hybrid
controller or new gain sweep was implemented in this experiment.
