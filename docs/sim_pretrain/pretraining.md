# AIRBOT Play pre-training operations

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## Normal Training Completed (2026-09-09)

The user requested normal-mode training first after paired collection completed.
The normal run in `archive/sim_pretrain/two_control_pretraining_v2/normal` finished all 200
epochs and saved both VAE checkpoints and `encoder.pt`. Evaluation reports latent
collapse: test MSE .0150553 versus mean-trajectory baseline .0069280, with 0/5
active latent dimensions. Export does not imply a validated property estimator.
Commands, artifacts and limitations are in
[control_comparison_pretraining.md](control_comparison_pretraining.md).
Impedance training has not started. Historical gate restrictions below do not
override the authorized research collection/training workflow.

## Authorized Two-Controller Research Collection (2026-09-08)

The user now explicitly authorizes skipping contact-motion acceptance for paired
normal/impedance data collection. Use the separate record-only workflow in
[control_comparison_pretraining.md](control_comparison_pretraining.md).
It retains acceptance failures as labeled research data; the production `collect`
gate below is unchanged. Per-mode datasets use the same 1000/100/100 parameter
assignments. This authorization does not mean either controller passed wiping
acceptance or that physical sponge parameters have been calibrated.

## Current Direct Wrist Layout (2026-09-08)

The default is now tabletop base + `tool_mount: direct_wrist_v3`, output
`archive/sim_pretrain/pretrain_paper_direct_wrist_v3`. The transverse gripper frame, camera
bracket and backing plate are removed from the assembly; the sponge meets the
cylindrical wrist directly. Link6 aggregate inertia is retained because a separate
housing inertia is unavailable. See [direct_wrist.md](direct_wrist.md) for geometry,
limitations, commands and validation. Historical results below do not validate
this new layout. All 33 tests pass; fresh gain300 free-space RMS is .452535 mm,
but contact motion is still 3/27 and mu=.9/k1000/width.02 also fails. Collection
remains blocked. No formal dataset or training run was started.

## Contact Motion Gate And Reference Experiment (2026-09-08)

The contact-motion gate now checks directional travel, lateral tracking, pose,
contact continuity and saturation in sanity and individual collection validation.
Old sanity provenance is stale after this change. The original reference remains
the production default; nominal-reference experiments are not collection approval.
Criteria, experimental guards and commands: [contact_acceptance.md](contact_acceptance.md).
Earlier statements below about the open gate describe historical source versions.
The completed four-way reference/FF experiment did not resolve high-friction
sliding; `archive/sim_pretrain/contact_reference_v2` has no eligible variant. The full original
controller rerun in `archive/sim_pretrain/contact_acceptance_sanity_v2` passes only 3/27
contacts, passes the static sensor check, and exits 2 with `passed=false`.
All 25 software tests pass. No formal collection or training has started.

This implementation follows `docs/sim_pretrain/pretrain.md`. It only controls simulation.

Previous layout: tabletop base + `compact_v2` tool, output directory
`archive/sim_pretrain/pretrain_paper_tabletop_compact_v2`. See [tabletop_compact.md](tabletop_compact.md)
for the authoritative placement, changed TCP, commands and validation. Bridge
and restored-inertia results below describe historical layouts.

Historical engineering tool mount: `tool_mount: bridge_v1`, with output directory
`archive/sim_pretrain/pretrain_paper_bridge_v1`. See [tool_mount_bridge.md](tool_mount_bridge.md)
for assembly dimensions, changed mass/sensor lever arm and fresh validation commands.
The inertia-profile results below predate this mount and are historical.

Dynamic viewer / animated export commands are documented in
[wiping_visualization.md](wiping_visualization.md). Run
`python -m scripts.sim_pretrain.experiments.visualize_wiping` from the project root for a fresh simulation
followed by repeatable playback, without changing controller or contact physics.

## Restored Inertia Profile (2026-09-08)

The current configuration uses `simulation.inertia_profile: discoverse_standard`.
The historical run is preserved in `archive/sim_pretrain/pretrain_paper_discoverse_inertia`;
new outputs use the configured `runs/sim_training/` directory. At XML assembly time,
only the six arm-link inertials (mass, center of mass, principal inertia and its
orientation) are copied from
`asserts/references/discoverse/airbot_play.xml`, and all six arm
joint armatures are explicitly set to zero. The source file is included in
provenance hashes. The audited DISCOVERSE commit is
`d67f47c084aba0e0cf422a8725235f8b9238655a`; the byte-preserving copy and its original
source path are recorded in `asserts/manifest.json`. See [model assets](models.md).

Robot source XML, collision geometry, body transforms, tool inertia, damping,
frictionloss, IK, velocity feedforward, gains, torque limits and exploration
speeds are unchanged. Zero armature reproduces the published ordinary DISCOVERSE
MJCF assumption; it is not a measured motor-inertia claim. The `legacy` profile
retains the previous robosuite inertials/armature for isolated comparisons.

```bash
conda activate clean
python -m unittest discover -s tests -t . -p 'test_pretraining.py' -v
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
```

Previous results and reports remain in their old output directories. Verification
is complete: 18 tests passed, free-space gain 300 passed, all 27 contact scans and
the static sensor checks passed the existing automated checks (`sanity` exit 0).
The historical failure results below do not describe this restored profile.
Regression tests now also check the compiled inertial pose/tensor against the
reference, zero armature after reset, unchanged collision geometry and control
limits, unchanged source files, and rejection of incomplete source/target models.

At gain 300, free-space RMS changed from 1.253 to 0.447 mm, peak error from 6.306
to 2.296 mm, maximum orientation error from 0.676 to 0.256 degrees, and saturation
fraction from 5.75% to zero. Gain 100 RMS was 1.026 mm and still failed. The gate
selects the first passing gain, so 1000 was not tested in this run.

IMPORTANT: Do not interpret the automated pass as approval of contact tracking
quality. Existing contact checks validate finite signals, effective solver
parameters, tool-only contact and unloading; unlike the free-space check, they
do not impose position/orientation error thresholds under load. The 27 scans
had RMS 0.590-25.944 mm, maximum position error 55.151 mm and maximum orientation
error 10.055 degrees, with no torque saturation. The worst position case was
`mu=3.5, stiffness_direct=1000, width=0.02`. These deviations need investigation
before full collection. No official dataset or training run was started.

Current artifacts: `archive/sim_pretrain/pretrain_paper_discoverse_inertia/sanity.json`,
`sanity_traces/*.npz`, `inertia_validation.png` and `contact_tracking_warning.png`.
The automatic gate is open under its existing rules, but full collection is not
recommended until the contact-motion issue is resolved. No acceptance rule or
gain was silently changed to obtain this result.

## Contact Control-Chain Diagnostics

Completed results and interpretation: [contact_diagnosis.md](contact_diagnosis.md).
All three cases reproduce their uninstrumented reference exactly; the new contact
test and all 18 existing tests pass. The diagnostic identifies persistent IK
correction limiting and reference/force coupling under load, without actuator
clipping. No controller fix or acceptance-rule change has been made.

Run the isolated contact diagnostic at the currently selected gain 300:

```bash
conda activate clean
python -m unittest discover -s tests -t . -p 'test_contact_diagnosis.py' -v
python -m scripts.sim_pretrain.experiments.diagnose_contact --config configs/sim_pretrain/pretrain_paper.yaml --output runs/sim_data/contact_diagnosis_repeat
```

The output directory must not exist; choose a new `--output` for repeat runs.
This compares mu=0/k=1000, mu=3.5/k=1000 and mu=3.5/k=0.5, all with width=0.02.
Each case runs an uninstrumented reference and a traced rollout. The script
rejects trajectory/FT differences caused by instrumentation. No production
controller, configuration, dynamics parameters or collection gate is changed.
`--gain` is an explicit diagnostic-only override; the default is 300.

Outputs include `report.json`, per-case `*_physics.npz`, `*_rollout.npz`,
`*_baseline.npz`, `*_contacts.npz`, and `contact_control_chain.png`.
At each 2 ms pre-integration state, the trace records the actual P/D/feedforward/
bias torque, IK raw and bounded correction, joint-goal forward kinematics,
clipping, passive/constraint forces, Jacobian and mass matrix. A separate copied
MuJoCo data state is forwarded with the newly applied command to obtain aligned
contact forces, sensor values and accelerations; the live state is not forwarded
or modified by the diagnostic. These are instantaneous copied-state force
solutions, not separately measured post-integration samples.

Contact wrench is world-frame force ON the tool and torque about the TCP.
Sensor wrench is raw local parent-on-child support wrench, not contact force;
the shifted/rotated contact wrench is also saved for comparison. The feedback
equivalent wrench solves `J.T @ wrench = requested - bias`; it is an algebraic
representation of control torque, not a direct force command or measured load.
Contact-to-generalized force mapping and full dynamics balance are checked or
reported. Per-contact columns are time, contact index, geom1, geom2, dimension,
distance, position XYZ, contact-frame force XYZ, torque XYZ, mu and
`norm(tangential_force)/(mu*normal_force)`. This last ratio measures translation
only, not full pyramidal/elliptic cone utilization or definitive slip state.
IK correction velocity is distinct from the finite difference of successive
joint goals used by feedforward. Full collection remains discouraged pending
contact-motion validation, even though the existing automatic gate is open.

## Historical Reversal Diagnostics

An independent instrumented subclass records every 2 ms physics substep without
changing the production controller or configuration:

```bash
conda activate clean
python -m scripts.sim_pretrain.experiments.diagnose_reversal --config configs/sim_pretrain/pretrain_paper.yaml
```

It repeats the existing free-space experiment with and without feedforward at
gains 100/300/1000. Outputs are `archive/sim_pretrain/reversal_diagnosis/report.json`, physics
and 100 Hz rollout NPZ files, and reversal plots. Use `--output` with a new path
for repeat runs; existing reports are not overwritten. These are diagnostic
artifacts only, never training data, and do not update the physics gate.
Physics timestamps refer to the state before integration and the torque about
to be applied; rollout timestamps refer to the state after integration. The
first reverse command takes effect at physics time 3.00 s (target endpoint 3.01 s).
The script checks that P + D + feedforward + bias equals requested torque, and
that applied torque equals the unchanged actuator clipping operation. It also
compares positions against previous rollouts to detect instrumentation effects.
The diagnostic is complete; see `docs/sim_pretrain/reversal_diagnosis.md`. It identified
unconfigured AIRBOT armature inherited from RobotModel as the dominant contributor
to the large requested joint-1 torque. This is a model-audit finding, not evidence
that the physical AIRBOT cannot execute the task. At that historical stage no
corrective model change had been applied; current results are described above.
The extended inertia audit additionally records compiled joint armature and its
algebraic contribution to requested torque. Reproduce it separately with:

```bash
python -m scripts.sim_pretrain.experiments.diagnose_reversal --config configs/sim_pretrain/pretrain_paper.yaml --output runs/sim_data/reversal_diagnosis_inertia
```

This does not zero or otherwise change armature. Subtracting its contribution
at a recorded state is an attribution calculation, not a prediction of a new
closed-loop trajectory. The base RobotModel source is hashed in this report too.

## Historical Feedforward Profile
The earlier profile enabled `simulation.target_velocity_feedforward: true` and
writes to `archive/sim_pretrain/pretrain_paper_velocity_ff`; old no-feedforward diagnostics in
`archive/sim_pretrain/pretrain_paper` are preserved. CLI commands below are unchanged.

The scoped joint controller adds `M @ (Kd * qdot_target)` to the original torque
(or `Kd * qdot_target` when mass compensation is disabled). Target velocity is the
backward difference of consecutive absolute IK joint goals at 100 Hz, held across
the five physics substeps. The first goal after reset has zero feedforward.
Velocities are uniformly scaled to the existing IK max joint speed (2 rad/s);
actuator torque limits remain unchanged. No acceleration feedforward, gain
increase, exploration speed change or acceptance-threshold relaxation is added.
This is an engineering controller extension, not a claim about the paper's code.
Set the flag to false and choose a separate output directory for an ablation;
never mix data from the two controller profiles.

Feedforward validation (2026-09-07): gains 100/300/1000 have free-space position
RMS 1.458/1.253/1.194 mm, respectively. None meets the unchanged 1 mm threshold.
The maximum position errors are 6.375/6.306/6.243 mm; orientation maxima are
0.365/0.676/0.827 degrees. Saturated-control-interval fractions are 3.50/5.75/8.50%.
For comparison, without feedforward gain 300/1000 RMS was 3.971/2.194 mm and
saturation was 0/2.75%. Thus lower overall lag does NOT imply better transient
behavior. The contact scan, official collection and training remain blocked.
Run from the Adaptive_Wiping root, using the existing `clean` conda environment:

```bash
conda activate clean
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain collect --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain train --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain evaluate --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain export --config configs/sim_pretrain/pretrain_paper.yaml
```

The CLI is implemented. Run sanity first: a nonzero exit blocks collection.
CPU training is the default; no GPU is required for this small VAE.
Dependencies: Python 3.10, the local robosuite 1.5.2 fork, MuJoCo, NumPy,
SciPy, PyYAML, h5py, PyTorch, matplotlib. No robot SDK or physical sensor is used.
The existing `clean` environment supplies robosuite's dependencies and PyTorch.
This is an overlay for that environment, not a replacement for robosuite setup:

```bash
python -m pip install -r requirements/sim_pretrain.txt
```

For a new environment, install the local robosuite fork's dependencies first and
install PyTorch >=2.6 (a CPU build is sufficient). The verified interpreter is
`/home/wp/miniconda3/envs/clean/bin/python`; use it in place of `python` when conda
activation is unavailable. No private-macros file, robosuite_models or Mink is
needed by this explicitly configured AIRBOT IK path; their startup warnings are
nonblocking. Do not replace this fork with an unrelated pip robosuite release.

## Model conventions

The pre-training environment replaces position actuators with torque motors in
its in-memory XML only. Existing robot and mouse demonstration files are unchanged.
Joint frictionloss is zero; torque limits are simulation limits, not hardware ratings.
The three sampled fields are sliding friction, direct constraint stiffness, and
solimp width. Stiffness is NOT a measured N/m material property. Damping is
`2*sqrt(stiffness)`. The remaining solimp parameters are `[0.2,0.9,width,0.5,2]`.
All collision patches receive the same parameters with priority over the table.

## Verification

The original 12 software tests passed (including a two-epoch temporary training/export
roundtrip and the actual tool's static force/torque check). Full collection must
not precede the physics gate. This is NOT yet a completed paper training run.
Three additional controller regression tests cover finite-difference velocity,
speed limiting, reset, invalid inputs, mass-scaled torque correction, substep hold,
and the disabled/original-controller path. All 15 tests passed after the extension.
Run the same test command below. The current `sanity` and `collect` commands both
exit with code 2 because the physical tracking threshold is not met, as intended.
`archive/sim_pretrain/pretrain_paper_velocity_ff/tracking_comparison.png` compares the recorded
before/after position-error curves for gains 300 and 1000.
Run software and static sensor checks with:

```bash
python -m unittest discover -s tests -t . -p 'test_pretraining.py' -v
```

The learning/collector tests use explicitly synthetic temporary fixtures and two
epochs only. They validate the software, not successful paper reproduction, and
cannot create official training artifacts in `archive/sim_pretrain/pretrain_paper`.
The original no-feedforward diagnostic found that all three prescribed gains failed the 1 mm
free-space tracking requirement. The implementation does not increase torque
limits, alter exploration speed, or relax the acceptance threshold automatically.
Observed free-space RMS: gain 300 = 3.971 mm, gain 1000 = 2.194 mm. Gain 100
failed initial settling (24.524 mm position error, 4.628 degrees orientation error).
Consequently the 27-parameter scan and full 1000/100/100 collection have not run;
there is no official 200-epoch checkpoint or encoder yet. Zero-reference-velocity
joint damping was a likely contributor to ramp-tracking lag. The authorized
feedforward extension and its new validation results are described above.
Even after tracking passes, the 27-contact scan, unloading checks and an independent
eccentric-load sensor check must pass. The sensor check uses a separate welded
clone of the model to compare central/eccentric external loads with the exact
expected local support wrench. It never alters exploration states or data.
Exploration checks contact parameters and non-tool collisions at every 2 ms
physics substep; each 10 ms saturation flag is the OR over its five substeps.

`archive/sim_pretrain/pretrain_paper_discoverse_inertia/sanity.json` records each gain, errors and provenance.
`sanity_traces/*.npz` contains full successful diagnostic rollouts (not training data).
Collection fixes parameter assignments first and stores invalid rows as NaN,
never as synthetic zero trajectories. Re-running collect resumes pending rows;
`collect --retry-failed` explicitly retries the same failed parameter assignments.
Configuration/source/environment changes require a new sanity run and cannot be
mixed into an existing dataset. Do not manually edit the gate report to bypass it.
Complete datasets also have `dataset_integrity.json`; training verifies the file
SHA256, and evaluation/export verify their parent artifacts.

Once validated, the stages produce `dataset.h5`, `collection_failures.json`,
`vae_last.pt` (final epoch), `vae_best.pt`, `history.json`, `preprocessing.json`,
`evaluation.json`, `training.png`, `reconstruction.png`, `test_embeddings.npz`,
and `encoder.pt`. Failed physics checks do not produce trained models.

Frozen encoder usage, after a successful export:

```python
from scripts.sim_pretrain.learning import FrozenSpongeEncoder
encoder = FrozenSpongeEncoder("runs/sim_training/pretrain_paper_discoverse_inertia/encoder.pt")
mu = encoder.encode(raw_ft)  # finite numpy array [B,400,6] -> [B,5]
```

The raw input is local sensor wrench, in N and N*m, without per-episode tare.
Filtering is offline zero-phase filtering of complete four-second sequences.
Validation and test use training-only statistics without clipping. Reconstruction
RMSE is against filtered physical signals, not the original unfiltered signal.
Validation uses the latent mean; stochastic training losses are not directly
comparable to deterministic validation reconstruction losses. A collapse warning
in evaluation is an experimental result, not permission to change beta silently.
