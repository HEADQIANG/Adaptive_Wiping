# Two-Controller Data Collection And Pretraining

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## New Friction-Range Run

For the user-requested fresh normal collection with mu=[0,1.2] and the original
VAE training profile, see [normal_mu1p2_pretraining.md](normal_mu1p2_pretraining.md).
This independent run preserves all original two-mode data and checkpoints below.

## Training Signal Plots

For raw, filtered and normalized six-axis normal-mode training FT, see
[training_ft_visualization.md](training_ft_visualization.md). The independent
plotter writes to `archive/sim_pretrain/normal_ft_visualization_v1`, preserves frozen outputs,
and does not run a model or read validation/test trajectories for plotting.

## Subsequent Normal-Mode Ablation

The isolated reconstruction/collapse experiment is documented in
[vae_ablation.md](vae_ablation.md). Its 2026-09-09 run completed with both bounded
small-AE checks failing; no new VAE or impedance model was trained. Frozen
property probes did detect predictive information in the original encoding,
PCA5 and full FT. This does not repair the original decoder or validate contact
motion. Original collection and 200-epoch baseline artifacts remain unchanged.

## Current Authorization

On 2026-09-09 the user requested continuing with normal-mode training first.
The normal 200-epoch run, evaluation and encoder export are now complete; see
Normal-Mode Training below. Impedance training has not been started.

The user explicitly requested skipping contact acceptance and collecting data
on 2026-09-08. This is a separate AIRBOT simulation research dataset, not a
passed production wiping dataset. Production collection and its gate are
unchanged. `--record-only-contact-gate` is required to acknowledge this policy.
Motion/contact acceptance is saved per trajectory but does not reject it.
Finite values, 400 samples at 100 Hz, effective contact parameters, unexpected
collisions and successful unloading are still checked. Broken trajectories are
not padded or replaced with new random parameters.

## Configuration

Both modes use the same immutable train/validation/test assignments, with
1000/100/100 trajectories per mode. The 1000 training samples, 4 s exploration,
100 Hz six-axis local FT, press .01 m/s for 2 s and lateral +/-.05 m/s for 1 s
each follow paper sections III-A, IV-C1 and IV-D. Parameter ranges remain
mu=[0,3.5], direct solref k=[.5,1000], solimp width=[.02,.3].

Normal is the existing IK + joint PD + goal velocity feedforward at gain300.
Impedance is the already-tested `cart_6000` controller, with XY K=6000 N/m,
Z K=150 N/m and orientation K=120 Nm/rad. Preparation remains production IK;
exploration/unloading use the selected controller. Robot, sponge, inertia,
contact mapping, target trajectory and torque limits are otherwise unchanged.

The paper uses UR5e; AIRBOT and this controller comparison are engineering
variations. Extra validation/test samples, uniform sampling, batch32, seed42
and second-order 10 Hz zero-phase filtering are documented engineering choices.
The VAE pretraining profile remains latent5, beta=.06, Adam1e-4, 200 epochs.
It learns a representation, not automatically calibrated physical properties.

## Collect

Run from the project root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -p test_comparison_collection.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.collect_control_comparison --output runs/sim_training/two_control_pretraining_v2 --record-only-contact-gate --workers 2
```

This opens no viewer and requires no graphics backend. Two worker processes
collect one mode each, with one writer per HDF5 file. Re-run the same command
to resume unfinished assignments; completed samples are not collected again.
`--retry-failed` retries the exact failed assignments and preserves failure history.
`--max-trajectories 1` collects one new sample per mode as a resumable pilot.
`--mode normal` or `--mode impedance` selects just one mode.
Each `collection.json` is updated after every attempted trajectory and can be
read for progress without opening a live HDF5 writer. Do not launch duplicate
writers on the same mode; a file lock enforces single-writer access.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.collect_control_comparison --output runs/sim_training/two_control_pretraining_v2 --record-only-contact-gate --verify-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.summarize_control_data --output runs/sim_training/two_control_pretraining_v2
```

Exit2 means the selected collection is incomplete, including an intentionally
limited pilot. Contact acceptance failures alone do not produce this exit code.
Source/configuration changes reject resuming into the existing dataset.

### Interrupted HDF5 Recovery

An abrupt process termination can leave a variable-length HDF5 string with an
invalid global-heap reference, even if all committed trajectories remain readable.
Do not retry repeatedly into such a file. Stop collection and confirm its writer
has exited first. The separate recovery script requires the same exclusive lock,
copies the original file into a unique `recovery-*/original.h5` backup and rebuilds
the HDF5 storage. It verifies every numeric field and committed trajectory before
atomic replacement. Only unreadable strings in uncommitted rows can be cleared;
their locations/errors are retained in `recovery.json`. Corruption in a committed
record aborts recovery. Parameters, statuses, attempt counts, readable strings and
failure history remain unchanged. No controller or collection source is changed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -p test_collection_recovery.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.recover_control_dataset --output runs/sim_training/two_control_pretraining_v2 --mode impedance --record-only-contact-gate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.collect_control_comparison --output runs/sim_training/two_control_pretraining_v2 --record-only-contact-gate --verify-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.collect_control_comparison --output runs/sim_training/two_control_pretraining_v2 --record-only-contact-gate --workers 2 --retry-failed
```

Use `--retry-failed` only after inspecting the failure history. It retries the
same assignment and retains prior errors. Verification before completion exits2
by design; require `verified: true` and `source_unchanged: true` in its report.

On 2026-09-09, impedance train index232 had this interrupted-string error.
The recovery backup is `impedance/recovery-qegspxh8/original.h5`, with its hash
and full audit in the adjacent `recovery.json`. All 241 committed impedance
trajectories and all numeric fields were preserved. The same assignment was
successfully retried; its storage failure remains in `failure_history.jsonl`.
The complete regression suite passed all 49 tests, including three recovery tests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

`two_control_pretraining_v1` is the first one-trajectory-per-mode pilot, retained
with a collector source snapshot. v2 vectorizes repeated contact-parameter checks
at the same physics substeps. It does not alter dynamics or acceptance rules;
the regression test compares all recorded fields exactly against both v1 pilots.

## Outputs

`manifest.json` contains controller settings, paper/source hashes and all paired
parameter assignments. Each `normal/` and `impedance/` directory contains:

- `config.json`: a JSON configuration also readable by the existing YAML loader.
- `dataset.h5`: raw local FT, actual/target poses, joints, timestamps, saturation,
  world contact wrench, normal loads, requested/applied torque and nominal twist.
- `motion_passed` and `acceptance_json` in each split: unchanged motion gate results.
- `valid`, `status`, `unloaded`, `error`, `attempts`: collection integrity, NOT
  successful wiping acceptance. Status0=pending, 1=interrupted/in-progress,
  2=recorded, -1=failed.
- `collection.json`, `dataset_integrity.json` on completion and optional
  `failure_history.jsonl`. No failures are silently discarded.

`collection_verification.json` independently re-scores saved trajectories,
checks paired assignments, integrity hashes, metadata and unchanged sources.
After both writers finish, `summarize_control_data.py` writes `data_summary.json`,
`raw_ft_distribution.png` and `paired_sample.png`. It checks disjoint split
assignments, matched nominal targets/clocks and reports the full motion-failure
distribution without discarding any trajectory. It does not train a network.
The comparison tests also check rejection of incomplete/corrupt summary inputs,
paired summary counts and preservation of failed-motion statistics.
An actual-simulation regression also compares environment reuse across two
parameter assignments with a fresh environment in both control modes, guarding
against controller history leaking into subsequent collected trajectories.

No training is automatically launched by the collection command. Training uses
each mode's own config, training-set-only preprocessing and separate checkpoints.
The latest request prioritizes normal-mode training, documented below.

## Completed Collection: 2026-09-09

`archive/sim_pretrain/two_control_pretraining_v2` is complete. Both writers exited and released
their locks. At collection completion training had not yet started; the later
normal-mode training result is documented below.

| Mode | Train | Validation | Test | Total | Motion-pass labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal IK+FF gain300 | 1000 | 100 | 100 | 1200 | 0 |
| Impedance cart_6000 | 1000 | 100 | 100 | 1200 | 248 |

All 2400 trajectories contain finite `(400, 6)` raw local FT and completed
unloading. Current failed/pending counts are zero. Normal accumulated 1202
attempts and impedance 1203, retaining interrupted attempts and the recovered
storage failure without substituting parameters. The 241 committed impedance
records in the recovery backup were compared against the final dataset: every
field is unchanged, and the backup SHA256 still matches its recovery audit.

`collection_verification.json` reports `verified`, `complete` and
`source_unchanged` all true. `data_summary.json` verifies 1200 unique parameter
assignments across the three splits, paired between controllers. Paired target
position differences are exactly zero in every split, and timestamps match.
Both final dataset hashes match their integrity files and both mode configs
load successfully through the existing `learning.load_data()` API.

The generated `raw_ft_distribution.png` and `paired_sample.png` were visually
checked. Raw sensor signals retain tool weight; unloading does not imply a zero
raw FT signal. The normal mode still exhibits insufficient travel and orientation
errors; the impedance mode still loses contact. Its motion-pass label counts are
212/15/21 for train/validation/test. These labels do not filter the collection.
Complete data is not evidence of passed wiping acceptance or a trained material
estimator. The full regression suite passed 49 tests.

Collection itself produces no checkpoints or calibrated sponge-property estimates.

## Normal-Mode Training: 2026-09-09

Run from the project root using the existing per-mode configuration. The normal
run below is already complete: do not run `train` again in the same directory.
The trainer rejects an existing `vae_last.pt`; do not delete it to bypass this
guard. `evaluate` and `export` can be repeated for the saved model.

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock --nonblock runs/sim_training/two_control_pretraining_v2/normal/.training.lock python -m scripts.sim_pretrain train --config runs/sim_training/two_control_pretraining_v2/normal/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain evaluate --config runs/sim_training/two_control_pretraining_v2/normal/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain export --config runs/sim_training/two_control_pretraining_v2/normal/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -p test_pretraining.py -v
```

The run retains the frozen configuration: 200 epochs, five-dimensional latent,
beta=.06, Adam1e-4, batch32, seed42, CPU/one thread, second-order 10 Hz filtering.
Preprocessing is fitted on the 1000 training trajectories only. All trajectories
remain included regardless of their contact-motion label. No data, network,
controller, physics, hyperparameter or collection provenance source was changed.

Artifacts in `archive/sim_pretrain/two_control_pretraining_v2/normal`:

- `vae_best.pt`, `vae_last.pt`: best validation and final checkpoint, both epoch200.
- `encoder.pt`: frozen encoder, preprocessing and evaluation metadata.
- `history.json`, `preprocessing.json`: all 200 epochs and train-only statistics.
- `evaluation.json`, `test_embeddings.npz`: 100-test-trajectory diagnostics and
  posterior means/log variances; means have shape `(100, 5)`.
- `training.png`, `reconstruction.png`: inspected learning curves and the first
  test trajectory reconstruction in physical units.

| Test Diagnostic | Value |
| --- | ---: |
| VAE normalized reconstruction MSE | 0.0150552569 |
| Training-mean-trajectory baseline MSE | 0.0069280341 |
| Fixed-latent MSE | 0.0150552252 |
| Shuffled-latent MSE | 0.0150552122 |
| KL | 0.0005790935 |
| Active latent dimensions at variance >1e-4 | 0/5 |

`beats_mean_baseline=false` and `collapse_warning=true`. The reconstruction is
worse than the training-mean baseline and barely changes when latent codes are
fixed or shuffled. This run is a saved normal-control baseline, not a validated
sponge-property estimator. Encoder export is an artifact operation, not a quality
approval. Small latent variance does not alone prove that absolutely no property
information exists; no held-out property readout has been evaluated yet.

All 200 history entries and losses were checked, best-epoch selection matches
the history, checkpoint configuration/runs/sim_training/source hashes match, and recomputing
preprocessing from training data reproduces the saved statistics. The exported
encoder is frozen and repeatable on the 100 test trajectories; it agrees with
saved test means within float32 precision (maximum observed difference <1e-8).
The 18 pretraining regression tests passed. No training process remains running.

Impedance is still data-only. Further diagnosis should preserve this baseline
and separately test data informativeness and the reconstruction/KL balance;
the current diagnostics do not isolate a single cause. Do not silently change
the frozen beta, architecture or dataset, or overwrite this run to improve metrics.
