# Normal-Mode Pretraining With Friction 0-1.2

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## Authorized Extension

The subsequent request to continue for another200 epochs is handled separately
by [continue_pretraining.md](continue_pretraining.md). This original200-epoch
dataset/model directory remains immutable; the new target is epoch400.

## Completed Run: 2026-09-09

`archive/sim_pretrain/normal_mu1p2_pretraining_v1` completed fresh normal-mode collection,
200-epoch training, evaluation and frozen encoder export. The completed collection,
training, evaluation, export and plotting stages exited0; the intentionally
incomplete10-row serial pilot exited2. A post-export
bitwise assertion initially failed on2/500 embedding elements by9.313226e-10;
the float32-tolerant check below passed without any model/source change.

Collection contains1000/100/100 train/validation/test trajectories. All1200 are
finite, unique, successfully unloaded and different from their old FT records;
each assignment was attempted once, with no data failures or pending rows.
Actual mu extrema are .0014796750143 and1.1989256765691. Every saved stiffness
and width assignment exactly matches the previous manifest. No exact FT or
parameter duplicates occur across splits. The old dataset/model/config hashes
and original source provenance remain unchanged. The serial10-row pilot and
six-worker replay matched every recorded field and unloaded wrench exactly.

Motion acceptance passed0/1000 training,1/100 validation and0/100 test rows.
The other1199 rows remain in their assigned splits under the authorized
record-only policy. Data integrity is not contact-motion acceptance.

The best-validation-loss and last checkpoints are both epoch200. Saved
preprocessing exactly matches a fresh fit on this run's training split only,
and differs from the previous dataset's scaler. Losses and parameters are finite.

| Normalized MSE | Validation | Test |
| --- | ---: | ---: |
| Epoch200 VAE (mu decode) | 0.0168675184 | 0.0161118694 |
| Training-mean-trajectory baseline | 0.0082746344 | 0.0080603920 |
| VAE / baseline | 2.03846 | 1.99889 |

Test fixed-latent MSE is0.0161119215; original single-shuffle MSE is
0.0161118507. An additional20 shuffles with dedicated seeds42-61 increase
test MSE by only0.00000231% on average, and fixed encoding by0.000324%.
Neither validation nor test satisfies20% reconstruction improvement or10%
latent-utilization improvement. Test KL is0.0005656302; mu variances are
5.71e-6 to1.26e-5. The original evaluator reports `collapse_warning=true`.
This is evidence of negligible decoder use of sample-specific mu, not proof
that the encoding contains no property information. No new property probes
or further hyperparameter selection were performed.

| Test Physical RMSE | VAE | Mean-Trajectory Baseline |
| --- | ---: | ---: |
| Fx (N) | 0.591052 | 0.338039 |
| Fy (N) | 0.059019 | 0.035642 |
| Fz (N) | 0.419333 | 0.222062 |
| Tx (N m) | 0.007276 | 0.006155 |
| Ty (N m) | 0.022137 | 0.019938 |
| Tz (N m) | 0.012277 | 0.007169 |

Test press/forward/reverse MSE is .012418638/.014572668/.025037535;
the corresponding baseline is .003207685/.009756142/.016070059.
Stage-resolved physical RMSE and validation metrics are saved in
`training_verification.json`. These are diagnostics, not paper-reported scores.
Shrinking the friction range did not fix the reconstruction/collapse problem.
Normalized scores across old/new datasets use different scalers and should
not be interpreted as a direct quality improvement or regression.

Artifacts in the run root: `range_change_audit.json` (completed),
`collection_verification.json` (normal-only), `normal_data_summary.json`,
`parallel_pilot_verification.json`, `parallel_collection.json` and
`training_verification.json`. Models, history, preprocessing, evaluation,
test embeddings and training/reconstruction plots are under `normal/`.
The six training FT plots and their numerical arrays are separately under
`archive/sim_pretrain/normal_mu1p2_ft_visualization_v1`; no old subset manifest is reused.
Training curves, test reconstruction and training FT overview were inspected.
The reconstruction still visibly differs from the filtered target.

- Dataset SHA256: `d8e49d88ed7e755eef324a346dfdea0be99358e5e3c7baa9f20491da979d6b82`.
- Epoch200 SHA256: `b27293567a210fbeacf2630ca1bb2f14d34f73c05de96753877009aca459e75c`.
- Parallel wrapper SHA256: `75e84398c888e17ed1e22f3b5bf05daaa98f00c09ea27eae3fd6070db904eb7d`.

Three new parallel-collector tests and the full92-test regression suite passed
(full run109.682 seconds). No production/control/learning source was edited
after these tests. All collection, training, evaluation and plotting sessions
have exited. No impedance training, architecture change or automatic extension
beyond200 epochs was started. Do not overwrite this completed output directory.

## Scope And Configuration

The 2026-09-09 request authorizes fresh collection and training with the friction
randomization changed from[0,3.5] to[0,1.2]. This run follows the current normal
AIRBOT Play mode (IK + goal-velocity feedforward, gain300). It does not collect
or train impedance mode and does not overwrite the previous runs/sim_pretrain/model.

`configs/sim_pretrain/pretrain_paper_mu1p2.yaml` differs from `configs/sim_pretrain/pretrain_paper.yaml`
only in friction bounds and output location. Stiffness_direct=[.5,1000],
width=[.02,.3], seed42, all control/physics parameters and the exploration are
unchanged. Sampling uses the same seeds/order, so stiffness/width assignments
are exactly the same as the previous dataset; friction draws use the new range.
All FT trajectories are newly simulated, not rescaled old sensor signals.

Paper III-A, IV-C1 and IV-D (PDF pages3-5) were checked using text and rendered
pages. The1000 training trajectories,400x6 FT samples,4 seconds at100 Hz,
press .01 m/s for2 s and lateral .05 m/s for1 s per direction, latent5,
decoder ReLU/dropout.1, beta.06, Adam1e-4 and200 epochs follow the existing
paper-derived profile. Validation/test sets of100 each, batch32, seed42,
uniform independent sampling, train-only min/max normalization and the exact
second-order10 Hz filter are engineering choices. AIRBOT replaces the paper's
UR5e. Friction[0,1.2] is the user-requested deviation, not the original paper range.
MuJoCo stiffness_direct is not calibrated N/m and width is not geometric width.

The original collection/learning/model/preprocessing source is unchanged.
No LeakyReLU, GELU, beta warmup, extra epochs, supervised property labels or
previous checkpoint weights are used. A smaller friction range does not by
itself guarantee improved reconstruction, latent utilization or contact motion.

Contact acceptance remains record-only as previously authorized. Keep all
complete trajectories even if motion acceptance fails; still require finite
400-frame traces, effective parameter checks, no unexpected collisions and
successful unloading. Never substitute easier random parameters after failure.

## Collect

Run from the repository root. The first run must use a new output directory;
subsequent identical collection commands resume pending assignments only.

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.collect_control_comparison \
  --config configs/sim_pretrain/pretrain_paper_mu1p2.yaml \
  --output runs/sim_pretrain/normal_mu1p2_pretraining_v1 \
  --mode normal --record-only-contact-gate
```

An optional `--max-trajectories 10` makes a resumable pilot and exits2 while
incomplete. Inspect `normal/collection.json` for progress without opening the
live HDF5 writer. Do not run duplicate writers. `--retry-failed` retries the
same failed assignments, and should be used only after reviewing failure history.
The reused two-mode initializer creates an `impedance/config.json` placeholder;
this is NOT collected impedance data and must not be reported as completed.

### Parallel Collection

The initial10 serial trajectories took29.924 seconds. The independent
`collect_normal_parallel.py` can precompute rollouts in separate spawned
processes while feeding the UNCHANGED `collect_mode` single HDF5 writer in
manifest order. Workers never open/write the dataset. Each worker reuses the
original CollectionWipe with its standard reset and unloading. Bounded prefetch
uses at most twice the worker count of queued results; the writer closes the
HDF5 file after every50 attempted records. A short limited run may compute a
few prefetched trajectories that are not committed; these do not enter the data.

Before continuing collection, replay the committed serial pilot and require
exact equality of every recorded field and unloaded wrench:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.collect_normal_parallel \
  --config configs/sim_pretrain/pretrain_paper_mu1p2.yaml \
  --output runs/sim_pretrain/normal_mu1p2_pretraining_v1 \
  --workers 6 --record-only-contact-gate --verify-pilot 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.collect_normal_parallel \
  --config configs/sim_pretrain/pretrain_paper_mu1p2.yaml \
  --output runs/sim_pretrain/normal_mu1p2_pretraining_v1 \
  --workers 6 --record-only-contact-gate
```

Identical commands resume without rewriting committed rows. The parallel
wrapper does not retry failures or substitute assignments; it stops after a
batch with a data failure for inspection. The original serial retry command
remains available. `parallel_pilot_verification.json` records replay evidence;
`parallel_collection.json` records wrapper source hash and invocations, separate
from original production provenance. No original collection source is edited.
Its tests cover bounded ordered prefetch, resume gaps, failure propagation and
normal-only scope:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_normal_parallel.py' -v
```

## Verify Normal Only

The original CLI `--verify-only` expects both modes. For this normal-only run,
use its unchanged verification function with only the normal configuration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from collect_control_comparison import verify
from scripts.shared.common import write_json

out = Path("runs/sim_pretrain/normal_mu1p2_pretraining_v1")
manifest = json.loads((out / "manifest.json").read_text())
cfg = json.loads((out / "normal/config.json").read_text())
result = verify(out, manifest, {"normal": cfg})
result["verification_scope"] = "normal only; impedance not collected"
result["paired_parameter_assignments"] = None
write_json(out / "collection_verification.json", result)
print(json.dumps(result, indent=2))
assert result["verified"] and result["complete"] and result["source_unchanged"]
PY
```

Do not use the paired `summarize_control_data.py` CLI on this single-mode run.
Its `split_statistics` function can summarize each completed normal HDF5 split.
The range-change audit should verify only the two intended config differences,
identical stiffness/width assignments,1200 unique assignments across splits,
unchanged frozen sources and original runs/sim_pretrain/model hashes before training.

## Train, Evaluate And Export

Only proceed after all1000/100/100 trajectories are complete and verified.
The new training-set preprocessing is fitted on this run's1000 training rows;
do not reuse the old dataset's normalization. The trainer initializes a fresh
original VAE and refuses to retrain where `vae_last.pt` already exists.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock --nonblock runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/.training.lock \
python -m scripts.sim_pretrain train \
  --config runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain evaluate \
  --config runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain export \
  --config runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json
```

The original trainer saves best-validation-loss and epoch200 checkpoints;
its evaluate/export stages use epoch200 (`vae_last.pt`). Evaluation includes
test normalized MSE, training-mean-template baseline, fixed/shuffled-latent
diagnostics, physical channel RMSE, training curves and one test reconstruction.
Encoder export preserves the baseline interface and does not imply quality
acceptance. Compare against each run's own mean-template baseline; normalized
MSE values from different data ranges/scalers are not directly interchangeable.

Verify the frozen interface and saved embeddings without rewriting artifacts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.sim_pretrain.learning import FrozenSpongeEncoder

torch.set_num_threads(1)
folder = Path("runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal")
encoder = FrozenSpongeEncoder(folder / "encoder.pt")
with h5py.File(folder / "dataset.h5", "r") as h5:
    raw = h5["test/ft"][:]
mu = encoder.encode(raw)
assert mu.shape == (100, 5) and np.isfinite(mu).all()
assert all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
np.testing.assert_array_equal(mu, encoder.encode(raw))
with np.load(folder / "test_embeddings.npz") as saved:
    np.testing.assert_allclose(mu, saved["mu"], rtol=1e-6, atol=1e-8)
    print("export max abs difference:", np.abs(mu - saved["mu"]).max())
PY
```

## View Training Signals And Test

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.plot_training_ft \
  --dataset-config runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json \
  --indices 0 83 --output runs/sim_pretrain/normal_mu1p2_ft_visualization_v1
xdg-open runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/training.png
xdg-open runs/sim_pretrain/normal_mu1p2_pretraining_v1/normal/reconstruction.png
xdg-open runs/sim_pretrain/normal_mu1p2_ft_visualization_v1/training_overview.png
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -q
```

The visualization output must be new. No old VAE ablation subset is supplied,
because its manifest belongs to another dataset. No control or architecture
tuning should be inferred from this dataset-range comparison.
