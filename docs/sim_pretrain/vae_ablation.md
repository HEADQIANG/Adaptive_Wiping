# Normal-mode VAE ablation

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This is a bounded engineering experiment, not a claim to reproduce the paper's
training configuration. The original 200-epoch model, dataset, collection code,
controller, physical parameters, filter and split assignments remain unchanged.
All contact-acceptance failures remain in the dataset. No impedance training or
hardware work is performed.

The subsequent authorized ReLU/LeakyReLU paired diagnostic is documented in
[decoder_activation_experiment.md](decoder_activation_experiment.md). Its separate
output preserves this experiment. LeakyReLU improved small-AE MSE but still failed
the unchanged threshold; no full-data AE or new VAE was trained in that follow-up.

## Run

From the repository root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.vae_ablation \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --output runs/sim_training/normal_vae_ablation_v1
```

The output must not exist. There is no resume or overwrite option. A failed or
interrupted experiment is retained; a separately authorized rerun needs a new
directory. Exit 0 means stable improvement (at least two of three seed-specific
test checks pass), 2 means the finite protocol completed without that result,
and other failures raise an exception. Do not modify experiment source during a
run: its hashes are fixed in the manifest. CPU execution uses one Torch thread.

## Stages And Criteria

1. Select 32 training rows with `default_rng(42).choice(..., replace=False)`.
   Deterministic AE, mu decoding, no KL or dropout, Adam 1e-3, full batch,
   at most 2000 steps. Stop on training MSE strictly below both 1e-3 and 10%
   of this subset's mean-template MSE. This is not a generalization check.
2. If passed, train a fresh full-data AE for 200 epochs, batch32. Select by
   validation MSE across all epochs. Require at least 20% improvement over the
   training-mean-template validation baseline.
3. If passed, train six fresh VAE variants for 200 epochs: fixed beta .06,
   fixed .001, warm .001, warm .01, warm .06, warm .001 without dropout.
   All use Adam 1e-3, batch32, seed42, latent5, original loss reductions;
   dropout is .1 except the last. Training always samples z. Warm beta is 0
   at epoch1, reaches the final value at epoch50 and stays constant. Select
   checkpoints by validation MSE only at epochs50-200; also keep epoch200.
4. Require validation MSE <=80% of the mean-template baseline and both fixed
   code MSE and the mean of 20 shuffled-code MSEs >=110% of normal MSE.
   Fixed code is the evaluated split's mean mu, used only for diagnostics.
   Shuffle seeds are 4200-4219. Variance and KL are auxiliary, not five-dimension
   activation requirements. Ties use the table order above.
5. Only if these stages fail, try one encoder fallback: per-frame 6-16-5,
   GELU after16, then flattened 2000-64-10, GELU after64. Decoder unchanged.
   Repeat the same checks and sweep. Stop if this also fails.
6. Lock the eligible minimum-validation-MSE configuration before testing.
   Repeat it from scratch for seeds43/44 with identical splits. Report each
   seed's best eligible epoch on test. Do not pick a seed by test performance.

The original baseline is reused, not continued or overwritten. Deterministic
AE checks are never reported as a repaired VAE. Physical RMSE uses the filtered
FT target, separately for Fx/Fy/Fz (N) and Tx/Ty/Tz (N m). Phases are frames
0:200 (press, .01-2 s), 200:300 (forward), 300:400 (reverse).

## Frozen Property Probes

Compare original mu, selected seed42 mu (if eligible), train-fitted PCA5 and
flattened normalized FT. Constant predictions are the training label means.
Feature and label standardizers fit only training rows. Tiny nonzero latent
variance is preserved. Ridge alphas are 1e-4, 1e-2, 1, 100. MLP is input-64-32-3
with ReLU, Adam1e-3, weight decay1e-4, batch32, max400 epochs, validation patience50.
Pick the predictor by average validation standardized MSE, not test results.
Supervised models are saved separately and never backpropagate into encoders.

Test reports include RMSE, MAE and R2 per property, with paired bootstrap1000
95% intervals for RMSE improvement against the training-mean constant baseline.
Report detected predictive information only if R2>0 and the improvement interval
is strictly above zero. This interval is conditional on the fitted predictor,
not an estimate of across-seed training uncertainty. `stiffness_direct` is a
MuJoCo contact parameter, not calibrated physical N/m. A failed probe is not a
proof of absolute absence of information. No labels enter the VAE training API.

## Artifacts And Checks

`protocol.json`, `manifest.json`, `original_hashes.json`, `source_hashes.json`
record settings, subset rows, train-only preprocessing checks, split uniqueness
and original/source/runs/sim_training/config hashes. `progress.json` and per-run `status.json`
record progress and failures. Each training group saves `history.json`, `best.pt`,
`last.pt`, training and beta plots; non-finite losses or gradients stop that group.
Baseline checkpoints are referenced, not copied or rewritten.

`selection.json` precedes all test scoring. Selected seed runs contain
`validation.json`, `test.json`, `encoder.pt`, latent diagnostics and six-channel
reconstruction plots for fixed test rows 0,10,...,90. No failed candidate is
exported as a successful new representation. `probes/` contains independent
supervised predictor artifacts, validation selection and property scatter plots.
`report.json`/`report.md` summarize the finite experiment and original-file audit.

`scripts/sim_pretrain/experiments/vae_property_probe.py` is called by the main entry, not a separate
training CLI. `probes/<input>/selected_predictor.pt` contains its input/label
standardizers and predictor type. `FrozenPropertyPredictor(path).predict(features)`
expects the recorded feature representation (mu, PCA5 or normalized flattened
FT), not raw sensor trajectories. `probes/pca5.npz` preserves training-fitted PCA
mean/components. Per-input reports retain all four ridge fits and the MLP result;
test metrics never change which predictor is selected. `property_scatter.png`
plots only validation-selected predictors. No scikit-learn dependency is needed.

The independent loader does not replace `FrozenSpongeEncoder`:

```python
import sys
sys.path.insert(0, "scripts")  # repository root; avoids the ROS package named scripts
from vae_ablation import FrozenAblationEncoder

encoder = FrozenAblationEncoder("runs/sim_training/normal_vae_ablation_v1/<selected-run>/encoder.pt")
mu = encoder.encode(raw_ft)  # raw local FT [B,400,6] -> [B,5], no gradients
```

Use `from vae_property_probe import FrozenPropertyPredictor` with the same
script-directory setup for supervised predictor loading. Do not import through
`scripts.*` on this machine: ROS also installs a package with that name.

Run software checks (synthetic fixtures are not research data):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_vae_ablation.py' -v
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

The focused test module checks exact original model/loss equivalence, warmup
boundaries, AE versus sampled VAE paths, held-out isolation, both architecture
exports, frozen gradients, finite-loss failure handling, selection epochs and
tie order, overwrite protection, original runs/sim_training/model/source hashes, PCA/ridge,
and synthetic informative/collapsed reconstruction and property diagnostics.
Test-only short runs do not expose a production CLI shortcut around the protocol.
An end-to-end synthetic probe test also checks all four representations, five
predictor candidates, plots, reloads and that saved selections contain no test
scores. Truly constant features use unit scale, without discarding tiny but
nonzero latent variance. Interrupted runs retain their failure status.

## Completed Run: 2026-09-09

`archive/sim_pretrain/normal_vae_ablation_v1` has completed with exit2 and
`outcome=bounded_experiments_failed`. This is a completed negative experiment,
not an execution crash. Do not rerun the command into this existing directory.

| Small AE, seed42 | Steps | Training MSE | Subset template MSE | Required strict MSE | Pass |
|---|---:|---:|---:|---:|---|
| Original linear encoder | 2000 | 0.008913830854 | 0.006650017574 | <0.000665001757 | No |
| Nonlinear GELU encoder | 2000 | 0.008819349110 | 0.006650017574 | <0.000665001757 | No |

Both deterministic checks failed with zero KL weight and zero dropout. By the
predeclared bounds, neither full-data AE nor the six VAE configurations nor
seeds43/44 were started. Those branches are implemented and software-tested but
have no real-data results from this run. No improved encoder is exported, and
there is no claim of a stable VAE improvement. This does not establish that the
data lack information or that five latent dimensions are intrinsically inadequate.

The independent property comparison completed for the available original mu,
PCA5 and full FT inputs; new mu is explicitly unavailable. Validation selected
MLP for each input. Best/stop epochs were 139/189, 139/189 and 68/118 respectively.

| Frozen input | Friction test R2 | stiffness_direct test R2 | Width test R2 |
|---|---:|---:|---:|
| Original mu + MLP | 0.775473 | 0.729512 | 0.266425 |
| Train-fitted PCA5 + MLP | 0.946052 | 0.996886 | 0.335419 |
| Full FT + MLP | 0.974327 | 0.994737 | 0.520139 |

All nine selected-predictor/property checks have positive R2 and a positive
lower paired-bootstrap RMSE-improvement bound. Width is substantially weaker:
the original-mu MLP often predicts near the training mean for larger widths.
These are in-distribution simulation readouts with fixed splits, not accurate
real-world material calibration. In particular, low latent variance and a decoder
that ignores mu do not imply that deterministic mu contains no predictive signal.
Predictor standardization uses only training statistics and preserves its small
nonzero variance; the VAE weights were never updated by property supervision.

Original baseline test MSE is 0.015055255 (minor float32 batching difference from
the old evaluation); training-template MSE is 0.0069280341. It still fails the
reconstruction and latent-use checks. Phase MSE is 0.0129313 / 0.0129052 /
0.0214533 for press / forward / reverse. Six-channel physical errors and all
per-property RMSE/MAE/R2/bootstrap intervals are retained in the reports.

Inspected the generated training curves, baseline reconstruction and latent plot,
and the nine-panel property scatter. All ten fixed-index baseline reconstructions
exist. Both AE histories contain 2000 records, best and last are step2000, and
checkpoints reload successfully. All 40 protected original files and both
experiment source hashes match their pre-run snapshots. The complete regression
suite passed 65 tests in 51.470 seconds, including the 16 new experiment tests.
No training, simulation or test process from this task remains running.

Read the completed results or open the saved figures (no new training):

```bash
less runs/sim_training/normal_vae_ablation_v1/report.md
less runs/sim_training/normal_vae_ablation_v1/probes/report.md
xdg-open runs/sim_training/normal_vae_ablation_v1/probes/property_scatter.png
xdg-open runs/sim_training/normal_vae_ablation_v1/baseline/reconstructions/test_000.png
```

Do not interpret the failed small AEs as successful new VAE models or automatically
extend the bounded protocol. Further decoder/optimization diagnostics require a
separately chosen experiment, with these negative results retained.
