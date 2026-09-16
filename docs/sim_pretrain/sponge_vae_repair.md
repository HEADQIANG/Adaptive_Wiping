# Offline Encoder Repair

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This is an engineering variant, not the original paper model. It preserves the
existing encoder interface and frozen loader, changes the decoder to Linear(5,
2400), initializes using training-only channel/trajectory PCA and least squares,
and warms beta from 0 to 0.0001 over 50 epochs. Training samples the posterior;
validation and downstream inference use deterministic mu. No clipping, real-data
normalization fitting, sensor tare, robot command or calibration change occurs.

The old decoder confines all six-channel samples to one affine plane of dimension
at most five, with additional ReLU restrictions. The new decoder removes that
shared channel-plane restriction, not the five-dimensional trajectory bottleneck.
PCA initialization gives a working representation before stochastic optimization.
These combined changes are a repair candidate, not a controlled attribution of
all previous failures to any single cause. Old mu had demonstrated predictive
information despite low variance; it must not be described as entirely useless.

## Run

From the repository root, using the existing clean environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover \
  -s tests -p 'test_sponge_vae_repair.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain.experiments.repair_sponge_vae \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --output runs/sim_training/sponge_vae_repair_v1
```

Output must not already exist; failed/interrupted runs are retained. There is no
resume. Use a new directory for a rerun. Exit 0 means the simulation checks passed,
2 means a completed negative experiment; exceptions are recorded in status.json.
Do not edit experiment sources while it is running. All 1000/100/100 simulation
rows and original splits are retained, including contact acceptance failures.

Three seeds (42/43/44), 200 epochs each, batch32, Adam0.0001, CPU single-thread.
PCA initialization is deterministic and common to seeds; stochastic samples and
batch orders differ. Select minimum validation MSE from epochs50-200, lock all
selections before test scoring. Require MSE <=80% of training-template baseline,
and fixed/shuffled latent MSE >=110% of normal reconstruction MSE. Both validation
and test must pass for seed42 and at least two of three seeds. Export seed42 only,
never pick a seed by its test score. The historical test split is not newly blind.

Files: protocol/manifest/status/selection/report JSON, report.md, per-seed
initialization/history/best/last/validation/test artifacts; encoder.pt only on
success. Protected baseline checkpoints, dataset, real raw data, calibration and
prior experiment checkpoints are hashed before/after. The original code and model
files are not overwritten. Export is checked against FrozenSpongeEncoder.

Simulation success does not fix sensor-frame mismatch or real exploration being
outside simulation normalization bounds. One exploration shared across all eight
demos supplies only one embedding: changing the encoder cannot provide missing
cross-material training variation. Rebuild prepared data and retrain/evaluate XY
and feedback in a separate output after successful export; retain all OOD warnings
and hardware_ready=false. Do not deploy merely because reconstruction improved.

## Downstream Comparison

The new config references the same untouched real raw data, but the repaired
encoder and a separate prepared/training/output directory. Run only after the
encoder experiment has succeeded. Each stage must exit successfully before the
next stage. Existing output is not overwritten; use `--resume` only for interrupted
train/cross-validate runs with unchanged sources and inputs.

```bash
conda activate clean
python -m scripts.real_training prepare --config configs/real_training/real_training_airbot_native_repaired.yaml
python -m scripts.real_training train --config configs/real_training/real_training_airbot_native_repaired.yaml
python -m scripts.real_training evaluate --config configs/real_training/real_training_airbot_native_repaired.yaml
python -m scripts.real_training cross-validate --config configs/real_training/real_training_airbot_native_repaired.yaml
python -m scripts.real_training export --config configs/real_training/real_training_airbot_native_repaired.yaml
```

No deployment configuration is switched. `policy.pt` remains an offline artifact.
Check `final/evaluation.json` and `cross_validation/evaluation.json`; training
loss alone is not acceptance evidence. Full software regression (no robot):

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -p 'test_airbot_model_chain.py' -v
```

## Completed Encoder Results (2026-09-11)

`archive/sim_pretrain/sponge_vae_repair_v1` completed successfully. Do not rerun into it.
All three seeds passed validation and historical test reconstruction/latent-use
checks; selected epochs were 51, 51 and 53. Seed42 was exported as predeclared.

| Historical simulation test metric | Original | Repair seed42 |
|---|---:|---:|
| Normalized reconstruction MSE | 0.015055255 | 0.000416822 |
| Training-template baseline MSE | 0.006928034 | 0.006928034 |
| Active mu dimensions at variance >1e-4 | 0 | 5 |
| Fixed-code MSE | See original report | 0.006920033 |
| Mean shuffled-code MSE, 20 permutations | See original report | 0.012896406 |

Repair seed43/44 test MSE: 0.000416490 / 0.000418150. Seed42 improves MSE
97.23% relative to the original and 93.98% relative to the mean template.
The affine channel-plane lower bound was 0.000160064 on training data, much
less than the original error: that restriction alone does not explain the entire
failure. PCA-initialized full-trajectory reconstruction MSE was 0.000381495 before
training. This supports the combined initialization/decoder/loss repair, not a
claim that the KL schedule by itself solved the issue.

All original files in the manifest and experiment sources passed their hash
audit. Frozen raw-FT export predictions matched the trained encoder exactly.
The full regression ran 311 tests in 98.216s: 308 passed and three PyKDL tests
skipped in clean; those three passed separately under system python3. Five new
repair tests cover initialization, stochastic versus mean decoding, export gate
and frozen compatibility, invalid/rank-deficient input, overwrite protection and
epoch50 checkpoint selection.

The real exploration is still outside the unchanged simulation normalization
range on [100, 100, 100, 92.5, 100, 100]% of channel samples. Its repaired mu is
[-25.732, 69.086, 25.742, 28.286, -71.925], while simulation test latent variances
are approximately [1.006, 0.999, 0.632, 0.729, 0.852]. This is not repaired by
passing simulation reconstruction checks. The real-data comparison keeps the
entire input, its raw units, gravity/bias and all out-of-distribution warnings.

## Completed Downstream Results (2026-09-11)

`archive/real_training/real_training_airbot_native_repaired_v1` contains completed full-data
and eight leave-one-demo-out model pairs, each trained for 10000 XY / 2000 feedback
epochs. Input raw hash is unchanged:
`ebab153802001126466aa8b98b2606bcaaa2d69319048113171005b30c1aa0ff`.

| Metric, mm | Original train | Repair train | Original leave-one-out | Repair leave-one-out |
|---|---:|---:|---:|---:|
| XY RMSE | 15.522 | 15.709 | 17.777 | 18.418 |
| Height increment RMSE | 0.547 | 3.445 | 3.733 | 3.397 |
| Height increment MAE | 0.379 | 1.731 | 2.083 | 1.884 |

The leave-one-out zero-increment baseline has RMSE3.554mm / MAE1.711mm.
Repaired feedback improves RMSE about9.0% relative to the old model and4.4%
relative to zero, but its MAE still loses to zero. Visual inspection shows that
many height peaks are missed; the new feedback predictions remain near zero.
XY worsens about3.6% versus the original and loses to the train-mean-trajectory
baseline (leave-one-out RMSE17.737mm). This is **not an accepted overall policy
improvement**. Do not promote this policy or claim that all model issues are fixed.

With identical sponge embeddings across all eight demos, a deterministic
sponge-only XY network necessarily predicts one shared trajectory. No encoder
repair can distinguish those demonstrations without additional varying inputs.
The real embedding remains OOD; coordinate alignment and the runs/sim_training/model contract
must be resolved before claiming physical adaptation. No inference-time clipping,
undocumented recentering, new calibration flags or automatic policy replacement
has been applied. The old model and its outputs remain available.

`policy.pt` was exported strictly for offline comparison. Reload predictions
match exactly. Eight recorded-FT replays (1001 ticks and20 feedback predictions
each) match batch predictions with maximum error0.0m. Replay is not a closed-loop
or hardware test. Replay report: `archive/real_deploy/airbot_native_repaired_replay_v1/report.json`.

After a successful new export, the offline replay command is:

```bash
conda activate clean
python -m scripts.real_deploy replay \
  --training-config configs/real_training/real_training_airbot_native_repaired.yaml \
  --output runs/real_deploy/airbot_native_repaired_replay_v1
```

That replay output already exists for this completed run; use a fresh path for
any repeat. No training or test process remains running after completion.
