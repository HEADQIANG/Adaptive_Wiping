# Continue Original VAE Training

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## Completed 200 To 400 Epochs: 2026-09-09

`archive/sim_pretrain/normal_mu1p2_continue400_v1/status.json` is `completed`. The original
200 epochs were replayed with all metrics and every final weight exactly
equal to the source. Adam was NOT reset: recovered step counts are6400 and
final counts12800 for all parameters (32 batches/epoch). All400 history rows
are finite, and rows1-200 equal the original history. Best added and final
checkpoints are both epoch400. Data, source200-epoch artifacts, config,
normalization and original learning/control sources are unchanged.

| Metric | Epoch200 | Epoch400 | Reduction |
| --- | ---: | ---: | ---: |
| Validation MSE | 0.0168675184 | 0.0094989669 | 43.68% |
| Test MSE | 0.0161118712 | 0.0091815591 | 43.01% |

The unchanged mean-template test baseline is0.0080603920, so the final VAE
still has13.91% higher MSE. Fixed mu gives0.0091817975, and20 shuffled-mu
experiments average0.0091818023: relative increases are only0.002597% and
0.002648%. Test KL is3.5350e-5, mu variances1.679e-6 to6.719e-6. Both
validation/test fail reconstruction and latent-utilization quality checks.
Continued training improved reconstruction but did not resolve collapse;
it does not establish absence of property information in the encoder.

| Test Physical RMSE | Epoch200 | Epoch400 |
| --- | ---: | ---: |
| Fx (N) | 0.591052 | 0.364746 |
| Fy (N) | 0.059019 | 0.043318 |
| Fz (N) | 0.419333 | 0.248379 |
| Tx (N m) | 0.007276 | 0.006361 |
| Ty (N m) | 0.022137 | 0.020622 |
| Tz (N m) | 0.012277 | 0.007839 |

Epoch400 test press/forward/reverse MSE is .004268540/.010630360/.017558802.
The corresponding baseline remains .003207685/.009756142/.016070059.
Training curves, the200/400 reconstruction comparison and latent diagnostic
plots have been inspected. There are14 saved PNGs, including10 fixed-index
test reconstructions. Numerical diagnostics and model files are retained;
no further training is automatically authorized.

The first process reached epoch400 and wrote all training/evaluation artifacts,
then exited1 on the export check: batch100 versus batch32 differed by at most
1.937151e-7. Verification was corrected to compare matching batch sizes and
also verify exact weights/scaler. Both same-batch comparisons differ by at
most3.725290e-9 and pass rtol1e-6/atol1e-8. `--verify-export-only` then exited0,
without retraining, changing model files or regenerating plots. The original
failure and the verification-only source revision are recorded in
`postprocessing_recovery.json`; original input/source hashes remain unchanged.

- Epoch400 checkpoint SHA256: `7e240d40b10781929b5ffa94b4a8a0b666abe1ff444644af71f57d58b3cc2d57`.
- Encoder SHA256: `9f2464c006aa5734979ede12b0bc71df344dad65cc22ab59c6502d351482c9e3`.
- Training script at execution: `2a50f90fe4399b1ba81ae729d5db946679c992d029ae8a004df9c24f2161051a`.

The nine focused tests passed, including uninterrupted versus resumed training,
export roundtrip, verification-only recovery and no weight/history writes.
The final full101-test regression suite passed in63.245 seconds. All training,
export verification and test sessions have exited; the requested extension
ends at400 epochs and the original200-epoch result remains preserved.

The user authorized another200 epochs on the new mu=[0,1.2] normal dataset.
This is a400-epoch engineering extension, not the paper's200-epoch result.
No recollection, filtering of contact failures, architecture change, learning
rate/beta/dropout change, new normalization or impedance training is performed.

## Recover And Continue

The original checkpoint contains weights but no optimizer or random state.
`scripts/sim_pretrain/experiments/continue_pretraining.py` first replays its original training on CPU
with the original seeds, loader ordering, stochastic VAE and Adam settings.
EVERY epoch's metrics and every final weight must match the saved run exactly.
Any mismatch aborts before additional training; there is no optimizer-reset
fallback. The recovered Adam and Python/NumPy/Torch random states are saved
before continuing from epoch201 through400 without reseeding.

Run from the project root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.continue_pretraining \
  --dataset-config archive/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json \
  --additional-epochs 200 --output runs/sim_training/normal_mu1p2_continue400_v1
```

The output must be new and cannot overlap either source dataset or checkpoint
directory. Original data, config, sources, models and diagnostics are hashed
and never overwritten. The original config retains epochs200; the separate
`run_config.json` explicitly records source_epoch200, additional_epochs200 and
target_epoch400. `status.json` records replay, training, evaluation and failure
states. This script does not silently resume a partially written directory.
Non-finite losses, gradients or updated parameters stop the run.

## Artifacts And Evaluation

- `vae_recovered.pt`: the exact source weights with recovered optimizer/RNG.
- `vae_last.pt`: last added epoch, including optimizer/RNG for further continuation.
- `vae_best.pt`: best validation total loss WITHIN the added201-400 epochs.
- `history.json`: original1-200 records followed by the added201-400 records.
- `recovery_verification.json`, `protected_hashes.json`, `status.json`: audit trail.
- `evaluation.json`: source/final comparisons using identical diagnostic code,
  same dataset/scaler, mean-template baseline,20 fixed-seed latent shuffles,
  six physical RMSE channels and press/forward/reverse metrics.
- `encoder.pt`: compatible with unchanged `FrozenSpongeEncoder.encode`,
  raw[B,400,6] to deterministic[B,5], frozen, no property supervision.
- `training.png`, `beta.png`, `latent_diagnostics.png`,
  `reconstruction_comparison.png` and fixed test0,10,...,90 reconstructions.

Evaluation uses the requested final epoch400, not a test-selected checkpoint.
The20 shuffle seeds4200-4219 and float32 inverse-target RMSE follow the existing
ablation diagnostics helper; the source epoch200 is reevaluated identically
for comparison. Minor rounding differences from the original single-batch
evaluation are expected. No outcome automatically triggers extra training.
The20% baseline improvement and10% latent-utilization thresholds are engineering
diagnostics, not paper-reported criteria. Reusing an already inspected test set
does not create a new untouched benchmark.

```bash
xdg-open runs/sim_training/normal_mu1p2_continue400_v1/reconstruction_comparison.png
xdg-open runs/sim_training/normal_mu1p2_continue400_v1/training.png
xdg-open runs/sim_training/normal_mu1p2_continue400_v1/latent_diagnostics.png
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_continue_pretraining.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -q
```

Do not use the original train/evaluate/export CLI with the continuation folder:
its config/data coupling intentionally remains unchanged. Load complete models
with `load_continued_model` from the new script, or the exported encoder with
`scripts.sim_pretrain.learning.FrozenSpongeEncoder`.

If a further extension is separately authorized, point `--checkpoint` to this
run's `vae_last.pt`, retain the ORIGINAL `--dataset-config`, and choose another
new output directory. A native continuation checkpoint restores optimizer/RNG
directly and does not replay its previous epochs. A failed/interrupted run must
first have matching checkpoint and history epochs before reuse; inconsistent
files are rejected rather than automatically rewritten.

The focused tests compare a2+2+2 legacy/native continuation with the unchanged
trainer's uninterrupted6 epochs, including exact weights/history and unchanged
source artifacts. Other checks cover mismatch rejection, all three RNG states,
output protection, train-only stochastic/gradient branches, exclusion of the
test loader and non-finite loss/gradient termination. The full-run exported
encoder is also checked against saved test embeddings and repeat inference.

## Export Verification Recovery

Export checks compare exact encoder weights/scaler and then outputs at MATCHING
batch sizes, using rtol1e-6/atol1e-8. Full-batch and batch32 float32 matrix
reductions can differ slightly, especially near zero, and their difference is
reported separately. This does not change training, weights or predictions.

If training reached its final epoch and only the already-written export check
failed, the following command verifies existing artifacts without retraining,
rewriting weights or regenerating plots. It refuses unfinished training or an
already-completed run. Source/runs/sim_training/history/checkpoint provenance must match.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.continue_pretraining --verify-export-only \
  --output runs/sim_training/normal_mu1p2_continue400_v1
```

`postprocessing_recovery.json` preserves the previous failure, training versus
postprocessing script hashes and unchanged model/history hashes. Only a change
to this independent script is permitted for recovery; original protected
inputs and learning sources must remain unchanged. The new export tests also
verify that recovery cannot call training or modify checkpoint/encoder/history.
