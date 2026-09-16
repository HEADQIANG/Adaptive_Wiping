# Decoder Activation Experiment

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This is the separately authorized next experiment after both small AEs in
`normal_vae_ablation_v1` failed. It changes only the original decoder's ReLU
to LeakyReLU(.01), retaining the original linear encoder and all linear layers.
It does not edit the previous experiment, production code, original data or model.

## Run

From the repository root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.decoder_activation_experiment \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --previous-output archive/sim_pretrain/normal_vae_ablation_v1 \
  --output runs/sim_training/normal_decoder_activation_v1
```

The output directory must not exist. No overwrite or resume is supported. Do
not place it inside the previous experiment or frozen dataset directory. Do
not edit experiment source during execution. CPU Torch uses one thread. Exit0
means the gated VAE continuation reached the existing three-seed stability
criterion. Exit2 means a bounded check failed without an execution exception.
Exit3 means the ReLU control did not reproduce the previous experiment, so B
and all later stages were not run. Other errors raise exceptions and save status.

## Fixed Protocol

1. Verify original data, model, production source and previous experiment source
   hashes. Reuse the original train-only preprocessing and exactly the 32 training
   indices saved in the previous manifest. No contact-failure rows are removed.
2. Save seed42 initial parameter tensors once. Both A=ReLU and B=LeakyReLU(.01)
   load these exact tensors. Both use mu decoding, beta0, dropout0, Adam1e-3,
   full batch32, maximum2000 steps. The only A/B intervention is the activation.
3. Run A first. Compare every step's pre-update loss/MSE/KL, post-update MSE,
   and all final parameters against the previous original-linear small AE.
   Record bitwise equality; tolerances are fixed at rtol1e-6/atol1e-8. A mismatch
   stops execution of B and all continuation stages pending investigation.
4. Run B. Require MSE strictly below both .001 and 10% of the subset's average
   trajectory MSE (previously .000665001757). Stop at success or step2000. If
   failed, finish the report; do not extend the budget or try another decoder.
5. Only on B success, train a fresh full-data LeakyReLU AE for 200 epochs,
   batch32. Select by validation MSE; require at least20% improvement over the
   training-template validation baseline. On failure, stop.
6. Only on full AE success, run the same six fresh 200-epoch VAE controls from
   `vae_ablation.md`, retaining LeakyReLU for every group. Training always samples
   z. Select checkpoints by validation MSE at epochs50-200; save epoch200 too.
   Keep the same reconstruction and fixed/shuffled-code gates and table tie order.
7. If a VAE qualifies, lock its configuration before fresh seed43/44 repetitions
   and final test reports. At least two of three test passes are required for
   stability. Test data never select hyperparameters or control an earlier gate.
   These existing test data were already examined in prior work, so final test
   reports are not a newly blinded benchmark. No candidate means no test scoring.

Neither a direct 5-to-2400 decoder, a nonlinear encoder, a mean-trajectory residual
connection nor property supervision is included. The previously trained property
readouts remain independent baselines. No recollection, controller/material change,
impedance training or hardware experiment is enabled.

## Artifacts

`shared_initialization.pt` records the actual shared tensors and their canonical
content hash. Manifests and snapshots record dataset/config/source hashes and all
protected files in the previous experiment. `reproduction.json` records the A
audit. Each run saves spec, status, history, best/last activation-tagged models,
training curves and activation diagnostics at initialization, step1, every100
steps and the final step. Non-finite loss/gradient/parameter checks stop that run.

`activation_comparison.png` compares training MSE, never-positive preactivation
fraction and mean-trajectory error. Paired six-channel reconstructions use only
subset indices0,10,20,30 (original dataset indices are in their filenames).
Never-positive on these32 rows is not proof of global neuron inactivity, and a
LeakyReLU unit still has a nonzero local derivative on negative inputs.

`report.json`/`report.md` record the actual stopping condition. Conditional full
runs save validation diagnostics and, only after selection, test diagnostics and
frozen encoder exports. No failed small AE is presented as a repaired VAE.

Load full models with their activation-aware loader, not the original v1 loader:

```python
import sys
sys.path.insert(0, "scripts")  # repository root; avoids ROS's scripts package
from decoder_activation_experiment import load_activation_model

model, metadata = load_activation_model(
    "runs/sim_training/normal_decoder_activation_v1/leaky_small_ae/best.pt"
)
```

The model expects already preprocessed FT. `metadata['preprocessing']` preserves
the preprocessing state. A conditional successful encoder export uses the existing
independent `FrozenAblationEncoder.encode(raw_ft)` interface, because the encoder
architecture itself is unchanged. The decoder's activation is recorded separately.

## Verify And View

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_decoder_activation.py' -v
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -v
less runs/sim_training/normal_decoder_activation_v1/report.md
xdg-open runs/sim_training/normal_decoder_activation_v1/activation_comparison.png
```

Focused tests cover equal initialization, the negative-input derivative, exact
agreement with the previous trainer, corrupt-history rejection, activation-aware
checkpoint and frozen-encoder roundtrips, conditional stops, the six validation-only
VAE specs, warmup/checkpoint epochs, finite-failure handling, overwrite/nesting
protection, paired training plots and preservation of prior source/data hashes.
Short synthetic tests do not expose a CLI override of the real experiment budget.

## Completed Run: 2026-09-09

`archive/sim_pretrain/normal_decoder_activation_v1` is complete. The process exited2 with
`outcome=small_ae_failed`; this is the planned stopping condition, not a crash.
Do not rerun into the existing directory.

| Small AE | Steps | Training MSE | Required strict MSE | Passed |
|---|---:|---:|---:|---|
| A: ReLU | 2000 | 0.008913830854 | <0.000665001757 | No |
| B: LeakyReLU(.01) | 2000 | 0.004072058015 | <0.000665001757 | No |

A reproduced all2000 steps of the previous loss history and every final parameter
bit for bit: both maximum differences are zero. A and B shared initial parameter
hash `0096997413cf89bc246c29a93846283f7c289b6791a3dc4ebc4b3f5e442b5713`.
Both best and last checkpoints are step2000 and reload successfully.

B reduced training MSE by54.3175% relative to A but is still about6.12 times the
strict threshold. The subset average-trajectory baseline remains0.006650017574.
The mean-trajectory fitting component of error fell from0.005097893 to0.000923228.
Never-positive preactivation on the fixed32 rows fell from46.15% to31.05%; unlike
ReLU, LeakyReLU has a nonzero derivative for negative inputs, so31.05% is not an
inactive-neuron count. Exact-zero activation fraction was64.57% for A and0% for B.

This controlled change improved optimization in this run but did not establish
the only cause of poor reconstruction, convergence, generalization or a repaired
VAE. Both histories remained finite. No budget extension, direct linear decoder,
full-data AE, VAE sweep, seed43/44 repetition, test scoring or property retraining
was performed. `selection.json` is null and `test_scored` is false. The full-data
and VAE branches are implemented and software-tested, not executed on real data.

Verified all128 protected original/previous-experiment files and both current
source hashes unchanged. Inspected the comparison plot and a paired six-channel
training reconstruction; four reconstruction plots exist for original training
rows83,836,640,126. The complete regression suite passed75 tests in53.530 seconds,
including10 new activation tests. No training or test process remains running.

View the paired reconstruction without retraining:

```bash
xdg-open runs/sim_training/normal_decoder_activation_v1/train_reconstruction_083.png
```

The next candidate discussed is a separate direct `Linear(5,2400)` decoder
diagnostic with the encoder and data held fixed. It has not been implemented or
run by this experiment and must not be silently enabled as a fallback.
