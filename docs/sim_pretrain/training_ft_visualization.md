# Training FT Visualization

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

The independent `scripts/sim_pretrain/experiments/plot_training_ft.py` plots recorded training signals,
not model reconstructions. It reads only the HDF5 training split, verifies the
dataset/config hash and saved training-only preprocessing, and writes into a
new visualization directory. No model inference, optimization, recollection or
controller/material changes are performed. Failed-contact trajectories are kept.

## Generate

Run from the repository root:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.plot_training_ft \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --indices 0 83 \
  --subset-manifest archive/sim_pretrain/normal_vae_ablation_v1/manifest.json \
  --output runs/sim_pretrain/normal_ft_visualization_v1
```

The output must not already exist or be inside the frozen dataset or subset
experiment directory. To inspect another training row, change `--indices` and
choose a new output directory. Indices are zero-based (normal mode:0-999).
The optional `--subset-manifest` adds a distribution plot for the exact32 rows
used in the prior AE experiments. It does not filter the full training overview.

## Figures

All figures have forces Fx/Fy/Fz in the left column and torques Tx/Ty/Tz in the
right column. Physical forces are N and torques are N m, in the local `ft_frame`,
not world axes. Each trajectory has400 samples at100 Hz, from .01 to4 seconds.
Vertical boundaries at2 and3 seconds separate press, forward and reverse phases.

- `sample_000.png`, `sample_083.png`: gray raw FT and blue filtered FT in physical
  units. Filtering exactly reuses the original second-order10 Hz zero-phase filter.
- `normalized_sample_000.png`, `normalized_sample_083.png`: the actual six-channel
  normalized network inputs. These are dimensionless, not force or torque units.
- `training_overview.png`: all1000 filtered training trajectories, median, mean
  and pointwise10-90% quantile band. The band is trajectory variability, not a
  statistical confidence interval or one example trajectory.
- `normalized_overview.png`: the same full training distribution after the saved
  training-set min/max scaling to0-.9. No clipping or new normalization is applied.
- `subset_overview.png`: the optional fixed32-row AE subset in physical units.

Raw signals include tool weight and are not bias-subtracted pure contact forces.
Plot titles show simulation friction, `stiffness_direct` and width; the stiffness
parameter must not be interpreted as calibrated physical N/m. Collection validity
is not proof of successful contact-motion acceptance.

`plotted_data.npz` retains the selected exact raw/filtered/normalized arrays, time,
parameters and plotted bands. `summary.json` records input/source hashes, selection,
units, channel ranges, retained counts and post-run input-integrity verification.

## View And Test

```bash
xdg-open runs/sim_pretrain/normal_ft_visualization_v1/sample_083.png
xdg-open runs/sim_pretrain/normal_ft_visualization_v1/training_overview.png
xdg-open runs/sim_pretrain/normal_ft_visualization_v1/normalized_sample_083.png
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_training_ft_plot.py' -v
```

The synthetic tests cover force/torque panel order and units, exact preprocessing,
training-only reads, retention of failed-contact rows, integrity/stale-preprocessing
rejection, output protection, plot generation and saved numerical plot arrays.
Filtered and normalized arrays are checked exactly against the training pipeline;
the additional physical-unit roundtrip check accounts for float32 normalization
and inverse-division rounding using a conservative maximum physical channel range.

## Generated Output: 2026-09-09

`archive/sim_pretrain/normal_ft_visualization_v1` now contains all seven figures listed above,
each1950x1350 pixels, plus `summary.json` and `plotted_data.npz`. Generation
completed with exit0. The full overview retains all1000 normal-mode training
trajectories, including their failed-contact labels; the optional subset has
the same32 indices as the prior AE experiment. No validation/test trajectories
were used for these figures.

Selected training parameters are:

| Training row | Friction | stiffness_direct | Width |
|---|---:|---:|---:|
| 0 | 2.708846 | 62.532075 | 0.255965 |
| 83 | 0.584405 | 843.401224 | 0.070770 |

These two examples differ in all three parameters and are not a controlled
friction-only comparison. Row83 was also part of the fixed32-row AE check.
For a separate fixed-stiffness/width comparison at mu=0,0.5,0.9,2.5,3.5,
see [Normal-Mode Friction FT Sweep](friction_ft_sweep.md). Those freshly simulated
diagnostic trajectories are not inserted into the training dataset.
The raw/filtered arrays and normalization were verified against the existing
preprocessor; normalized arrays remain dimensionless. The dataset, config,
integrity record, saved preprocessing and subset manifest hashes are unchanged.
Figure titles, force/torque units, phase markers and plot rendering were checked.
The four visualization tests and complete79-test regression suite passed; the
final full suite took48.594 seconds. All plotting/test processes have exited.

To read existing images, use the `xdg-open` commands above. Do not rerun generation
into the same directory; use a new output name for additional training indices.
