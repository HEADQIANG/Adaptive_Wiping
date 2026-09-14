# Normal-Mode Stiffness FT Sweep

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This independent diagnostic compares stiffness_direct=0.5,10,100,250,500,1000
with mu=0.9 and width=0.02 fixed. It uses the existing normal AIRBOT Play
IK + velocity feedforward controller, gain300, seed42, direct-wrist mounting
and4-second exploration. Each condition starts with a fresh environment.
No original data, models, physics/controller sources or previous sweep are changed.

The existing mapping is `solref=[-k,-2*sqrt(k)]`. Thus the damping term varies
with k; this is a scan of the training pipeline's stiffness parameter, NOT a
physical stiffness-only experiment with fixed damping. Do not interpret k as
calibrated N/m. Width is the solimp parameter, not geometric sponge width.

## Generate

Run from the repository root:

```bash
conda activate clean
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.stiffness_ft_sweep \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --stiffness 0.5 10 100 250 500 1000 --mu 0.9 --width 0.02 \
  --output runs/sim_pretrain/normal_stiffness_ft_v1
```

The directory must not exist or be inside the frozen dataset directory.
Choose a new output name for additional runs. No resume or automatic parameter
substitution is provided. Motion-acceptance failures are recorded, not discarded;
data-integrity, contact-parameter, unexpected-collision and unloading checks remain.
Exit0 means all data were generated, not that all conditions wiped successfully.

The script reuses `friction_ft_sweep.run_sweep` with one fixed mu per k, without
editing that earlier script. The extra per-k directories preserve its source
audits and failure reporting. The new outer summary identifies the stiffness
sweep, explicit solref values and paths to each saved trajectory.

## Results And Units

- `ft_raw_comparison.png`: six stiffness conditions, unfiltered sensor FT.
- `ft_filtered_comparison.png`: same traces after the original second-order
  10 Hz zero-phase filter, without normalization or bias subtraction.
- `k_0.5/ft.png`, `k_10/ft.png`, `k_100/ft.png`, `k_250/ft.png`, `k_500/ft.png`,
  `k_1000/ft.png`: individual raw (gray) and filtered (blue) comparisons.
- `motion_contact.png`: actual/target world-Y positions, summed contact normal
  load, orientation error and Z tracking error. Force alone does not prove sliding.
- `sweep.npz`: numeric data stacked in stiffness order0.5,10,100,250,500,1000.
  `ft` and `ft_filtered` have shape[6,400,6], `parameters` has shape[6,3] with
  columns mu/stiffness_direct/width, and `time` has shape[6,400].
- `k_<value>/mu_0.9/trajectory.npz`: individual raw/filtered trajectory and
  diagnostics. The per-k runner additionally saves a one-condition `sweep.npz`
  with leading dimension1. No saved data are inserted into the training set.
- `summary.json`: configuration, source/input hashes, actual contact solref,
  per-condition motion/unloading results, raw per-channel absolute maxima and
  per-phase signed means/RMS/absolute maxima. All six cases are attempted even
  when one fails; there is no combined success archive for an incomplete sweep.

FT channels are Fx,Fy,Fz (N), Tx,Ty,Tz (N m), in local ft_frame coordinates.
They include tool weight and dynamics. `contact_wrench` is a different signal:
the world-frame contact wrench ON the tool, moment about TCP. `normal_sum` is
the sum of individual contact normal forces. Do not confuse sensor Fz with
normal_sum or compare torques at different origins without the proper transform.

Each trajectory has400 samples,100 Hz, .01-4 s. Press occupies samples[0:200]
(.01-2 s), forward[200:300] (2.01-3 s), reverse[300:400] (3.01-4 s).
Commands press at10 mm/s for2 s and slide50 mm/s for1 s in each direction.
Soft settings can produce large simulated penetration; retaining a complete
trace is not proof of realistic sponge deformation or material calibration.

## View And Read

```bash
xdg-open runs/sim_pretrain/normal_stiffness_ft_v1/ft_filtered_comparison.png
xdg-open runs/sim_pretrain/normal_stiffness_ft_v1/ft_raw_comparison.png
xdg-open runs/sim_pretrain/normal_stiffness_ft_v1/motion_contact.png
```

```python
import numpy as np

with np.load("runs/sim_pretrain/normal_stiffness_ft_v1/sweep.npz", allow_pickle=False) as data:
    print(data["parameters"])  # mu, stiffness_direct, width
    time = data["time"][2]     # k=100, seconds
    ft = data["ft"][2]         # [400,6], physical units
    filtered = data["ft_filtered"][2]
```

## Test

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_stiffness_ft_sweep.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -q
```

Focused tests cover parameter isolation/order, damping mapping, fresh resets,
exact stacked/per-condition exports and filtering, figure generation, invalid
inputs and output guards, and preservation of unsuccessful conditions.

## Generated Results: 2026-09-09

`archive/sim_pretrain/normal_stiffness_ft_v1` completed with exit0. All six400-frame
trajectories are finite and successfully unloaded. Nine PNGs were generated:
six individual plots, two FT comparisons and one motion/contact comparison.
FT figures are1950x1350; motion/contact is1950x1050. Raw/filtered comparisons
and the low-stiffness individual plot were visually checked.

Raw per-channel absolute maxima over the full4 s are below. The maxima need
not occur at the same time and must not be combined into a simultaneous wrench.

| stiffness_direct | Fx (N) | Fy (N) | Fz (N) | Tx (N m) | Ty (N m) | Tz (N m) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 2.9402 | 0.5906 | 3.0377 | 0.03738 | 0.11537 | 0.06148 |
| 10 | 1.9614 | 0.1380 | 1.9285 | 0.01386 | 0.04562 | 0.05013 |
| 100 | 1.6612 | 0.1163 | 1.5685 | 0.01319 | 0.04786 | 0.04083 |
| 250 | 1.6744 | 0.1473 | 1.6652 | 0.01980 | 0.05757 | 0.03877 |
| 500 | 1.6901 | 0.1853 | 1.7359 | 0.02732 | 0.06666 | 0.03686 |
| 1000 | 1.7130 | 0.2173 | 1.7877 | 0.03310 | 0.07273 | 0.03410 |

| stiffness_direct | Forward travel (mm) | Reverse travel (mm) | Max orientation error (deg) |
|---|---:|---:|---:|
| 0.5 | 15.558 | -4.935 | 6.690 |
| 10 | 10.976 | -3.442 | 7.444 |
| 100 | 7.064 | -2.332 | 7.901 |
| 250 | 5.897 | -2.161 | 7.948 |
| 500 | 5.239 | -2.276 | 7.894 |
| 1000 | 4.787 | -2.281 | 7.843 |

All six fail the original full motion acceptance. Travel uses signed endpoint
differences, not integrated path length; negative reverse travel means the
endpoint is farther forward at4 s than at3 s despite a reverse command.
The k=.5 condition builds normal load more slowly and has a reverse-phase peak
normal load of3.36155 N at3.68 s, with sensor force-norm peak4.26864 N over the
full trajectory. Lower k does not ensure a smaller transient peak under this
coupled contact/controller dynamics. No unique mechanical cause is inferred.

All18 original recorded fields at k=.5/1000 exactly match historical
`runs/sim_pretrain/contact_cartesian_v2/baseline/contact_02/03.npz`, respectively.
At k1000, all19 fields including filtered FT exactly match
`archive/sim_pretrain/normal_friction_ft_v1/mu_0.9/trajectory.npz`. Stacked and individual
arrays agree exactly, all six target arrays match, and protected source/data/
configuration/preprocessing/model hashes are unchanged. Three new tests and
the full86-test suite passed (58.546 s). All simulation/test sessions exited;
no VAE training, original data recollection or impedance experiment was started.
