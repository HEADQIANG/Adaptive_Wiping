# Normal-Mode Friction FT Sweep

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This isolated simulation compares mu=0,0.5,0.9,2.5,3.5 under the existing
AIRBOT Play normal controller (IK + velocity feedforward, gain300). It fixes
stiffness_direct=1000 and width=0.02. The frozen normal configuration provides
the existing mounting, inertia, controller reference, seed42 and exploration.
Each coefficient uses a fresh environment and identical target positions.
These are new diagnostic trajectories, not selected or averaged training rows.
No training dataset, model, production controller or gate is modified.

For k=0.5,10,100,250,500,1000 with mu=0.9 and width=0.02 fixed, see
[Normal-Mode Stiffness FT Sweep](stiffness_ft_sweep.md). It reuses this recorder
without changing the earlier friction results or script.

## Run

From the repository root:

```bash
conda activate clean
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.friction_ft_sweep \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --mu 0 0.5 0.9 2.5 3.5 --stiffness 1000 --width 0.02 \
  --output runs/sim_pretrain/normal_friction_ft_v1
```

The output directory must not exist or be inside the frozen dataset directory.
Choose a new output name for another run. There is no automatic retry or resume.
All five conditions are attempted; failed cases retain their parameters and
failure reasons. Exit0 means data generation completed, NOT wiping acceptance.
There is no viewer and no GPU/rendering requirement for simulation or PNG plots.

## Signals And Files

- Each trace has400 samples, .01-4.00 s, at100 Hz. Phase slices are press
  [0:200] (.01-2.00 s), forward [200:300] (2.01-3.00 s), reverse [300:400]
  (3.01-4.00 s). Commands press at10 mm/s for2 s and move50 mm/s in each
  horizontal direction for1 s. Actual motion is recorded separately.
- `ft_raw_comparison.png` and `ft_filtered_comparison.png`: five conditions
  overlaid on six axes, forces in the left column and torques in the right.
  Filtered plots reuse the existing second-order10 Hz zero-phase filter;
  no scaling, clipping, bias subtraction or across-condition averaging.
- `mu_0/ft.png`, `mu_0.5/ft.png`, `mu_0.9/ft.png`, `mu_2.5/ft.png`,
  `mu_3.5/ft.png`: individual raw (gray) and filtered (blue) traces.
- `motion_contact.png`: actual/target world-Y position, summed contact normal
  load, orientation error and Z tracking error. Contact does not prove sliding.
- `sweep.npz`: numeric arrays stacked in the requested mu order. `ft` and
  `ft_filtered` are[5,400,6], `parameters` is[5,3], and `time` is[5,400].
  Other arrays retain positions, orientations, contact counts/loads, joint
  torques and initial sensor bias. Per-condition `trajectory.npz` has the
  same fields without the leading condition dimension.
- `summary.json`: configuration, source/input hashes, units, per-case status,
  raw per-channel absolute peaks and per-phase signed mean/RMS/absolute peak,
  force/torque norm peaks, full motion acceptance and unloading results.

Sensor channels are Fx,Fy,Fz (N), Tx,Ty,Tz (N m) in local `ft_frame` coordinates.
They include tool weight and dynamics, not just contact friction. Saved
`contact_wrench` is a different quantity: world-frame contact wrench ON the
tool with moment about TCP; `normal_sum` sums individual normal contact loads.
Do not directly compare their torque components without shifting reference
points and transforming frames. No friction ratio is estimated here.

At requested mu=0 MuJoCo uses a sliding-friction floor of1e-5. Torsional and
rolling coefficients remain0.005 and0.0001, respectively, across all cases.
Zero requested sliding friction does not imply zero sensor force/torque.
`stiffness_direct` is a MuJoCo contact parameter, not calibrated N/m; width is
the existing solimp parameter, not the sponge's geometric width.

## View And Read

```bash
xdg-open runs/sim_pretrain/normal_friction_ft_v1/ft_filtered_comparison.png
xdg-open runs/sim_pretrain/normal_friction_ft_v1/motion_contact.png
xdg-open runs/sim_pretrain/normal_friction_ft_v1/mu_0.9/ft.png
```

```python
import numpy as np

with np.load("runs/sim_pretrain/normal_friction_ft_v1/sweep.npz", allow_pickle=False) as data:
    print(data["parameters"])  # mu, stiffness_direct, width
    time = data["time"][2]     # mu=0.9, seconds
    ft = data["ft"][2]         # [400,6], raw physical units
    filtered = data["ft_filtered"][2]
```

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_friction_ft_sweep.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

Focused tests cover fixed parameters/order, fresh environments, exact filtering,
phase boundaries and statistics, finite-value checks, plot/data export, protected
outputs, preservation of motion failures and explicit simulation-failure reporting.

## Generated Results: 2026-09-09

`archive/sim_pretrain/normal_friction_ft_v1` completed with exit0: five finite400-frame
trajectories, all successfully unloaded, eight PNGs, five per-condition NPZs,
the stacked `sweep.npz`, and overall/per-condition JSON summaries. The six-channel
figures are1950x1350; the motion/contact figure is1950x1050. Figures were inspected.

Per-channel absolute maxima below use RAW samples over the full4 s. Different
channels can reach their maxima at different times; these are not one wrench.

| mu | Fx (N) | Fy (N) | Fz (N) | Tx (N m) | Ty (N m) | Tz (N m) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.0274 | 0.0172 | 1.9985 | 0.02069 | 0.02604 | 0.00621 |
| 0.5 | 1.1459 | 0.2173 | 1.9163 | 0.03579 | 0.07864 | 0.02632 |
| 0.9 | 1.7130 | 0.2173 | 1.7877 | 0.03310 | 0.07273 | 0.03410 |
| 2.5 | 1.8160 | 0.2174 | 1.7877 | 0.02843 | 0.07456 | 0.03626 |
| 3.5 | 1.8165 | 0.2174 | 1.7877 | 0.02843 | 0.07443 | 0.03583 |

| mu | Forward travel (mm) | Reverse travel (mm) | Max orientation error (deg) |
|---|---:|---:|---:|
| 0 | 47.997 | 45.092 | 1.662 |
| 0.5 | 32.776 | 14.886 | 1.945 |
| 0.9 | 4.787 | -2.281 | 7.843 |
| 2.5 | 0.625 | 0.289 | 8.870 |
| 3.5 | 0.605 | 0.297 | 8.870 |

Travel is the signed world-Y endpoint difference using the original acceptance
definition, not integrated path length. Negative reverse travel at mu=.9 means
the endpoint moved farther in the forward direction during the reverse command.
Mu0 meets the travel/orientation limits but still fails X/Y tracking checks.
No condition passes all original motion checks; all observations are retained.
The near-overlap of mu2.5/3.5 FT accompanies very small motion, not equal material
friction coefficients or successful high-friction sliding.

All18 common recorded fields at mu0/.9/3.5 exactly reproduce historical
`runs/sim_pretrain/contact_cartesian_v2/baseline/contact_01/03/05.npz`, respectively.
The stacked and individual NPZ arrays agree exactly; all fields are finite.
Source, dataset, configuration, preprocessing and baseline model hashes are
unchanged. Four focused tests passed. The first full regression was terminated
with exit143 near its end; a complete rerun passed all83 tests in51.723 s (exit0).
All simulation and test sessions have exited. No model training was performed.
