# Normal-Mode Width FT Sweep

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Independent diagnostics at width=0.02,0.05,0.1,0.2,0.3. Defaults fix mu=0.9,
stiffness_direct=1000 and the existing normal IK+velocity-feedforward controller,
gain300. Width is the MuJoCo solimp transition parameter, NOT geometric sponge
width. The stiffness parameter is not calibrated N/m.

The new entry point reuses the unchanged friction-sweep collector once per width,
with a fresh environment, seed42 and the same400-frame exploration. No training
runs/sim_pretrain/model, production control, previous script or previous results are modified.
Contact-motion failures are recorded and retained, not used to reject data.

## Run

From the repository root:

```bash
conda activate clean
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.sim_pretrain.experiments.width_ft_sweep \
  --dataset-config archive/sim_pretrain/two_control_pretraining_v2/normal/config.json \
  --mu 0.9 --stiffness 1000 --width 0.02 0.05 0.1 0.2 0.3 \
  --output runs/sim_pretrain/normal_width_ft_v1
```

Output must be new and outside the frozen dataset directory. There is no resume,
retry or automatic training. Exit0 means all diagnostic data were generated,
not that wiping passed acceptance. No viewer or GPU rendering is required.

## Outputs And Interpretation

- `ft_raw_comparison.png`, `ft_filtered_comparison.png`: six channels overlaid
  by width; forces in the left column, torques in the right.
- `width_0.02/ft.png` (and other widths): individual raw/filtered FT comparison.
- `motion_contact.png`: actual/target world-Y position, summed contact normal
  load, orientation error, actual-minus-target Z error.
- Root `sweep.npz`: arrays stacked in requested width order. `ft`, `ft_filtered`
  are[5,400,6]; `time` is[5,400]; `parameters` is[5,3] ordered mu,stiffness,width.
- Each `width_*/sweep.npz` is the reused collector's one-condition[1,400,6]
  export. `width_*/mu_0.9/trajectory.npz` holds unstacked arrays, including
  position, targets, quaternion, contact force and actuator torque diagnostics.
- Root and per-width `summary.json` record configuration, source/input hashes,
  states, failures, raw per-channel absolute peaks, three-phase signed mean,
  RMS and absolute peaks, motion acceptance and successful unloading.

Samples are .01-4.00 s at100 Hz: press .01-2.00 s, forward2.01-3.00 s,
reverse3.01-4.00 s. The command presses at10 mm/s, then travels50 mm/s in each
horizontal direction for1 s. Actual displacement can differ significantly.
FT is local `ft_frame`: Fx,Fy,Fz in N and Tx,Ty,Tz in N m. It includes tool
weight and dynamics; no bias subtraction or normalization. Filtering uses the
original second-order10 Hz zero-phase filter, without across-width averaging.
`contact_wrench` is separately the world contact wrench ON tool about TCP;
its torques must not be compared with local sensor torques without transforms.

## View And Read

```bash
xdg-open runs/sim_pretrain/normal_width_ft_v1/ft_filtered_comparison.png
xdg-open runs/sim_pretrain/normal_width_ft_v1/ft_raw_comparison.png
xdg-open runs/sim_pretrain/normal_width_ft_v1/motion_contact.png
```

```python
import numpy as np

with np.load("runs/sim_pretrain/normal_width_ft_v1/sweep.npz", allow_pickle=False) as data:
    print(data["parameters"])
    time = data["time"][2]  # width=0.1
    ft = data["ft"][2]      # [400,6], raw physical units
    filtered = data["ft_filtered"][2]
```

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -t . -p 'test_width_ft_sweep.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -q
```

Focused checks cover width-only assignments, fresh environments, exact filtered
and stacked exports, plots, invalid/duplicate widths and colliding directory
names, overwrite/frozen-directory protection, and retaining failed conditions.

## Generated Results: 2026-09-09

`archive/sim_pretrain/normal_width_ft_v1` completed with exit0. Five400-frame trajectories
are finite and successfully unloaded; eight PNGs and combined/per-condition
NPZ/JSON files are available. Both six-channel comparison figures and an
individual trace, as well as the motion/contact plot, were visually inspected.

Raw per-channel absolute peaks over the whole4 s (not simultaneous samples):

| Width | Fx (N) | Fy (N) | Fz (N) | Tx (N m) | Ty (N m) | Tz (N m) |
|---|---:|---:|---:|---:|---:|---:|
| 0.02 | 1.71296 | 0.21735 | 1.78772 | 0.033097 | 0.072735 | 0.034098 |
| 0.05 | 1.70564 | 0.21430 | 1.81167 | 0.032557 | 0.071261 | 0.035418 |
| 0.1 | 1.70441 | 0.21385 | 1.81088 | 0.032439 | 0.071106 | 0.035617 |
| 0.2 | 1.70411 | 0.21374 | 1.81076 | 0.032406 | 0.071064 | 0.035677 |
| 0.3 | 1.70405 | 0.21371 | 1.81075 | 0.032403 | 0.071057 | 0.035686 |

Forward travel is4.787,4.795,4.796,4.798,4.799 mm in width order; reverse travel
is-2.281,-2.270,-2.268,-2.268,-2.268 mm. A negative reverse endpoint difference
means net movement was still forward during the reverse command, not successful
return motion. Maximum orientation errors are7.843-7.881 degrees. All five
conditions fail full motion acceptance, and all are retained.

Width0.02 exactly reproduces all19 arrays (including filtered FT) from the prior
`archive/sim_pretrain/normal_friction_ft_v1/mu_0.9/trajectory.npz`. Combined and individual
exports match exactly. Protected source/configuration/dataset/model hashes are
unchanged. Three focused tests passed; full regression passed89 tests in62.398 s.
This includes the separately added stiffness-sweep tests. This task's simulation
and test sessions exited0, with no background tasks from this task remaining.

The curves are close, especially at width0.1/0.2/0.3. This is weak sensitivity
for this particular controller, mu, stiffness and exploration; it does not prove
that width is irrelevant across the whole parameter distribution.
