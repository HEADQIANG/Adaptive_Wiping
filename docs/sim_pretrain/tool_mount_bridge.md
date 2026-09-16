# Geometrically Consistent Engineering Tool Mount

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

`simulation.tool_mount: bridge_v1` is now selected by `configs/sim_pretrain/pretrain_paper.yaml`.
New results go to `archive/sim_pretrain/pretrain_paper_bridge_v1`; old gate/data cannot be reused.
This is an engineering mount on the existing ordinary AIRBOT gripper housing,
NOT a claimed reconstruction of the manufacturer's bare flange. Source audit:
[tool_mount_audit.md](tool_mount_audit.md). No source robot XML is modified.

## Assembly

The rigid hierarchy is right_hand -> mount_adapter -> wiping_gripper (backplate
and sponge). No free/hinge joint is added. The adapter has an interface plate
seated against the front of the original housing mesh and two supporting rails.
The original gripper housing/camera geometry and six link inertias are retained;
they are not silently replaced by parts from the incompatible force-model variant.

All dimensions below are engineering assumptions in metres. In link6 coordinates:

| Part | Dimensions | Axial extent |
| --- | --- | --- |
| Housing interface plate | X=.050, Y=.130, Z=.002 | -.0524 to -.0504 |
| Two rails | .008 x .008 x .0634 each; Y=+/- .052 | -.0504 to +.013 |
| Sponge backing plate | .120 x .050 x .002 in tool coordinates | +.013 to +.015 |
| Existing sponge appearance | .120 x .050 x .030 in tool coordinates | +.015 to +.045 |

The tool's -90 degree rotation about Z swaps its X/Y extents in link6 coordinates.
The interface seats 1 mm into the visible mesh envelope (frontmost vertex z=-.0514)
as a modeled fit tolerance, not a measured flange or bolt pattern. Rails touch both
plates exactly; the backing plate touches the sponge rear face. The existing
sponge collision patches and TCP remain at their original local/world transforms
for a given joint pose; the script does not fake connection by shifting only mesh.

## Dynamics And Sensing

All added parts have matching visual and box collision geometries. Collision
geometries have explicit zero mass because inertial elements represent each rigid
assembly exactly once. Added mass/tensor uses uniform aluminium density 2700 kg/m3
and the box formula plus the parallel-axis theorem. Adapter mass is 0.05701104 kg;
backplate mass is 0.0324 kg. Total added mass is 0.08941104 kg.

The original sponge mass 0.03 kg and principal inertia `[.01,.01,.01]` kg*m2 are
retained to avoid bundling an unrelated sponge inertia recalibration. These are
inherited simulation assumptions, NOT a physically calibrated uniform foam box.
The backplate is combined with that inertial at the correct COM and full tensor.

The FT site is moved from inside the sponge to its backing interface:
tool-local `[0,0,-.017]`, link6-local z=+.013. Axes and raw sign convention are
unchanged. The sensor now represents support wrench for sponge+backplate; adapter
mass is upstream. Torque lever arm and gravity bias have changed, so old FT data
is incompatible. No additional physical sensor housing mass is invented.

Only original sponge patches receive sampled mu/k/width. Rigid mounting parts
are NOT sponge material and are NOT included in the accepted tool-table contacts;
mounting-part penetration into the table is rejected by the existing unexpected
contact check. Same welded assembly self-overlap does not create spurious joints.

## Run And Validate

From the project root:

```bash
conda activate clean
python -m unittest discover -s tests -t . -p 'test_tool_mount.py' -v
python -m unittest discover -s tests -t . -p 'test_pretraining.py' -v
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 0 --stiffness 1000 --width 0.02
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
```

Optional headless preview (new output filename required):

```bash
MUJOCO_GL=egl python -m scripts.sim_pretrain.experiments.visualize_wiping --no-viewer --gif runs/sim_training/wiping_bridge_v1.gif --mu 0 --stiffness 1000 --width 0.02 --speed 1
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if conda activation is unavailable.
`tool_mount: legacy` in an explicitly separate config disables this assembly for
historical comparisons. Do not edit or mix old artifacts to make them pass new
provenance checks. Do not collect/train until current mechanical AND contact-motion
checks pass; a visually connected tool alone does not fix high-friction tracking.

## Completed Verification (2026-09-08)

- Two new mount tests, 18 existing pretraining tests and the visualization
  regression test passed (21 tests). Compiled adapter mass and backing mass,
  rigid hierarchy, geometry alignment, six DOFs, unchanged original sponge/TCP
  kinematics and central/eccentric sensor loads are checked.
- `archive/sim_pretrain/wiping_bridge_v1.gif` was rendered with EGL and visually inspected.
  The housing, interface plate, two rails, backing plate and sponge are visibly
  connected while the real simulated trajectory moves.
- Fresh sanity selected gain 300: free-space position RMS 0.44832 mm, maximum
  orientation error 0.25377 degrees, no saturation. Gain 100 RMS 1.01112 mm failed.
- All 27 parameter cases completed with finite metrics and unloading. The
  independent sensor check passed; no mounting-part contact error was reported.
- Only 3/27 cases passed the currently implemented contact-motion acceptance.
  Maximum position error was 55.270 mm, maximum orientation error 10.228 degrees.
  This geometry change does not solve the contact-control tracking problem.
- Sanity exited 2 and `archive/sim_pretrain/pretrain_paper_bridge_v1/sanity.json` has
  `passed:false`. Unlike the older historical gate, the current collection code
  checks contact-motion acceptance and blocks this configuration. No dataset
  collection or formal VAE training was started.
