# Tabletop Base And Compact Sponge Mount

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Superseded default: see [direct_wrist.md](direct_wrist.md). This document describes
the historical compact_v2 layout. To reproduce it, use a separate config selecting
`tool_mount: compact_v2` and a fresh output directory; current default commands
without that config render the direct wrist layout instead.

The historical configuration selects `base_mount: tabletop`, `base_xy: [0.07,0]`
and `tool_mount: compact_v2`. Output: `archive/sim_pretrain/pretrain_paper_tabletop_compact_v2`.
This implements the agreed tabletop placement and tight sponge attachment, not
a calibrated manufacturer flange or the original paper robot geometry.

## Mechanical Layout

- The existing table remains 0.5 x 0.8 m, centered at XY=[.15,0], top Z=.9 m.
- The robot loads with NullMount. Wipe ignores its base_types argument internally,
  so the pretraining subclass selects NullMount before robot loading. No previously
  registered pedestal geometry is deleted and the global robot model is unchanged.
- Robot base XY=[.07,0]. The existing base mounting collision box has bottom
  Z=.0025 in robot coordinates, so robot root world Z=.8975. The installation
  plane is exactly Z=.9. This is an installation-plane convention; the visible
  mesh lies about 1.4 mm above that plane due to the inherited mesh/proxy difference.
- The full base collision footprint is checked against the table before compile.
  It spans approximately X=[-.0875,.1875], Y=[-.0806,.0806] on the tabletop.
- Exploration center moves to [.31,0], clear of the base and within the table.
- The long rails and upper interface plate are absent. The 2 mm backing plate
  alone connects the existing wrist housing to the sponge, as a fixed child.
- Tool translation relative to right_hand changes from [0,0,.015] to
  [0,0,-.0494]. The complete tool, colliders, FT site and TCP move together by
  -64.4 mm in the wrist's local Z, with no change to their local geometry or axes.
- In link6 coordinates the backplate support plane is Z=-.0514, matching the
  audited housing front envelope. Backplate spans [-.0514,-.0494]; sponge
  appearance spans [-.0494,-.0194]. This is an engineering mounting assumption.

## Dynamics And Compatibility

The 57.01104 g bridge adapter and its inertia are removed; the 32.4 g aluminium
backing plate and 30 g sponge remain. Their combined inertia calculation and the
inherited sponge inertia assumption are unchanged. The raw FT site remains on the
backing support face at tool-local [0,0,-.017]. Its world pose, and the robot's
effective tool lever arm, change with the compact installation.

Controller, reference mode, speeds, contact parameter mapping, force limits and
acceptance thresholds are not modified. The historical `stand` base and
`legacy`/`bridge_v1` tool modes remain supported through a separate config.
Never reuse old data or gate reports after changing the mount/config/source hashes.

## Commands

From the project root:

```bash
conda activate clean
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 0 --stiffness 1000 --width 0.02
python -m scripts.sim_pretrain.experiments.visualize_wiping --closeup --mu 0 --stiffness 1000 --width 0.02
python -m unittest discover -s tests -t . -p 'test_tabletop_compact.py' -v
python -m unittest discover -s tests -t . -p 'test_pretraining.py' -v
python -m unittest discover -s tests -t . -p 'test_tool_mount.py' -v
python -m unittest discover -s tests -t . -p 'test_wiping_visualization.py' -v
python -m unittest discover -s tests -v
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if conda activation is unavailable.
Optional headless exports require EGL and new output filenames:

```bash
MUJOCO_GL=egl python -m scripts.sim_pretrain.experiments.visualize_wiping --no-viewer --gif runs/sim_training/tabletop_compact.gif --mu 0 --stiffness 1000 --width 0.02 --speed 1
MUJOCO_GL=egl python -m scripts.sim_pretrain.experiments.visualize_wiping --closeup --no-viewer --gif runs/sim_training/tabletop_compact_closeup.gif --mu 0 --stiffness 1000 --width 0.02 --speed 1
```

Reset and pose tests verify six DOFs, absence of the old pedestal/rails, mounting
height, full footprint containment, compact tool transforms and sensor loads.
The complete fresh free-space/contact scan remains mandatory; a visually correct
assembly is not proof of contact tracking. Failures must remain failures, with
no data collection or threshold relaxation.

## Verified Results (2026-09-08)

All 31 tests passed, including three new tabletop/compact regression tests and
the existing dynamics, contact acceptance, reference, VAE and playback tests.
Both GIFs above were exported and visually inspected. Each has 81 frames;
first-to-middle changed pixels are 4542 (overview) and 15976 (close-up).
The low-friction displayed rollout (mu=0, k=1000, width=.02, gain=300) has
position RMS 14.582 mm, maximum 18.980 mm, angle maximum 1.610 degrees and
zero torque saturation. Its Y tracking error is 6.481 mm, exceeding the 5 mm
contact gate: the animation verifies assembly, not tracking acceptance.

Fresh `archive/sim_pretrain/pretrain_paper_tabletop_compact_v2/sanity.json`: gain 100 fails
free-space tracking; gain 300 passes with RMS .464307 mm. All 27 contact cases
complete unloading, and the independent sensor load check passes. Only 3/27
contact motion checks pass. Maximum position error is 55.192668 mm, maximum
orientation error is 9.956722 degrees, and maximum saturation fraction is zero.
The command exits 2 with `passed: false`; production collection remains blocked.
Production source hashes were checked against the report. No formal data
collection or VAE training was started. Next work is contact-control diagnosis
under this new installation, not further cosmetic relocation or relaxed gates.
