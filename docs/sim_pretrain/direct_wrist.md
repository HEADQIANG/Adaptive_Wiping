# Direct Wrist Sponge Mount

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

The default `configs/sim_pretrain/pretrain_paper.yaml` now selects `tool_mount: direct_wrist_v3`
and writes new output under `runs/sim_training/`. Historical results remain in
`archive/sim_pretrain/pretrain_paper_direct_wrist_v3`. The tabletop base is unchanged.
This is a paper-like engineering layout, not the paper's UR5e or a calibrated
AIRBOT bare-flange model.

## Geometry And Dynamics

- Preserve all six robot joints and their coordinate transforms.
- Use MuJoCo to parse `asserts/robosuite_models/robots/airbot_play/meshes/link6.obj`, recover source-frame
  vertices, and retain complete disconnected mesh components ending at or before
  link6-local Z=-79.5 mm. Check the retained envelope before applying the change.
  The original cylindrical wrist remains; the transverse gripper frame is absent.
  No source OBJ, robot XML or DISCOVERSE asset is changed on disk.
- Replace the camera bracket visual with a 0.2 mm cylindrical end cap, entirely
  inside the wrist envelope. Replace the old rectangular collision proxy with a
  cylindrical envelope, radius 28.5 mm, Z=[-195.95,-79.5] mm. Registered geom names
  are reused so robosuite instance/ID mappings and reset remain valid.
- Remove the engineering backing plate and rails. Sponge rear face meets the
  wrist end face directly at link6 Z=-79.5 mm. Tool origin is right_hand-local
  [0,0,-79.5] mm. Relative to compact_v2 the sponge/TCP move 30.1 mm toward the
  wrist; all sponge visual and contact geometry move together unchanged.
- FT site is tool-local [0,0,-15] mm, exactly on the wrist/sponge interface.
  Tool TCP remains on the wiping face. The 1 mm preparation gap is recomputed
  from the actual collision envelope, not from the visual surface alone.
- Tool mass falls from 62.4 g to the inherited 30 g sponge mass, removing the
  known 32.4 g backing plate and its inertia. Original sponge inertia is retained.
- IMPORTANT: the source provides no separate gripper-housing inertial model.
  Link6 retains its original aggregate 538.55 g, center of mass and inertia;
  other arm inertials are also unchanged. This is a geometry/attachment
  simplification, not a dynamically calibrated physical gripper removal.
  Do not infer real bare-wrist force capability from these results.
- Controller, gains, torque limits, contact mapping and acceptance thresholds
  are unchanged. Old data and sanity reports must not be reused.

## Run

From the project root:

```bash
conda activate clean
python -m scripts.sim_pretrain.experiments.visualize_wiping --closeup --mu 0.9 --stiffness 1000 --width 0.02 --gain 300
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 0 --stiffness 1000 --width 0.02 --gain 300
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if conda activation is unavailable.
Headless preview, with a new filename for every export:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.visualize_wiping --closeup --no-viewer --gif runs/sim_data/direct_wrist_closeup.gif --mu 0.9 --stiffness 1000 --width 0.02 --gain 300 --speed 1
```

Add `--collisions` to inspect collision proxies; omit `--closeup` for an overview.
Playback is recorded physics, with Space to pause and R to restart. Looping does
not physically unload the sponge. Acceptance is evaluated separately by sanity,
including actual unloading. Exit 2 denotes failed acceptance, not a valid dataset.
This change does not authorize collection, training, or hardware control.

Historical `compact_v2` and `bridge_v1` profiles remain available in separate
configs. Do not switch an old layout into the new output directory or overwrite
historical reports. The current mesh source hash is included in provenance.

## Verified Results (2026-09-08)

- All 33 tests pass, including geometry, reset, unchanged arm inertials, whole-tool
  transforms, independent sensor loads, historical profiles and recorded playback.
- `archive/sim_pretrain/direct_wrist_mount.png` shows the initial assembly.
  `archive/sim_pretrain/direct_wrist_closeup.gif` was visually checked at the start, middle
  and end: 81 distinct encoded frames, 14,200 changed pixels from first to middle.
  No transverse frame, camera bracket or backing plate remains in the rendered
  assembly. The animation is mu=.9, k=1000, width=.02, gain300.
- Fresh sanity: `archive/sim_pretrain/pretrain_paper_direct_wrist_v3/sanity.json`.
  Gain100 fails free-space; gain300 passes with RMS .452535 mm and maximum
  orientation error .542584 degrees. Contact motion passes only 3/27 cases
  (mu=0, k=.5, all three widths). All 27 unload; the sensor check passes;
  no actuator saturation occurs. Overall `passed=false`, CLI exit 2.
- Additional `mu0p9_check.json` and `mu0p9_contact.npz` in that directory record
  mu=.9/k1000/width.02/gain300: forward travel 4.786724 mm, reverse travel
  -2.280773 mm, return error 7.067497 mm, maximum Y error 45.211335 mm,
  maximum orientation 7.843257 degrees. Both sliding phases maintain 100%
  geometric contact and unloading succeeds, but the motion gate fails.
  Negative reverse travel means continued net forward motion during return.
- `verification.json` records re-scoring of all 27 saved contact traces, finite
  data, matching source/config hashes, preserved robosuite/DISCOVERSE source
  assets and rejection of collection by the failed production gate.

To repeat the extra mu=.9 motion and unloading check without writing a dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import json
from scripts.shared.common import load_config
from scripts.sim_pretrain.simulation import PretrainingWipe
from scripts.sim_pretrain.collection import validate_trajectory
from scripts.sim_pretrain.acceptance import contact_motion_acceptance
cfg = load_config("configs/sim_pretrain/pretrain_paper.yaml")
env = PretrainingWipe(cfg, 300, (0.9, 1000, 0.02))
try:
    data = env.rollout()
    validate_trajectory(data)
    result = contact_motion_acceptance(data)
    result["unloaded_wrench"] = env.unload(data)
    print(json.dumps(result, indent=2))
finally:
    env.close()
raise SystemExit(0 if result["passed"] else 2)
PY
```

Assembly is complete; high-friction sliding is not solved. Do not relax the gate
or reinterpret the simplified wrist as a validated hardware model.
