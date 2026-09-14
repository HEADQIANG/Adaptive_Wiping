# Dynamic Wiping Visualization

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

The default configuration places the robot directly on the table and attaches
the sponge directly beneath the cylindrical wrist, without the transverse gripper
frame, camera bracket or backing plate. Geometry, inertial limitations and current
commands are documented in [direct_wrist.md](direct_wrist.md); the previous
backing-plate layout is in [tabletop_compact.md](tabletop_compact.md).
The viewer itself does not change
installation geometry; it renders whichever model the selected config builds.

Run from the Adaptive_Wiping project root in a desktop terminal:

```bash
conda activate clean
python -m scripts.sim_pretrain.experiments.visualize_wiping
python -m scripts.sim_pretrain.experiments.visualize_wiping --closeup --mu 0.9 --stiffness 1000 --width 0.02 --gain 300
```

If conda activation is unavailable, use `/home/wp/miniconda3/envs/clean/bin/python`.
The script computes a fresh rollout with the current `PretrainingWipe` and then
replays it in an independent MuJoCo data object. Preparation happens before the
window opens. The displayed 4-second exploration is press (2 s), +Y slide (1 s)
and -Y slide (1 s). Half-speed playback and looping are the defaults, with a
short frozen end pose between repetitions. A repeat resets playback, not physics;
it is not a simulated lift-and-return motion. No hardware is accessed.

Space pauses/resumes; R restarts. Use the native viewer mouse controls to rotate,
pan and zoom. Close the window to exit. `--once` exits after one playback.
`--speed 1` uses nominal playback speed; allowed range is 0.1 to 4. This option
never changes simulation timestep, command timing or exploration velocities.

The default parameters are mu=1.75, k=500.25, width=0.16 and gain=300. To compare
the diagnosed low/high-friction cases with otherwise identical settings:

```bash
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 0 --stiffness 1000 --width 0.02
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 3.5 --stiffness 1000 --width 0.02
```

High friction is expected to show the currently observed sticking and tilt,
not a corrected or artificially animated target-following motion. Defaults use
`configs/sim_pretrain/pretrain_paper.yaml`; `--config` can explicitly select another profile.
The script respects the current configuration, including reference mode if set.
It never changes the controller, tool installation offset, source model or gate,
and does not write training data. Metrics printed before playback describe the
real rollout, not the display interpolation or wall-clock playback speed.

Collision group 0 is hidden by default, showing the original robot meshes and
gray tool appearance. `--collisions` shows green arm / blue tool collision shapes.
Hiding collisions does not remove them from physics or repair missing visible
mounting geometry. The direct wrist profile changes the model before simulation;
the viewer does not move the sponge independently or hide colliders to fake fit.

Red markers/path show the commanded TCP, cyan the actual TCP, and a yellow arrow
shows resultant world contact force ON the tool, with a visual scale of 2 cm/N
capped at 12 cm. To avoid occlusion it is drawn beside TCP at world offset
`[0,-0.10,0.04]` m, with a gray line back to its true reference point. The offset
does not represent a tool part or an actual force application point.
This is not the raw local FT sensor or a contact moment
visualization. The target can go below the opaque tabletop during pressing; it
is not moved upward just to make the marker visible. Paths show past samples.
The playback data is display-only; viewer perturbations are not a force-control
experiment and do not propagate back to the recorded simulation.

## GIF Export

Pillow (already available in `clean`) supports optional animated GIF export.
Desktop OpenGL can use the default backend; for headless export select EGL before
Python imports MuJoCo:

```bash
MUJOCO_GL=egl python -m scripts.sim_pretrain.experiments.visualize_wiping --no-viewer --gif runs/sim_pretrain/wiping_preview.gif
```

Choose a new GIF filename for repeat exports; existing files are not overwritten.
GIF output is 640x480 at 20 fps and loops. The export speed follows `--speed`.
The GIF uses a shared palette with reserved annotation colors so thin force
arrows are not quantized into the gray background.
Use `--gif ...` without `--no-viewer` to export then open the interactive window
on a desktop with a compatible OpenGL backend. EGL availability depends on the
machine driver; it is not required for the normal desktop viewer command.

## Verification

```bash
python -m unittest discover -s tests -t . -p 'test_wiping_visualization.py' -v
```

The test compares fresh recorded and unrecorded high-friction rollouts exactly
for position and raw FT, checks all 401 frames (start plus 400 samples) are present,
checks representative replay TCP positions, and verifies that replay does not
modify live robot state. It also checks overlay construction and collision display
options. Existing pretraining and contact diagnostics remain independent.

Custom overlay geometry is display-only (`mjCAT_DECOR`), not a model collision
shape. No extra body, inertia, tool connector or constraint is added.

Verified on 2026-09-08: the new visualization regression and all 18 existing
pretraining tests passed. The desktop viewer completed a high-friction playback
with `--once --speed 2` and exited normally. A nonfatal GLFW Wayland window-position
warning was emitted. EGL export succeeded; the final low-friction preview is
`archive/sim_pretrain/wiping_visualization_final.gif`. Multiple frames were inspected, with
nonblank pixels, robot/tool motion and a visible yellow contact-force arrow.
Earlier `wiping_preview*.gif` and `wiping_visualization.gif` files are intermediate
display checks, not training data; use the final preview above.
