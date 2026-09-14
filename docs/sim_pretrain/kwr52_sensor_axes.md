# KWR52 Sensor Axis Alignment

## Convention And Limits

The operator defines left at robot joint zero, looking from the base toward
the tool. The four-screw cable cover faces left. Using the manufacturer's
nominal KWR52 drawing, the cover outward normal lies 30 degrees from +X
toward +Y. This gives sensor +X left/down, +Y left/up and +Z toward the sponge:

```text
sensor +X = cos(30 deg) * left - sin(30 deg) * up
sensor +Y = sin(30 deg) * left + cos(30 deg) * up
sensor +Z = forward
```

Source: https://kunweitech.com/products/609.html and
https://kunweitech.com/upload/20250930/202509301107107542.jpg.
The cover-to-hole-group alignment follows the drawing geometry. The public
page lists A/B/G, while the installed sensor is KWR52-TiS; this is the
assumed nominal convention, not serial-specific factory calibration. Confirm
the TiS axes and channel signs with known-direction loading before hardware use.

At simulation q=0, forward/left/up are approximately base +X/+Y/+Z.
The existing tool axes are right/down/forward. Consequently the fixed rotation
is `tool_from_sensor = Rz(+150 deg)`, with MuJoCo wxyz quaternion
`[0.258819045103, 0, 0, 0.965925826289]`. The source robot quaternions have
small rounding errors; zero-pose tests allow 1e-5 directional error.
This is a local attachment rotation, not a constant world/SDK transform.

Only `ft_frame` orientation changes. Tool geometry, TCP, sensor origin,
contacts, masses, inertias, controller and gravity remain unchanged. Both force
and torque are reported in the new axes by MuJoCo; no second channel rotation
or sign flip is applied. Existing `sensor_quat_xyzw` records the actual sensor
orientation, which now differs from the TCP orientation. MuJoCo still reports
parent-on-child support wrench; matching hardware signs requires a separate
known-load check. This change does not align the physical sensor origin,
calibrate electronic bias, or certify hardware readiness.

## Configuration And Compatibility

Both current `pretrain_paper*.yaml` files select
`simulation.ft_frame_profile: kwr52_left_v1`. The output directories are now
`runs/sim_pretrain/pretrain_paper_kwr52_left_v1` and
`runs/sim_pretrain/normal_kwr52_left_v1` respectively.
Missing `ft_frame_profile` or explicit `legacy` preserves the old orientation.
UR5e comparisons explicitly retain `legacy`. The profile is independent of
the AIRBOT tool-mount geometry. Original XML/mesh assets are not edited.

Old FT data, normalization statistics, sanity reports and checkpoints must not
be mixed with the new axes or resumed into these directories. The config hash
and existing source provenance change. No automatic historical-data conversion
or hardware calibration flag update is performed.

## Run And Verify

From the repository root:

```bash
conda activate clean
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest tests.sim_pretrain.test_sensor_axes -v
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain.experiments.visualize_wiping \
  --config configs/sim_pretrain/pretrain_paper.yaml --closeup --mu 0 --stiffness 1000 --width 0.02
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if conda activation is unavailable.
The viewer command is unchanged; geometry looks unchanged because only the
measurement axes changed. Tests inspect the compiled sensor frame directly.
They check zero-pose directions, moving-frame rotation, force and torque
conversion, central/eccentric loads, reset and unchanged physical dynamics.
Run a fresh sanity before collection; axis alignment does not fix contact
tracking or relax any existing acceptance criteria. Do not start hardware motion
or training solely because these offline coordinate tests pass.

## Verification Results (2026-09-11)

The focused regression passed all 30 tests:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest tests.sim_pretrain.test_sensor_axes \
  tests.sim_pretrain.test_direct_wrist tests.sim_pretrain.test_tool_mount \
  tests.sim_pretrain.test_tabletop_compact tests.sim_pretrain.test_pretraining \
  tests.sim_pretrain.test_ur5e_comparison tests.sim_pretrain.test_wiping_visualization -v
```

An in-memory four-second rollout comparison (400 samples, mu=0.9,
stiffness=1000, width=0.02, gain=300) found zero TCP/joint trajectory difference
and maximum wrench rotation residual below 1.8e-12 in per-channel SI units.
No production training dataset was collected.

Full unittest discovery ran 399 tests with 2 failures, 3 errors and 4 skips.
Four unsuccessful calibration tests use archived real exploration data rejected
by the current press-protocol check, before simulation frame handling. The
remaining failure is the asset inventory check: `asserts/connector/p6_ft_fl.STL`
is absent from the manifest. These unrelated data/assets were left unchanged.
This is not a full-suite pass or physical calibration acceptance; a fresh
production sanity run and hardware known-load verification remain required.
