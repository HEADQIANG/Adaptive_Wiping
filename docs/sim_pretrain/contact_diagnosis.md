# Contact Control-Chain Diagnosis (2026-09-08)

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## Scope And Reproduction

Completed simulation-only diagnosis with the current DISCOVERSE ordinary inertia
profile, explicit armature=0, gain=300, velocity feedforward enabled, unchanged
2 ms physics / 100 Hz commands, 10 mm/s press and 50 mm/s slide. No controller,
robot XML, contact parameter mapping, torque limits or collection gate changed.
The restored inertias are retained; this is not a reason to restore the previous
unconfigured robosuite armatures. No full dataset or VAE training was started.

From the project root:

```bash
conda activate clean
python -m unittest discover -s tests -t . -p 'test_contact_diagnosis.py' -v
python -m unittest discover -s tests -t . -p 'test_pretraining.py' -v
python -m scripts.sim_pretrain.experiments.diagnose_contact --config configs/sim_pretrain/pretrain_paper.yaml --output runs/sim_pretrain/contact_diagnosis_repeat
```

The verified interpreter is `/home/wp/miniconda3/envs/clean/bin/python` if conda
activation is unavailable. Use a previously nonexistent output directory; the
completed results are in `archive/sim_pretrain/contact_diagnosis`. No hardware SDK is invoked.
Missing private macros, robosuite_models and Mink warnings are nonblocking for
this explicitly configured AIRBOT IK. Matplotlib can use a temporary cache if
the default user cache directory is not writable.

The directory contains a JSON report, a control-chain PNG and four NPZ files per
case: uninstrumented baseline, instrumented 100 Hz rollout, 500 Hz physics trace
and individual contacts. Definitions and column order are in `pretraining.md`.

## Results

All cases use solimp width=0.02. The Y range is full-rollout max minus min, not
total path length or endpoint error.

| mu | Direct k | Actual Y range (mm) | Max position error (mm) | Max orientation error (deg) | IK-limited slide commands / 200 | Torque-clipped substeps |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1000 | 50.976 | 18.191 | 0.870 | 200 | 0 |
| 3.5 | 1000 | 0.806 | 55.151 | 9.643 | 200 | 0 |
| 3.5 | 0.5 | 0.638 | 51.507 | 10.055 | 182 | 0 |

The mu=0 case uses MuJoCo's effective minimum friction 1e-5. Its position error
is largely the persistent vertical contact offset, not failure to traverse Y.
The high-friction cases fail the intended sliding motion, not merely a transient
reversal criterion. The different maxima of 55.151 mm and 10.055 deg belong to
different k values, not one identical rollout.

## IK Limits The Error Sent Downstream

The local solver (`scripts/sim_pretrain/robosuite/robosuite/utils/ik_utils.py`, solve) computes a
damped inverse correction and uniformly scales all six joint velocities when
any component exceeds 2 rad/s. It then starts from ACTUAL joint position at
every command, not the preceding desired joint position:

```text
dq_raw = damped_inverse(J) @ [0.95 * position_error / 0.01,
                             0.95 * orientation_error / 0.01] + nullspace_term
scale = min(1, 2 / max(abs(dq_raw)))
q_goal = clip_to_joint_ranges(q_actual + 0.01 * scale * dq_raw)
```

The actual solver outputs and the reconstructed expressions match at every
command. At command instants, each joint position error is at most 0.02 rad
(about 1.146 deg); it can change during the following physics substeps. Increasing
the Cartesian error no longer proportionally increases this bounded joint error.
This is a controller architecture limitation under sustained load, not an
actuator torque clipping event.

At physics t=2.99 s, the current command is the endpoint intended for t=3.00 s:

| Quantity | mu=0, k=1000 | mu=3.5, k=1000 |
| --- | --- | --- |
| Final Cartesian target Y (mm) | 50.000 | 50.000 |
| Actual TCP Y before integration (mm) | 49.215 | 0.672 |
| Y from forward kinematics of q_goal (mm) | 49.463 | 7.064 |
| IK uniform scale | 0.35073 | 0.13627 |
| Largest raw IK correction (rad/s) | 5.702 | 14.676 |
| Largest bounded IK correction (rad/s) | 2.000 | 2.000 |

For high friction, joint 4 is the limiting component at this instant. Splitting
the same damped inverse into position and orientation inputs gives joint-4 raw
contributions 0.154 and 14.522 rad/s respectively. The orientation correction
therefore scales down all translation-related corrections too. This is a
same-state algebraic decomposition, not a test with orientation control removed.
Peak raw correction over this run is 15.815 rad/s (minimum scale 0.12646).

IK limits alone are not sufficient to explain failure: the low-friction case
also limits throughout sliding and still moves laterally. The observed failure
combines sustained contact error, the bounded actual-state-based IK reference,
finite contact-control authority, and high friction / orientation coupling.

## Joint Controller And Feedforward

The actual torque decomposition is verified at all 2000 substeps per run:

```text
tau = M @ (Kp * (q_goal - q) - Kd * dq + Kd * dq_goal) + bias
dq_goal = difference_of_successive_q_goals / 0.01
```

Kp=300 is a mass-scaled joint-controller gain, not Cartesian stiffness of
300 N/m or an unscaled torque stiffness of 300 Nm/rad. The bounded joint errors
produce relatively small restoring torques at the current mass matrix.

For mu=3.5/k=1000 at t=2.99, joint-1 P/D/FF/bias terms are approximately
0.34483/-0.82548/+0.81376/+0.00037 Nm, totaling 0.33347 Nm. Joint-4 P is
0.14014 Nm, and total torque is 0.01499 Nm after the other terms including bias.
The full-run peak absolute requested torques are
`[0.5032, 6.5052, 3.8312, 0.1244, 0.0290, 0.0060]` Nm, below the unchanged
`[10, 10, 10, 5, 5, 5]` limits. The arm is not exhausting those torque limits.

The feedforward path is working and is not clipped, but its velocity reference
is not the bounded IK correction velocity. At the same instant, joint-4 IK
correction is +2 rad/s, while the difference of successive joint goals is
-0.16891 rad/s and actual velocity is -0.16851 rad/s. Because each q_goal starts
from actual q, this difference includes actual motion under contact. In the
recorded state, it largely cancels velocity damping despite unwanted tool tilt.
It is not a sustained friction-force compensation term. This establishes the
reference coupling; whether disabling or replacing it improves the complete
contact trajectory requires a separate ablation, not inference from subtraction.

## Contact Loads And Tilt

For mu=3.5/k=1000, averaged over physics t in [2.8,3.0):

- Contact force ON the tool in world coordinates is approximately
  `[-0.013, -0.663, +0.965]` N.
- Controller feedback torque expressed as an equivalent TCP wrench has
  Fy=+0.763 N and Fz=-0.958 N. This representation excludes bias and is not a
  force command or a sensor reading; acceleration and passive forces explain
  why it is not exactly opposite contact force.
- `mu * sum(Fnormal)` is about 3.376 N. This is an aggregate translational upper
  bound, not proof that every patch sticks or an exact net-force threshold when
  moments and unequal contact loading are present. Some patches do reach their
  individual friction limit; the maximum per-contact ratio alone is misleading.

At t=2.99, the normal-force-weighted contact point is 58.124 mm ahead of the TCP
in world Y, near the tool edge. Only 13 contacts carry more than 1e-9 N, versus
70 in the low-friction comparison. Contact point velocities can be reconstructed
without another simulation:

```text
omega = J_rot @ dq
v_point = J_pos @ dq + cross(omega, contact_position - TCP_position)
weighted_slide_speed = sum(Fnormal * norm(v_point_XY)) / sum(Fnormal)
```

This gives 0.342 mm/s for high friction versus 50.125 mm/s for low friction at
this instant. The table is fixed, so these are relative tangential speeds.
High-friction TCP vz is +10.807 mm/s and world angular vx is -0.16855 rad/s:
the tool is lifting/tilting with a nearly stationary loaded edge rather than
performing the intended 50 mm/s translation. This is not a perfectly static
contact or proof of zero local slip.

At t=3.00 after integration, the high-friction position error is
`[+0.347, -49.312, +24.696]` mm, whose norm is 55.151 mm. The vertical error is
not all material compression. Even mu=0 leaves approximately 17.25 mm vertical
offset. A commanded 20 mm downward displacement does not establish that the
rigid tool / soft-contact model actually compresses by 20 mm.

The raw FT sensor measures local support wrench, not world contact force.
For example, low-friction mean local sensor Fz is +1.998 N while world contact
normal is +2.292 N. Coordinate orientation, the 0.03 kg tool's approximately
0.294 N gravity, and dynamics matter. Sensor readings were not tared or altered.

## Verification And Limits

- All three instrumented rollouts have exactly zero position and raw-FT
  difference from separately run uninstrumented references.
- One new contact test plus all 18 existing pretraining tests pass. The new test
  covers contact ordering/sign reversal, moment-arm shifting, force mapping to
  generalized coordinates, and the empty-contact case.
- Summed tool contact generalized force matches qfrc_constraint to below
  1.8e-15 Nm across runs. No joint-range clipping occurred, so no unaccounted
  joint-stop constraint explains the loads.
- Maximum full dynamics balance residual is 8.74e-6 Nm (soft case); high-k
  cases are below 1.10e-8 Nm. No large missing-force term was identified.
- Minimum singular value of the recorded mixed-unit spatial Jacobian stays
  above 0.1328; no rank-loss event or discontinuous IK branch change is indicated.
  Singular values mix translation and rotation units and are not an absolute
  universal singularity criterion.
- Feedforward limiting and actuator clipping are both zero in all three runs.
  Joint-goal command jumps remain below 0.00291 rad; the 2 rad/s IK correction
  limiter is a distinct mechanism and was active as reported above.

These findings identify concrete reference and load-path limitations. They do
not establish a unique root cause through controlled controller ablations or
validate any proposed replacement controller. Parameters are simulation
assumptions, not measured AIRBOT hardware capability or sponge material data.

## Recommended Next Work (Not Implemented)

1. Separate nominal trajectory velocity feedforward from contact-error feedback.
   Compare a reference path that does not continually reduce persistent Cartesian
   error to a small offset from actual q. Do not blindly remove the speed limit
   or accumulate an unreachable reference without anti-windup / feasibility checks.
2. With the existing torque limits, compare contact-appropriate Cartesian pose
   impedance or a revised IK/joint reference scheme. Measure actual lateral
   travel, orientation, normal load and transient saturation before gain selection.
   Increasing Kp alone is not yet a verified solution.
3. Agree on and add explicit contact-motion acceptance criteria, including actual
   forward/backward travel and orientation. Separate expected normal compliance
   from lateral tracking failure; do not simply apply free-space XYZ error limits.
4. Rerun free-space and all 27 contact checks before collecting the fixed dataset.
   The existing automatic gate is OPEN because it does not threshold contact
   tracking. This diagnostic does not close it; collection remains discouraged.

Do not lower mu, enlarge actuator limits, restore artificial armature, discard
the hard samples, or change paper parameter ranges to conceal this failure.
