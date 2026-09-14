# Attended Unloaded Pose Capture

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

## Interactive Recording (Recommended)

From the repository root, run in a NEW terminal; keep the existing
`AIRBOT Attended Calibration Teaching` terminal open. The teaching terminal
maintains the already-authorized gravity-compensation session. This separate
recorder never requests a mode change and does not replace that session.

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.force_sensor.tools.record_unloaded_session
```

Commands: `r` then Enter confirms the displayed contact-free/stationary/support
conditions and records3s; `s` prints progress; `q` closes only this recorder.
Blank Enter does not confirm. Each capture contains61 raw FT/pose observations.
Numbering continues after every existing pose file, including incomplete files;
no recording is overwritten. Pose001 and pose002 already exist and passed the
single-pose checks, so the next default capture is pose003. Pose002 differs from
pose001 by approximately106.4degrees in SDK-expressed gravity direction.

Between captures, reposition only under the previously approved supported manual
procedure, with sponge clear of contact and no force on the sensor-side assembly
during measurement. Plan6-8 distinct safe orientations plus a repeat of the first.
The script reports gravity-direction differences and warns about near-duplicates
and rank deficiency; these are coverage diagnostics, not a fitted calibration.
Do not force motion to reach a requested angle. Pure translation or rotation
about gravity contributes little new information. Existing raw samples are never
silently excluded, relabeled, tared or used to approve production flags.

On completion: `q` in the recorder does NOT exit gravity compensation. Continue
supporting the arm; only after support is secure, type `q` in the SEPARATE teaching
terminal to request idle. Idle does not hold the arm. A second person handling
the keyboard is recommended. Never close the teaching terminal while unsupported.

Read saved progress without opening any device:

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.force_sensor.tools.record_unloaded_session --status
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -t . -p 'test_unloaded_session.py' -v
```

Use `--directory /path/to/new/session` only for a genuinely separate setup/session.
Do not mix recordings after a mount/sponge/tare change. A directory lock rejects
another instance of this interactive recorder; do not run the single-pose tool
concurrently. Sensor-port ownership is checked separately before each capture.

## Single-Pose Tool

This tool observes one already-positioned robot pose for3s. It never acquires
control, enables the arm, changes controller, moves, homes or sends a robot stop.
It starts/stops the manufacturer serial stream but never calls tare. Do not run
it alongside another force reader. It rejects an occupied port before connecting.

Before any eligible calibration capture, the onsite operator must confirm that
the same installed tool/sponge is fully clear of external contact, nobody is
touching/loading the sensor-side assembly, the robot is stationary and supported
by its existing safe mode, and trained supervision/emergency stop are available.
The script does not authorize manual repositioning or choose a safe motion mode.
Do not move an idle arm by force or assume it holds its weight automatically.

Run from the repository root:

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover \
  -s tests -p 'test_unloaded_pose_capture.py' -v
# Connection observation only: no contact-free claim.
/home/wp/miniconda3/envs/clean/bin/python -m scripts.force_sensor.tools.capture_unloaded_pose \
  --output runs/force_sensor/real_robot/unloaded_calibration/connection_001.json
# Only after explicit current onsite confirmation and a stationary pose:
/home/wp/miniconda3/envs/clean/bin/python -m scripts.force_sensor.tools.capture_unloaded_pose \
  --contact-free-confirmed \
  --output runs/force_sensor/real_robot/unloaded_calibration/pose_001.json
```

Every output path must be new. Failures retain incomplete records. The3s record
contains61 asynchronous robot/raw-FT observations at nominal20Hz, with distinct
receive-time counts, ages, mean/std and robot stationarity checks. This is not a
full-rate FT recording and does not claim hardware timestamp synchronization.
No contact-free status is inferred from low variance or a stable pose.

For subsequent poses, follow an individually agreed, supervised positioning
procedure. Record approximately6-8 sufficiently distinct safe orientations plus
a repeated initial orientation; a later conditioning check determines whether
the samples identify gravity and bias. The first pose alone is never a completed
calibration. This tool does not fit parameters or write production calibration
JSON, training inputs or deployment verification flags. Preserve mount and tare
state across the entire session; record any operator-side change explicitly.

## Connection Observation Completed

`archive/force_sensor/real_robot/unloaded_calibration/connection_001.json` contains61 valid
stationary observations and61 distinct force receive timestamps. Maximum force
age was17.70ms; raw mean force norm25.437N. Controller remained `idle`, service
valid, motor codes [0,0,0,1,1,1]. These readings are not evidence of physical
support, absence of contact, mass or calibrated zero. Current contact-free status
was not confirmed, so `eligible_unloaded_pose=false` and `complete_calibration=false`.
The serial stream and client were closed after observation; no process from the
capture remains running. Three synthetic assessment tests passed.

Before pose001, obtain current onsite confirmation of reliable support, a fully
suspended/untouched sensor-side tool, and operator readiness. Do not directly
drag the idle arm or remove support based on this observation. No controller
switch or movement is included in this tool.

## Offline Fit And Validation

After recording, run these commands without connecting to hardware:

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -t . -p 'test_unloaded_fit.py' -v
/home/wp/miniconda3/envs/clean/bin/python -m scripts.force_sensor.tools.fit_unloaded_calibration \
  --output runs/force_sensor/unloaded_calibration_fit_v1.json
```

Output must be new. Original pose files are hashed and never changed. Failed
stationarity records are retained but do not enter fitting. Each accepted pose
contributes one mean wrench and one mean SDK orientation, with equal pose weight.
The model assumes SDK Z-up and constant mount/bias: F=b+sign*w*R*u, where u is
gravity direction expressed in SDK end axes, R is a proper rotation and w is
sensed weight. Both global force signs are diagnostic hypotheses; they do not
verify manufacturer axis convention. Torque fits bias plus COM cross gravity.
An unconstrained affine force fit is retained only as a diagnostic comparison.

Validation excludes whole connected groups of gravity directions closer than
10degrees, so repeats cannot leak into a held-out orientation group. Report
conditioning, held-out errors, fitted mass/bias variability and the first/last
repeat discrepancy before accepting a physical interpretation. Mass, rotation,
COM and bias are candidates only; the script never approves production or writes
calibration flags. A good training residual alone is insufficient. A repeat at a
slightly different angle is compared against the fitted gravity change, not
assumed to have exactly the same true force. Unexplained change may include drift,
load/contact, mounting change or model error; it is not uniquely electronic drift.

## First Multi-Pose Fit Result

`archive/force_sensor/unloaded_calibration_fit_v1.json` records the first completed diagnostic.
Ten pose files were retained. Eight passed stationary/freshness checks; pose003
failed J5 peak speed0.051282rad/s and pose009 failed J4 peak0.197802rad/s, versus
the existing0.05rad/s gate. These files were not deleted or silently accepted.
The accepted orientations form four connected10degree gravity-direction groups:
[001,010], [002], [004,005,006], [007,008]. Thus eight accepted files are not eight
independently distributed orientations. Full design condition number is45.44;
leave-group-out values reach117.26.

The unapproved rigid-load candidate gives sensed weight1.133N (mass equivalent
0.1155kg), force bias[-8.658,17.145,17.794]N and training force-vector RMSE0.265N.
These are fitted candidates, not independently verified mass/electronic zero or
permission to subtract that bias from historical training data. Held-out-group
force-vector RMSEs are1.264,0.740,0.355 and0.106N. The first/last gravity-direction
difference is2.147degrees; after accounting for the candidate gravity change,
their unexplained force difference is0.601N. Constant bias/rigid-load assumptions
are not sufficiently supported to publish a production calibration.

Before gathering more random orientations, check repeatability: only with safe
support, the same untouched suspended tool and no change to mode/mount/tare,
hold one pose, allow it to settle, and use the existing recorder for three new
captures, roughly10s apart, without repositioning between them. The next numbers
are pose011 onward. Do not remove support or force the idle arm. This isolates
short-term repeatability from orientation changes; it cannot by itself prove
absence of contact, thermal stability or the source of any drift. Recheck the
result before deciding how many additional distinct safe orientations are needed.

Three new synthetic fit tests passed (known signed rigid load/COM, degenerate
orientation rejection, repeated-direction grouping). No hardware command, model
replacement or production calibration update was made during this offline fit.
