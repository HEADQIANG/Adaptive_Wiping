# Real Demonstration Training

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Manual raw-data collection is now available: see [AIRBOT eight-demonstration
collection](airbot_demonstrations.md). It does not yet produce calibrated HDF5;
the schema, physical calibration and exploration requirements below still apply.

This pipeline is offline only: it does not connect to or command a robot.
The separate attended [AIRBOT exploration runner](airbot_exploration.md) now
implements the nominal press/slide motion with hardware guards. Its sensor-frame
JSONL diagnostics are not yet calibrated training HDF5 inputs; the raw-data
contract below and the eight-demonstration requirement remain unchanged.
Real raw recordings now exist; see [AIRBOT native-data training](airbot_native_training.md)
for the audited, explicitly uncalibrated offline experiment. They are not yet
calibrated inputs for this profile. The software verification results are
listed at the end of this document; they do not establish real force-control performance.

The separate [AIRBOT calibrated conversion workflow](airbot_calibrated_pipeline.md)
can now produce this profile's physical-coordinate inputs after measured
calibration evidence is supplied. Its HDF5 explicitly distinguishes the derived
100 Hz processing grid from the approximately 56 Hz original received data,
retains both streams, and records that acquisition difference from the paper.

From the repository root, use the existing `clean` environment, or install
PyTorch >= 2.6 and `requirements/real_training.txt` in a separate Python 3.10+
environment. ROS, robot SDKs, MuJoCo and robosuite are not required.

The entry point is `python -m scripts.real_training`; configuration is
`configs/real_training/real_training_paper.yaml`. The default encoder is the existing 400-epoch
continuation with friction range [0,1.2], not the paper's 200-epoch pretraining.
Its existing latent-utilization warnings remain relevant to downstream research.

Raw HDF5 logs use schema version 1, `source_kind=real`, one `explorations` group
entry and eight `demonstrations` entries. `write_raw_log` in the new data module
serializes named episodes without overwriting an existing file. Every stream
requires timestamps in a shared monotonic clock, declared nominal rate >=100 Hz,
and coverage of its complete segment including the initial sample. Gaps and
sample ages over 20 ms are rejected. Median period must agree with nominal rate
within 10 percent. No clock offset is inferred from data.

Position is in base_link meters, orientation is a unit xyzw quaternion. FT must
already be converted to N/N*m at the `ft_frame origin` in `ft_frame local` axes.
Electronic sensor bias may be calibrated out, but gravity is retained to match
the existing exploration encoder; there is no automatic per-episode zeroing.
The recorded rigid `sensor_to_ft_frame` transform is provenance, not an operation
applied a second time. Physical calibration remains the future collector's job.

Prepared demonstration times are 0.4 through 10.0 seconds. A separate initial
height is recorded at zero. Windows use FT indices k-4 through k and predict
height[k+1]-height[k], k=4..23: 20 windows per demonstration, 160 total.
Exploration times are 0.01 through 4.00 seconds, with 400 samples. The batch
filter and streaming filter share identical causal SOS processing and initial
conditions; no early histories are padded with invented measurements.

The XY branch is input Dropout(.1), Linear(5,50), reshape(25,2). The FT branch
uses two left-padded kernel-3 dilation-1 convolutions (6->25->25), each followed
by ReLU and Dropout(.1), then projects the last feature to six dimensions.
The height head is Linear(11,128), ReLU, Dropout(.1), Linear(128,1).
Convolution details and dropout placement are explicit engineering defaults.

Training is deterministic single-thread CPU float32 with seed42, Adam lr.001,
standard betas/epsilon, zero weight decay, no scheduler/early stopping. XY uses
10000 epochs, batch8; feedback uses2000 epochs, batch32. Normalization is fitted
per channel over only the training episodes. Constant channels have a recorded
mask and unit denominator, preserving out-of-range observations rather than
erasing them. Neither normalizers nor predictions are silently clipped.
Exploration encoding also fixes the thread count to one, keeping cached
embeddings consistent between the prepare and train command processes.

Final training uses all eight demonstrations. Leave-one-demonstration-out
validation fits fresh scalers and fresh models on seven complete episodes per
fold. Its held-out errors are not mixed with final training-set reconstruction.
Checkpoints preserve both models, optimizers, random states and epochs every100
epochs and at branch boundaries. Resume requires identical data, encoder,
configuration and software provenance; no optimizer-reset fallback exists.

`OfflinePolicy` loads an exported bundle without the original encoder file.
Its interfaces use batch arrays: `encode_exploration([B,400,6])->[B,5]`,
`predict_xy([B,5])->[B,25,2]` meters, and
`predict_delta_h([B,5],[B,5,6])->[B,1]` meters. The latter requires *causally
filtered*, unnormalized physical FT. Histories shorter than five frames return
None; histories longer than five are rejected. Use `make_ft_filter(sample_hz)`
for the exported causal filter; sample its outputs at the 0.4s policy times,
as documented above. This API performs no robot motion or safety limiting.

## Commands

All relative configuration paths are resolved from the repository root, not
from the configuration file's directory. Set `raw_data` to your future HDF5
log and choose a fresh `output_dir`; the source files must live outside it.

```bash
conda activate clean
python -m pip install -r requirements/real_training.txt
python -m scripts.real_training inspect --config configs/real_training/real_training_paper.yaml
python -m scripts.real_training prepare --config configs/real_training/real_training_paper.yaml
python -m scripts.real_training cross-validate --config configs/real_training/real_training_paper.yaml
python -m scripts.real_training train --config configs/real_training/real_training_paper.yaml
python -m scripts.real_training evaluate --config configs/real_training/real_training_paper.yaml
python -m scripts.real_training export --config configs/real_training/real_training_paper.yaml
```

Cross-validation is a separate diagnostic and is not used to select settings
or the final checkpoint. It trains eight additional pairs of networks using
the full configured epoch counts. It can be run independently of final training.

Without data, `inspect` and `train` exit2 with missing-input details, without
creating a formal checkpoint. To test the software now with the selected encoder:

```bash
python -m scripts.real_training smoke-test --config configs/real_training/real_training_paper.yaml
python -m unittest discover -s tests -t . -p 'test_real_training.py' -v
```

Smoke tests run two epochs per branch and all eight validation folds in a
temporary directory. `--keep-smoke-artifacts` retains this new temporary
directory for visual inspection. Data, checkpoints, policy and prediction
files retain `source_kind=synthetic`; plots state SYNTHETIC SOFTWARE TEST.
There is no formal-command flag that allows synthetic data. Provenance is
propagated and checked by the pipeline, not a claim of tamper-proof attestation
against a person deliberately rewriting source files and all integrity records.

Resume only with the same inputs, code, environment and configuration:

```bash
python -m scripts.real_training train --config configs/real_training/real_training_paper.yaml --resume
python -m scripts.real_training cross-validate --config configs/real_training/real_training_paper.yaml --resume
```

At most99 completed epochs since the last checkpoint are replayed after an
interruption. The checkpoint is authoritative if a sidecar history write was
interrupted. Completed CV folds load their checkpoints rather than retraining.
Changing paths is permitted only when the bound content remains identical;
changing data, code or training settings requires a fresh output directory.

Artifacts: `prepared.h5` and `prepared_integrity.json`; `final/training.pt`,
`history.json`, `status.json`, `evaluation.json`, `predictions.npz`,
`predictions.png`, `training.png`; separate `cross_validation/fold_01..08/`
artifacts and pooled held-out metrics; finally `policy.pt` and `export.json`.
Final evaluation reports training-set reconstruction, not unseen-test success.
Reported XY RMSE/MAE pool coordinate errors across both axes and time samples;
they are not Euclidean-distance RMSE. All reported position errors use meters.
FT-based predictions are conditioned on recorded measurements, not the forces
that would result from executing predicted actions. Exports are always offline
and `hardware_ready=false` regardless of numerical prediction error.

## Raw Data Dictionary

Root attributes: `schema_version=1`, `complete=true`, `source_kind=real`,
`metadata` as a JSON object. Episode IDs contain only ASCII letters, digits,
underscores and hyphens. Dataset values are float64; no pickle is used.

| Location | Required fields |
| --- | --- |
| metadata identity | robot_id, sensor_id, calibration_id, tcp_definition, sensor_frame: nonempty strings |
| metadata geometry | position_frame="base_link", position_units="m", orientation_order="xyzw" |
| metadata FT | channels=[Fx,Fy,Fz,Tx,Ty,Tz], ft_units=[N,N,N,N*m,N*m,N*m], ft_frame="ft_frame local", ft_reference_point="ft_frame origin" |
| metadata calibration | compensation="sensor_bias_only_gravity_retained", sensor_to_ft_frame: right-handed rigid 4x4 matrix |
| metadata timing | clock="shared_monotonic_seconds" |
| metadata exploration_protocol | press_speed_m_s=.005, press_duration_s=2, lateral_speed_m_s=.05, lateral_duration_each_s=1 |
| explorations/ID attributes | sponge_id="normal", complete=true, start_time, ft_hz, source_kind="real" |
| explorations/ID datasets | ft_time[N], ft[N,6], covering start_time through start_time+4s |
| demonstrations/ID attributes | sponge_id="normal", complete=true, start_time, ft_hz, pose_hz, exploration_id, surface_id, source_kind="real" |
| demonstrations/ID datasets | ft_time[N], ft[N,6], pose_time[M], tcp_position[M,3], tcp_quaternion[M,4], covering start_time through start_time+10s |

All eight `exploration_id` values must refer to the sole exploration; all
`surface_id` values must be identical. N and M may differ. Preserve the actual
measured position rather than a controller target. A valid quaternion is not
evidence that the physical TCP and force frame have been correctly calibrated.

Use the helper from the future collector after assembling arrays:

```python
from scripts.real_training import write_raw_log

# metadata follows the table; actual calibrated arrays come from the collector.
# Each episode dictionary contains an "attrs" dict and the named numeric arrays.
write_raw_log("runs/real_training/real_training/raw.h5", metadata,
              explorations={"normal_exp": exploration_episode},
              demonstrations=demonstration_episodes, source_kind="real")
```

This helper will not overwrite an existing file. It marks the file complete
only after all groups are written; inspect still checks individual episode
completion and all numerical requirements. An interrupted/incomplete file
must not be relabeled or padded to bypass inspection.

## Failure Handling And Limits

- Missing data: collect and serialize the required logs, then inspect again.
- Frame/unit/compensation mismatch: fix and document the collector's conversion;
  do not rename metadata without transforming the actual measurements. A wrench
  translation requires the torque lever-arm term, not just rotation.
- Timestamp gaps: inspect sensor/robot synchronization and recollect invalid
  episodes. The importer does not fill gaps, infer clock drift or use future FT.
- Source encoder or normalization warnings: retain the diagnostic evidence.
  Real exploration beyond the simulation scaler range remains unmodified.
- Non-finite training: execution stops before replacing the last finite
  checkpoint. Inspect the data and numerical cause rather than relaxing gates.
  Resume also validates saved model/optimizer finiteness and epoch/history consistency.
- Existing output or stale evaluation: use the appropriate resume command,
  rerun evaluation for the same completed checkpoint, or choose a new output
  directory for a genuinely changed experiment. Export refuses to overwrite.
- Hardware deployment remains a separate task requiring real sensor/robot
  calibration, independently tested safety stops and supervised low-force tests.
  The paper's instruction to push as hard as possible is not an AIRBOT safety
  specification and is not implemented as a command or acceptance condition.

## Verification

Verified on 2026-09-09 in the existing `clean` environment: the full121-test
regression suite passed in70.959s, including20 focused real-training tests.
The selected 400-epoch encoder's SHA256 remains
`9f2464c006aa5734979ede12b0bc71df344dad65cc22ab59c6502d351482c9e3`.
No formal real-data training directory or checkpoint was created. Software is
ready for compliant real logs; hardware operation and force-control performance
remain unverified. The generated prediction/training plots were visually checked.

The no-hardware acceptance consists of the focused test module, the existing
repository regression suite, and a smoke test using the selected 400-epoch
encoder. Smoke runs use only generated logs in a temporary directory, two
epochs per branch, eight leave-one-out folds and160 feedback windows. They
check export roundtrip and retain the original encoder content hash.

Focused tests also cover asynchronous200 Hz FT, causal/latest-past sampling,
stream/batch equivalence, future-data independence, reset between episodes,
scaler isolation, malformed logs, synthetic-data rejection, unchanged frozen
weights, resume equality across both branches, completed CV resume without
optimizer steps, corrupted checkpoint rejection and missing-data exits.

To run the full regression suite (this includes existing simulation tests and
therefore uses the existing `clean` environment with its simulation dependencies):

```bash
python -m unittest discover -s tests -v
```

Existing simulation tests emit optional robosuite/Mink dependency warnings but
pass in `clean`. A read-only home directory may make Matplotlib use a temporary
cache; this does not affect the saved figures or training results.
