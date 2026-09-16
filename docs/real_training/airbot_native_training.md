# AIRBOT collected-data training and deployment

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This run uses the real completed exploration `exploration_ft_007.jsonl` and
the eight accepted demonstrations in `session_record_only_002`. Original logs
are never edited. The importer verifies accepted-file SHA256, complete segment
coverage, actual received timestamps, finite values and 20 ms freshness/gaps.

Important: this is a native-frame offline experiment, not a calibrated paper
reproduction or hardware acceptance. Poses are the SDK configured end frame,
not a calibrated sponge TCP. Forces are raw sensor-origin loads including
gravity and unknown electronic bias. No sensor-to-simulation transform has
been measured. The frozen encoder therefore has an explicitly unverified
physical input mapping. The calibrated `real_training_paper.yaml` contract
remains unchanged and rejects these native logs.

The 100 Hz recording loop receives new `latest()` FT values at approximately
56 Hz. Repeated timestamps are deduplicated, not relabeled as independent
100 Hz measurements. A causal latest-past hold (maximum age 20 ms) creates
the 100 Hz processing grid. Demonstration FT uses a causal second-order 1 Hz
Butterworth filter and is sampled at 0.4..10 s. Exploration uses the frozen
encoder's existing second-order 10 Hz offline Butterworth preprocessing.
Neither future observations nor fabricated padding are used for causal holds.

## Run from the repository root

```bash
conda activate clean
python -m pip install -r requirements/real_training.txt
python -m scripts.real_training.import_airbot --audit-only
python -m scripts.real_training.import_airbot
python -m scripts.real_training inspect --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training prepare --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training train --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training evaluate --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training export --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training cross-validate --config configs/real_training/real_training_airbot_native.yaml
```

The selected encoder is `archive/sim_pretrain/two_control_pretraining_v2/normal/encoder.pt`,
200 epochs with friction [0,3.5], matching the paper's pretraining epoch/range
instead of the prior default 400-epoch [0,1.2] continuation. It is frozen.
Downstream architecture, Adam learning rate .001, XY 10000 epochs, feedback
2000 epochs, dropout .1, five-frame FT history, normalization [0,.9], eight
demonstrations and 25 output positions follow the paper. Kernel size 3,
dilation 1, batch sizes 8/32, seed 42 and filter cutoffs are documented local
defaults where the paper does not specify them. Twenty valid next-step FT
windows per demonstration give 160 windows, without invented early histories.

Results go to `archive/real_training/real_training_airbot_native_v1`: prepared data,
checkpoints, predictions, metrics, plots and self-contained `policy.pt`.
`cross_validation` holds eight independently trained leave-one-demo-out folds;
it measures prediction with recorded FT, not closed-loop force control.
Final evaluation is training-set reconstruction. A single sponge cannot
establish generalization to unseen sponge properties.

Import/prepare/export refuse overwrite. Resume interrupted training with the
same config, source code and environment using `train ... --resume` or
`cross-validate ... --resume`. Do not modify code during a training run.

## Verification

```bash
python -m unittest discover -s tests -t . -p 'test_real_training.py' -v
python -m unittest discover -s tests -t . -p 'test_airbot_native_training.py' -v
python -m scripts.real_training.tools.summarize_airbot_training
```

The importer and all commands above are offline and do not connect to hardware.
Actual frame calibration, onsite swept-path verification, approved force limits, emergency-stop tests
and supervised commissioning remain required before physical deployment.

## Deployment adapter

Measured-calibration conversion and the complete calibrated-data commands are
now documented in [airbot_calibrated_pipeline.md](airbot_calibrated_pipeline.md).
The original native-frame model remains unchanged and is still not executable.

```bash
python -m scripts.real_deploy replay --output runs/real_training/real_training_airbot_native_v1/deployment_replay
python -m scripts.real_deploy preflight --config configs/real_deploy/airbot_deployment.json
python -m unittest discover -s tests -t . -p 'test_airbot_deploy.py' -v
```

Replay uses all eight original received FT streams at a causal 100 Hz grid.
It verifies 20 next-step predictions per episode against batch inference and
saves target traces and numerical parity. These targets are driven by recorded
measurements, not measurements from executing the targets. Preflight is offline
and currently exits 2: the native model is intentionally NOT executable. There
is no `--force` or uncalibrated execution bypass.

`PolicyLoop` fixes the first five vertical waypoints at the initial height while
collecting actual history. At 2.0 s the first learned height increment predicts
the 2.4 s waypoint; the final increment predicts 10.0 s. XY follows 25 absolute
waypoints, linearly interpolated at 100 Hz. This warm-up/interpolation choice
is an engineering deployment default, not specified by the paper. At runtime
oversized predictions stop rather than being silently clipped or retried.

Before using the sensor-calibrated `run`, obtain an `airbot_sensor_calibrated_offline` policy without the
source encoder quality or out-of-range warnings, and fill the measured/approved
fields in `configs/real_deploy/airbot_deployment.json`. Do not change native metadata or
warning flags to make preflight pass. In particular, supply matching calibration
identity, exact reviewed policy hash, measured `sensor_to_ft_frame`, electronic bias,
`initial_sdk_position_m`, fixed SDK orientation, and
site-approved position/speed/current/force/torque limits. Transforms are 4x4
homogeneous matrices; `sensor_to_ft_frame` maps sensor coordinates into FT axes
and includes the torque lever-arm term. Bias is subtracted before transformation;
gravity is retained. Project XYZ workspace limits were removed on 2026-09-12;
onsite verification must cover the entire robot/tool/cable swept volume.
Supply the policy's original `prepared.h5` as `prepared_data`; its hash is bound
to the policy and proves that FT filtering during training also used 100 Hz.
This adapter refuses policies trained with a different filter sampling rate.

The new task exploration NPZ must contain `ft[400,6]`, `time_s=0.01..4.00`,
and scalar strings `source_kind=real`, `robot_id`, `sensor_id`, `calibration_id`,
`ft_frame`, `ft_reference_point`, `compensation`, matching training metadata.
It must be measured with the current sponge using the same exploration protocol
and converted using the physical calibration, not synthesized from a policy.
Use `python -m scripts.real_training.airbot_calibrated_data exploration` to create it;
hand-assembled label-only NPZ files are no longer accepted. The converter also
records calibration identity/hash, original received times/values and protocol;
the deployment loader recomputes the calibrated causal samples for comparison.
`calibration_record` names the evidence-backed physical calibration JSON used
during training. Geometry and bias nulls in the deployment setup are filled
from that record in memory; explicit differing overrides are rejected.

Run in an environment with PyTorch >=2.6, `requirements/real_training.txt`,
`requirements/robot_control.txt` and the existing pinned `arm-sdk==5.2.2`.
Do not replace the robot SDK while installing training dependencies. The existing
`clean` environment suffices for replay/preflight; actual execution also requires
the SDK. Start the AIRBOT server with `--no-return` as documented in
`airbot_initial_pose.md`, place the robot manually at the approved start, ensure
the physical emergency stop and support are available, then use a new log path:

```bash
python -m scripts.real_deploy run --config configs/real_deploy/airbot_deployment.json \
  --execute --output runs/real_exploration/policy_run_001.jsonl
```

The explicit `--execute` flag authorizes attended execution without a startup passphrase. Live preflight
checks robot/EEF identity, idle/stationary start, ownership and fresh FT. Every
tick checks state, joint limits, velocity, tracking, orientation and force limits.
Missed deadlines stop without catch-up. Errors/SIGINT/SIGTERM request the SDK
software stop without retract, recovery or automatic homing. This is NOT a
safety-rated stop. Successful completion holds servo until the operator supports
the arm and types `IDLE`; no automatic retract is inferred from training.

## Saved results (2026-09-11)

All nine model pairs completed the full configured 10000/2000 epochs (one
final pair plus eight validation pairs). No epoch reductions, early stopping,
hyperparameter selection from held-out data, or synthetic training data were
used. The encoder parameters and original collection hashes are unchanged.

| Metric | Training set | Leave-one-demo-out |
| --- | ---: | ---: |
| XY coordinate RMSE | 15.522 mm | 17.777 mm |
| Height increment RMSE | 0.547 mm | 3.733 mm |
| Zero height increment baseline RMSE | 3.554 mm | 3.554 mm |

Held-out height prediction is about 5.0% worse than the zero-increment baseline.
The source encoder reports zero active latent dimensions at the 1e-4 variance
threshold and fails its own reconstruction baseline. The raw real exploration
also falls outside the simulation normalization range for 92.5%-100% of samples
per channel. These diagnostics prevent a claim of learned transferable sponge
properties or deployment readiness. Do not report training reconstruction as
closed-loop accuracy or force-control success.

`report/summary.json` verifies all nine checkpoint epochs/hashes, software
provenance, frozen encoder weights, original source hashes and held-out episode
separation. `report/results.png` is the native-frame summary figure. The legacy
per-episode training plots use generic Base X/Y labels; for this profile the
recorded coordinates are SDK end positions, not calibrated sponge TCP positions.
Use the explicitly labeled native-frame summary for reporting this experiment.
The report command requires a new output directory; use `--output` to retain a
new audit after changes without overwriting prior reports.

Final regression: all 286 tests passed in 83.606 s, with no skipped tests.
The structured test result and complete output are saved in
`report/regression.json`. The final streaming replay has zero numerical
difference from batch inference for all 160 predictions; evidence is in
`report/replay_final_verification.json`. This proves software consistency,
not closed-loop physical behavior.

The `clean` environment has also been checked with the local AIRBOT SDK wheel,
without changing the existing `/home/wp/airbot-venv-5.2` environment:

```bash
conda activate clean
python -m pip install '/home/wp/下载/sdk_client_release/dist/x86_64/arm_sdk-5.2.2-py3-none-any.whl'
python -m pip install -r requirements/real_training.txt -r requirements/robot_control.txt
python -m scripts.real_deploy --help
python -m unittest discover -s tests -v
```

Verified together: arm-sdk 5.2.2, grpcio 1.83.1, protobuf 7.35.1, torch 2.13.0,
numpy 1.26.4, scipy 1.15.3, h5py 3.16.0 and pyserial 3.5. The import check
constructed the exact `MoveEndPoseRequest` message used by the adapter and
loaded the trained policy; it did not connect to a server or sensor.
