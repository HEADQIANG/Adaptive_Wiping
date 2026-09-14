# Accepted demonstration overview

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

From the repository root, use Python 3 with NumPy and Matplotlib installed:

```bash
python3 -m scripts.real_training.tools.plot_demonstration_overview
```

Default input: `archive/real_training/raw_data/manual_demonstrations/session_record_only_002`.
Default output: `archive/real_training/real_robot/demonstration_overview_002`:

- `trajectories.png`: 3D paths, XY/XZ projections and XYZ versus time.
- `force_torque.png`: eight demonstrations overlaid in six sensor channels.
- `summary.json`: source hashes, sample counts, position spans and FT extrema.

To select another session or destination:

```bash
python3 -m scripts.real_training.tools.plot_demonstration_overview --session archive/real_training/raw_data/manual_demonstrations/session_record_only_002 --output runs/real_training/real_robot/demonstration_overview_custom
```

Existing generated files at the destination are replaced. Source files are
read-only; this command never connects to hardware. All eight `demo_01.json`
through `demo_08.json` must be accepted real records with passed recorded
quality checks. Their referenced raw logs must match SHA256 and contain start
and finished events. Unaccepted attempts and continuous sensor CSVs are excluded.
This visualization is not a new training-quality or physical-calibration audit.

Positions are absolute SDK end-frame positions in mm, not calibrated sponge TCP
positions. All views retain spatial offsets between demonstrations. Each trace
uses elapsed host time relative to its own logged start; no time warping is used.
Pose time uses `host_monotonic_s`, FT time uses `sensor_receive_perf_s`. Adjacent
pose/observation timestamps are checked against the existing 20 ms bound; this
relies on the Linux CPython shared monotonic clock convention used by the importer,
not hardware synchronization. Identical FT receive timestamps are deduplicated;
conflicting values at a single timestamp are rejected.

FT values use `raw_sensor_wrench_si`, without filtering, interpolation, electronic
bias correction, gravity compensation or frame conversion. Forces are in N and
moments in N m, in sensor-local axes about the sensor origin. Sensor axes must not
be interpreted as SDK/world axes. Different tool orientations can change gravity
components; cross-demo differences alone do not establish contact quality.
