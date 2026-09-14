# AIRBOT User-Reported Tool Geometry

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Date: 2026-09-11. This is a partial measurement record, not a completed
calibration or permission to move hardware.

Follow-up model-chain and sensor-bias audit:
[airbot_frame_resolution_20260911.md](airbot_frame_resolution_20260911.md).

## Reported Measurements

Source: the operator's measurements and confirmations in this conversation.
Measurement uncertainty and instrument details have not been supplied.

| Item | Reported value |
| --- | --- |
| White wiping face length | 107 mm |
| White wiping face width | 65.78 mm |
| Reported sponge thickness | 31.26 mm |
| Robot-side thin square flange mounting plane to white wiping face | 79.7 mm |
| White face center aligned with circular flange center axis | Operator confirmed |
| White face parallel to the measured flange mounting plane | Operator confirmed |
| Mount, sensor and sponge unchanged since exploration and eight demonstrations | Operator confirmed |
| Sponge suspended without external contact at the reported initial position | Operator confirmed |

The chosen TCP is the geometric center of the uncompressed white wiping face.
Relative to the measured mounting-plane center, its displacement is 0.0797 m
along the flange axis toward the sponge. The 79.7 mm already reaches the wiping
face; do not add sponge thickness, sensor thickness or adapter thickness again.
Centering and parallelism are operator confirmations, not precision metrology.

The supplied photos show sensor markings X+ and Y+. They do not by themselves
establish the complete sensor-to-SDK rotation, sensor measurement origin or
electronic bias. Photos remain conversation attachments; this document does
not claim to contain or hash the original image files.

## Coordinate Checks Still Required

### Subsequent Operator Corrections and Nominal Choice

- Sensor identity: KWR52-TiS, S/N 1F06303, read from the clearer nameplate.
  The operator confirmed the older photo shows the same sensor.
- The operator confirmed the reader's six-channel multiplication by 9.81
  originates from the manufacturer code for this setup (kgf/kgf*m to N/N*m).
  This is an operator provenance statement, not an independent raw-protocol test.
- Corrected sensor sponge-side plane to wiping face: 46.84 mm.
- Sensor robot-side plane to wiping face: 69.68 mm.
- The earlier 69.68 mm sponge-side distance is superseded, not a second estimate.
- Implied measured sensor thickness: 22.84 mm. The operator chose to use the
  manufacturer's nominal 24 mm and actuator-side load reference plane instead.
- Nominal reference-plane-center to TCP displacement: [0, 0, 0.04684] m
  in SENSOR axes. Z+ points toward the sponge, as confirmed by the operator
  using the manufacturer axis diagram. This is not an SDK-frame transform.
- Nominal sensor robot-side plane to TCP: 70.84 mm. Its 1.16 mm difference
  from the measurement remains unresolved; adaptive feedback is not evidence
  that this error is compensated. Do not mix nominal and measured chains.
- Manufacturer source: https://kunweitech.com/products/609.html
  and https://kunweitech.com/upload/20250930/202509301107107542.jpg.
  The current product table lists A/B/G, not TiS. Applicability is a user-approved
  nominal assumption, not manufacturer confirmation of this serial number.

### Startup Log Correction

The supplied 14:07 startup log actually references
/usr/share/airbot_controllers/asset/play_arm/urdf/play.urdf, not the other
asset/play/urdf/play.urdf candidate discussed below. Kinematics reports a
fallback to default DH parameters. XML syntax of the referenced file parses,
but this does not establish why the driver's DH extraction failed.
The same log contains CAN write, craft flag, scheduling and servo YAML warnings;
successful read-only connection is not clearance for motion.

The offline audit described below compares the printed DH and end_convert
with recorded SDK poses. It does not identify physical flange location or
calibrate a force sensor. No calibration verification flags are changed.

The installed SDK's get_end_pose returns the service's arm_end_pos; its Python
docstring does not locate that frame on the physical flange.
The installed file /usr/share/airbot_controllers/asset/play/urdf/play.urdf
defines end_joint from link6 to end_link with xyz="0.0 0.0 0.0865" and
rpy="1.57 -1.57 0.0". This is a candidate model definition only, not proof that
the running service uses this exact chain or that its origin is the measured
mounting plane. Do not add or subtract 86.5 mm from the reported distance
without establishing both reference frames.

Still unresolved: service base/end frame binding, physical flange-to-SDK
transform, TCP axis convention, sensor axes and wrench reference point,
electronic bias, and measurement uncertainty. One suspended pose still
contains tool gravity and cannot independently identify electronic bias.

## Operational Status

No production code, training data, model, server setting or calibration JSON
was changed for this geometry record. No motion or sensor tare was performed.
Existing training/deployment commands remain unchanged; follow airbot_calibrated_pipeline.md.
Do not set calibration verification flags from this partial record alone.

## Reproduce the Offline Frame Audit

Run from the project root. The script imports no robot or serial SDK and
checks source hashes. It samples every twentieth recorded pose in each of
eight accepted demonstrations and compares six DH/conversion hypotheses.
It does not fit free offsets, overwrite existing reports or modify source data.

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.robot_control.tools.audit_airbot_frames \
  --startup-log /home/wp/.codex/attachments/8ac96f64-5ca8-4f09-8b83-3489f1c5f2fd/pasted-text.txt \
  --output runs/robot_control/airbot_frame_audit_20260911_v2.json
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -t . -p 'test_airbot_frame_audit.py' -v
```

First report: archive/robot_control/airbot_frame_audit_20260911_v1.json (408 poses).
Modified DH with right-multiplied end_convert yields pooled position RMS
0.096487 mm (maximum 0.738985 mm) and orientation RMS 0.029478 degrees
(maximum 0.204945 degrees). Equal sample counts permit pooling by mean squared
per-episode RMS. Omitting end_convert leaves approximately 120 degrees of
orientation discrepancy, despite similar position error.
Use a new output filename on each rerun. The startup log postdates the demos;
agreement is evidence of numerical consistency, not proof of historical
configuration, measurement accuracy, physical calibration or motion safety.
