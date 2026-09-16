# AIRBOT Model Frame Resolution

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This is an offline diagnostic, not physical calibration or authorization for
motion. No system files, controller settings, sensor settings, source data,
training models or production calibration flags were changed.

## Results

The archived /home/wp/airbot-logs-5.2/airbot::kdl.log contains the same default
DH and end_convert on 2026-09-10 (including 22:18) and on 2026-09-11.
The fallback therefore predates the current connection test. The log hash
at inspection was 0591cdb5d9450d3c88388d3971dad93787c84c95fb19aaa405d358a84bfbf869.
The log can be appended by the service; this hash is not a permanent binding.

Read-only binary inspection of /usr/bin/airbot-arm found that
check_and_update_dh_from_xml constructs end_joint/end_link names and contains
checks requiring the end joint parent link6 and child end_link. The referenced
URDF instead ends with arm_connnect_joint -> eef_connect_base_link. This is a
structural incompatibility consistent with the reported extraction fallback,
not malformed XML. No patched driver or modified URDF was executed to test a
fix; other parser constraints have not been exhaustively ruled out.
Binary SHA256: da2b6ef3698b818f0d3239ca96f8c53d1a173a004170b874463273e7ff7ea501.

Independent FK uses the installed Orocos PyKDL solver and an XML-parsed URDF
chain. It compares 408 recorded joint configurations to the printed modified
DH chain, not to external physical measurements. No free transform was fitted.

| Comparison | Position RMS | Orientation RMS |
| --- | --- | --- |
| Unadjusted URDF vs nominal SDK-to-connector relation | 1.000803 mm | 0.000267 deg |
| After explicit base-height difference removal for diagnosis | 0.008387 mm | 0.000267 deg |

URDF joint1 height is 0.1127 m; printed DH base height is 0.1117 m. The 1 mm
difference was removed only for the second diagnostic, not from recorded data.
This does not certify either model's physical base height.

The candidate transform mapping model connector coordinates into SDK end
coordinates is:

```text
 0  0  1  0.0865
-1  0  0  0
 0 -1  0  0
 0  0  0  1
```

Thus model connector Z+ corresponds to SDK X+, and its model origin is 86.5 mm
along SDK X+. This does NOT prove the connector model origin coincides with the
operator's measured square-flange surface. Do not add 79.7 mm to this offset
and call it a calibrated TCP without resolving that surface correspondence.
Sensor rotation about its central axis also remains unmeasured in SDK axes.

## Sensor Bias Findings

The current reader initializes its software bias to zero, sends start/stop
stream commands, and implements tare as a local array subtraction. It has no
implemented query for the acquisition box's internal zero state. Demonstration
and exploration collection preserve received loads with configured zero bias;
they do not call the reader's tare method. This establishes software handling,
not absence of a prior zero operation inside the box.

The operator confirmed the manufacturer supplied the 9.81 unit conversion for
this setup. Keep this conversion. Do not equate it with bias or gravity removal.
Reader hash at inspection:
018a379e5e28da640042e2d8904b29060bfbcc891da3249b0fcae60c854154fc.

## Minimum Remaining Physical Information

1. Relate the model connector origin to the actual measured flange surface and
   establish the sensor's complete installation rotation. Existing dimensions
   and photos are retained; no repeat sponge measurements are requested.
2. Establish the acquisition box's zero state and a justified electronic-bias
   measurement. A single suspended pose cannot separate gravity from bias.
   Existing contact demonstrations cannot be treated as unloaded calibration.

If manufacturer installation/zero records cannot settle these items, one
explicitly planned, attended calibration session is required. Do not improvise
robot motion or clear the sensor. Model-quality and motion-safety validation
remain separate later requirements.

## Reproduction

Run from the project root using system python3, where PyKDL is installed.
The clean environment does not contain PyKDL; it is unchanged. Dependencies
are NumPy, SciPy and python3-pykdl. The script imports no robot or serial SDK.

```bash
python3 -m scripts.robot_control.tools.audit_airbot_model_chain \
  --startup-log /home/wp/.codex/attachments/8ac96f64-5ca8-4f09-8b83-3489f1c5f2fd/pasted-text.txt \
  --output runs/real_deploy/robot_control/airbot_model_chain_audit_20260911_v2.json
python3 -m unittest discover -s tests -p 'test_airbot_model_chain.py' -v
python3 -m unittest discover -s tests -p 'test_airbot_frame_audit.py' -v
```

The first report is archive/robot_control/airbot_model_chain_audit_20260911_v1.json. Existing
reports cannot be overwritten; choose a fresh name for each run. Reports retain
input and script hashes. Training and deployment commands remain unchanged.
Model-chain tests skip only when PyKDL is absent; run with system python3 for
actual coverage, not the clean environment.
