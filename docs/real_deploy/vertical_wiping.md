# 垂直墙面擦拭

新增可选 `--wiping-mode vertical`，用于当前 `manual-tared` 现场起点流程。
不传该参数时仍是原来的水平 XY 轨迹和 Z 反馈；无需修改训练数据、权重或原配置。
支持 X/Z 和 Y/Z 两种映射：`±x` 面向平行于 SDK YZ 平面的墙，`±y` 面向平行于 SDK XZ 平面的墙。
SDK Z 应为现场竖直方向。
这不是任意斜墙的自动识别，也不自动转动工具或寻找接触。

## 运行命令

从项目根目录运行，使用原有 SDK/串口环境。先离线预检：

```bash
DEPLOY_PY=/home/wp/miniconda3/envs/clean/bin/python
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy preflight \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json \
  --wiping-mode vertical --wall-direction +x
```

现场运行使用相同方向参数：

```bash
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy run \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json \
  --wiping-mode vertical --wall-direction +x \
  --output runs/real_deploy/vertical_wiping/events.jsonl --execute
```

可用 `--policy /path/to/training/policy.pt` 选择其他清零人工示教模型，仍自动解析训练绑定。
垂直模式未指定 `--mode` 时使用 `manual-tared`；不支持旧 fixed-setup/calibrated 或旧 replay/shadow。
`--wall-direction` 必须显式指定：`+x` 表示沿 SDK +X 靠近墙，另一侧使用 `--wall-direction=-x`。
Y/Z 映射使用 `+y` 或 `--wall-direction=-y`，分别表示沿 SDK +Y 或 −Y 靠近墙。
参数不匹配会在连接硬件前报错。水平模式不接受墙面方向参数。

例如墙在 −Y 方向，先预检，再在现场运行：

```bash
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy preflight \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json \
  --wiping-mode vertical --wall-direction=-y
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy run \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json \
  --wiping-mode vertical --wall-direction=-y \
  --output runs/real_deploy/vertical_yz_wiping/events.jsonl --execute
```

墙在 +Y 方向时，以上两个命令均改成 `--wall-direction +y`。h/z/s/g 操作相同。

## 坐标映射与操作

| 墙面方向 | 原模型 X 轨迹 | 原模型 Y 轨迹 | 法向目标 | 工具相对桌面姿态 |
|---|---|---|---|---|
| `+x` | `Z₀ + (预测X − 首点X)` | `Y₀ + (预测Y − 首点Y)` | `实测X − Δh` | 绕 SDK Y 旋转 −90° |
| `-x` | `Z₀ − (预测X − 首点X)` | `Y₀ + (预测Y − 首点Y)` | `实测X + Δh` | 绕 SDK Y 旋转 +90° |
| `+y` | `X₀ + (预测X − 首点X)` | `Z₀ + (预测Y − 首点Y)` | `实测Y − Δh` | 绕 SDK X 旋转 +90° |
| `-y` | `X₀ + (预测X − 首点X)` | `Z₀ − (预测Y − 首点Y)` | `实测Y + Δh` | 绕 SDK X 旋转 −90° |

`Δh < 0` 沿指定方向压向墙面，`Δh > 0` 远离墙面。切向和法向符号成对变化，
对应右手坐标系的 90° 旋转；不是不带符号的反射交换。原模型路径跨度与时长保持不变。
X₀/Y₀/Z₀ 是按 h 捕获的本次 SDK 起点，实际反馈每 0.4 秒重新锚定法向实测位置（`±x` 取 X，`±y` 取 Y）。

1. 沿用 [现场部署准备](manual_tared.md)，进入重力补偿后，人工把工具转向墙面。
   工具、海绵、传感器保持原安装关系；姿态按上表旋转，使传感器局部法向和切向含义与训练一致。
2. 海绵离开墙面、撤去工具外载，按 `h` 捕获墙面起点与姿态并保持。
3. 按 `z` 在此姿态重新空载清零，不复用桌面姿态的重力偏置。
4. 按 `s`，保持捕获关节 2 秒采集五帧力历史，再执行完整 10 秒轨迹及法向反馈：
   `±x` 为 YZ 轨迹/X 反馈，`±y` 为 XZ 轨迹/Y 反馈。
   100 Hz 插值、25 次网络预测、固定捕获姿态、故障停止及结束保持均沿用原流程。
5. 结束后托住机械臂，按 `g` 进入重力补偿并退出；无自动回位或撤离。

六维力保持传感器局部通道及符号，扣除本次空载基线后沿用原滤波、归一化和网络。
不把 SDK 的 X/Z 或 Y/Z 位置映射应用到原始六维力通道。墙面姿态固定时去皮能扣除静态重力偏置，
不等于任意变姿态下的动态重力补偿。当前模式依赖人工摆正姿态，不测量墙面法向。
沿用原现场流程的项目阈值与 SDK 保护行为；模型在墙面的接触建立和泛化效果仍需实测。

## 离线验证

日志 `session_start.spec` 和 `reference_captured.wiping_frame` 记录模式、墙面方向、旋转矩阵、
法向轴及符号；`tangent_path_sdk_m` 记录起点法向位置处的完整切向路径。
逐帧 `delta_h_m` 始终为网络原始输出，`target_sdk_m` 和实测位置始终为真实 SDK XYZ。

自动生成的 `events.plots/fz_height.png` 在垂直模式显示预测的 SDK X 或 Y 法向端点；
为兼容原文件发现方式，文件名保留。CSV 使用 `measured_anchor_x_m`、`predicted_next_x_m`
和 `delta_h_to_sdk_sign`；Y/Z 映射对应 `measured_anchor_y_m`、`predicted_next_y_m`，
不将法向位置误标成 Z。Fz 曲线仍是传感器局部 Fz，不是基座 Fz。
历史水平日志缺少模式字段时按水平处理。

完整日志可重新绘图或分析法向执行误差（把路径换成实际输出）：

```bash
"$DEPLOY_PY" -m scripts.real_deploy.plot_inference \
  --events /path/to/events.jsonl --output /path/to/new_plots
"$DEPLOY_PY" -m scripts.real_deploy.plot_delta_h_error \
  --events /path/to/events.jsonl --output runs/real_deploy/vertical_delta_error
```

增量误差把实测 X 或 Y 变化按方向符号转换回模型法向后比较。专用于旧桌面 fixed/manual 对照的
`plot_run_comparison` 报告拒绝垂直日志，避免套用“下压 Z”的历史分析；旧桌面功能保持不变。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$DEPLOY_PY" -m unittest discover -s tests/real_deploy -t .
```

测试不连接机械臂或串口。垂直功能的验证覆盖四个墙面方向的轴映射、实测 X/Y 锚定、水平兼容、
姿态提示与旋转矩阵一致性、固定姿态/空载清零、两秒静止历史、完整轨迹、故障停止、
模式不匹配拒绝及 X/Y 法向日志绘图。
仅运行新增功能和自动绘图回归可用：

```bash
"$DEPLOY_PY" -m unittest tests.real_deploy.test_vertical_wiping tests.real_deploy.test_plot_inference
```

2026-09-16 Y/Z 扩展验证：全部 171 项部署测试通过，覆盖水平和 `±x/±y` 四个墙面方向。
默认配置绑定的真实模型 `runs/real_training/manual_tared/0915_171709/training/policy.pt`
本次以 `vertical/-y` 运行离线 preflight，返回 `offline_inputs_valid=true`、`blockers=[]`、
`hardware_connected=false`。这验证了软件入口和模型绑定，不代表墙面真机运动或接触效果已验收。
