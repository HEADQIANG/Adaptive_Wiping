# 历史人工示教候选审核

以下为历史步骤，不适用于当前默认配置；当前流程见 [manual_tared.md](manual_tared.md)。

最新修改：按用户要求，人工示教部署的原始力矩阈值改为仅记录，
`torque_limit_policy=record-only`、`guards.max_torque_nm=null`。
不再因力矩超过 0.8 Nm 阻止预检、回放或中止运动；实时读数及因果历史读数均适用。
六轴原始值完整保留，NaN、过期、合力 40 N（起点 32 N）、关节、深度和时序保护仍有效。
此例外仅限 manual-tared，旧部署和采集模式的力矩保护不变，不修改硬件保护或传感器量程。
新回放目录为 `runs/real_deploy/manual_tared/0915_194830/`；下方旧回放数值为历史记录。
本次回放中路径和合力检查通过，0.813025 Nm 原始力矩仍记录但不再阻塞。
整体仍因起点姿态不匹配而 `passed=false`，现场/模型确认未完成，不自动获得运动资格。

独立入口 `--mode manual-tared`，不修改旧 calibrated/fixed-setup 模式的输入契约。
本模式用于 `manual_recorded_baseline_10s_v1` 数据，绑定原模型、探索、人工会话和源码哈希。
训练权重与原始数据不修改，导出策略保持 `hardware_ready=false`。

## 路径处理

2026-09-15 按用户最新要求取消缩放和平移，保持网络输出的 25 个绝对 SDK XY 点原样，
范围约 46 x 183 mm。配置 `path_transform=original_xy_v1`，缩放比例 1、平移 0。
人工部署的项目侧笛卡尔速度改为 `record-only`：0.05 m/s 仅为历史参考，不再因超过它停止。
不减速、不逐点裁剪、不修改 Z 网络输出、训练文件或 SDK 速度设置。
SDK 命令关节速度 0.4 rad/s、实测关节超速 1.2 rad/s、合力、3 mm 高度增量、
20 mm 下压、5 mm 跟踪、0.02 rad 姿态及采样时序保护仍保留，可能因跟不上轨迹而停机。
这不是解除所有速度保护或保证硬件可以跟踪任意速度。

任务 XY 边界改为原预测路径与起点的包围盒，加 5 mm 跟踪余量，包括首段进入和最终回位。
不再套用旧程序示教的 X +/-5 mm、Y +/-50 mm；旧部署模式仍保持原限值。
新的完整路径及进入/回位段仍需现场检查净空。日志记录每段指令速度和 record-only 策略。

任务仍为 10 秒、100 Hz 因果力处理、0.4 秒策略间隔和 5 帧历史，不能直接减慢整条任务时钟。
前 2 秒名义下压 10 mm 是显式候选初始化，不是从人工示教学到的接触策略；现场未确认前不运行。
姿态取探索时的参考姿态，报告列出其与人工示教的姿态角差；安装一致不能证明方向/重力对齐。

## 离线命令

从项目根目录执行，使用 `/home/wp/miniconda3/envs/clean/bin/python`。
当前模型已 evaluate/export；不要覆盖已有 `policy.pt`，换模型必须建立新配置和审核绑定。

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_deploy preflight \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_deploy replay \
  --mode manual-tared --config configs/real_deploy/airbot_manual_tared.json \
  --output runs/real_deploy/manual_tared/0915_194830
```

preflight 不连接设备，现场项未确认时应返回 2，并明确 `run_enabled=false`。
上述输出路径是本次配置绑定的独立时间目录，仅可创建一次；已有目录不能覆盖。
replay 新建时间目录，生成 `report.json`、`replay.npz`、`path_review.png`。
回放按独立 FT 接收时间去重后的数据做因果重采样，核对 160 次流式/批量预测，
而不是把约 56 Hz 的原数组当作 100 Hz 使用。Z 路径仅假设完美跟踪并重用原始记录力，
不模拟接触动力学，也不是部署时真实力的预测。
报告分列 `path_limits_passed`、`recorded_force_guard_passed` 和姿态支持检查，
任何一项失败都不会生成可放行的 `passed=true`。
候选起点与全部已采样示教姿态均相差超过 0.02 rad 时，属于独立硬阻塞：
不能用 `training_input_alignment_confirmed=true` 绕过，需另行核验起点或解决力输入映射。

如需用回放解除预检中的“无报告”门槛，先把配置 `replay_report` 设为所选新时间目录的
`report.json`，再回放到同一个尚不存在的目录；配置、模型、输入或源码改变都会使报告失效。
不要先生成报告再改配置来伪造同一绑定。

## 现场门禁

所有新现场/模型审核项默认 false，不继承旧程序示教的批准。逐项审核：

- `training_input_alignment_confirmed`：审核姿态角差、力方向、清零和接触条件。
- `encoder_quality_exception_confirmed`：独立评估已有编码器质量警告，仅此模型/同海绵例外。
- `motion_limits_confirmed`：确认笛卡尔速度仅记录，其他原始载荷、深度、关节及姿态/跟踪保护保留。
- `startup_contact_confirmed`：核验候选起点无接触及 10 mm 初始化压入和制动余量。
- `policy_path_confirmed`：审核候选路径、抬升及回位全路径净空。
- `physical_estop_verified`、`support_handoff_confirmed`、`server_no_return_verified`：急停、承重交接、服务端不自主回位。

条件确认只是在配置中记录现场审核结论，不能把所有 false 自动改为 true。
确认变化后必须重新回放，当前绑定且 passed=true 才能进入 shadow/run。
软件条件通过不等于硬件安全认证或闭环稳定证明。

新入口不自动直达起点。操作者需安全地手动放置到冻结候选起点、idle 状态并支撑。
起点必须匹配位置 2 mm、朝向 0.02 rad、关节 0.03 rad；不匹配时不切 servo、不发送移动。
进入后先保持，再等待新的 `s`，采集静止非接触基线后执行一次候选路径。
运动跟踪和姿态偏差为 stop，不继承旧模式的 record-only。原始载荷保护不因扣零消失。
正常结束先抬后横移，实测回位后等待 `IDLE` 支撑交接；故障停止，不自动回撤或重试。

仅在上述审核和回放完成、现场允许时，由操作者在同时安装固定 SDK 5.2.2 与训练依赖的
环境交互执行。以下命令会连接硬件；本次开发不执行：

```bash
python -m scripts.real_deploy shadow --mode manual-tared \
  --config configs/real_deploy/airbot_manual_tared.json \
  --output runs/real_deploy/manual_tared_shadow/events.jsonl --execute
python -m scripts.real_deploy run --mode manual-tared \
  --config configs/real_deploy/airbot_manual_tared.json \
  --output runs/real_deploy/manual_tared_run/events.jsonl --execute
```

shadow 不获取控制权、不切模式、不发送目标、不触发停止/idle，不提供承重或替外部控制程序停机。
现场空载力与训练输入不同，预测超界即退出，不为完成 shadow 放宽限制。

## 历史原轨迹审核（保留力矩阈值时）

当前配置和报告：`runs/real_deploy/manual_tared/0915_184635/`。
`original_xy_preserved_exactly=true`，部署的 25 个 XY 点与原模型预测逐元素一致。
XY 跨度 46.203 x 183.413 mm，不缩放、平移或减速，任务仍为 10 秒。
原始路径理想跟踪回放峰值三维速度 0.097341 m/s，项目笛卡尔速度拦截已关闭。
XY 任务区域（SDK m）为 X [0.141180, 0.197383]、Y [-0.122899, 0.070514]，
包含起点与原始路径并加 5 mm 余量；旧小范围边界仅作图中参考。

路径检查通过、160 次预测一致，但整体回放仍未通过：原始力矩峰值 0.813025 Nm 超过保留的
0.8 Nm 上限，且候选起点姿态与示教不匹配。8 项现场/模型审核尚未确认。
这些条件不能因取消笛卡尔限速而自动解除。当前不连接机械臂、不允许直接执行运动。
要达到现场可执行，仍需核验无接触起点及力输入方向、确认物理载荷限制和现场保护，
按确认后的独立配置重新回放与预检，不能仅将失败标记改成通过。

## 历史缩放审核

本节为已废弃的缩放方案，不代表当前配置。历史文件保持原样，不可用于当前模式放行。

2026-09-15 审核目录：`runs/real_deploy/manual_tared/0915_180252/`。
模型已导出到 `runs/real_training/manual_tared/0915_171709/training/policy.pt`。
没有连接机器人，没有修改训练权重和原始数据，也没有设置新的现场批准。

- `path_review.png`：原预测、缩放候选、分段速度三联图。
- `report.json`：哈希绑定、路径/载荷/姿态分项结果和失败时刻。
- `replay.npz`：未改动的原预测 XY、候选 XY、理想跟踪候选 XYZ 及力反馈预测。

候选统一缩放比例 0.1731484，XY 跨度 8.0 x 31.76 mm，最大相对偏移 X 4.0 mm、Y 15.88 mm。
最大 XY 分段速度含进入段为 0.01685 m/s；理想跟踪回放最大三维速度为 0.01705 m/s。
保持 10 秒时间契约，没有扩大边界或在线裁剪，`path_limits_passed=true`。
160 次流式/批量力反馈预测最大差为 5.96e-10 m。

整体 `passed=false`、`run_enabled=false`，不是回放代码异常：

- 第 5 条记录 4.56--4.68 秒的 13 个因果网格点原始力矩超限，峰值 0.813025 Nm > 0.8 Nm；
  原始合力峰值 36.3997 N，没有超过 40 N。保留原始载荷保护，不扣零后再判定或放宽上限。
- 候选参考姿态与已采样示教姿态相差 0.2417--1.6823 rad，不能以安装一致推断力方向已对齐。
  此条件须核验候选起点/力输入映射，不能仅改确认标志。
- 9 项现场/模型确认仍为 false，缩放后真实接触力分布与闭环稳定性未验证。

因此当前只完成入口、候选路径和离线审核，不应执行 shadow/run。还需现场及物理输入审核，
不能通过改旧哈希、扩大限值或把失败报告改成 passed=true 放行。

## 软件回归

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_deploy.test_manual_setup tests.real_deploy.test_fixed_setup tests.real_deploy.test_airbot_deploy
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.robot_control.test_hardware_backend tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_programmed_demonstrations
```
