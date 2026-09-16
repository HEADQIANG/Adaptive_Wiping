# 同海绵、固定安装部署入口

2026-09-14：真机 TCP 标定代码已移除，固定安装流程仍直接使用 SDK 坐标。
代码来源改变会使已有策略源码绑定和旧回放失效；保留旧产物，不改写哈希放行。
重新生成兼容产物和验证的步骤见[传感器转换文档](../real_training/airbot_calibrated_pipeline.md)。
以下既有 v3 预检通过记录仅描述当时版本，不保证当前代码可直接运行。

适用日期：2026-09-13。显式使用 `--mode fixed-setup`，配置为
`configs/real_deploy/airbot_fixed_setup.json`。默认 `calibrated` 入口和旧标定门禁不变。
该入口仅绑定当前 `training_programmed_wide1200_v1/policy.pt`、宽范围仿真编码器、
`direct_start_session_004` 会话及其空载扣除、补齐到 10 秒的派生数据。
不修改原始采集、训练权重或模型的 `hardware_ready=false` 标记。

操作者已确认硬件、安装和海绵不变，并接受编码器塌缩在这一固定条件下的例外。
2026-09-13 后续确认：运动中的跟踪误差和姿态误差仅记录，其余列出的现场条件均可满足。
当前配置据此记录操作者确认，不表示程序已经执行或独立完成了真机验收。
例外绑定准确的策略、编码器和会话哈希，不适用于换海绵、换装、换起点或其他模型。
这不是坐标/重力标定，也不能说明条件泛化成立。固定深度示教没有提供主动纠正力误差的
示范，补齐尾段不是实测，训练误差和回放通过均不能证明闭环稳定。

## 输入与时间对齐

- 使用采集会话冻结的机器人身份、传感器、SDK 起点/朝向和保护配置，不读取后来修改的起点。
- 复用同一海绵、训练探索记录计算的冻结 embedding；不自动重做探索。
- 每次策略开始前，在无接触起点确认静止并重新采满 1 秒空载基线。静止阈值 0.1 rad/s；
  不稳则丢弃整段，在总计 10 秒内重新确认并重采。接触时禁止去皮。
- 在线六轴输入为 `raw_sensor_wrench - fresh_unloaded_baseline`，传感器坐标轴保持不变，
  不额外翻转 Y/Z，不取绝对值，不使用旧基线代替现场基线。
- 与训练相同：100 Hz 因果保持、二阶 1 Hz 因果低通，每 0.4 秒取样，5 帧历史预测下一高度增量。
  原传感器接收约 56 Hz，100 Hz 保持不代表有 100 Hz 独立测量。
- 原始六轴数据独立用于保护，不因空载扣除而放宽。当前冻结阈值为原始合力 40 N、
  起点 32 N、力矩 0.8 Nm；这些不是目标接触力。

## 运动流程与待确认项

只有 `run` 会运动：`--execute` 授权启动，现场检查通过后低速直接移向冻结起点，沿用程序示教的启动路径检查，
没有避障；到达后等待新的 `s`。`s` 前必须目视确认海绵完全离开表面。
随后采集基线，执行一次 10 秒策略。不会循环或自动开始第二条。

前 0–2 秒以 5 mm/s 下压至名义 10 mm，包含在 10 秒内，不另加预按压。
XY 从一开始就由网络的 25 个点插值，不强行改回示教的三段位移；前 2 秒也可能有微小横移。
首次高度预测发生在 2.0 秒，目标对应 2.4 秒，以当时实测高度加网络增量。
一条共 20 次高度预测。不增加额外 10 N 比例控制器，也不保证最后预测点正好回到起点。

正常结束后，在实测横向位置先抬到起点高度，再横向回到起点，回撤名义速度 5 mm/s。
回位位置、朝向和静止均实测验证，不以发送命令成功或 `record-only` 作为合格依据。
等待期间保持受监测 servo；最后安排支撑，重新输入 `IDLE` 才释放。
任何超限、失控、断流、命令拒绝或超时均中止，尝试软件停止，不自动回撤、重获控制权或清急停。

运动中的 `tracking_error_policy` 和 `orientation_error_policy` 均为 `record-only`，适用于
启动直达、策略执行和回撤。配置中的 5 mm / 0.02 rad 仅为报告参考值，不是这两个误差的停机阈值。
日志保存 `position_error_m`、`orientation_error_rad`、参考值、是否超出参考值及对应的
`tracking_target_sdk_m` / `tracking_target_quaternion_xyzw`；误差比较的是观测前已发送的目标，
不是本周期随后发送的新命令。启动插值仍限制速度与时长。
该设置不取消起点/最终回位的位姿资格检查和静止检查，也不取消实际下压及任务相对边界。
未正确回位仍会超时停止，不能仅因命令发完就去皮或释放 idle。
姿态偏差较大时空载扣除后的力可能不再与训练物理条件一致，必须审核日志，不能将继续运行视为效果合格。

以下六项现已按操作者逐项范围确认设为 `true`；换条件后需重新核实，不能沿用为通用批准：

| 配置项 | 所需现场确认 |
| --- | --- |
| `motion_limits_confirmed` | 确认 0.05 m/s 速度、3 mm/步高度增量；运动跟踪/姿态偏差仅记录，其余保护保留 |
| `startup_contact_confirmed` | 冻结起点非接触，批准前 2 秒名义下压 10 mm，留足行程和停机裕度 |
| `policy_path_confirmed` | 审核网络实际路径、当前位姿直达起点路径和先抬后横移回位路径，确认无障碍 |
| `physical_estop_verified` | 独立物理急停可用，操作者全程监护 |
| `support_handoff_confirmed` | 等待、故障和释放 idle 的承重/支撑交接方案明确 |
| `server_no_return_verified` | 服务端按探索流程以 `--no-return` 运行，不在退出时自主回位 |

策略阶段独立限制相对起点 X ±5 mm、Y ±50 mm、总下压 0–20 mm；实测下压同样不得超过 20 mm。
命令超过边界直接拒绝，不裁剪。该任务相对边界不恢复已经移除的全局 XYZ 工作空间。
软件检查不能保证机械运动绝不超程，现场仍需留出制动裕度和独立保护。
策略阶段保持严格 100 Hz：唤醒迟到超过 5 ms、发送前超过 9 ms、整周期超过 10 ms 均停止，不追赶。
这不是程序示教放宽后的调度配置；只读测试用于先检查当前机器性能，不应现场反复盲目放宽。

## 环境与离线检查

在仓库根目录执行。下面 `python` 必须来自同时具备 PyTorch、h5py、SciPy 和固定 SDK 5.2.2
的已验证环境；依赖见 `requirements/real_deploy.txt`，不要升级或替换现有 SDK。
本机离线测试使用 `/home/wp/miniconda3/envs/clean/bin/python`，不表示该环境已具备真机 SDK。

```bash
env -u PYTHONPATH python -m scripts.real_deploy preflight --mode fixed-setup
```

预检不连接设备，当前配置与最新回放一致时应显示 `offline_inputs_valid=true`、
`run_enabled=true`、`blockers=[]`，两项误差策略为 `record-only`，起点/回位检查为 `required`。
这表示软件入口允许显式运行，不等于硬件已经验证。
本次配置对应的最新离线回放为 `runs/real_deploy/fixed_setup_wide1200_replay_v3/report.json`。
回放逐条重建原始力扣基线后的在线输入，核对 160 次流式/批量预测及命令边界。
它使用记录中的实测状态，不模拟接触动力学，不是闭环验收。

修改配置（包括任何确认标志）、代码或绑定文件都会使旧回放失效。修改后先将配置的
`replay_report` 改为一个新目录下的 `report.json`，然后回放到该新目录。例如将其设为
`runs/real_deploy/fixed_setup_wide1200_replay_v4/report.json` 后运行：

```bash
env -u PYTHONPATH python -m scripts.real_deploy replay --mode fixed-setup \
  --config configs/real_deploy/airbot_fixed_setup.json \
  --output runs/real_deploy/fixed_setup_wide1200_replay_v4
env -u PYTHONPATH python -m scripts.real_deploy preflight --mode fixed-setup
```

输出目录必须不存在，不覆盖历史回放。配置在回放后再次修改，需要再次使用新目录回放。
换模型或安装不是重新计算哈希即可放行，需要重新审核适用范围与数据对齐。

## 只读在线推理

此步骤连接机器人和传感器，但不获取控制权、不切 servo、不发送目标、不触发停止或 idle。
由现场人员安全地将机械臂保持在冻结非接触起点、处于 idle 并安排支撑；此模式不提供承重保持。
现场确认只读操作可行后执行：

```bash
env -u PYTHONPATH python -m scripts.real_deploy shadow --mode fixed-setup \
  --output runs/real_deploy/fixed_setup_shadow_001/events.jsonl --execute
```

`--execute` 授权只读连接，不再等待启动口令；输入新的 `s` 开始。确认基线、原始/扣除/滤波力及推理耗时合理。
静止空载不等价于擦拭输入，预测可能触发轨迹边界而退出；不应为完成空载 shadow 而放宽边界。
shadow 不能验证接触纠偏方向、轨迹跟踪或闭环稳定，也不会因异常替操作者停止由外部控制的机械臂。

## 经批准后的单次运动

完成上表现场确认、重新回放，且预检显示 `run_enabled=true` 后，才人工执行：

```bash
env -u PYTHONPATH python -m scripts.real_deploy run --mode fixed-setup \
  --config configs/real_deploy/airbot_fixed_setup.json \
  --output runs/real_deploy/fixed_setup_attended_001/events.jsonl --execute
```

执行带 `--execute` 的命令后，现场检查通过即会自动直达起点，不是等 `s` 才进行启动移动。
到位后检查离面再输入 `s`；`q` 不执行策略，但仍须支撑并输入 `IDLE` 交接。
运动期间预先输入的命令被丢弃。每次使用新日志路径，保存 JSONL 与同名 `.sensor.csv`。
故障后查看 `aborted`/`stop_error`，确认实际停止并支撑；软件停止未确认时立即使用物理急停。
不得自动重新运行、清急停或在未知状态下回撤。

## 软件验证

```bash
env -u PYTHONPATH python -m unittest \
  tests.real_deploy.test_fixed_setup tests.real_deploy.test_airbot_deploy \
  tests.real_training.test_programmed_padding \
  tests.real_training.test_airbot_native_training \
  tests.real_training.test_real_training \
  tests.real_training.test_airbot_programmed_demonstrations
```

假设备测试不连接真机。覆盖空载扣除、重采基线、首段下压、历史时序、只读不发命令、
原始载荷保护、失控/拒绝/断流/姿态/深度/超时、实测回位失败、哈希和旧入口隔离。
另覆盖启动、擦拭和回撤中跟踪/姿态偏差仅记录并继续、非法策略拒绝，及 record-only 下其他保护仍有效。
软件测试与离线回放不替代现场验收。
