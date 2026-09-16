# AIRBOT Play 人工拖拽采集 8 条示教

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

入口 `scripts.real_training.airbot_demonstrations`，配置 `configs/real_training/airbot_demonstrations.json`。
仅人工拖拽，不发送位置目标、不自动回位、不回零。尚未进行真机验收。
2026-09-15：默认改为软件清零数据与实时六轴曲线。每次连接先在无接触状态输入 `z` 回车清零，
再输入 `s` 回车录制；清零只扣当前姿态的空载基线，不是跨姿态重力补偿。
新配置不能续采旧原始力会话，必须新建目录；详细步骤见 [清零与实时曲线](manual_tare_live_plot.md)。
未指定 `--mode` 时行为保持不变；新增的 `--mode programmed` 为独立协议，
配置、数据目录和操作步骤见 [程序示教采集](airbot_programmed_demonstrations.md)，不能混用旧人工会话。

## 论文对应与边界

依据 `docs/references/Adaptive_wiping.pdf` IV-B、IV-C.2、IV-D：同一 Normal 海绵，在固定的训练斜面上，
自然速度完成 8 次正确擦拭，每次 10 秒，末端位置及六维力最终形成 2.5 Hz、25 帧序列。
训练斜面不同于后续验证斜面。论文的“尽可能用力”不作为 AIRBOT 操作要求；仅使用现场确认的安全力。
论文探索动作的 0.01 m/s、0.05 m/s 不是人工擦拭的指定速度。

本入口以目标 100 Hz 记录实测 SDK 位姿、关节状态、传感器最近读数和各自主机时间戳，
同时保留传感器全速 CSV。这是为既有离线处理保留高频日志的工程选择，不是论文示教原始采样率。
10 秒有效片段后另保留至少 20 ms 尾部观测，保证最近 FT 接收时间覆盖片段终点，不增加训练帧数。
不进行接触去皮，不做跨姿态重力补偿或 FT 坐标变换。清零后的 `tared_sensor_wrench_si`
为原始读数减去本次 `z` 记录的 1 秒无接触均值；原始值和配置电子零偏修正值仍保留。
电子零偏默认 0，不代表已独立标定；清零不重复扣除该配置零偏。
重力补偿拖动控制器与力数据中的重力补偿是两件不同的事。

## 现场准备和配置

先完成 [拖拽安全检查与服务启动](../robot_control/airbot_initial_pose.md)。固定底座、工具和训练斜面，
核实工具负载与 SDK 重力补偿适配，验证急停；两人配合，一人全程托持机械臂，一人操作键盘和监护。
退出或故障请求 idle，不是位置保持，也不会主动抬起工具。电源或控制权丢失时不能依赖软件承重。

更新：默认 `workspace_force_policy="record-only"`，取消力/力矩阈值中止。
2026-09-12 起项目 XYZ 工作空间边界在所有模式中均已移除，旧字段会被忽略。
`workspace_force_policy` 保留历史名称，但现在只选择力/力矩策略；record-only 模式
不要求填写 `max_force_n/max_torque_nm`，这些字段可保持 `null`。
位置与六维力仍完整记录，传感器有效性/20 ms 断流、关节范围、控制权及电机状态检查保持不变。
关节速度另设 `joint_speed_policy="record-only"`，当前默认取消运动速度中止，
`joint_speed_limit_rad_s=null`；实测速度仍原样写入 `pose.joint_velocity_rad_s`，不裁剪。
仅此人工示教入口变化，自动探索和 SDK/固件保护不变。2026-09-15 起取消录制前静止检查：
输入 `s` 后只读取一次当前状态作为起点，不再进行 11 帧静止采样或检查采样期间位姿变化。
默认 record-only 配置允许运动中开始录制；若另设速度 stop 策略，其运动速度限制仍生效。
`setup_confirmed=false` 不会阻止此模式的 preview/run；每次 run 以 `--execute` 明确确认现场准备，不再输入启动口令，
这不是程序代为确认现场安全或标定。**超过力值或移出工作区不会自动中止，操作者须全程监护。**

如需恢复力/力矩阈值中止，设置 `workspace_force_policy="stop"`，并填写以下字段。
它不会恢复 XYZ 范围检查，完整运动路径仍须现场确认：

- `max_force_n`、`max_torque_nm`：传感器原点、原坐标轴下，扣除所填电子零偏后力/力矩的模长上限；包含工具重力，不是桌面法向接触力。由设备和现场批准，不照搬论文或探索时的 40 N。
- `setup_note`：记录训练斜面位置/倾角、海绵、含水状态、工具安装、负载、限值依据及标定状态。8 条中保持这些条件不变。
- 核对 `robot_sn`、`expected_eef_type`、`sensor_port` 和关节范围。要恢复关节速度中止，另设 `joint_speed_policy="stop"` 并填写 `joint_speed_limit_rad_s`（大于 0、不超过 0.5 rad/s，不是速度指令）。缺少策略字段的旧配置仍按 stop 处理。
- 核对 `surface_id`、`exploration_id`。默认 `normal_exp` 是关联名称，不会自动生成真实探索数据。
- 最后才将 `setup_confirmed` 改为 `true`。它确认现场采集条件，不表示 TCP/FT 已标定。

上述空值仅在 `stop` 模式阻止运行。不要用巨大限值代替 `record-only` 的明确记录。
两种模式均不修改 SDK/固件保护，不构成安全级保护。示教应始终在人员可控的低力条件下进行。

## 命令

所有命令从项目根目录执行，使用已有 SDK 5.2.2 环境，不改用旧 5.1.6 环境。

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install -r requirements/manual_demonstrations.txt
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate preview --config configs/real_training/airbot_demonstrations.json
```

`preview` 纯离线，当前默认 record-only 配置应退出 0，显示 `setup_error: null`。
如切回 stop 模式而未填写现场配置，则退出码 2 并列出原因。
不需要训练环境、PyTorch、ROS 或 MuJoCo。
默认图窗要求桌面 Tk/Qt 后端，缺失时在连接硬件前失败；离线 `preview/status` 不打开图窗。
无桌面时仅可显式 `--no-plot` 关闭显示，清零和数据记录不受影响。

终端 A：仅在没有已有服务且现场安全确认完成时启动。已有正常服务不要重复启动。

```bash
mkdir -p /home/wp/airbot-logs-5.2
env MALLOC_ARENA_MAX=2 AIRBOT_LOG_DIR=/home/wp/airbot-logs-5.2 \
  airbot-arm --address 127.0.0.1:50051 -i can0 -t airbot_play --no-return
```

终端 B：先只读检查，再采集。不要同时运行旧 teach、自动探索、传感器绘图或其他串口读取程序。

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose inspect
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate run \
  --config configs/real_training/airbot_demonstrations.json \
  --output runs/real_demonstrations/manual/session_001 --execute
```

1. 执行带 `--execute` 的命令前托稳机械臂；不再等待启动口令。仅看到 `Gravity compensation active` 后才拖动。
2. 首次先将工具移到无接触姿态，保持朝向稳定，输入 `z` 回车确认无接触并清零 1 秒。
   看到 `Tare complete` 后手动选择任务起点，输入 `s` 回车，无需先静止。
   显示 `RECORDING` 后记录自然擦拭 10 秒，并在图窗显示清零后的六轴曲线。
3. 看到 `STOP wiping` 停止擦拭，继续托稳。时序检查通过且现场确认整段无误时，输入 `a` 接受；其他输入拒绝重采。不要预先输入命令。
4. 每段之间手动选定起点和朝向，输入 `s`。第一次记录起点仍为本会话参考。
   起点位置现改为仅记录：显示偏差 mm，不再因为超过 2 mm 拒绝录制，也不会平移/裁剪实际轨迹。
   起点朝向也仅记录偏差 rad，不再因为与参考朝向相差超过 0.02 rad 拒绝录制。
   录制前静止检查已取消；四元数有效性、关节范围、传感器时效和控制状态检查仍保留。
5. 接受第 8 段后自动请求 idle，全程保持支撑。提前退出输入 `q`；采集中紧急中止用 Ctrl+C 并执行现场安全流程。
6. 必须看到 `Idle confirmed`。看到 `idle NOT confirmed` 时状态未知，不能放手或反复重启。
7. 退出并清理硬件连接后自动保存已接受数据的 XYZ 轨迹和六轴力/力矩图到会话 `plots/`。
   满 8 条绘制完整组，提前退出仅绘制已接受条目；重复导出使用新编号目录，不覆盖已有图片。
   运行命令不变，`--no-plot` 不关闭退出导出。详见 [图像文件与离线补画](manual_tare_live_plot.md)。

清零会话可以续采，必须在 `--output` 中使用首次输出的实际路径（包括时间戳目录），配置必须完全一致。
每次重新连接都要重新输入 `z`，不自动复用上次连接的基线；同一连接期间可在下一条开始前重新 `z`。
清零不在录制中执行。旧原始力会话不能用当前默认配置续采；原始文件和旧参考不改写。
关闭图窗只停止显示，不退出采集；安全退出仍使用终端 `q` 或 Ctrl+C。

不需要重启正常运行的 `airbot-arm` 服务，但旧采集进程必须先托稳退出，再执行新命令。
变更工具/海绵/斜面/限值需用新会话目录，不混合成同一组 8 条。未接受的尝试不会自动算入条数。
不要并发运行；输出目录有进程锁。同一设备仍须人工确保无其他控制客户端。

## 数据与验收

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate status \
  --output runs/real_demonstrations/manual/session_001
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot_demonstrations.py' -v
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot_initial_pose.py' -v
```

`status` 纯离线，校验已接受 JSONL 的 SHA256。8 条时退出 0，不足 8 条退出 1。
`demo_01.json` 至 `demo_08.json` 是人工接受记录，关联 `attempt_*.jsonl`；
`reference.json` 保存共同起点，`session.json` 保存配置和坐标语义，`connection_*.json` 保存固件身份，
新尝试的 `attempt_*.start.json` 保存初始位姿、原参考位姿及 `reference_start_deviation`，
其中 `position_delta_sdk_m` 为当前位置减参考位置的 XYZ 偏差（m），`position_distance_m` 为欧氏距离，
`position_policy="record-only"` 明确该次运行语义；接受条目也包含相同偏差字段。
朝向更新后还保存 `orientation_policy="record-only"` 和 `orientation_angle_rad`（最短旋转夹角，rad）；
四元数正负号等价不会产生虚假的朝向差。终端同步显示偏差，原始位姿不旋转、不校正。
不同工具朝向会改变传感器轴向及重力分量，应在后续数据审核和标定转换时考虑，不代表示教质量自动合格。
旧记录没有这些字段，不会被追溯改写或假定为新策略采集。
`tare_*.json` 保存每次基线及其观测样本；`sensor_<tare_id>.csv` 保留清零成功后直到重新清零或退出期间的全速读数，
`raw_*` 列是原始值，`net_*` 列是减去该次基线后的值（包含等待和手动调整，不等于单段 10 秒数据）。
每条 JSONL 的 `start.tare`、`.start.json` 和接受记录均带基线，样本 FT 带 `tare_id` 与 `tared_sensor_wrench_si`。
中断日志即使存在也不能手动补写结束标记作为成功数据。

质量检查要求完整 10 秒、位姿与 FT 接收间隔不超过 20 ms、位姿中位采样周期在 10 ms 的 ±10% 内、
单次状态读取不超过 20 ms。传感器陈旧超过 20 ms、无控制权或超关节范围会中止会话。
仅当 `joint_speed_policy="stop"` 时才因运动速度超限中止；默认只记录，但 NaN 等无效反馈仍拒绝。
只有 stop 模式还会因过力/力矩中止；record-only 不会。两种模式均不检查项目 XYZ 边界。
检查不能证明 SDK 缓存读数新鲜、真实接触持续或动作正确，仍需人工逐段确认。
SDK RPC 设置 deadline，但不是实时系统或独立急停，串行多次状态查询不保证能稳定达到 100 Hz。
时序不过关应先检查通信/负载并重采，不填充、复制帧或放宽训练门槛。

**采满 8 条原始日志不等于训练输入已就绪。** 当前 `training_ready=false` 是刻意保留的状态：
位置使用 SDK 末端坐标，不要求 TCP 标定；FT 位于传感器原点和原轴；机器人与传感器为主机接收时间，
没有硬件同步时间戳。CSV 同一串口批次的多帧共享接收时间，不能假设各帧具有独立硬件时间。
采集 JSONL 分别保留 pose `host_monotonic_s` 和 FT `sensor_receive_perf_s`，不能未核实时钟实现就混用。

新清零人工会话仍会被旧原始载荷导入器明确拒绝，避免忽略已记录的清零值；现在可使用
[清零人工示教导入与训练](manual_tared_training.md) 的显式参数，配对条件已确认一致的完整 manual-start 探索。
原会话的 `training_ready=false` 保持不变，训练就绪以新派生数据的 inspect/prepare 验收为准。
历史原始力会话进入 [真实离线训练](real_training.md) 前仍需按所选 profile 核对力输入处理和时钟约定。
仅传感器标定流程要求到 `ft_frame` 的变换（含力臂项）和电子零偏；还需同一 Normal 海绵的
4 秒真实探索，完成经过审计的导入，再由 `real_training inspect` 检验 `raw.h5`。
本入口不伪造标定矩阵、不生成合成数据代替真实探索，也不直接输出 `raw.h5`。
已有 `configs/real_training/real_training_paper.yaml` 的训练命令保持不变；不要对这些 JSONL 直接运行 prepare。

软件验证（2026-09-10）：SDK 5.2.2 环境下 `test_airbot*.py` 共 89 项测试通过，
包含 10 项新增示教测试；默认 preview 正确拒绝未确认的现场配置。未连接硬件或采集真实示教。
以上为初版 stop 配置验证；record-only 更新后的测试和默认 preview 按本节新说明执行。
旧无绘图版本不要求 matplotlib；当前默认实时曲线需要 `requirements/manual_demonstrations.txt` 中的绘图依赖。
