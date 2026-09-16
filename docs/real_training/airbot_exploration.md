# AIRBOT Play 5.2.2 真实探索动作

2026-09-15 输出规则更新：新结果自动增加 `MMDD_HHMMSS` 时间目录（如 `0915_142205`）。
后续关联、状态查询及续跑使用终端打印的实际路径；多阶段训练使用打印的 `run_config.yaml`。
下文未带时间层的历史路径示例不代表新文件的实际路径，完整新命令见 [runs 输出操作步骤](../run_outputs.md)。

2026-09-15：默认入口已改为 **manual-start**：拖拽后按 `h` 固定，按 `s` 清零并探索，
回撤后保持，输入 `IDLE` 退出。无独立起点文件、无起点匹配/静止/载荷阈值验收。
重力补偿拖拽期间（含 h 捕获参考）不执行项目侧速度阈值检查，速度防护仅依赖 SDK/固件；
切入 servo 后仍执行 1.2 rad/s 实测超速停止。其他健康、控制权、数据和力流时效检查不变。
完整命令、保留的保护与新示教训练关联见 [手动起点探索](manual_start_exploration.md)。
下文属于显式 `--mode force-guarded` 旧模式或历史记录；运行其命令须补上该模式参数，
不要把旧模式保护套给新的默认流程。

2026-09-12：已按用户要求移除项目 XYZ 工作空间边界。无需填写
`workspace_min_m/max_m`，也不再因这些边界拒绝起点、目标或实际反馈。
下方旧现场记录中的工作空间描述仅为历史说明；当前流程见
[边界移除与运行步骤](../robot_control/workspace_bounds_removed.md)。

实时显示清零后六轴力/力矩可在 `run` 命令中增加 `--plot`；默认无窗口。
显示在独立只读进程中运行，具体命令、依赖与关闭行为见
[探索实时曲线](real_exploration_live_plot.md)。

当前 force-guarded 探索会在进入 servo、发送运动目标之前进行约 1 秒软件清零。
必须在已确认的 1 mm 非接触间隙处保持静止，不能在海绵接触或压缩时清零。
清零后数据新增为 `force.tared_sensor_wrench_si`，原始值与原保护阈值含义不变。
运行步骤和日志字段见 [探索协议操作说明](../exploration_protocol.md#探索前软件清零)。

2026-09-11：当前按压已改为 `0.005 m/s × 2 s = 10 mm`，回撤同步为 `10 mm / 2 s`。
下方 2026-09-10 的现场记录及旧日志仍保留原 20 mm 行程信息；当前运行以第 1 节及
[探索协议操作说明](../exploration_protocol.md) 为准，旧数据不可直接用于新协议训练。

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

入口：`python -m scripts.real_training.airbot_exploration`。更新：2026-09-10。
默认仅离线预览，不连接机械臂或串口。本脚本没有经过真机运动验收，不能无人值守运行。
它执行训练前的按压和横移探索，不执行策略网络输出，不生成完整示教数据集。

### 现场进度与当前命令（2026-09-10，新起点 007）

最新规则：操作者要求所有运行跟踪误差只记录，当前配置
`tracking_error_policy=record-only`，优先于下方历史横移规则。
从开始发送目标后直到下压、横移、回撤及退出等待，全程不因相对目标的位置误差
（包括法向/侧偏）或姿态误差停止。0.15 rad 和 5 mm 仅作为报告参考，不是运行门槛。
起点匹配与初始观测仍检查，绝对工作空间、关节位置/实测速度、力/力矩、状态/控制权、
传感器时效及循环超时保护不变。只支持 force-guarded，未填该字段仍为旧 stop 行为。
不扩大实际工作空间，不改变目标轨迹或指令速度，误差记录并不证明其为正常现象。
取消姿态跟踪停止后，局部平移包围盒不能保护工具边缘倾斜扫掠，现场监护始终必需。

`exploration_ft_006.jsonl` 已保存协议 0.01–4.00 秒共 400 帧有限六维力，
在回撤第一条指令发送前因旧跟踪门槛停止，回撤采样为零。原始探索数据保留可供后续分析，
但不称为全会话成功或已标定的编码器输入；不要补写成功状态或为了回撤标志丢弃数据。
本次不执行硬件恢复、回撤或运动。若另做一轮完整会话，新文件为 `exploration_ft_007.jsonl`。

观测增加 `tracking_error_record_only`、`position_tracking_exceeds_limit` 和
`orientation_tracking_exceeds_limit`，保留实际数值及全部力/姿态数据。
`motion_complete` 仅表示目标序列发送完成，`completion_semantics` 明确该含义；
同时记录 `return_position_error_m`、`return_orientation_error_rad`、
`return_within_tracking_limits`。返回偏差不会阻止正常退出，但也不再打印“已回到起点”。
`session_complete` 仍表示 idle 交接完成，不代表实际轨迹完美或完整卸载。

历史调整：操作者指定运行姿态容差由 0.02 改为 0.15 rad（约 8.59 度），
配置字段为 `orientation_error_limit_rad`。仅当前 force-guarded 运行检查使用该值，
起点匹配仍为 0.02 rad；空中及无力接触模式不放宽。字段缺省值为原 0.02 rad。
每帧及故障观测记录 `orientation_error_rad` 和 `orientation_error_limit_rad`，
报错打印实际误差与阈值。目标姿态仍固定，不主动旋转工具。
0.15 是操作者指定的调试容差，不是安全额定值；工具边缘倾斜位移随力臂增大，
原平移工作空间不能证明边缘不会碰撞。现场需确认新增倾斜余量，软件不替代碰撞检查。
`exploration_ft_005.jsonl` 保留 356 帧，因原姿态门槛停止且未正常回撤；后续一轮
为 `exploration_ft_006.jsonl`，结果与当前规则见本节开头。

当前配置使用 `archive/real_training/real_robot/exploration_start_007.json`。操作者恢复后重新摆位，
于 22:18:55 保存 007，22:18:57 退出并确认 idle。
局部边界仅平移到 007，仍保留原 5 mm 监测余量；指令速度 0.4、实测速度停止阈值
1.2 rad/s、运行姿态容差 0.15 rad、横移记录规则及力/力矩限值保持不变。
`exploration_ft_004.jsonl` 已保存 232 帧，在 J6 实测速率 0.402930409 rad/s 时因原
0.4 rad/s 阈值中止，未正常回撤。随后一轮是 `exploration_ft_005.jsonl`，
当前下一轮文件名见本节开头。运行前仍须安全恢复并就位，通过只读 `check`。
操作者按记录步骤确认新起点的
1 mm 间隙、20 mm 下压的硬安装件余量和 +Y 方向 50 mm 横移空间。
工作空间按新起点平移，仍保留原先 5 mm 监测余量，不合并新旧工作空间或扩大范围。
操作者指定将实测速度停止阈值改为 1.2 rad/s，现已与指令限速分开：
`joint_speed_limit_rad_s=0.4` 仍只作为 SDK 指令关节限速，
`measured_joint_speed_stop_rad_s=1.2` 对任一轴实测速率绝对值执行单次超限停止。
不滤掉速度峰值、不等待连续超限；恰好等于阈值允许，超过立即停止。
1.2 是操作者指定的调试阈值，不是厂家安全额定值，是指令限速的三倍，保护余量明显减少。
SDK `set_arm_speed` 和每次 Cartesian 请求的 `vel` 均保持 0.4，不会发送 1.2 的速度目标。
未填写新增字段时仍按原指令速度停止；独立阈值仅允许用于 force-guarded，配置不得低于
指令速度或高于本次支持的 1.2 rad/s。该上界不是硬件安全认证。
笛卡尔目标仍为原 4 秒协议，其他力/力矩、位置、姿态、时效保护不变。
合力 40 N、合力矩 0.8 N*m、起点合力 32 N 和六轴 eff=8 不变。

`exploration_ft_001.jsonl` 是首次超速停止的 98 帧日志；触发超速的那次状态未写入，
不能据此认定为误报。`exploration_ft_002.jsonl` 因新位置超出旧配置工作空间而中止，
没有探索采样。`exploration_ft_003.jsonl` 保存了 210 帧，在横移起步时因旧版 5 mm
三维跟踪门槛停止。历史文件均保留，不覆盖旧日志。
录制新起点不会自动修改探索配置；以后重新摆位也需同步核对起点文件和局部边界。

以下为历史上仅横向跟踪记录规则，仅在 `tracking_error_policy=stop` 时生效：
`lateral_tracking_policy=record-only`。
该选项只允许用于 `force-guarded` 模式，缺省仍为 `stop`；空中和无力接触模式不放宽。
只在协议 2–4 秒横移阶段的发送前后检查中，将沿 `slide_direction_sdk` 的误差作为质量
记录，不据此停止。法向和横向侧偏组成的垂直于滑动方向误差仍执行 5 mm 模长限制；
这使用配置的桌面坐标轴投影，不假定 SDK Z 必然是桌面法向。
起点、下压、回撤、退出前仍执行原位置检查，工作空间、姿态、关节速度、电流、
六维力和时效保护不变，不增加速度或负载上限，也不改变 4 秒/400 个目标的轨迹。
若横移返回结束后仍离起点横向过远，会在回撤前停止，不自动追赶或回撤。

每次观测记录 `normal_error_m`、`slide_error_m`、`cross_slide_error_m`（实际减目标，
单位 m），以及 `slide_error_record_only`、`slide_tracking_exceeds_5mm`。
正常结束汇总最大横移误差及超过 5 mm 的采样数，1 mm RMS 仍仅作质量指标。
即使记录满 400 帧，也必须检查实际运动和力响应，不能把横移未发生称为论文动作完成。
观测触发停止时，先请求原软件急停，再在 `aborted.fault_observation` 保存已读取状态、
目标、误差和读取时刻，`context` 标明阶段、索引和发送前/后；超速报错同时显示轴号和速度。
若触发位置/速度保护时尚未读取 FT，该字段为 null，不为完善日志而延迟停止或补读。
SDK 状态读取本身失败时可能没有 fault_observation；旧日志不会补写或改造。

以下为先前连通性与起点 002 的记录：

已保存 `archive/real_training/real_robot/exploration_start_002.json`，退出拖拽后确认 idle。
随后只读检查与记录起点相差约 0.11 mm；这是当次测量，不保证后续位置保持。
配置已填写实际机械臂编号、NULL 末端类型、新起点文件和 FTDI 传感器稳定路径。
力传感器已完成约 3 秒连通性采集，3215 帧保存于
`archive/force_sensor/real_robot/ft_connection_20260910_001.csv`；未清零，不是探索轨迹。
随后操作者确认 1 mm 起点间隙、20 mm 下压及 +Y 方向 50 mm 横移空间，并确认
首轮调试限值：起点合力 32 N、运行合力 40 N、合力矩 0.8 N*m、六轴 eff 均为 8。
实际传感器为 KWR75A，各轴力量程 50 N、力矩量程 2 N*m。以上是用户批准的试验
停止阈值，不是厂家认证的装置安全额定值，不使用传感器过载能力作为运行余量。
`calibration_confirmed=true` 记录现场确认，不代表自动完成了 TCP 或电子零偏标定。
该次历史采集电子偏置为零，保护作用于包含重力与零偏的原始载荷，当时不在起点自动 tare。

当前局部工作空间由新起点的目标轨迹包围盒向各方向扩展 5 mm 计算，边界保持不变，
不复用空中模式的绝对工作空间，也不是整臂碰撞认证。
六轴关节位置边界沿用该机械臂已有配置；最初速度限值为 0.2 rad/s，
当前使用操作者确认的 0.4 rad/s，见本节开头。
名义可容许压缩按现场确认记为 19 mm；位置误差可能增加实际压缩，必须确保硬安装件
仍有余量。若现场只能容许恰好 19 mm 而没有误差余量，不要运行，先调整装置。

项目硬盘发生过 USB 断连重挂载。旧终端即使提示符路径正确，也可能仍引用失效目录，
表现为模块找不到或 SDK 创建相对 `logs` 目录时出现 I/O error。先重新进入目录：

```bash
cd /
cd '/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping'
env PYTHONPATH='/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping' /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose inspect
```

若仍出现 I/O error，停止运行并排查硬盘，不用 sudo 绕过。采集期间不要拔插项目硬盘。
串口重新插拔后，如访问被拒绝，核实 `/dev/ttyUSB0` 为上述 FTDI 设备后执行：

```bash
sudo setfacl -m u:wp:rw /dev/ttyUSB0
```

现场配置完成后，在项目根目录使用以下只读检查命令（不运动、不打开串口）：

```bash
env PYTHONPATH='/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping' /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check --mode force-guarded
```

此前起点更新只修改配置；本次按操作者确认取消所有运行位置/姿态跟踪停止并保留日志。
探索数据的 1 mm RMS 指标仅作质量记录，不作为丢弃数据的条件；不为追求仿真跟踪精度反复重采。
假硬件测试已隔离现场配置的 `expected_eef_type=NULL`，并固定测试速度为 0.2 rad/s，
避免现场调参改变假硬件故障用例；继续覆盖带夹爪与无夹爪场景。
测试命令仍为第 6 节的 AIRBOT 无硬件测试命令。

上次异常后没有正常回撤。先按现场规程支撑和恢复机械臂，不在软件中自动解除急停。
重新就位后执行上述 `check`，通过后才在现场终端执行下列命令。
若重录了新起点，需先更新配置与边界，不能只更换输出文件名。

```bash
env PYTHONPATH='/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping' /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run --mode force-guarded --execute --time-scale 1 --output runs/real_exploration/exploration_ft_007.jsonl
```

`--execute` 授权连接硬件，检查通过即尝试运动，不再等待启动口令。4 秒探索（400 个目标采样）后，另用 2 秒
回撤 10 mm；回撤不属于编码器输入。回撤后仍为 servo，按提示安排安全支撑再输入
`IDLE`，看到 `session_complete` 才完成退出。中止后不自动重试、不自动提高限值，
已有日志禁止覆盖；保留中止日志分析原因。当前控制代码仍有 5 ms 迟到及 20 ms FT
过期停止条件；此次没有放宽这些实时保护，不能保证本机一定完成全程。
JSONL 保留原始六维力、实际位姿、目标、时间戳与阶段，仍为 sensor local / origin。
后续属性编码需要处理与训练坐标系的对应，不能把 `encoder_ready=false` 改成 true
或将本次原始日志直接当作已经标定的编码器输入。

当前选择的无接触运动验证使用独立 `--mode air`，操作见 [空中运动验证](../robot_control/airbot_air_motion.md)。
需要重新记录空中起点，当前下移 10 mm、横移 50 mm 往返；不使用旧接触起点、不读取力。

默认模式仍为 `force-guarded`。“不读取六维力、仍接触海绵”的验证（当前下移 10 mm）
使用独立 `--mode contact-no-ft`，配置与步骤见 [接触运动验证](airbot_contact_motion.md)。
该模式不是空中运动，轨迹不缩短，不能作为力控或训练数据验收。

## 1. 与仿真的对应

真机 `exploration_protocol.exploration_offset()` 对应仿真 `PretrainingWipe.rollout()`
调用的 `simulation.exploration_offsets()`，通过包含起点的 401 个采样点逐点相等测试。
2026-09-11 按用户要求将按压改为 `0.005 m/s × 2 s`，名义下压 10 mm。
历史检查点锁定的是旧源码；不更新或伪造其来源记录，不与新协议数据混用。
离线检查与运行注意事项见 [探索协议操作说明](../exploration_protocol.md)。
仿真原来的运行命令不变，两份 `pretrain_paper*.yaml` 都使用以下默认轨迹。

| 协议时间 | 桌面坐标系相对位移 `[X,Y,Z]`，m | 动作 |
| --- | --- | --- |
| 0 s | `[0,0,0]` | 海绵未压缩底面距桌面 1 mm |
| 2 s | `[0,0,-0.010]` | 5 mm/s 下压 |
| 3 s | `[0,0.050,-0.010]` | 保持高度，50 mm/s 沿 +Y 移动 |
| 4 s | `[0,0,-0.010]` | 保持高度，50 mm/s 沿 -Y 返回 |

姿态固定为记录起点的 SDK `xyzw` 四元数，不绕工具局部轴解释平移。
`table_normal_sdk` 是桌面向上的单位法向；`slide_direction_sdk` 是仿真 +Y 在
SDK 参考坐标系的单位向量，两者必须正交。目标为：

```text
p_sdk(t) = p_recorded + slide_direction_sdk * dy(t) + table_normal_sdk * dz(t)
```

固定工具、固定姿态下，SDK 末端和海绵 TCP 的平移增量相同；但绝对接触间隙仍须实测。
不要把仿真的世界坐标、桌高 0.9 m、关节初值直接发给真机。
完成 400 个探索采样后，另用 2 秒沿法向上退 10 mm 回到起点，回撤数据单独标记。
**4 秒时仅横移返回，Z 仍处于下压状态，并未卸载。**
不复制仿真的 80 mm 卸载、仿真 reset 或从任意当前位置移至起点的准备路径。

100 Hz 是目标发送/采样节拍，不是已经测得的硬件实时性能。
SDK servo 的伺服增益、逆解、速度限制、真实海绵和接触动力学不等于 MuJoCo 控制器。
本脚本对照验证目标定义，并记录实际跟踪误差；不能保证真实位姿和六维力逐点等同仿真。
不要为追求误差指标增大电流、力或速度限值。

## 2. 环境和服务

在项目根目录运行，继续使用已安装 SDK 5.2.2 的独立环境，不混用旧版 `airbot_py`：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install -r requirements/robot_control.txt
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore preview --mode force-guarded
```

预览只依赖标准库，可在安装 numpy/pyserial 前运行。`run` 复用现有 KWR75 串口读取器，
必须安装上面的两项依赖；SDK 环境原先没有它们。不要向 SDK 环境安装 MuJoCo 或训练依赖。
服务启动和急停/支撑要求见 [初始位姿操作](../robot_control/airbot_initial_pose.md) 第 2–4 节。
已正常运行的服务不要重复启动，必须确认其启动参数含 `--no-return`。
脚本不启动/重启服务，不使能或清错，不解除急停，不调用 `return_zero()`。

KWR75 不能同时被绘图器或其他采集进程占用。确认 `/dev/ttyUSB0` 确实是力传感器，
而非机械臂 USB-CAN；建议在配置中使用经核实的 `/dev/serial/by-id/...` 稳定路径。
本脚本不自动选择或修改串口/CAN 参数。

## 3. 初始位置和现场配置

先按 [初始位姿操作](../robot_control/airbot_initial_pose.md) 手动摆位并记录真机起点：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose teach \
  --execute --output runs/real_exploration/initial_pose_001.json \
  --label exploration_start --tool-note "Actual installed sensor, mount and sponge"
```

起点必须是底座固定、海绵面平行于桌面、**未压缩海绵底面实际离桌 1 mm** 的姿态。
早先记录的高处准备位姿不能直接当作探索起点。旧记录若已存在，使用新文件名，不覆盖。
单个 SDK 位姿无法证明 TCP、法向、海绵厚度或间隙正确，必须测量并记录标定依据。
拖拽结束要退出 teach 并释放控制权；idle 不承诺保持位置，先安排安全支撑。

编辑 `configs/real_training/airbot_exploration.json`，或准备独立 JSON 并通过 `--config` 指定。
模板中的 `null` 和 `calibration_confirmed=false` 是有意设置的拒绝执行门槛，
不能照抄示例或只把布尔值改成 true 来绕过现场确认。

| 字段 | 现场需要填写/确认的内容 |
| --- | --- |
| `calibration_id`, `robot_sn`, `sdk_frame_and_tool_note` | 标定记录编号、与起点文件及在线固件一致的机械臂序列号、SDK 参考系/末端定义及实际工具说明 |
| `initial_pose_file` | 正确的真机起点 JSON；相对路径始终相对项目根目录 |
| `table_normal_sdk`, `slide_direction_sdk` | 实测单位向量；模板 Z 向上/Y 横移只适用于确已对齐的安装 |
| `initial_gap_m` | 固定为 0.001，且必须物理测量，不是用于修改起点高度的参数 |
| `verified_compression_allowance_m` | 新协议名义压缩为 0.009 m，现场允许量还须覆盖误差和安全余量，且硬安装件不会触桌；保留现场已确认限值，不自动重写 |
| `joint_min_rad`, `joint_max_rad` | 六轴现场保守关节界限，保留距机械限位和奇异位姿的裕度 |
| `joint_speed_limit_rad_s` | 默认 0.2 rad/s，脚本上限 0.5；是 SDK 关节限速，不是笛卡尔速度或厂商安全额定值 |
| `measured_joint_speed_stop_rad_s` | 当前 1.2 rad/s，任一轴实测速率绝对值超过即停止；不修改指令速度，未填写时沿用 `joint_speed_limit_rad_s` |
| `orientation_error_limit_rad` | 当前运行姿态容差 0.15 rad；缺省 0.02，仅 force-guarded 可调整，起点匹配仍为 0.02 rad |
| `tracking_error_policy` | 当前 record-only，运行位置/姿态误差仅记录；上述姿态值仅作报告参考。起点及绝对安全边界仍检查 |
| `joint_current_limits`, `eef_current_limit` | 根据实际电机/工具和厂家规定选择的六轴及夹爪电流限值；不是 N 或 N*m，不能抄仿真的 torque_limits |
| `sensor_port`, `sensor_bias_si` | 传感器串口及独立标定的电子零偏 `[N,N,N,N*m,N*m,N*m]`；不是接触后 tare，也不去除工具重力 |
| `max_force_n`, `max_torque_nm`, `max_initial_force_n` | 传感器原点处去电子偏置后的三维力/力矩模长阈值和起点力阈值，包含工具重力；根据工具和传感器额定值选定 |
| `calibration_confirmed` | 上述条件、硬件急停、支撑、负载和路径确已验收后设 true |

没有通用安全按压力，故代码没有替用户虚构阈值。10 mm 名义下压与海绵厚度/硬度不适配时，
本脚本应当拒绝使用；缩短轨迹需要同步修改仿真并重做相应数据，不能冒充原探索协议。
配置不构成碰撞检测器，也不做整条路径的 IK/关节连续性预求解，必须先按厂家流程验收
该工具、姿态和局部运动范围。首次接触试验可用更慢模式，但减速本身并不降低最终压缩力。

## 4. 只读检查与执行

填写配置后先检查（连接 SDK 仅读取状态，不申请控制权，不打开传感器）：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check --mode force-guarded
```

要求控制器为 idle。当前位姿与记录的差距须不大于 1 mm、0.02 rad、任一关节 0.03 rad；
偏差超限只报错，不自动移动到起点。也不因容差合格就证明 1 mm 间隙合格，间隙仍需现场核实。
`check` 输出实际 SDK 目标关键点供操作者核对，不验证力传感器或物理标定。
`run` 在申请控制权前还检查 11 个静止样本和实时起点力，切换 servo 后再核对起点。

首次经过现场运动验收后，可用 5 倍时间进行有人监护的低速调试：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run --mode force-guarded \
  --execute --time-scale 5 --output runs/real_exploration/exploration_slow_001.jsonl
```

该模式 20 秒探索、10 秒回撤，采样仍为 100 Hz。它不匹配 4 秒训练协议，不能用于原编码器。
实际速度改变会改变接触力，低速试验通过不代表正式速度已经验收。
正式目标协议命令为：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run --mode force-guarded \
  --execute --time-scale 1 --output runs/real_exploration/exploration_001.jsonl
```

1. 执行带 `--execute` 的命令前完成现场检查；不再等待启动口令。全程必须有急停监护和安全支撑方案。
2. 正常通过检查后开始探索和回撤。禁止同时拖拽、运行其他控制脚本或拔插连接。
3. 正常回撤后 servo 仍激活，持续检查位姿和力。不要关终端或假设断开后可保持。
4. 按现场规程安排安全支撑后输入 `IDLE`，脚本请求 idle、确认响应，再关闭客户端。
   不要把手伸入夹点内托接运动中的机械臂。若无法安全支撑，先按硬件规程处理。
5. 日志末尾出现 `session_complete` 才表示整个会话和 idle 交接完成。

SDK servo 指令同时携带夹爪目标。脚本读取并保留当前夹爪开度，无法读取时拒绝运动。
例外仅为配置、记录和服务均明确确认 `expected_eef_type="NULL"` 的无夹爪装置，
使用不含夹爪字段的六轴 servo 请求；详情见上述接触运动验证文档。其他缺失反馈仍拒绝。
已知夹爪型号的反馈还必须在 SDK 5.2.2 的对应开度范围内，避免 SDK 静默截断后意外开合。
固定工具且无夹爪的服务配置必须先按 SDK/厂商要求核实，不能随意伪造夹爪反馈。

## 5. 停止和日志边界

每次目标发送前后检查六维力、服务/电机状态、控制权、关节位置/速度、
跟踪误差及运行姿态误差。当前 `tracking_error_policy=record-only` 下运行位置/姿态误差
只记录而不停止，起点匹配仍检查。默认 stop 模式使用旧跟踪保护，具体适用阶段见开头。
FT 超过 20 ms 未更新、命令拒绝、超过 5 ms
采样迟到或下一周期发送前已超时均中止，不跳点、不追赶补发，不静默放宽阈值。
两类电机的无错误码 0/1 均接受，其他码拒绝。

运行异常、Ctrl+C、SIGTERM 或终端 EOF 在已尝试切换 servo 后请求 SDK 软件急停；
不清急停，不重新争抢控制权，不在状态未知时自动回撤。停止失败会显式告警。
已有控制权但尚未尝试切换模式时仅释放客户端，不向其他控制器发送停止命令。
日志写入失败同样走停止路径，日志既有文件永不覆盖。

**软件保护不是安全回路。** SDK 5.2.2 的私有 stub 经项目内代理加入 RPC deadline：
一般调用 50 ms，控制器切换 1.5 s；不改已安装 SDK。deadline 仅限制客户端等待，
不保证服务端撤销已收指令。初始化前的 SDK 连接/固件查询沿用厂商实现。
SDK 状态有缓存、关节与位姿不是原子读取、传感器使用主机接收批次时间而非硬件时间；
这些检查不能证明底层传感器新鲜性，也不能保证 100 Hz 硬实时。
线程调度、RPC、磁盘卡顿、SIGKILL、断电或总线故障都可能阻止及时保护；
独立物理急停、现场监护及防坠支撑始终必需。不要通过删除 deadline/力检查来使运行通过。

JSONL 包含配置、原起点及 SHA256、固件、初始测量、每次目标、实际位姿/关节、
接收时间、原始/去电子偏置六维力、跟踪误差、阶段和结束/失败状态。
`motion_complete` 中分别报告名义协议匹配、位置 RMS 是否在仿真 1 mm 阈值内；
这些标志不代表动力学、传感器标定或完整训练数据验收。
回撤不计入探索。部分日志不能通过补齐或改完成标志当作有效探索。

当前力日志是 **真实 sensor local / sensor origin**，不是仿真的 `ft_frame`。
全部日志固定 `encoder_ready=false`。后续接入 [真实训练](real_training.md) 还需要
传感器到 `ft_frame` 的刚体 wrench 变换（包含力矩力臂项）、共享时钟验证、合格采样，
以及八段示教和对应 HDF5 schema。本次脚本不能直接把 JSONL 当成 `raw.h5`，
新增的起点软件 tare 只提供独立的相对载荷字段，不执行跨姿态重力补偿，
也不通过对齐/插值伪造 400 帧训练数据。

## 6. 无硬件验证

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot*.py' -v
/home/wp/miniconda3/envs/clean/bin/python -m pip install 'pyserial>=3.5,<4'
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -v
```

第一条使用假客户端/假传感器，不连接真机或串口。第二条还验证原有仿真/训练回归，
使用现有 `clean` 环境；不要用 SDK 环境运行需要 MuJoCo/PyTorch 的测试。

2026-09-10 验证结果：SDK 环境 44 项 AIRBOT 无硬件测试全部通过；`clean` 全量回归
214 项，213 项通过、1 项 SDK 类型测试跳过（该项已在 SDK 环境通过），耗时 77.516 s。
401 个采样点与原仿真函数完全相等。`simulation.py` 的 SHA256 保持为
`695074e69f1d24f9568e7387110ecdad346e06ac23f3f2a84d330cb44806f452`，历史来源检查通过。
SDK 保持 5.2.2，仅补装 numpy 2.2.6、pyserial 3.5，`pip check` 通过；
`clean` 补装 pyserial 3.5。预览和未标定配置拒绝执行的入口均已验证。

本次只读检查 `127.0.0.1:50051` 连接超时，`runs/real_exploration` 中未找到起点记录。
未启动服务、未修改 CAN/SDK 配置、未执行运动或传感器串口操作。上机前需要按现场实际
服务地址和起点文件重新检查，软件测试通过不能替代真机验收。
