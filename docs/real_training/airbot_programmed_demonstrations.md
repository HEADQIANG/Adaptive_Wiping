# AIRBOT 程序示教采集

当前为独立的 `airbot_programmed_fixed_depth_v2` 协议：与探索一样，位置目标只由时间和指定深度决定，
六维力用于记录及探索已有的超限停止，**不进行恒力或高度反馈控制**。
只采集、审核和续采，不修改人工示教、探索或训练模型。
已移除 10 N 目标力、闭环增益、滤波、死区和力轴确认等配置门槛。
2026-09-12 操作者明确确认非接触起点、完整左右路径、12 mm 下压、急停与支撑，当前配置已记录
`setup_confirmed=true`。这是操作者确认，不是软件独立验收；仍须执行下文只读起点检查。
未确认配置（`setup_confirmed=false`）的 preview 仍退出 2，run 拒绝。

## 动作与数据边界

配置：`configs/real_training/airbot_programmed_demonstrations.json`。
通过 `exploration_config` 读取探索配置，使用其中同一起点文件、机器人、传感器、方向、
关节速度/范围、电流及原始载荷阈值。会话冻结完整配置、起点记录及三个源文件的 SHA256。
擦拭期间位置/姿态跟踪策略沿用探索配置，包括其中的 `record-only`；程序不会修改探索配置。
启动自动到位段独立强制执行跟踪检查，不采用 `record-only`。

### 启动自动到位

操作者已确认允许直达移动，配置 `startup_approach.mode=direct`。每次运行（包括续采）在
`PROGRAMMED` 确认后，先检查当前姿态静止和原始载荷，进入 servo 并保持实测当前位置，
再从当前位置直线移动到探索起点，同时用最短旋转姿态插值。无需手动精确调到起点。
采用五次平滑时间曲线，峰值平移速度 `speed_m_s=0.01`（10 mm/s）、峰值角速度
`angular_speed_rad_s=0.1`，不是擦拭横移速度；擦拭仍为 0.05 m/s。
计算时长超过 `max_duration_s=60` 时拒绝启动，不截短轨迹或提高速度。
移动目标的位置误差上限 `tracking_tolerance_m=0.005`、姿态误差上限
`orientation_tolerance_rad=0.02`，保留原始载荷、关节速度、控制权和调度保护。
启动到位段使用 100 Hz 固定调度：唤醒迟到超过 5 ms 停止；状态读取、目标计算、命令发送、
CSV 和 JSON 写入必须在该轮计划开始后的 10 ms 内完成。超过周期立即停止，不跳帧或追赶发送。
5 ms 是调度迟到限制，不再作为上述所有操作的累计完成期限。原有 RPC 超时、20 ms 状态读取和
传感器陈旧检查保持不变；初始保持指令的观测与发送也检查 10 ms 预算。
日志 I/O 移到该轮命令发送之后，但仍计入周期，磁盘过慢仍会停止，不会静默漏记。
起始实测下压量已超过相对探索起点 20 mm 时也拒绝，不从未知接触状态自动解脱。
到位后仍须通过原有位置、姿态、关节和静止检查；逆解分支不一致时停止，不自动换分支修正。
到位完成不启动示教，清除提前输入，等待新的 `s`；空载基线仅在到位后逐条采集。

**直达不是避障规划。** 每次确认 `PROGRAMMED` 前，必须核实当前工具、机械臂各连杆到目标的
整个运动空间无障碍，不只核实末端直线。当前位置变化时须重新检查；有接触、障碍或路径不明时不要启动。
`direct_path_confirmed=true` 只记录操作者本次对直达方式的确认，不代表软件验证了碰撞安全。
更换现场布局应撤销确认并重新核实。程序不会自动启动机器人服务或清除急停。

| 组别 | 初始下压 | 条数 | 下压+横移名义时长 |
| --- | --- | --- | --- |
| nominal | 10 mm | 2 | 6.0 s |
| under | 8 mm | 3 | 5.6 s |
| over | 12 mm | 3 | 6.4 s |

固定顺序：nominal、nominal、under、over、under、over、under、over。
下压速度取探索协议的 5 mm/s；横移速度取探索协议的 0.05 m/s。
相对横移坐标为 `0 -> +50 -> -50 -> 0 mm`，对应左 5 cm、右 10 cm、左 5 cm。
“左”是探索首段方向，目前 SDK +Y，不是观察者视角。全路径含原探索未覆盖的反向一侧，必须重新现场核实。

先位置控制下压到分组深度，再在 4 秒横移阶段始终保持该名义深度。
即使实测力低于或高于 10 N，只要未触发探索原有保护，程序也不会加深或抬起。
`under/nominal/over` 仅表示 8/10/12 mm 三种深度，不保证实测接触力分别不足、适中或偏重。
下压和横移纳入有效片段；另保留 20 ms 尾部实测记录保证 FT 覆盖终点。
空载基线、等待、回撤及回位检查独立标记，不拼进擦拭片段。
回撤速度同下压速度，三组名义回撤时长分别为 1.6/2.0/2.4 秒，随后最多等待 2 秒确认停稳回位。
回撤现在独立允许 `retraction_lateness_limit_s=0.01`：发送前原为错过计划时刻即停止，
现在允许迟到最多 10 ms；发送返回、采样和日志完成也检查该轮计划时刻后的 10 ms 上限。
采样原来的 5 ms 迟到限制仅在回撤段放宽为 10 ms，擦拭段不变。
延迟后后续节拍顺延，目标步长保持 0.05 mm，相邻发送间隔至少 10 ms，不追赶补发；
因此实际回撤可能慢于名义 5 mm/s，并按实测时间记录，不保证回撤实测采样始终为 100 Hz。
整段命令回撤额外时间上限 `retraction_max_extension_s=1.0`，三组分别最多 2.6/3.0/3.4 秒，
不含随后原有的停稳检查。超过单轮或总时间预算仍停止，不自动重试或恢复回撤。
上述是软件检查预算，无法保证阻塞通信或系统故障时精确按时停止。
检查 11 帧、50 ms 间隔的静止状态，不能把发送回位指令当成实测回位。
起点和回位均复用探索的起点匹配检查，目前位置 2 mm、朝向 0.02 rad、各关节 0.03 rad。
等待接受、下一条或 idle 交接时继续检查回位；漂移超限中止，不继续接受或启动。

每条力报告保存横移开始前的六维力，以及横移期间原始/空载扣除后六轴均值、最小值、最大值和三轴合力统计。
不再报告“10 N 误差”“闭环恢复”或自动判断条件是否成立。
统计基于 100 Hz 观测，重复最近 FT 值不代表新的独立传感器测量。
没有有效接触、擦拭错误或实际轨迹不合格时，操作者应拒绝该次，不凭“命令完成”接受。

## 保护与现场确认

`motion_policy=fixed-depth`、`max_depth_m=0.02` 和三组深度固定，不再填写独立 `control` 参数。
每条开始前在同一静止、非接触姿态采集 1 秒空载基线；禁止接触去皮。
按操作者要求，程序示教独立使用 `stationary_speed_limit_rad_s=0.1` 作为静止检查的关节速度上限，
适用于本模式的启动检查、到位、基线和回位静止验证。探索、人工示教及起点记录工具的默认判据不变。
该值不是运动速度命令，也不是运行中的关节超速停止阈值；关节位置变化量 0.01 rad、
末端位置变化量 2 mm 和姿态变化量 0.02 rad 的静止判据仍保留，起点匹配检查也不放宽。

`baseline_retry_timeout_s=10.0` 是每条基线采集的总时间窗口，包含最初等待静止、重试和完整 1 秒采样，
不是每次失败重新获得 10 秒。若采样中静止判据不通过，记录 `tare_discarded` 并丢弃整段基线，
保持原有 servo 起点目标，在持续监测下重新取得 11 帧静止窗口，再从空数据开始采集完整基线。
无需再次按 `s`，成功后继续当前这条示教，不增加示教条数；丢弃的原始数据保留作审计，不参与去皮。
总时间到期仍未获得合格基线时停止该条并按原流程请求软件急停。
只重试静止检查失败；起点偏离、过力、运行超速、断流、控制权丢失和 I/O 异常仍立即传播至停止流程。
软件超时不能打断所有阻塞 I/O，10 秒不是硬实时安全保证。
六维力按原始传感器轴向保留符号、不滤波、不裁剪；空载扣除量仅作记录，不用于驱动高度。
传感器轴不等于已标定桌面法向，空载扣除不是完整重力补偿。

在 `setup_note` 记录实际表面、海绵、工具和现场验收情况；只有核实非接触起点、
完整正反向路径、12 mm 最大下压指令、保护限值、急停及支撑后，才设置 `setup_confirmed=true`。
探索配置的旧确认值不能自动批准新路径；只有操作者明确确认才记录为 true。

20 mm 是相对探索起点的总下压检查上限，不是计划下压量；实际指令最多 12 mm，实测超过 20 mm 中止。
探索配置的压缩允许量加初始间隙须覆盖 12 mm 计划下压，目前 1 mm 间隙意味着至少 11 mm 压缩允许量。
原始合力/力矩、控制权、关节、电机状态、20 ms FT 陈旧、5 ms 调度迟到检查仍保留。
当前探索 `tracking_error_policy=record-only`，擦拭运行位置和朝向偏差仅记录、不因其超限中止；
如探索配置改为 stop，本入口也按其规则中止。起点、实测回位与 20 mm 下压边界不因 record-only 取消。
不恢复全局 XYZ 工作空间检查。
当前继承原始合力 40 N、合力矩 0.8 N*m、起点合力 32 N 阈值；这些不是目标接触力或厂家安全额定值。
空载扣除不会绕过上述阈值，“只记录力”不等于取消过力保护。

## 运行步骤

从项目根目录使用 SDK 5.2.2 环境。已有正常 `airbot-arm --no-return` 服务不要重复启动，
服务及支撑要求见 [初始位姿操作](../robot_control/airbot_initial_pose.md)。
不能同时运行人工拖拽、探索、其他机器人控制客户端或占用传感器串口的程序。

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install -r requirements/robot_control.txt
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate preview --mode programmed
```

`preview` 不连接硬件、不创建采集目录，显示固定深度轨迹、时长、继承策略及现场确认状态；
当前已记录现场确认，preview 应退出 0；未确认时退出 2。软件检查通过不证明真机安全。
完成现场确认后，执行只读起点检查：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate check --mode programmed
```

`check` 读取 SDK 身份和 idle 状态，不申请控制权、不切换控制器、不打开传感器。
偏离起点不再直接拒绝，输出 `start_matches` 以及当前到目标的距离、转角和计划到位时长。
超出自动到位时长或边界仍拒绝。`check` 本身不移动，也不验证静止、力、避障或现场标定。

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate run --mode programmed \
  --config configs/real_training/airbot_programmed_demonstrations.json \
  --output runs/real_training/programmed_demonstrations/direct_start_session_004 --execute
```

1. 核实从当前位置直达起点的完整路径、非接触目标起点、擦拭路径及人员防护。机械臂停稳后输入 `PROGRAMMED` 并回车，才连接设备并自动到位；此后可能立即运动。
2. 自动到位并实测停稳后，看到等待提示再输入 `s` 并回车，表示确认非接触起点，开始本条静止检查和空载基线采集。到位运动期间提前输入的 `s` 作废。
3. 程序下压、固定深度横移、回撤；此时不要触碰机器人，不要提前输入下一步命令。
4. 实测回位停稳后查看报告。输入 `a` 并回车仅表示人工接受，其他输入拒绝重采，`q` 结束采集。
5. 接受后仍等待新的 `s`。运动中、上一提示期间提前输入的完整/部分命令会被清除，不排队执行。
6. 采满 8 条或输入 `q` 后，servo 仍保持受监测状态。按现场规程安排安全支撑，再输入大写 `IDLE` 并回车。
7. 必须确认 `Idle confirmed`。idle 不是位置保持，不能据此认为机械臂会承重。

同一命令、同一目录可续采，只补未接受的条件，拒绝的尝试保留但不计数。
程序配置、探索配置或起点文件变化（包括文件摘要变化）必须使用新目录。
本次新增自动到位配置已改变摘要，请用上面的新目录，不可续接更新前的固定深度会话。
启动时序修正另以 `startup_timing_revision=2` 冻结，不能续接旧时序会话。
本次增加 0.1 rad/s 静止判据和限时基线重试，配置摘要已改变，
基线重试版本使用 `direct_start_session_003`。本次回撤阈值更新又改变了配置，当前请使用命令中的
`direct_start_session_004`。不要覆盖旧目录；`003` 中已接受的第一条保留，不能自动跨冻结配置合并计数，
新会话从 0/8 开始。
本协议不能续接旧 `airbot_programmed_v1` 闭环会话；旧配置、记录不会被重写。
带有 `control`、`target_force_n` 或 `force_direction_confirmed` 的旧配置明确拒绝，不静默忽略这些字段。
已满 8 条的会话不会再次连接硬件。不要并发采集；目录锁不是设备级控制互斥。

异常、Ctrl+C、SIGTERM、终端 EOF 或写盘错误在尝试进入 servo 后请求软件急停，
不自动回撤、不清急停、不重新争抢控制权。若软件停止未确认，应使用现场硬件安全流程。
传感器延迟、RPC 缓存、系统调度、断电或 SIGKILL 都可能使软件保护失效；不是安全级控制系统。

## 数据审核与测试

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate status --mode programmed \
  --output runs/real_training/programmed_demonstrations/direct_start_session_004
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest tests.real_training.test_airbot_programmed_demonstrations -v
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.real_training.test_airbot_demonstrations \
  tests.real_training.test_airbot_exploration \
  tests.robot_control.test_airbot_initial_pose -v
```

`status` 纯离线，校验连续条目、条件顺序、原始文件 SHA256、完成事件、样本序号和实际时序覆盖，
显示每条力报告与回位误差。8 条退出 0，不足 8 条退出 1。
使用训练环境可补跑需要 h5py/PyTorch 的导入拒绝测试及旧训练回归：
SDK 环境缺少 h5py 时，新增测试中的导入拒绝项会跳过；不要在此环境用全目录
`test_airbot*.py` 通配符代替上述定向命令，因为它还会加载依赖 h5py 的部署、标定和训练测试。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.real_training.test_airbot_native_training \
  tests.real_training.test_real_training -v
```

`session.json` 保存冻结配置；`connection_*.json` 保存设备身份；`sensor_*.csv` 保存全速原始传感器数据。
`approach_*.jsonl` 独立保存启动自动到位的初始实测状态、计划时长、移动目标、实测位姿、原始力和停稳结果，
不计入擦拭片段或接受条数；异常关联该到位文件，不自动回撤或重试。
到位 sample 的 `timing` 区分唤醒迟到、完整观测、SDK 状态读取、目标计算、命令响应和 CSV 写入耗时。
当前 JSON 写入耗时只能写完后测得，因此保存在下一帧的 `previous_logging_s` 中，末帧保存在
`approach_motion_complete.last_logging` 中。`commanded_position_m` 和 `commanded_quaternion_xyzw`
是本轮已发目标，原有 `target_position_m` 是观测时用于检查跟踪的上一目标。
时限异常的 `fault_*.json.approach_timing` 保存失败阶段和已完成步骤的耗时，在请求软件停止后写入。
阶段名 `command` 代表命令返回后已超出本轮总预算，不等于仅命令本身耗时超限；须查看各项耗时。
`tare_start`、`tare` 样本、`tare_discarded` 和 `tare_complete` 通过 `baseline_attempt` 区分基线重试；
仅最终 `tare_complete.tare_bias_si` 用于该条示教。`tare_discarded.reason` 保存静止失败原因。
回撤样本增加 `due_perf_s`、`send_perf_s`、`lateness_s`、`schedule_shift_s` 和观测/命令耗时；
回撤时限故障在 `fault_*.json.retraction_timing` 记录阶段、步号、迟到量和总耗时。
发生超时后先按现场规程确认停止、支撑和故障原因，不要连续清急停重跑，也不要直接调大保护阈值。
`attempt_*.jsonl` 保存所有阶段，`demo_01.json` 至 `demo_08.json` 保存人工接受记录，
`fault_*.json` 关联异常及当前尝试。不能补写完成标志来接纳中断数据。
会话和尝试标记 `motion_policy=fixed-depth`、`force_feedback_enabled=false`，
并保留 `demonstration_source=programmed`、`training_ready=false`。
旧 `import-airbot` 默认明确拒绝此变长协议，不允许改标签冒充实测 10 秒。
操作者现已明确要求保持末状态补齐，可通过新增显式选项生成独立派生训练副本，
操作及来源标记见[补齐训练说明](programmed_hold_last_training.md)；不裁剪或拉伸原有效片段，不修改原始文件。
软件验证不连接真机，实际运动、接触效果与安全性必须另行现场验收。
测试包含同一组在不同力幅值/符号下运动命令完全一致、所有横移帧深度固定、
探索力保护和跟踪策略继承、旧闭环配置拒绝、完整回位、逐条按键、续采与日志校验。

2026-09-12 固定深度更新验证：程序示教测试 21 项；SDK 定向回归运行 131 项，
130 项通过、1 项因缺少 h5py 跳过；clean 环境回归 53 项全部通过，包含该导入拒绝项。
上述固定深度更新测试时，未确认配置仅报告 `setup_confirmed=false`，无闭环参数缺失。
随后根据操作者明确确认将当前配置改为 true，离线预览通过；未连接机器人或串口。

2026-09-12 启动直达到位更新：程序示教测试增至 26 项，覆盖偏移和姿态插值、峰值速度、
到位后等待新命令、只读检查允许偏移、过长路径拒绝、模式切换漂移、过力、断流、控制权丢失、
调度超时、命令拒绝和回位失败。SDK 定向回归 136 项中 135 项通过、1 项缺少 h5py 跳过；
clean 环境 58 项全部通过。离线 preview 退出 0；未连接机器人或传感器，未进行真机运动验收。

启动时序修正验证：程序示教测试 28 项；SDK 定向回归 138 项中 137 项通过、1 项缺少 h5py 跳过，
clean 环境 60 项全部通过，离线 preview 退出 0。新增测试验证 6 ms 命令响应加日志在 10 ms 周期内
可运行，并分别注入观测、命令、CSV、JSON 和唤醒超时，确认失败阶段入盘且不发送下一条目标。
这些是假设备时序测试，不能证明此前真机超时仅由 5 ms 检查造成，也不保证实际机器可满足 100 Hz。
本次未连接硬件、未解除急停；再次现场运行若仍超时，应依据新增分项日志诊断，不能绕过保护。

静止判据和基线重试更新验证：程序示教测试 32 项；SDK 定向回归 142 项中 141 项通过、1 项缺少 h5py
跳过；clean 环境 64 项全部通过；离线 preview 退出 0。覆盖本模式允许 0.1 rad/s 而共享默认仍为
0.05 rad/s、丢弃后完整重采且不混入旧样本、持续不稳的总超时，以及过力、断流、起点偏离、
失去控制权和运行超速不重试。保留原有回位等待行为；本次未连接真机或解除急停。

回撤时限更新验证：SDK 定向回归 144 项中 143 项通过、1 项缺少 h5py 跳过；clean 环境 66 项全部
通过，离线 preview 退出 0。测试覆盖发送前短时迟到可完成回撤、发送间隔不小于 10 ms、擦拭时序
仍合格，以及单次大延迟和持续延迟超出总预算时停止。未连接真机或解除急停，软件测试不替代现场验收。
