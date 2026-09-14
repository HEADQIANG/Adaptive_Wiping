# 无六维力采集的接触运动验证

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

当前用户已改为先验证无接触动作，请使用 [空中运动验证](../robot_control/airbot_air_motion.md) 的
`--mode air` 和新空中起点。本页接触模式保留原有检查，不用于绕过未确认的接触条件。

此模式对应当前要求：海绵厚 28 mm，用户给定最大允许压缩 20 mm，执行与仿真相同的
**下移 10 mm、横移 50 mm 再返回**，不读取六维力传感器。不是空中运动。
2026-09-11 按压速度改为 `0.005 m/s`，持续 `2 s`；现场允许压缩限值不自动改动。
它验证位置控制动作，不证明接触力安全、力控性能或训练数据合格。尚未进行真机运动验收。

入口仍为 `scripts.real_training.airbot_exploration`，必须显式加 `--mode contact-no-ft`。
默认 `force-guarded` 模式没有改变，传感器缺失/断流时仍拒绝运动，不会自动切换至无力模式。
新的配置是 `configs/real_training/airbot_contact_motion.json`，不会替换正式探索配置。

## 1. 固定轨迹和压缩量

当前仿真从海绵底面**离桌 1 mm、没有预压缩**的位置开始，目标末端下移 10 mm。
因此平面接触的名义压缩是 9 mm，并非精确实测值。不要把起点设在已经接触或
已经压缩的位置，否则会改变协议并消耗压缩裕量。

| 时间 | 桌面坐标系目标增量，mm |
| --- | --- |
| 0 s | `[0, 0, 0]` |
| 2 s | `[0, 0, -10]` |
| 3 s | `[0, 50, -10]` |
| 4 s | `[0, 0, -10]` |

姿态固定，100 Hz；之后另用 2 s 上退 10 mm，等待人工安全支撑并输入 `IDLE` 后退出。
`--time-scale 1` 是上述时间；`--time-scale 5` 仅减速至 20 s 探索、10 s 回撤，
下移和横移距离完全不变。慢速调试日志不能当作原 4 秒训练协议，减速也不会减少最终压缩。

## 2. 先记录真实起点

服务连接已由用户提供的只读输出确认：SDK 5.2.2，机械臂
`PZ60C02603000943`，电机状态正常，`eef_type="NULL"`。不需要因此重启正常服务，
但要确认原服务带 `--no-return`。先完成负载、急停、支撑和整臂/线缆运动路径的现场检查。

在项目根目录、有人监护的交互终端运行：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose teach \
  --execute --output runs/real_training/real_robot/contact_motion_start_001.json \
  --label contact_motion_start --tool-note "28 mm sponge; actual mount and SDK end-frame definition"
```

执行前安全支撑机械臂，再输入 `DRAG`。待显示重力补偿已激活，缓慢拖到目标起点，
海绵面与被接触平面平行、未压缩底面距平面 1 mm。确认整个 50 mm 横移路径的表面高度
及间隙误差，避免突出物、斜面和硬安装件触碰。静止且已安全支撑后按 `S` 或 `s`，
无需回车，检查通过即保存并自动退出，不再需要 `Q`。idle 不保证保持位置，
后续脚本若发现起点偏移会拒绝运动，不自动归位。
拖拽操作的详细安全要求见 [初始位姿文档](../robot_control/airbot_initial_pose.md)。

## 3. 填写配置，不猜测安全参数

`configs/real_training/airbot_contact_motion.json` 已写入用户给出的机械臂序列号、NULL 末端、
海绵厚度 0.028 m 和允许压缩 0.020 m。其余现场未知值保留 `null` 或 `false`。
仍需确认/填写以下项目，不能只修改确认开关来绕过它们：

- `calibration_id`、`sdk_frame_and_tool_note`：现场几何检查记录与实际工具/SDK 末端说明。
- `table_normal_sdk`、`slide_direction_sdk`：上法向和横移正方向，在 SDK 参考坐标系中表达。
  模板的 Z/Y 方向仅在实际坐标确实对齐时成立。
- `joint_min_rad/max_rad`：现场允许的关节界限。项目 XYZ 边界已移除，
  整臂、工具和线缆的完整路径须现场确认；压缩预算和上界检查仍保留。
- `joint_current_limits`：六轴经厂家规定和现场负载确认的电流限制，不是 N 或 N*m，
  不能抄仿真扭矩限制。关节限流不是末端力限制。
- `contact_geometry_uncertainty_m`：整个路径上接触几何误差的已验证上界，必须为正数。
  包括起点间隙、桌面高度变化、SDK/TCP 定义误差，以及允许姿态偏差下最远海绵边缘的位移。
  不能只填测量仪器分辨率，也不能直接采用下面的预算上限作为测量结果。
- `calibration_confirmed`、`no_force_contact_confirmed`：完成以上检查、确认有独立硬件
  停止/支撑措施且接受没有实时超力保护后才设为 true。

无力模式要求：

```text
9 mm 名义压缩 + 位置跟踪误差限值 + 接触几何误差上界 + 额外压缩余量 <= 20 mm
```

模板位置误差阈值为 0.5 mm，额外余量为 0.25 mm，因此几何误差预算最多 10.25 mm。
这两个模板数值只是工程起始设置，**不是厂商安全额定值、已验收精度或停止距离保证**。
当前名义压缩至允许上限的总裕量为 11 mm，如果工具角度误差、横移表面起伏或停止过程不满足它，
不要降低误差记录值、扩大允许压缩值或放宽保护来运行完整行程；需要先改善/验证现场条件
或启用力保护。软件阈值检查发生在状态返回后，不能保证故障发生时不越过物理压缩界限。

无力模式起点位置误差还要求不超过位置跟踪阈值（模板为 0.5 mm），相比原模式更严格。
运行中记录根据 SDK 位移和声明的几何误差得到的压缩上界估计；这不是实测压缩或力数据。
达到 `允许压缩量 - 额外余量` 的边界后中止，异常不盲目回撤。

## 4. 预览、检查、运行

预览完全离线，不需要串口/numpy/pyserial，也不会把缺失六维力补成零：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore preview --mode contact-no-ft
```

配置完成后，只读检查起点及目标坐标，不申请控制权、不打开串口：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check --mode contact-no-ft
```

现场条件满足后，可以先以慢速验证完整距离：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --mode contact-no-ft --execute --time-scale 5 --output runs/real_training/real_robot/contact_motion_slow_001.jsonl
```

之后，在现场批准正式速度时使用同一轨迹的原时间参数：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --mode contact-no-ft --execute --time-scale 1 --output runs/real_training/real_robot/contact_motion_001.jsonl
```

必须输入 `CONTACT-NO-FT` 才启动连接/验证；原 `EXPLORE` 确认词不能用于无力模式。
过程中保留起点、工作空间、关节速度/位置、电机错误、姿态、跟踪误差、控制权和超时检查。
Ctrl+C/SIGTERM/EOF 或故障请求 SDK 软件停止，但无法替代物理急停、防坠支撑和人工监护。
回撤完成后，先按现场规程安全支撑，再输入 `IDLE`；日志有 `session_complete` 才表示正常交接。
不要通过拔电或直接关终端退出位置保持状态。

NULL 末端必须由配置、起点记录和在线缓存固件一致确认。仅在明确为 NULL 时，
使用 SDK 5.2.2 原笛卡尔 servo RPC 的六轴请求，不查询夹爪、不发送夹爪目标/速度/电流字段。
这避免原封装无条件注入夹爪速度并在每周期报告 NULL 型号警告；没有改动已安装 SDK，
也没有伪造 G2 型号。六轴序列化字段已经用真实 SDK protobuf 类型和假 RPC 验证，
仍需现场验证服务端实际运行行为。已安装夹爪时继续保留实测开度，类型不一致拒绝执行。

## 5. 后续接入力传感器

保持相同轨迹，填写原 `configs/real_training/airbot_exploration.json` 的传感器、标定和力限值，
将正确起点路径与 NULL 末端设置（`expected_eef_type="NULL"`）同步过去后，
使用原 `--mode force-guarded`（默认）入口。带夹爪时需要真实的夹爪电流限制。
只有独立标定完成且数据新鲜才会允许运动；读取力不等于已经实现闭环力控制。

本模式日志明确为 `mode=contact-no-ft`、`force_monitoring=false`、`force=null`、
`encoder_ready=false`，没有串口操作，也没有零力占位数据。它不能用于原探索编码器训练。
`nominal_protocol_match` 只表示目标轨迹和时间是否匹配，不证明真实接触动力学一致。
即使以后接入采集，也仍需完成传感器坐标变换、时间同步和训练数据格式验收。

无硬件测试命令保持不变：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot*.py' -v
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -v
```

2026-09-10 软件验证：SDK 环境 52 项 AIRBOT 测试全部通过；`clean` 全量回归
222 项，220 项通过、2 项 SDK 类型测试跳过，耗时 88.115 s；两项已在 SDK 环境通过。
覆盖无力模式不导入传感器、不查询力、完整 400 点轨迹、模式隔离、压缩预算、
跟踪/超时/中断停止、NULL 六轴 protobuf 请求。离线预览及未确认配置拒绝执行已验证。
原 `simulation.py` 哈希和历史训练来源检查保持不变。本次未连接硬件、未执行运动，
`runs/real_training/real_robot` 尚无起点文件；这些测试不替代物理接触验收。
