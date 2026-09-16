# AIRBOT Play 5.2.2 自由拖拽与初始位姿记录

当前交互：拖到起点并静止、托稳机械臂后，按 `S` 或 `s` 即开始采样；无需回车。

探索起点示教使用 `--label exploration_start` 时，保存成功且确认 idle 后，程序自动
更新 `configs/real_training/airbot_exploration.json` 的 `initial_pose_file`。
无需再手工执行“更新探索配置”这一步，路径采用本次 `--output` 的实际文件名：

```bash
python -m scripts.robot_control.airbot_initial_pose teach \
  --execute --output runs/real_exploration/exploration_start_008.json \
  --label exploration_start --tool-note "Installed KWR75A and sponge"
```

在项目根目录、已激活 SDK 虚拟环境且完成下述硬件准备后执行。`--execute` 授权启动，不再输入口令，
静止并支撑好机械臂后按 S。终端显示 `Updated exploration config` 表示更新成功。
随后直接运行 `python -m scripts.real_training explore check --config configs/real_training/airbot_exploration.json`。
取消、保存失败或 idle 未确认时不会更新配置；配置写入失败会报错，但保留已保存的位姿文件。
其他 label 和 `capture-idle` 不自动修改探索配置。此功能不自动归位，也不启动探索。
静止检查通过后保存并自动切回 idle 退出，不再有 `Q` 退出选项。
按键前必须准备好安全支撑，idle 不保证保持位置。`Ctrl+C` 可取消；
运行命令与 JSON 格式不变，执行前先托稳；检查通过即进入拖拽，详见 [统一交互规则](interaction.md)。

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

离线回归说明：`capture-idle` 不再提前导入仅拖拽使用的 `Controller`；真实连接仍
通过 `open_client` 校验 SDK 版本。现有硬件命令不变，纯模拟客户端测试无需安装 SDK。
部署与模型回放操作见 [airbot_native_training.md](../real_training/airbot_native_training.md)。

后续真实探索按压/移动脚本见 [airbot_exploration.md](../real_training/airbot_exploration.md)。
该脚本只接受已在记录起点附近的机械臂，不自动回起点；原 teach/inspect 命令不变。
探索起点需另行实测海绵底面距桌面 1 mm，本文的高处准备位姿不能直接替代。

更新：2026-09-09。入口：`scripts/robot_control/airbot_initial_pose.py`。
本次完成代码和无硬件测试，尚未切换真机拖拽模式或获得真实起点文件。
用户于 2026-09-09 20:30 首次启动服务时崩溃；只读诊断与待验证的启动规避办法见第 8 节。
后续服务已运行，可只读获取机械臂状态；`(0,0,0,1,1,1)` 的状态误判修复见第 9 节。

## 1. 任务边界与论文对应

`docs/references/Adaptive_wiping.pdf` 第 4 页 IV-B、第 5 页 IV-C.2 使用自由拖拽进行动觉示教，
记录末端位置及六维力。本文先完成其中的准备环节：手动选定起点，记录关节角、
末端位置和姿态，为后续重复示教提供参考。论文没有提供 AIRBOT 可直接使用的起点坐标。

这里的“设定”是保存参考位姿，不是更改电机零点，也不是发送位置目标。
不回零、不自动回到起点、不执行擦拭、不采集坤维六维力，不修改重力补偿参数。
单个起点 JSON 不是 `real_training` 所需的完整示教数据集。

## 2. 上机前必须确认

- 底座固定，工作区无人和障碍物，急停及厂家规定的停机方法已验证。
- 固件与 5.2.2 配套，实际末端配置正确，海绵、传感器和安装板的负载适合拖拽。
  默认重力补偿不表示已补偿新增工具；若出现明显下坠、自行运动、振动或过热，立即停止。
- 操作者从切换前到整个会话结束均需能托住机械臂；建议第二人操作键盘并监护急停。
  不把手放在夹点内，不强推关节限位，不照搬论文“尽可能用力”的要求。
- 初次仅选无接触、留有安全间隙的准备位姿，工具朝向未来擦拭面，关节远离限位和奇异姿态。
  接触起点按固定安装、工具姿态、实测间隙和现场力保护确认；不要求 TCP 标定。
- `idle` 不等于位置保持。退出、异常、断电或控制权丢失可能导致失去支撑；不能依赖软件急停或重力补偿承重。
- 不运行官方 `record_and_replay` 示例来做这一步，该示例停止录制会调用 `return_zero()`。
  官方部分名称为 `get_*` 的示例也会主动切换重力补偿；本项目的 `inspect` 才是只读入口。

## 3. 本机环境核验

所有项目命令在 `Adaptive_Wiping` 根目录执行。使用已安装的独立 SDK 环境，
不要使用旧 `/home/wp/airbot-venv` 的 `airbot_py 5.1.6`，也不必安装训练依赖。

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
dpkg-query -W -f='${Package} ${Version} ${Status}\n' airbot-arm
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -c 'from importlib.metadata import version; print(version("arm-sdk"))'
ip -details link show can0
airbot-arm --help
```

本次核验主程序软件包和 `arm-sdk` 都为 `5.2.2`，`can0` 为 UP / ERROR-ACTIVE。
ERROR-ACTIVE 是 CAN 正常错误处理状态名，不代表当前有总线故障，也不证明电机通信正常。
本机显示 `bitrate 0`，可能由串行 CAN 接口上报方式导致；不可据此盲目修改 CAN 参数。
`airbot-arm --version` 本机显示 `1.0`，安装包版本应以 `dpkg-query` 为准。
这些软件/接口检查均不代表固件、负载和急停已经验收。

## 4. 启动服务并只读检查

完成上述现场确认后，在终端 A 执行。启动服务本身涉及硬件初始化，必须有人监护。
若已有服务在运行，先确认其参数及用途，不要启动第二个或直接杀掉现有进程。

```bash
mkdir -p /home/wp/airbot-logs-5.2
env MALLOC_ARENA_MAX=2 AIRBOT_LOG_DIR=/home/wp/airbot-logs-5.2 \
  airbot-arm --address 127.0.0.1:50051 -i can0 -t airbot_play --no-return
```

此为本机启动崩溃后的待验证规避命令，原因见第 8 节。不要使用 `sudo airbot-arm`。
环境变量仅作用于这次启动，不修改系统限制或全局配置。

**必须保留 `--no-return`**，它禁止服务退出时自动回零；并不保证无初始化动作，
也不是“退出后保持当前位置”。脚本不会启动或停止此服务。

在终端 B、项目根目录执行：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose inspect
```

此命令不申请控制权、不使能、不清故障、不切换控制器，仅读取状态、关节角、末端位姿和缓存固件信息。
确认六个关节读数与实际姿态合理对应，末端位置单位 m、四元数顺序 xyzw，电机无错误。
输出 `state.service_state.motor_status_codes` 保留六轴原始状态码；`0`、`1` 均为官方定义的无错误值，
不能要求它们必须全部为 `0`。其他码仍拒绝后续拖拽。
开始拖拽要求服务当前 `controller_state` 为 `idle`；不是 idle 时，先按厂家流程查清当前任务，
脚本不会接管正在运行的其他控制器。缓存固件信息须与设备资料核对。

连接超时表示服务未启动或地址不匹配，不要用运动示例测试连接。
状态字段与预期不一致时保留 `inspect` 输出用于排查，不删除校验强行拖拽。

## 5. 拖拽、记录、退出

终端 B 执行，输出文件必须尚不存在：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose teach \
  --execute \
  --output runs/real_deploy/robot_control/real_robot/initial_pose_001.json \
  --label wiping_start \
  --tool-note "Describe the mounted sensor, plate and sponge here"
```

1. 执行带 `--execute` 的命令前托稳机械臂并完成现场检查；程序不再等待启动口令。
2. 程序申请独占控制权，调用 SDK `enter_gravity_compensation_mode()`，并检查 `gravity_comp` 状态。
   只有显示 `Gravity compensation active` 后才开始缓慢手动拖动。
3. 拖到选定的初始位置和朝向，保持支撑并静止。按 `S` 或 `s`，无需回车。
4. 程序读取 11 组样本，间隔至少 0.05 秒，检查关节速度绝对值不超过 0.05 rad/s、
   任一关节角范围不超过 0.01 rad、末端位置跨度不超过 2 mm、姿态跨度不超过 0.02 rad。
   这些只是工程上的记录稳定性阈值，不是机器人安全限值或标定精度保证。
   运动过大时拒绝保存、继续等待，可稳定后再次按 `S`，无需回车。
5. 显示 `Saved initial pose` 才表示文件写入完成。保存的是最后一个实测样本，不做姿态平均。
   保存成功即自动退出，不再等待 `Q`；重录需重新启动并指定新文件名。
6. 脚本自动请求 `Controller.idle`、检查响应，再关闭客户端并释放控制权。
   **按 S 前就要托稳机械臂，保存和退出都不会锁住起点。** 看到 `idle NOT confirmed`
   时，记录可能已保存，但不能视为安全退出，按现场硬件安全流程处理。
7. 确认安全支撑后，按厂商流程关闭终端 A 服务和电源。服务必须仍是带 `--no-return` 启动的那一个。

Ctrl+C、终端 EOF 或读取/保存异常会尝试相同的 idle 清理，因此全程都要准备支撑。
看到 `idle NOT confirmed` 时，实际硬件状态未知，立即使用现场硬件安全流程，不要放手机械臂。
控制权丢失时不自动重新抢占、不自动恢复拖拽。本模块固定 SDK 版本，并仅覆写
5.2.2 的自动申请控制权钩子来禁用这一默认行为，不修改安装的 SDK 文件。

## 6. 文件内容与限制

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m json.tool runs/real_deploy/robot_control/real_robot/initial_pose_001.json
```

- `initial_pose.joint_position_rad`：六个实测关节角，rad；另存速度 rad/s。
- `initial_pose.sdk_end_position_m`：SDK 末端位置 `[x,y,z]`，m。
- `initial_pose.sdk_end_orientation_xyzw`：SDK 末端姿态四元数 `[qx,qy,qz,qw]`。
- `host_time_utc`、`host_monotonic_s`、`read_duration_s`：主机读取时间及耗时，不是电机硬件时间。
- `metadata`：标签、工具说明、服务地址、缓存固件信息；`samples` 保留静止检查的原始样本。
- `coordinate_semantics`：记录 SDK 参考坐标系和末端定义，不再保存 TCP 标定状态。

SDK 两次读取关节与位姿，并非同一个原子快照。静止窗口降低时序偏差，但无法证明底层电机数据新鲜。
服务缓存有效性和电机错误检查只是辅助诊断，不构成独立安全保护。
SDK 若干 RPC 没有客户端 deadline；读取返回后才检查耗时，网络卡死时软件清理可能无法及时执行。
请在本机有线 CAN 环境有人值守使用；不得把它用于无人监控的力控或远程安全停机。
掉电/强制杀进程无法保证文件写完或执行退出清理；不要将不完整 JSON 当作有效起点。
本次不实现自动回到记录起点；后续需要先设计并验收限速、路径、碰撞和停止策略。

## 7. 无硬件测试

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot_initial_pose.py' -v
```

使用假客户端验证状态校验、静止窗口、四元数、文件防覆盖、显式确认、控制权拒绝/丢失、
失败退出、Ctrl+C/EOF 和只读检查。测试不启动服务、不发 CAN 帧，不替代真机验收。

## 8. 启动崩溃：日志权限与锁定内存上限

2026-09-09 的现场启动日志先出现 `Failed to create logger`，随后出现
`pthread_create failed: Resource temporarily unavailable`，最终 gRPC
`Could not create grpc_sync_server worker-thread` 触发 SIGABRT。此时服务未成功启动，
不能继续拖拽。先确保机械臂有安全支撑，程序崩溃不能保证退出清理已执行。

只读诊断依据：

- `/home/wp/runs/real_deploy/robot_control/all.log`、`airbot.controllers.log` 等旧日志为 `root:root 644`，
  当前 `wp` 用户无法追加写入；`/tmp/zlog.conf` 的输出规则使用 `AIRBOT_LOG_DIR`。
  第 4 节改为用户新建的独立日志目录，不删除旧日志、不修改其所有权。
- `/var/crash/_usr_bin_airbot-arm.1000.crash` 中 `Threads: 26`，
  `VmLck: 1998628 kB`。启动它的父终端锁定内存上限为 `2054737920` 字节，
  即 `2006580 KiB`，剩余仅 `7952 KiB`；默认线程栈是 `8192 KiB`。
- 已安装二进制的 `realtime_tools::lock_memory()` 调用 `mlockall(3)`，
  即 `MCL_CURRENT | MCL_FUTURE`。后续新线程栈也受锁定内存上限约束，
  这解释了为何仍有可用内存却无法创建 gRPC 工作线程。
- 用户任务总数约 2600，用户 cgroup 上限 40838；实际终端 scope 上限 18562，
  相关 `pids.events` 均未触顶。没有依据要求杀掉后台程序或调高进程数限制。

第 4 节加入 `MALLOC_ARENA_MAX=2`，限制 glibc 内存分配 arena 数量，减少多线程时
大量虚拟地址空间被锁定的压力；不降低线程栈、不关闭内存锁定、不取消硬件保护。
这是有诊断依据的启动规避办法，**尚未通过重启真机验证，也不保证控制实时性已验收**。
本次诊断没有重启服务、发送运动指令、修改系统限制或修改 SDK。

按第 4 节在现场监护下重试。若服务保持运行且无上述错误，再执行 `inspect`；
只有状态正常后才进入第 5 节。若仍失败，保留新的终端输出和新日志目录内容，
不要继续拖拽或反复重启，也不要直接改成 `sudo`、`ulimit -l 0` 或关闭保护。

## 9. `(0, 0, 0, 1, 1, 1)` 被误报为 Motor fault

这是本项目早期 `health()` 将所有非零状态码都视为故障的错误，不表示后三轴电机损坏。
官方 [ArmMotorState 文档](https://docs.airbots.online/airbot-play/sdk/api/types/arm-state.html)
将 `error_ids` 标注为“状态码/错误码”；官方硬件库的
[MotorState::error_id 定义](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware/blob/56ea4901144bdf43ad4dcb231cff4c7bb54fc8dc/airbot_hardware/utils.hpp#L327)
明确写明 `0x00 or 0x01 means no error`，其他数值表示具体错误。

本机只读诊断确认电机类型为 `OD, OD, OD, DM, DM, DM`，DM 固件均为 `5015`，
三次读取都得到 `(0,0,0,1,1,1)`，服务为 `Idle / idle`，电机温度约 26-30 摄氏度。
此结果仅确认通信与本次状态，不代表工具负载、坐标标定或自由拖拽已经验收。

修复后仅接受 `0`、`1`，其他码仍中止操作；没有删除检查、清除电机错误、
修改固件/保护阈值或自动切换模式。新增假客户端测试覆盖真实混合状态码、只读检查、
每一轴的其他 8 位状态码均阻止申请控制权，以及记录保留原始状态码。

命令未改变，不需要重新安装 SDK，也不需要重启已正常运行的服务。
在终端 B 明确切换到项目目录并重新检查：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose inspect
```

确认状态正常、现场安全条件满足后，按第 5 节进行交互式拖拽记录。

## 10. 静止时反复提示 Joint speed exceeds

2026-09-09 只读采样约 1 秒，末端位置跨度约 0.12 mm，关节角跨度约 0.00038 rad，
后三轴瞬时速度反馈仍达到约 0.022-0.037 rad/s；操作者同时确认肉眼稳定。
原记录阈值 0.02 rad/s 易受此类反馈噪声影响，现调整为 0.05 rad/s（绝对值，等于阈值允许）。
这仅是保存起点的稳定性检查，不发送速度指令、不修改 SDK、重力补偿或硬件保护参数。
11 帧采样及关节角、位置、姿态跨度限制均保持不变；任一条件不满足仍拒绝保存。

超速提示现在列出所有超限关节（J1-J6）、各自采样窗口内的最大速度绝对值及阈值，例如：

```text
Not saved: Joint speed exceeds limit=0.05 rad/s: J4 peak=0.07 rad/s; support and stop the arm. Stop moving and retry 's'.
```

命令和 JSON 结构不变。新记录的 `stationarity_check.max_joint_speed_rad_s` 为 `0.05`，
原始速度样本照常保留；既有记录不改写，其阈值以文件内记录为准。
使用第 7 节的同一测试命令验证噪声场景、正负边界、超限诊断及其他跨度检查。

旧的运行中进程不会自动加载修复：先托稳机械臂，在旧会话输入 `q` 并回车，确认退出，
再在项目根目录重新执行第 5 节的 `teach` 命令。不需要重启正常运行的 `airbot-arm` 服务。
现场准备完成后重新运行带 `--execute` 的命令，拖到起点并静止、托稳后按 `S` 保存并自动退出，无需回车。
如果仍失败，保留完整提示；不要继续放宽阈值，先确认是否存在实际下沉或抖动。
# 退出拖动后的只读起点记录

机械臂已在 idle 静止且现场确认不会继续下沉时，可执行：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose capture-idle \
  --output runs/real_deploy/robot_control/real_robot/air_motion_start_idle_001.json \
  --label air_motion_start --tool-note "28 mm sponge; settled idle air start"
```

不加 `--execute`。命令只读硬件并新建记录，不申请控制权、不切换控制器、不回位。
全程要求 idle 和原有静止检查通过，已有输出文件会被拒绝。新起点仍需核对净空及工作空间，
再更新探索配置并运行 `airbot_exploration check --mode air`；不能将采样成功当作位置保持保证。
