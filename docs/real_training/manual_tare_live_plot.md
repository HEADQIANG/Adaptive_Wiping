# 人工示教清零与实时力曲线

默认配置：`force_recording=software_tared`、`live_plot=true`。仅影响人工拖拽示教。
软件清零基线是当前无接触姿态下的六轴原始读数均值，不是硬件置零，也不是跨姿态重力补偿。
改变工具朝向仍可能改变重力分量。不得在接触状态清零，程序无法自动识别接触。

## 准备

先完成 [现场与服务检查](../robot_control/airbot_initial_pose.md)，使用 `--no-return` 服务。
两人配合、全程支撑并确认急停；默认不因超力或运动超速停止，不检查 XYZ 工作空间。
不要另开力传感器绘图 CLI，它会与采集程序争用同一个串口。

桌面终端执行：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
source /home/wp/airbot-venv-5.2/bin/activate
unset PYTHONPATH
python -m pip install -r requirements/manual_demonstrations.txt
python -m scripts.real_training demonstrate preview --mode manual \
  --config configs/real_training/airbot_demonstrations.json
python -m scripts.robot_control.airbot_initial_pose inspect
```

图窗依赖 matplotlib 与桌面 Tk/Qt 后端，自动后端不可用时可在本次运行前 `export MPLBACKEND=TkAgg`。
若 Tk 模块缺失，按系统版本安装对应 tkinter 包；不要更换固定的 `arm-sdk==5.2.2`。
图窗启动失败会在采集客户端连接机器人前退出。`preview/status` 不连接硬件、不打开图窗。

## 新建并录制

先托稳机械臂。新配置不得混入旧原始力会话。

```bash
SESSION_DIR="runs/real_demonstrations/manual/$(date +%m%d_%H%M%S)/session_001"
echo "$SESSION_DIR"
python -m scripts.real_training demonstrate run --mode manual \
  --config configs/real_training/airbot_demonstrations.json \
  --output "$SESSION_DIR" --execute
```

1. 看到 `Gravity compensation active` 后才拖动，保持全程支撑。
2. 工具移到无接触姿态，保持朝向稳定，输入 `z` 回车确认无接触，采集 1 秒基线。
3. 看到 `Tare complete` 后，手动选择任务起点，输入 `s` 回车录制 10 秒。没有录制前静止门槛。
4. 图窗显示 `Fx/Fy/Fz`（N）和 `Tx/Ty/Tz`（Nm），最近 10 秒，约 10 Hz 刷新；采集仍目标 100 Hz。
5. 看到 `STOP wiping` 后停止、托稳；时序合格且动作正确才输入 `a` 回车接受，否则 `r` 回车拒绝。
6. 再次输入 `s` 可复用本次连接的基线；需要重新清零时先无接触，再输入 `z`。
7. 满 8 条或等待输入时 `q` 回车退出，录制中取消用 Ctrl+C。必须确认 `Idle confirmed`；idle 不承重、不保持位置。

清零期间保留设备健康、控制权、关节范围及 FT 20 ms 时效检查，不恢复 11 帧静止门槛。
至少 40 个不同接收时刻、跨度至少 0.9 秒；断流、冲突或覆盖不足时中止并请求 idle，不使用旧基线冒充成功。
配置开启的力/速度 stop 检查仍生效，力保护仍按原始值扣配置电子零偏计算，不能被清零抵消。

## 数据与续采

已采完的会话可离线补画 XYZ 轨迹与六轴力/力矩曲线：

```bash
python -m scripts.real_training.tools.plot_demonstration_overview \
  --session runs/real_demonstrations/manual/0915_163046/session_001
```

默认保存到该会话的 `plots/`：`trajectories.png` 包含 XYZ 时间曲线、三维轨迹和 XY/XZ 投影，
`force_torque.png` 包含六轴曲线，`summary.json` 记录来源哈希与力字段。
按每条记录的开始时间对齐时间轴，位置保留 SDK 绝对坐标（mm），不平移或平滑。
当前清零会话只绘制已记录的清零值，缺失或与基线不符时拒绝，不回退到原始力。
仅使用 `demo_01.json` 至 `demo_08.json` 接受的数据，拒收尝试不混入。
重复绘图创建 `plots_002/` 等新目录，不覆盖原图；不足 8 条可加 `--allow-partial`，图中明确标出实际条数。
采集程序退出时现在自动调用同一导出功能，无需新增运行参数；在请求 idle、关闭客户端/传感器及实时图窗之后绘图。
采满 8 条绘制完整组；`q`、Ctrl+C 或异常退出时仅绘制已经接受的条目，0 条则不生成图。
`--no-plot` 只关闭实时窗口，不关闭退出导出；自动保存使用 Agg 后端，不需要桌面。
导出失败只打印错误，不修改原始日志、不改变采集返回码；可使用上述离线命令补画。
再次 `run` 一个已满 8 条且配置一致的会话也会补画，但不会连接硬件；`preview/status` 不生成图片。
代码更新后旧采集进程需要按现场流程退出再启动；当前采集配置不变，可继续使用配置一致的新清零会话。

- `attempt_*.jsonl`：主力字段 `ft.tared_sensor_wrench_si = raw_sensor_wrench_si - start.tare.raw_baseline_si`，同时保留原始值和电子零偏修正值。
- `tare_<id>.json`：无接触确认、基线均值、不同接收数、跨度及完整基线观测样本。
- `sensor_<id>.csv`：全速 `raw_*` 和清零后的 `net_*`，基线在每个文件中固定；重新清零时关闭旧 CSV 并创建新文件。
- `session.json`：标记主力字段与配置；每条接受记录和起点记录也保留对应清零信息。

```bash
python -m scripts.real_training demonstrate status --mode manual --output "$SESSION_DIR"
```

续采使用同一实际 `SESSION_DIR` 和同一配置，再运行上面的 `run`。重新连接必须再次 `z`，不自动复用旧基线。
新会话仍 `training_ready=false`。当前旧原始载荷训练导入器明确拒绝软件清零人工会话，不能直接套用旧训练命令。
清零人工示教现在有独立的[导入与训练步骤](manual_tared_training.md)：配对完整 manual-start 探索，
明确确认采集条件一致，通过基线和时序审计后生成新数据，不修改本会话。

图窗使用独立进程，只读取日志，不控制机械臂、不打开串口。关闭图窗或图窗运行故障不会停止采集，终端会提示；退出需使用终端安全流程。
无桌面情况下显式添加 `--no-plot` 可只采集清零数据。显示不是实时安全保护，不能据图窗响应速度判断 FT 是否正常。

## 软件验证

```bash
python -m unittest tests.real_training.test_airbot_demonstrations tests.real_training.test_manual_live_plot \
  tests.real_training.test_demonstration_overview -v
MANUAL_PLOT_GUI_TEST=1 MPLBACKEND=TkAgg python -m unittest \
  tests.real_training.test_manual_live_plot.ManualProcessTests.test_desktop_reads_new_attempt_and_closes -v
```

上述测试使用假机器人和合成曲线，不连接设备；不等于真机 100 Hz 时序或现场安全验收。

2026-09-15 软件验证：SDK 环境相关回归 117 项无失败（6 项跳过）；另行启用的桌面图窗测试通过，
`clean` 环境的清零导入保护测试及历史 native/程序补齐 21 项测试通过。六轴曲线完成渲染检查，未连接真机。
