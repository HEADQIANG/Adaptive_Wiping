# 探索时实时显示清零后的力/力矩

在探索命令后增加 `--plot`，打开六轴实时窗口。适用于默认 `manual-start` 和显式 `force-guarded`；
`air` 和 `contact-no-ft` 没有力传感器数据，不能开启此功能。
不加 `--plot` 时维持原有无窗口行为。

## 环境与运行

在项目根目录、带桌面显示的终端运行。真机 SDK 环境需有 Matplotlib 和 Tk/Qt：

```bash
/home/wp/airbot-venv-5.2/bin/python -m pip install 'matplotlib>=3.7,<4'
```

当前环境已安装 Matplotlib，已有 Tk 8.6。其他机器若缺少 Tk，需由管理员安装
与 Python 版本匹配的 Tk 包。无图形桌面的 SSH 会话不要加 `--plot`。

仅查看协议和参数，不连接机器人、串口或打开窗口：

```bash
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore preview --plot
```

完成现场安全检查，在有人监护的终端执行默认手动起点流程：

```bash
env MPLBACKEND=TkAgg \
  PYTHONPATH='/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping' \
  /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --config configs/real_training/airbot_exploration_manual.json \
  --execute --time-scale 1 --plot --plot-window 10 --plot-hz 10 \
  --output runs/real_exploration/exploration_tared_plot_001.jsonl
```

每次选择尚不存在的输出文件名，不覆盖旧数据。带 `--execute` 启动后不再等待口令，先验证绘图
进程能打开窗口，再连接硬件；缺少依赖或无法打开窗口会在硬件连接前退出。
连接后拖拽至人工确认的 1 mm 非接触间隙，按 `h` 固定，看到提示后按 `s` 清零并探索。
新模式无起点匹配、静止或载荷阈值验收；按压仍为 `0.005 m/s × 2 s`，探索 `4 s / 400` 帧，
另用 `2 s` 回撤，返回偏差仅记录。保持到 `IDLE` 人工退出。详见 [完整操作](manual_start_exploration.md)。
旧流程需同时指定 `--mode force-guarded --config configs/real_training/airbot_exploration.json`，原保护不变。

## 窗口与数据

- 开启 `--plot` 后，完成终端 `IDLE` 交接并退出程序时，自动在 JSONL 同目录保存
  同名 PNG，例如 `exploration_tared_plot_001.png`，命令无需增加参数。
  图片只包含探索阶段的六轴清零后曲线（`--time-scale 1` 时为 4 秒），
  不包含回撤，不受 `--plot-window` 限制。JSONL 仍完整保存探索和回撤数据。
  即使提前关闭实时窗口，退出时仍从日志生成图片；正常处理的中止保存已有采样并标注
  `aborted`，无有效采样则不生成图片。强制杀进程或断电不能保证自动保存。
  图片生成在硬件清理后进行，不进入控制循环。已有同名 PNG 不覆盖，保存失败会在
  终端提示，不修改 JSONL。每次运行请继续选用新的日志文件名。
- 左列 Fx/Fy/Fz，单位 N；右列 Tx/Ty/Tz，单位 Nm。
- 只读取 JSONL 中发送后 `force.tared_sensor_wrench_si`，不再重复清零，
  不用原始值或固定电子偏置值代替。重复的传感器时间戳不会重复绘制。
  实时曲线也只绘制 `phase=exploration` 的采样；回撤时保留探索曲线，不追加回撤数据。
- 横轴为传感器接收时间相对探索开始的秒数；慢速调试时显示真实经过的时间，
  不把时间压缩为 4 秒。状态栏区分清零、按压、正向横移、返回、回撤和结束/中止。
- `--plot-window` 为滚动窗口，默认 10 秒，范围 0.5–300 秒。
  `--plot-hz` 为显示刷新频率，默认 10 Hz，范围 1–30 Hz；不改变 100 Hz 控制节拍。
  显示有延迟时可降到 5 Hz。持续 0.5 秒没有新日志采样时提示数据停止更新。
- 探索与回撤完成后保留曲线，终端仍执行原有安全支撑和 `IDLE` 交接。
  每次显示刷新会提交全部已读样本，包括结束前不足一个刷新周期的最后几个点。
  程序退出时关闭窗口并回收绘图子进程；中止也不会让绘图拖延原有停止流程。

绘图在独立进程中只读已刷新的 JSONL，不打开串口、不连接 SDK，也不向控制循环
增加队列发送或 GUI 刷新。它仍会占用计算资源，因此不承诺显示延迟或硬实时性能；
原迟到、超力、超力矩及其他停止检查继续有效。

关闭绘图窗口只关闭显示，不是停止运动的命令。运行期间绘图崩溃也不会替代
原有控制和安全机制；停止仍按原终端流程与现场急停规程执行，不能以曲线代替安全监测。
清零后曲线不是跨姿态重力补偿。训练导入、原始日志和原离线绘图入口的数据含义不变。

## 无硬件验证

```bash
/home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.real_training.test_exploration_live_plot \
  tests.real_training.test_airbot_exploration \
  tests.force_sensor.test_kwr75_live_plot
```

在桌面会话中验证独立窗口的打开、合成日志读取及退出，不连接任何硬件：

```bash
EXPLORATION_PLOT_GUI_TEST=1 /home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.real_training.test_exploration_live_plot.PlotProcessTests.test_desktop_process_reads_synthetic_log_and_closes
```
