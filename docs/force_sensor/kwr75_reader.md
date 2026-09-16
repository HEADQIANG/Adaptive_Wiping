# KWR75 实时查看、CSV 记录与离线绘图

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

部署新增 `Kwr75Reader.latest_before(timestamp, net=False)`：返回不晚于策略时刻的
最后一帧接收数据，不使用未来帧。重复接收时间采用该批次最后一帧，与 `latest()`
语义一致；调用者仍须检查数据年龄。原有记录、绘图命令不变。
训练与部署操作见 [airbot_native_training.md](../real_training/airbot_native_training.md)。

## 环境与串口权限

从项目根目录执行（已创建环境时无需重复创建）：

```bash
python3 -m venv .venv-kwr75
source .venv-kwr75/bin/activate
python -m pip install numpy pyserial matplotlib
python -m scripts.force_sensor.kwr75_reader --help
```

Linux 用户需要串口读写权限。若 `id -nG` 不包含 `dialout`：

```bash
sudo usermod -aG dialout wp
newgrp dialout
id -nG
source .venv-kwr75/bin/activate
```

`wp` 是当前项目使用的用户名，其他机器需替换。`newgrp` 启动新 shell；也可注销并重新登录。
核对设备接线和 `ls -l /dev/serial/by-id/` 后选择传感器端口，不要将机械臂通信端口传入。
脚本以 460800、8N1 访问传感器，会发送开始/停止采样指令，但不控制机械臂运动。

## 实时查看并保存

工具固定姿态、悬空且无人接触时执行：

```bash
python -m scripts.force_sensor.kwr75_reader \
  --port /dev/ttyUSB0 --tare --secs 60 \
  --csv "runs/force_sensor/kwr75_$(date +%Y%m%d_%H%M%S).csv"
```

- `--csv` 指定新文件，自动创建父目录；已有同名文件会报错，绝不覆盖。
- `--tare` 先用约 1 秒静止数据清零，再开始 CSV 记录；清零前的等待与清零采样不写入 CSV。
- 不清零时去掉 `--tare`，此时 CSV 的 `raw_*` 与 `net_*` 相同。
- `--secs` 为显示阶段时长，必须为有限正数，不包含等待首帧与清零时间。
- 到时自动停止；`Ctrl+C` 会停止采样、写完队列并关闭文件，退出码为 130，不打印完整阶段统计。
  中断在安全的轮询边界处理；清零期间按下时，会等待当前约 1 秒清零结束再退出。
- 正常退出显示文件路径和写入行数。未指定 `--csv` 时只显示、不保存，原用法仍有效。
- 指定 `--csv` 后默认在结束时自动生成同目录的全程六轴 PNG，详见下一节；
  只需要 CSV 时添加 `--no-save-plot`，此时无需安装 Matplotlib（除非同时使用 `--plot`）。

每个成功解析的帧各写一行，不受约 100 Hz 终端刷新频率限制。传感器约 1000 Hz 时，
60 秒通常约 60000 行；实际数量受速率、调度及停止时正在读取的末批帧影响。
原始值保留浮点精度，不采用终端显示的小数截断；CSV 中无 10 ms 平均结果。

## 结束采集后自动保存曲线

带 `--csv` 的采集命令默认自动保存曲线，无需额外运行绘图脚本：

```bash
python -m scripts.force_sensor.kwr75_reader \
  --port /dev/ttyUSB0 --tare --secs 60 \
  --plot --plot-window 10 \
  --csv "runs/force_sensor/kwr75_$(date +%Y%m%d_%H%M%S).csv"
```

到时结束、按 `Ctrl+C`、关闭实时窗口均按以下顺序处理：停止串口读取，
将队列数据写完并关闭 CSV，关闭实时窗口，读取完整 CSV 生成 PNG。
等到终端打印 `PNG: ...` 并返回提示符，才表示图片保存完成。

- `--tare` 对应输出 `原CSV文件名_net.png`；不清零对应 `原CSV文件名_raw.png`。
- 图片包含本次 CSV 的全部帧和全部时间，不是最后 10 秒窗口截图，不做滤波或 10 ms 平均。
- `Ctrl+C` 的退出码仍为 130；图形正常关闭或到时结束为 0；保存失败返回 1。
- 不加 `--plot` 也会自动保存 PNG，且不需要桌面显示环境。实时窗口与结束后的导出互不依赖。
- 不加 `--csv` 则不自动生成文件。仅保存 CSV、不自动绘图时添加 `--no-save-plot`。
- PNG 与 CSV 均不覆盖同名文件。开始采集前会检查目标 PNG；若采集中目标 PNG 被其他程序创建，
  导出时也会拒绝覆盖。更换文件名后可使用离线绘图命令重试。
- 首帧前退出、CSV 没有数据行、读取或 CSV 保存失败时不生成 PNG，避免把不完整记录当作正常结果。
  PNG 导出失败不会修改已保存的 CSV。非有限数据会被离线绘图校验拒绝，应检查 CSV 中的异常。
- 自动导出需要 Matplotlib，缺少时会在打开串口前报错，提示安装或使用 `--no-save-plot`。
  强制杀进程、拔盘或断电不属于正常结束，无法保证自动保存。

## CSV 字段与时间含义

| 字段 | 含义 |
| --- | --- |
| `frame_index` | 从读取器启动以来累计的解析帧序号，从 1 计数；记录前已有等待/清零，因此首行通常不为 1 |
| `timestamp_utc` | 主机处理串口批次的 UTC 时间，ISO 8601 格式，`+00:00`；北京时间加 8 小时 |
| `host_time_ns` | 同一主机时间的 Unix 时间戳，纳秒整数；纳秒表示精度不等于真实采样精度 |
| `elapsed_s` | 自启用 CSV 记录以来的单调时钟秒数，不受系统校时影响 |
| `raw_Fx_N`、`raw_Fy_N`、`raw_Fz_N` | 未清零的三个力分量，N |
| `raw_Tx_Nm`、`raw_Ty_Nm`、`raw_Tz_Nm` | 未清零的三个力矩分量，N·m |
| `net_Fx_N` 等六个 `net_*` 字段 | 原始值减去开始记录时固定的零偏，单位与对应原始列一致 |

**协议不含设备采样时间戳。** 一次串口读取通常包含约 9 帧，同批帧共享主机时间和
`elapsed_s`，按 `frame_index` 保留顺序。不要将这些时间当作独立的逐帧硬件时间，
也不要从相邻行时间差直接推算 1 kHz 频率。`host_time_ns` 在部分表格软件中可能因
数字精度限制被舍入，需要精确保留时按文本导入。

记录线程只入队，主线程写盘；缓冲每约 1 秒刷新，结束时写完剩余数据。
断流不会重复写入旧值；终端会显示 `[STALE]`，仅静默断流时仍等到指定时间才退出。
串口读取异常、队列满或写盘失败会报错并返回非零退出码，此时文件可能不完整。
异常断电、强制杀进程无法保证末尾缓冲数据落盘。不要同时运行两个程序读取同一端口。

清零不是跨姿态重力补偿。单位乘数沿用原代码的 `9.81` 假设，仍需对照实物协议核实。
没有新增校验和、NaN/Inf 过滤或安全闭环控制。终端 `std polled` 与峰值仅基于轮询数据，
完整逐帧分析应使用 CSV；`std window` 是 10 ms 窗口均值的轮询统计，不表示严格 100 Hz。

## 边采集边查看六轴滚动曲线

在桌面终端激活环境，首次使用安装 Matplotlib（已安装时无需重复安装）：

```bash
source .venv-kwr75/bin/activate
python -m pip install matplotlib
```

保持工具静止、悬空且无人接触，执行以下命令，采集 60 秒、实时显示最近 10 秒并保存 CSV：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping

env -u PYTHONPATH .venv-kwr75/bin/python \
  -m scripts.force_sensor.kwr75_reader \
  --port /dev/ttyUSB0 --tare --secs 60 \
  --plot --plot-window 10 \
  --csv "runs/force_sensor/kwr75_$(date +%Y%m%d_%H%M%S).csv"
```

- `--plot` 开启六轴窗口，左列 Fx/Fy/Fz（N），右列 Tx/Ty/Tz（Nm）；不加该参数仍只显示终端数值。
- 使用 `--tare` 时曲线显示 `net`；去掉 `--tare` 时显示 `raw`。不做 10 ms 平滑。
- `--plot-window` 指定滚动时间范围，默认 10 秒，允许 0.5–300 秒；启动后横轴从 0 开始，
  超过窗口长度后向前滚动，纵轴根据可见数据自动缩放。
- `--plot-hz` 为图形刷新上限，默认 10 Hz，允许 1–30 Hz，与传感器采样率不同。
  绘图卡顿时可降低到 `--plot-hz 5`。
- 图形显示的是主循环轮询到的最新帧，目标轮询间隔约 10 ms，实际受绘图、写盘和调度影响，
  不保证覆盖所有 1 kHz 帧，不能用图上峰值替代全量 CSV 的峰值统计。
- 读取线程仍独立解析串口，CSV 记录每个已解析帧，不进行图形降采样或滤波；写盘跟不上导致
  CSV 队列满时会报错，不会静默丢帧。图形缓冲有界，仅保留可见窗口内的轮询数据。
- 超过 50 ms 没有新帧时窗口显示红色 `STALE`，旧值不会作为新样本重复绘制；恢复后断开曲线
  再绘制新数据。非有限值显示 `INVALID`。静默断流时仍等到指定时间或用户结束；串口读取异常会退出。
- 关闭曲线窗口会结束本次采集、保存队列剩余 CSV 并关闭串口，返回 0；`Ctrl+C` 返回 130。
  到达 `--secs` 后也会自动关闭窗口。GUI 窗口不在采集结束后继续驻留，回看请用离线绘图脚本。
- 窗口会在打开串口前初始化，等待首帧时保持响应；清零期间约 1 秒内可能暂不刷新，结束后处理关闭请求。
- 使用 `--csv` 时，实时窗口关闭后默认自动保存全程 PNG；未指定 `--csv` 时只显示、不保存。

实时窗口需要桌面会话和 Tk/Qt 后端；程序在打开串口前检查图形后端。缺少 Tk 时，在此
Python 3.10 环境中可执行 `sudo apt install python3.10-tk`，然后在上面的采集命令前加
`MPLBACKEND=TkAgg`。SSH 无显示环境时应去掉 `--plot`，采集后再离线绘图。

若项目所在 USB 磁盘刚发生断连，先重新进入项目目录并用 `--help` 确认脚本可读；
反复出现 I/O 错误时停止向该磁盘采集，先备份数据并检查存储设备。

## 采集结束后的 CSV 离线绘图

绘图是独立入口，不打开串口、不控制机械臂，也不修改 CSV。先等待采集程序退出，
确保文件已关闭，再绘图。首次在采集虚拟环境中安装绘图依赖：

```bash
source .venv-kwr75/bin/activate
python -m pip install matplotlib
python -m scripts.force_sensor.plot_kwr75_csv --help
ls -lt runs/force_sensor/*.csv
```

将以下 `runs/force_sensor/kwr75_时间戳.csv` 替换为实际 CSV 路径。默认绘制清零后六轴曲线，无需桌面环境。
不指定 `--output` 时输出为输入文件同目录下的 `原文件名_net.png`；若采集时已自动生成该图，
再次离线绘图需指定不同输出路径，例如：

```bash
python -m scripts.force_sensor.plot_kwr75_csv runs/force_sensor/kwr75_时间戳.csv \
  --output runs/force_sensor/plots/kwr75_review.png
```

原始值与清零后值对比，叠加 10 ms 平均，并打开可缩放、平移的 Matplotlib 窗口：

```bash
python -m scripts.force_sensor.plot_kwr75_csv runs/force_sensor/kwr75_时间戳.csv \
  --mode both --window-ms 10 --show
```

只查看清零后第 5 到 15 秒，指定输出位置：

```bash
python -m scripts.force_sensor.plot_kwr75_csv runs/force_sensor/kwr75_时间戳.csv \
  --mode net --start 5 --end 15 --window-ms 10 \
  --output runs/force_sensor/plots/kwr75_net_5_15s.png
```

- 三行两列分别为左侧 `Fx/Fy/Fz`（N）、右侧 `Tx/Ty/Tz`（Nm），横轴为 CSV 的 `elapsed_s`。
- `--mode net` 为默认清零后值，`raw` 为原始值，`both` 为对比；未做清零的 CSV 两组曲线重合。
- `--window-ms 0`（默认）不平均；正数叠加向前回看指定毫秒的逐帧等权均值，淡色保留所有帧。
  同一批时间戳内的帧先整体纳入窗口，再输出该时间点的均值，不伪造逐帧硬件采样时间。
  这是滑动平均，不是固定 100 Hz 重采样，也不保证与在线 `window_mean()` 的轮询结果完全相同。
- `--start/--end` 按原始相对秒数截取（包含边界），横轴不重新归零；窗口平均只使用所选区间数据。
- 大于 50 ms 的时间间隔或帧序号跳号处断开曲线，不跨缺失区间连接。重复时间戳正常保留。
- PNG 自动创建父目录，不覆盖同名文件；再次绘图请更换 `--output` 或使用不同绘图参数。
- `--show` 需要桌面会话及可用的 Tk/Qt 后端；没有 GUI 时仍可直接生成 PNG。
  在此 Python 3.10 venv 中缺少 Tk 时可安装 `sudo apt install python3.10-tk`，
  然后在桌面终端执行 `MPLBACKEND=TkAgg python -m scripts.force_sensor.plot_kwr75_csv ... --show`。
  此处 `...` 需替换为实际 CSV 路径及参数；SSH 无显示环境时不要强制 TkAgg。
- 空文件、缺少字段、截断行、NaN/Inf、倒序时间或重复帧号会报错，不会静默删除异常样本。
- 旧版 `tee` 保存的 `.log` 不是 CSV，不能直接用于这个入口。

## 统计 net 六轴最大值和最小值

在项目根目录、已激活的采集环境中执行：

```bash
python -m scripts.force_sensor.kwr75_net_peaks archive/force_sensor/historical_logs/kwr75_dry_desk_hand_col_brush.csv
```

其他采集文件只需替换 CSV 路径，也支持模块入口：

```bash
python -m scripts.force_sensor.kwr75_net_peaks archive/force_sensor/historical_logs/kwr75_dry_desk_hand_col_brush.csv
```

脚本复用现有 CSV 读取与校验逻辑，只需要已有的 NumPy，不需要 Matplotlib、
串口或机械臂连接，不修改输入文件。使用全部 `net_*` 帧，不重新清零，不做 10 ms 平均。

脚本文件名和运行命令保持不变，现改为每轴独立计算 `max(net)` 和 `min(net)`，
保留正负号，不求绝对值。终端只输出六行统计表，字段如下：

- `Axis`：Fx、Fy、Fz、Tx、Ty、Tz。
- `Unit`：力为 N，力矩为 Nm。
- `Max`：该轴清零后值的最大值。
- `Min`：该轴清零后值的最小值。

不再输出绝对值峰值、帧号或时间。例如 Fz 的范围可能为 `Max = +0.278059 N`、
`Min = -18.383786 N`。六轴极值可能出现在不同时间，不是同一时刻的六维向量，
也不是力/力矩合量。计算保留浮点精度，终端数值显示六位小数。
空数据、缺列、非有限数值或非法帧序会报错并返回非零退出码。

## 无硬件验证

程序接口支持 `reader.start_csv(path, background=True)`：CSV 由单独线程写盘，
此时 `flush_csv()` 只检查记录器/串口错误，不同步等待落盘；`stop_csv()` 或 `stop()`
先分离记录器，再等待剩余记录写完，最长等待 2 秒，超时或写入失败会明确报错。
队列仍为 1024 批，满队列不阻塞串口采集，但会标记记录不完整，调用方必须停止当前任务。
默认 `background=False`，现有传感器 CLI 命令和行为不变。现场起点部署自动开启后台模式，
见 [部署运行说明](../real_deploy/manual_tared.md)。

```bash
.venv-kwr75/bin/python -m unittest discover -s tests -t . -p 'test_kwr75_reader.py' -v
.venv-kwr75/bin/python -m unittest discover -s tests -t . -p 'test_kwr75_plot.py' -v
.venv-kwr75/bin/python -m unittest discover -s tests -t . -p 'test_kwr75_net_peaks.py' -v
.venv-kwr75/bin/python -m unittest discover -s tests -t . -p 'test_kwr75_live_plot.py' -v
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest tests.real_deploy.test_recording -v
```

测试使用模拟串口，覆盖分片帧、整批记录、时间戳、清零值、防覆盖、队列溢出、
正常退出、Ctrl+C 保存、写盘失败和串口异常，不连接传感器或机械臂。
绘图测试使用合成 CSV，覆盖六轴排布、重复时间、窗口均值、时间区间、PNG 导出、
断流断线、防覆盖、异常输入及无图形界面运行。
极值测试覆盖六轴带符号最大/最小值、单帧正负数和零值、精简输出、输入不变和错误输入。
实时绘图测试覆盖六轴渲染、窗口滚动、刷新限频、缓冲上限、陈旧数据提示及恢复断线，
并用模拟串口检查图形与 CSV 同时工作、关闭窗口及 Ctrl+C 的保存和清理。
自动导出测试覆盖 CSV 关闭后生成全程图片、三种结束方式、禁用导出、空数据、防覆盖和保存失败。
