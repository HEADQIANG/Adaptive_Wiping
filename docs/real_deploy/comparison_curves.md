# 真机部署与示教参考曲线

## 本次人工起点与旧固定安装运行对比

以下命令只读取两份已完成的部署事件日志，不连接机器人、不重放运动、不修改原来的示教对比工具。
`--events` 必须是现场起点 workflow，`--reference-events` 必须是旧 fixed-setup。
同一命令兼容历史 10 秒日志及 2026-09-16 起的“静止 2 秒 + 运动 10 秒”日志；
替换为本次实际日志和新输出路径即可。新日志显示全部 12 秒和 25 次高度预测，
旧日志仍为 10 秒和 20 次预测，不截掉新日志末尾，也不将两次轨迹拉伸到相同时长。
本次操作者已确认在受压/接触状态去皮，因此显式传入 `--tare-contact loaded`；
其他运行不能照抄该事实标记，未知时省略或用 `unknown`，确认空载时用 `unloaded`。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_deploy.plot_run_comparison \
  --events runs/real_deploy/manual_tared_run/0915_210143/events.jsonl \
  --reference-events runs/real_deploy/fixed_setup_attended_001/events.jsonl \
  --tare-contact loaded \
  --output runs/real_deploy/manual_tared_run/0915_210143/comparison_fixed_setup_v1
```

新增工具使用已有 NumPy、SciPy 和 Matplotlib，无需安装 SDK、串口或训练依赖。
输出目录必须不存在；重复绘图改用 `comparison_fixed_setup_v2` 等新目录，不覆盖历史产物。
遵循项目时间目录规则；本例已在 `0915_210143` 内，不再增加一层时间目录。

输出：`position_xyz.png`（绝对位置）、`position_relative_xyz.png`（相对起点）、
`force.png`、`torque.png`（去皮及在线滤波）、`wrench_raw.png`（策略原始载荷与基线）、
`height_feedback.png`（初始化与反馈分离）、中文 `analysis.md` 和 `summary.json`。
图中青色为本次 manual，红色为旧 fixed，目标用虚线，实测用实线；灰底为前 2 秒初始化，
浅黄底为 2–2.4 秒首次高度反馈段。高度预测图实心点为预测时刻，空心方块为下一目标端点时刻。

位置反馈用 SDK 实测时间，目标用策略计划时间；各自策略启动为 t=0，不拉伸时间、不校正延迟。
相对图对同一次的实测和目标减去同一个参考起点，不消除二者的初始偏差。
力/力矩为策略采用的 100 Hz 因果保持值，不是串口 CSV 全帧；保留传感器局部轴与符号。
六轴先独立扣除各自基线，再核对二阶 1 Hz 因果滤波，不把传感器 Fz 等同于桌面法向力。

工具校验两种日志格式、完整周期、完成状态、有限数值、时间戳、去皮/滤波复算以及
`下一端点 Z = 预测时实测 Z + delta_h`；报告分开统计 0–2、2–2.4、2.4 秒至各自结束。
另将第 2 秒至结束的目标 Z 净变化分解为预测增量总和与实测重新锚定偏移总和，避免把预测简单累加当作实际运动。
新流程额外核对前 2 秒 XYZ 目标恒定及运动 tick；`summary.json` 记录每次运行的时长、周期数与预测数。
不会用配置中的旧 initialization 标签覆盖现场起点日志记录的实际初始化方式。
缺少桌面高度和工具几何时不计算间隙/压缩量，不将受压去皮后接近零的值解释为空载。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_deploy.test_plot_run_comparison tests.real_deploy.test_plot_comparison -v
```

生成后逐张打开六张 PNG，确认小幅 Z 变化、图例、单位和阶段标记可读。

## 原有部署与示教对比

从项目根目录执行，仅离线读取数据，不连接设备，不修改控制配置或训练结果：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_deploy.plot_comparison \
  --events runs/real_deploy/fixed_setup_attended_001/events.jsonl \
  --raw-data runs/real_training/programmed_hold_last_tared_004/raw.h5 \
  --evaluation runs/real_training/training_programmed_wide1200_v1/final/evaluation.json \
  --output runs/real_deploy/fixed_setup_attended_001/comparison_v1
```

需要 NumPy、SciPy、h5py、Matplotlib。输出目录必须不存在；重绘请改用 `comparison_v2` 等新目录。

- `position_xyz.png`：部署实测位置、部署下发目标，以及八条示教实测位置。
- `force.png`：Fx、Fy、Fz 与八条示教参考的比较。
- `torque.png`：Tx、Ty、Tz 与八条示教参考的比较。
- `wrench_filtered.png`：双方均采用训练相同的二阶 1 Hz 因果低通后的六轴对比。
- `summary.json`：输入文件 SHA256、来源、各示教真实有效时长和图像列表。

参考来自本次模型绑定的程序固定深度示教 `direct_start_session_004`，并非另一套人手拖动示教。
八条参考逐条显示，没有用均值替代，也不是力控制设定值。
各自运动开始作为 t=0，不对齐横移启动时刻，不进行时延校正或时间拉伸。
位置用 SDK 位姿读取时间，命令用策略计划时间；这是曲线叠加，不将随后发送命令当成之前观测的跟踪目标。
位置单位毫米，保留基座绝对坐标，不当作海绵 TCP 或海绵压缩量。

部署力来自策略实际使用的因果采样 `raw_ft - baseline`，不是 CSV 中未去皮的 net 字段。
示教使用已保存的 `ft_raw_before_baseline - recorded_unloaded_baseline`；两者保留传感器局部轴、
原点和符号，不取绝对值，不翻转轴，不把 Fz 直接当作表面法向力。
未滤波图保留 100 Hz 保持值，不意味着传感器有 100 Hz 独立观测。

八条示教的原始有效时长为 5.6、6.0 或 6.4 秒，随后保持末状态补齐至 10 秒。
每条参考真实段用实线、补齐段用虚线；浅灰区为部分示教开始补齐，深灰区为全部补齐。
部署 10 秒均是实际运行记录，不能把部署后段与虚构补齐段的吻合当作真实示教验证。

脚本校验部署 tick 完整性、独立去基线、在线滤波复算、训练原始数据哈希及示教会话绑定。
运行后逐张打开 PNG，确认九个物理分量、图例、单位和补齐标记完整。

回归测试（不连接设备）：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m unittest tests.real_deploy.test_plot_comparison
```
