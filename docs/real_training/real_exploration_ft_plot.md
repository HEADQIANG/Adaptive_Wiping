# Real exploration force/torque plots

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Run from the repository root with Python 3, NumPy and Matplotlib installed:

```bash
python3 -m scripts.real_training.tools.plot_real_exploration_ft archive/real_training/real_robot/exploration_ft_007.jsonl
```

The default output is `archive/real_training/real_robot/exploration_ft_007_curves.png`.
Use `--output PATH.png` to choose a different destination. An existing output
image at that path is replaced. The input log is not modified. No robot or
sensor connection is made.

Only `event=sample`, `phase=exploration` records are plotted, using the
post-send `force` fields and `protocol_time_s`. Initial observations,
pre-send observations and retraction samples are excluded. Six panels show
Fx, Fy, Fz in N and Tx, Ty, Tz in N m, in the sensor-local frame at the
sensor origin. No filtering, gravity removal or frame conversion is applied.
Bias-corrected data is overlaid only when different from the raw data.

Background stages denote commanded protocol time: press 0-2 s, +Y slide
2-3 s, -Y slide 3-4 s. They do not certify actual trajectory tracking.
For time-scaled runs this axis remains protocol time, not elapsed wall time.
