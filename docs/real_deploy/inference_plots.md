# 推理结束自动保存曲线

当前 manual-tared 的现场起点部署默认开启，无需修改配置或安装新依赖。
退出旧部署进程后，继续使用 [manual-tared 启动命令](manual_tared.md) 和 h/z/s/g 顺序。
按 s 后静止采样 2 秒，执行 10 秒轨迹；出现 policy_complete 后自动后台绘图，不需要按 g 才开始。
绘图期间继续末端关节保持与状态监测，g 仍可正常切重力补偿退出。

输出位于本次实际时间目录中的 `events.plots/`（如果日志改名，目录使用相同文件名主干）：

- `fz_height.png`：清零后及滤波 Fz 时间曲线、预测下一法向 SDK 位置（水平 Z、垂直 X/Y）和 Δh；下方按预测时刻配对 Fz 与 Δh。
- `fz_ratio.png`：`100 × Fz(t) / 参考Fz`，同时显示清零后与滤波后的比值，100% 虚线表示等于参考。
- `fz_height_predictions.csv`：每次预测时间、端点时间、当前 Fz、五帧 Fz 历史、Δh、实测法向锚点及预测端点、坐标映射符号。
- `fz_reference_ratio.csv`：全部 1201 个处理点的 Fz 和比值。
- `summary.json`：参考力来源和哈希、统计口径、单位、预测次数及完成推理日志快照的哈希。
- `inference_events.jsonl`：截至 policy_complete 的日志快照。它不表示随后 g 交接或整个会话成功。

水平模式预测下一高度为 `实测Z(t) + Δh(t)`；垂直模式为 `实测X/Y(t) + sign × Δh(t)`，
`sign` 由墙面方向决定。目标端点在 `t+0.4s`；图上预测值放在发出预测的 t 时刻。
Fz 与 Δh 的对应点取预测时最新滤波 Fz；网络实际使用五帧六轴历史，不能把图当作仅由 Fz 决定高度的函数。
法向位置是 SDK 坐标，不是海绵压缩量或离表面距离。完整保留按 s 后 0～12 秒，灰色区域表示前 2 秒历史采样。

参考力来自当前策略绑定的 8 条人工示教：各自扣除记录基线，在 [0,10) 秒按 100 Hz 因果保持取样，
分别计算未滤波、有符号 Fz 均值，再对 8 条均值等权平均。启动前从绑定 raw.h5 计算并写入会话头，
随 --policy 自动切换，不使用旧程序示教的四秒窗口，也不使用论文中的固定数值。
参考力是绘图比较基准，不增加控制目标或反馈逻辑。比值不取绝对值、不裁剪；分母为零时标记未定义，
CSV 留空，报告 ratio_defined=false；近零均值与较大 RMS 并存时提示正负抵消。
绘图时核对会话中的数据/示教哈希绑定与各条均值，拒绝不一致的参考值。

绘图使用独立进程，完成事件写盘后读取快照，不在 100 Hz 控制循环中绘制。
启动或绘图失败会报告错误，原始日志保留；不会因绘图失败主动改变末端保持/交接流程。
按 g 退出并清理设备后，主程序最多等待绘图 30 秒；失败或超时返回非零，详情在同目录 `events.plots.log`。
推理中断且没有 policy_complete 时不生成完整推理图。

重绘仅适用于包含 plot_reference_fz 会话头的新日志；在项目根目录运行，指定新的输出目录：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_deploy.plot_inference \
  --events runs/real_deploy/manual_tared_run/MMDD_HHMMSS/events.jsonl \
  --output runs/real_deploy/manual_tared_run/MMDD_HHMMSS/events.plots_v2
```

现有输出目录不覆盖。离线重绘不连接硬件，沿用日志中的本次绑定参考力。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests/real_deploy -t .
```

## 垂直模式兼容说明（2026-09-16）

现场起点入口新增[垂直擦拭](vertical_wiping.md)。自动绘图命令和文件名不变，
按日志模式使用 X、Y 或 Z 法向端点及正确增量符号；垂直 CSV 字段为
`measured_anchor_x_m` / `predicted_next_x_m`，Y/Z 映射时改用
`measured_anchor_y_m` / `predicted_next_y_m`；`delta_h_to_sdk_sign` 记录方向符号。
Fz 始终指传感器局部通道，不随基座位置交换。旧日志默认为水平模式。
