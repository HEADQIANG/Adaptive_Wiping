# 论文参考值比值格式

## 定义与区别

核对 `Adaptive_wiping.pdf` 第 5 页表 I、第 6 页表 II 及图 6：论文报告的是
`100 * mean(Fz_deployment) / mean(Fz_demonstration)`，100% 虚线表示平均力与参考一致。
图 6 横轴是不同桌面高度或海绵，不是时间；论文没有报告六轴力矩逐时刻比值图。

本工具按同一固定分母定义扩展时间曲线：`R_j(t) = 100 * wrench_j(t) / reference_j`。
并非逐时刻除以示教曲线，也并非相对误差 `(actual-reference)/reference`。
本次仅有一条真机部署，不虚构多海绵/多桌面曲线或基线、AC 对照。

参考来自同一海绵的八条绑定程序示教，并非论文中的人手拖动示教。
每条使用真实的四秒横移段 `[measured_duration-4, measured_duration)`，各 400 个保持采样，
先取各条有符号均值，再等权平均。排除启动下压及所有补齐段，避免补齐时间改变分母。
该时间窗口是本数据的明确分析选择，不声称论文使用了此窗口。
不使用论文的 Normal=-12.6 N 代替本机参考。

部署曲线保留 0～10 秒，0～2 秒启动和 6.4 秒后末段保持用灰色标识。
双方均减各次独立空载基线，不额外低通，保留传感器局部轴和符号，不当作已标定法向力。
100% 表示当前值等于示教的有符号均值；不是 100% 成功率，也不是误差为零。

## 力矩比值限制

往复运动中 Fx/Fy 和力矩会正负抵消，均值可能接近零。
工具仍按真实分母计算，不取绝对值、不偷偷换成 RMS，也不截断大比值。
仅当分母绝对值不大于 1e-12 时标记未定义，报告写入 null，不加 epsilon 伪造数值。
当 `abs(mean)/RMS < 0.1` 时图中给出抵消警告，该阈值仅是本工具的提示，不是论文标准。
这些通道的大百分比不应作为控制质量、实际载荷增大倍数或安全性判断。

## 运行

项目根目录，使用 clean 环境，只读分析，不连接机器人，不改变控制或模型文件：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_deploy.plot_reference_ratio \
  --events runs/real_deploy/fixed_setup_attended_001/events.jsonl \
  --raw-data runs/real_training/programmed_hold_last_tared_004/raw.h5 \
  --evaluation runs/real_training/training_programmed_wide1200_v1/final/evaluation.json \
  --output runs/real_deploy/fixed_setup_attended_001/ratio_v1
```

输出目录必须不存在，重绘改为 `ratio_v2` 等新目录。
生成 `fz_ratio.png`、`force_ratio.png`、`torque_ratio.png` 和 `summary.json`。
报告保留六轴分母、抵消诊断、各条参考均值及部署各时间段的均值比，明确窗口为左闭右开。
依赖与 [原始对比曲线](comparison_curves.md) 相同。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_deploy.test_reference_ratio tests.real_deploy.test_plot_comparison
```

运行后逐张检查 PNG 中百分比轴、100% 参考线与近零分母提示。
