# 新预训练模型的高度与轨迹训练

2026-09-15：下文是已有实验，保持原样。新的 manual-start 探索只能配采集前已绑定它的
新程序示教，并使用独立数据/训练输出；命令见 [手动起点探索与训练](manual_start_exploration.md)。
本实验已经使用去基线数据，不是原始载荷训练。

本次使用新完成的 `runs/sim_training/pretrain_wide_1200_v1/vae_last.pt`（第 1000 轮），
按现有导出流程生成并冻结 `encoder.pt`，不使用旧归档编码器，也不使用未经本轮最终测试评估的
`vae_best.pt`（第 997 轮）。`pretrain_wide_2000_v1` 是提前结束采集的来源，不是已完成的模型。
输入来源为 8 条末状态保持至 10 秒的独立派生数据。本次为了匹配新仿真输入，另生成
`programmed_hold_last_tared_004/raw.h5`，按各自实际记录的空载基线扣除六轴偏置。
旧的 `programmed_hold_last_004/raw.h5` 原始载荷副本保持不变。
新编码器必须重新生成 embedding 和 prepared 文件，不能复用旧编码器的 prepared 数据。

## 论文核对

参考 `Adaptive_wiping.pdf` 第 3 页 III-B 和第 5 页 IV-C2、IV-D，已检查原文页面。

| 项目 | 本次配置 | 依据 |
| --- | --- | --- |
| 示教输入 | 8 条，每条 25 个时刻，间隔 0.4 秒 | IV-C2 |
| 海绵编码器 | 5 维，冻结权重，使用后验均值 | III-A/B；均值推理沿用仓库 |
| 轨迹分支 | 单全连接层，5 输入、50 输出，dropout 0.1 | III-B1，输出为 25x2 XY |
| FT 分支 | 5 帧历史，2 层 TCN，每层 25 通道，dropout 0.1，输出 6 维 | III-B2、IV-D |
| 高度分支 | 拼接 5+6 维，128 隐藏维，ReLU，dropout 0.1，输出下一步高度增量 | III-B2 |
| 优化器及损失 | 两分支独立 Adam，lr=0.001，MSE | IV-D、III-B |
| 训练轮数 | XY 10000；FT+height 2000 | IV-D |
| 预处理 | Butterworth，通道归一化 [0,0.9] | IV-D |

论文未在以上段落规定 batch size、随机种子、TCN kernel size 或滤波截止频率等实现细节。
保留仓库实现：XY batch=8、FT batch=32、seed=42、kernel=3、单线程 CPU；
探索编码器采用保存的二阶 10 Hz 离线滤波，下游力采用二阶 1 Hz 因果滤波。
不为“与论文一致”虚构原文未给出的参数。

本次只对齐下游训练超参数和公开结构，不宣称完整实验复现：新预训练为 1200 条总数据、1000 轮、
扩展随机化；论文为 1000 条、200 轮及另一组范围。示教为程序固定深度加常值尾部，而非 10 秒人工示教；
AIRBOT 坐标和传感器尚未完成仿真坐标标定。配置使用独立的 `airbot_native_tared_offline`，保留全部警告。
新编码器测试报告还有 `collapse_warning=true`、有效潜变量维数 0，且不优于均值重建基线。
训练可作为离线拟合实验，但不能证明学会海绵属性泛化或真机力反馈。

## 执行步骤

项目根目录，使用 `clean` 环境。下列命令均不连接机器人。

首次导出新编码器（本次已完成，不要重复覆盖）：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain export \
  --config runs/sim_training/pretrain_wide_1200_v1/run_config.json
```

生成去基线副本（新数据目录只导入一次）：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_programmed_wide1200.yaml \
  --exploration runs/real_exploration/exploration_tared_plot_003.jsonl \
  --session runs/real_demonstrations/programmed/direct_start_session_004 \
  --programmed-hold-last --subtract-recorded-baseline
```

新仿真模型元数据为 `ft_frame local with output Y/Z reversed`，不同于旧模型的 `ft_frame local`。
此前 `scripts/sim_pretrain/experiments/match_real_amplitude.py` 已使用真机 tared 字段对比调整后的仿真输出，
本次沿用此约定：真机轴保持原样，不再次翻转；探索及每条示教减去各自成功 `tare_complete` 的基线。
新副本保存扣除前的 `ft_raw_before_baseline`、`recorded_unloaded_baseline` 和所有原文件摘要，
加载时验证 `ft = raw - baseline`，不把新模型重命名为旧坐标约定。此处理仅用于离线实验，不是坐标标定。
两个 native profile 分别校验各自模型 frame 和补偿方式，不能静默交叉混用；真机部署仍要求已标定 profile。

核查与准备（新输出目录只 prepare 一次）：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect \
  --config configs/real_training/real_training_programmed_wide1200.yaml
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare \
  --config configs/real_training/real_training_programmed_wide1200.yaml
```

训练和训练集回放评估：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train \
  --config configs/real_training/real_training_programmed_wide1200.yaml
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training evaluate \
  --config configs/real_training/real_training_programmed_wide1200.yaml
```

如训练中断，确认配置、数据、编码器和代码未变化后，给 train 命令加 `--resume`。
每 100 轮保存检查点；不要重建 prepared 数据或覆盖旧输出以尝试续训。
结果目录为 `runs/real_training/training_programmed_wide1200_v1/`，其中 `final/training.pt` 同时保存
轨迹分支、高度反馈分支、冻结编码器、归一化参数和优化器状态。
`final/status.json` 记录完成轮数，`final/history.json` 记录训练损失；evaluate 生成误差报告和图。
这里的评估使用全部训练示教及记录的力，不是独立测试或闭环仿真，更不是真机验收。
补齐帧参与原有损失和指标，需结合真实运动段与补齐段分别解读，不能仅依赖整体误差。

代码修改后回归：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_programmed_padding \
  tests.real_training.test_airbot_native_training \
  tests.real_training.test_real_training \
  tests.real_training.test_airbot_programmed_demonstrations
```

## 本次结果

已完成导出、去基线副本导入、prepare、完整训练和 evaluate，不必重复执行首次生成命令。
轨迹分支 10000 轮、高度反馈分支 2000 轮，训练记录共 12000 条，无 NaN 或提前终止。
已验证最终检查点中的海绵编码器权重与新导出的源编码器完全相同，所有原始采集文件摘要未改变。

| 训练集回放 RMSE | 全部时刻 | 非补齐目标时刻 | 补齐目标时刻 |
| --- | --- | --- | --- |
| XY 坐标分量 | 6.0353 mm | 7.7430 mm | 1.0625 mm |
| 下一步高度增量 | 0.05253 mm | 0.07355 mm | 0.01045 mm |

这里 XY 是各坐标分量合并后的 RMSE，不是二维欧氏距离的平均值；高度是下一步增量误差，不是累积高度误差。
分段统计使用 prepared 的 `is_padding`：XY 对应全部 25 时刻，高度对应第 6 至 25 时刻，
仅按目标时刻分类，未要求历史窗口完全落在同一段。
原有均值轨迹基线的全时刻 XY RMSE 为 6.03525 mm，轨迹网络几乎与其相同。
全部示教共享同一次探索 embedding，轨迹分支没有深度组别输入，因此不能区分不同下压时长造成的
横移开始时间差异，不应把它理解为已学会三组各自独立的时间轨迹。
高度零增量基线 RMSE 为 0.34839 mm；模型在训练集记录力上的拟合更好，但没有闭环执行证据。

去基线后，本次探索六通道的编码器归一化越界比例均为 0；潜变量坍塌和非论文预训练配置警告仍存在。
最终归一化训练 MSE（含训练时 dropout）与回放 RMSE 不同，勿直接混用。
主要文件：

- `runs/real_training/training_programmed_wide1200_v1/final/training.pt`
- `runs/real_training/training_programmed_wide1200_v1/final/status.json`
- `runs/real_training/training_programmed_wide1200_v1/final/evaluation.json`
- `runs/real_training/training_programmed_wide1200_v1/final/training.png`
- `runs/real_training/training_programmed_wide1200_v1/final/predictions.npz`

回归：clean 定向测试 75 项全部通过；SDK 采集定向测试 144 项中 143 项通过、1 项缺少 h5py 跳过。
没有启动真机，没有执行留一交叉验证，没有导出或批准可部署策略；`hardware_ready=false` 保持不变。
