# 程序示教补齐到 10 秒

本流程按操作者要求，生成独立的 `programmed_hold_last_10s_v1` 派生数据。
原始采集文件、审核记录、失败尝试和 SHA256 不修改；不自动训练或连接机器人。
10 秒格式兼容不代表完整复现论文采集过程，也不代表新增了真实的静止接触测量。

## 补齐规则

只使用 8 条已接受且通过完整性、时序和回位校验的示教。
有效片段是下压加三段横移，排除等待、空载基线、20 ms postroll、回撤及回位检查。
原片段按实际时间因果对齐到 100 Hz，不拉伸时间，不填内部断流；最大采样年龄仍为 20 ms。

| 条件 | 原有效时长 | 尾部补齐 | 补齐帧数 |
| --- | --- | --- | --- |
| nominal | 6.0 s | 4.0 s | 400 |
| under | 5.6 s | 4.4 s | 440 |
| over | 6.4 s | 3.6 s | 360 |

每条输出时间为 0 至 10 秒，间隔 0.01 秒，共 1001 帧（包含 t=0）。
补齐段 XYZ、四元数及六维原始力始终等于最后一条有效横移观测的实测值，
不是最后的回撤状态，也不把实测值改成理想中心或零力。
原片段末端网格点仍使用因果取样；末条观测通常比该网格点晚数毫秒，
补齐自下一个网格点开始使用末条观测，并保留它实际的源时间戳，不将它回填至过去。
现有训练的因果力滤波保持不变，所以滤波输出在恒定尾部输入下可能仍有短暂收敛过程。

`is_padding` 在原有效片段为 0、派生尾部为 1。
`pose_source_time`、`ft_source_time` 保存对齐使用的原始相对时间；补齐段源时间不增长。
元数据标记 `contains_derived_samples=true`、`paper_equivalent_collection=false`，保留所有来源摘要。
`source_kind=real` 表示来源是真机，不表示所有输出帧均为实测；必须结合派生标记理解。

## 命令

在项目根目录使用 `clean` 训练环境，不使用机器人 SDK 环境。
配置为 `configs/real_training/real_training_programmed_hold_last.yaml`，引用本次探索起点对应的
`exploration_tared_plot_003.jsonl` 和完整 `direct_start_session_004`，不使用中断的探索记录。

先审计（不生成正式输出）：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_programmed_hold_last.yaml \
  --exploration runs/real_exploration/exploration_tared_plot_003.jsonl \
  --session runs/real_demonstrations/programmed/direct_start_session_004 \
  --programmed-hold-last --audit-only
```

首次生成副本：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_programmed_hold_last.yaml \
  --exploration runs/real_exploration/exploration_tared_plot_003.jsonl \
  --session runs/real_demonstrations/programmed/direct_start_session_004 \
  --programmed-hold-last
```

输出 `runs/real_training/programmed_hold_last_004/raw.h5` 和 `import_report.json`。
已生成时不要重复导入，已有目标文件会拒绝覆盖。更换数据时复制配置并使用新输出路径。
不带 `--programmed-hold-last` 的旧导入入口仍拒绝原始变长程序协议；人工导入不变。

验证和准备训练输入：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect \
  --config configs/real_training/real_training_programmed_hold_last.yaml
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare \
  --config configs/real_training/real_training_programmed_hold_last.yaml
```

准备后得到每条 25 个策略时刻（0.4 秒间隔），并保留对应 `is_padding` 标记和完整派生元数据。
训练输出目录为 `runs/real_training/training_programmed_hold_last_004`，不覆盖旧模型。
正式训练是单独操作，本次转换不执行：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train \
  --config configs/real_training/real_training_programmed_hold_last.yaml
```

## 解释边界与验证

按此次要求，补齐帧参与原有训练和指标计算，mask 只标记来源，不自动剔除或降权。
约 36% 至 44% 的时长是恒定尾部，容易使高度网络和整体指标偏向零高度变化。
这批固定深度示教没有展示恒力纠偏，不能把低训练误差视为学会力反馈。
当前审计还报告探索力输入超出旧编码器归一化范围、源编码器质量警告及未标定坐标问题，
未裁剪这些输入，也未自动更换编码器；必须在评估中处理，训练完成不等于可上机部署。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_programmed_padding \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.real_training.test_airbot_native_training \
  tests.real_training.test_real_training
```

覆盖三种时长、常值尾部、内部缺帧拒绝、回撤排除、显式启用、训练形状、mask 与尾部篡改拒绝。

本次已执行审计、正式导入和 prepare，所列 `raw.h5`、`import_report.json`、`prepared.h5` 及
`prepared_integrity.json` 已生成，不必重复导入或 prepare。训练输入已通过 `load_prepared` 验证：
力为 `(8,25,6)`，XY 为 `(8,25,2)`，高度和补齐标记为 `(8,25)`；原文件摘要全部保持不变。
clean 定向回归 73 项全部通过；SDK 采集定向回归 144 项中 143 项通过、1 项缺少 h5py 跳过。
未启动正式训练，未连接机器人；上述编码器范围和质量警告仍未解决。
