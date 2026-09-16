# 清零人工示教离线导入与训练

新数据推荐使用一键入口 `python -m scripts.real_training train --demonstrations <会话目录>`
`--exploration <探索日志> --confirm-same-setup`，无需修改 YAML。自动导入、预处理、训练、评估并导出。
完整示例、输出和续训说明见[路径切换操作步骤](../path_driven_workflow.md)，下文分阶段命令继续可用。

适用于 8 条实测 10 秒人工示教和一条完整的原速 manual-start 探索。
2026-09-15 本批数据的探索与示教条件已由用户确认一致。
确认作为导入时的事后人工声明保存到派生数据，不补写原会话，不冒充采集前哈希绑定或标定。
机器人、末端类型、传感器、海绵和探索编号仍必须在记录中匹配，不能用确认参数跳过。

## 导入与预处理

在项目根目录执行，使用已有 clean 环境。缺少依赖时按本目录 README 的训练环境要求安装。
所有命令均离线，不连接机器人。先只读审计，再正式生成新数据：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_manual_tared.yaml \
  --exploration runs/real_exploration/0915_152957/manual_exploration_001.jsonl \
  --session runs/real_demonstrations/manual/0915_163046/session_001 \
  --manual-tared --subtract-recorded-baseline --confirm-same-setup --audit-only

/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_manual_tared.yaml \
  --exploration runs/real_exploration/0915_152957/manual_exploration_001.jsonl \
  --session runs/real_demonstrations/manual/0915_163046/session_001 \
  --manual-tared --subtract-recorded-baseline --confirm-same-setup
```

不要使用 `real_robot/manual_exploration_001.jsonl`，那是中止日志。
不要加 `--programmed-hold-last`，人工示教有完整的实测 10 秒，不需要末状态补齐。
新输出统一位于 `runs/real_training/manual_tared/MMDD_HHMMSS/`，包含 `raw.h5`、
`import_report.json`、`run_config.yaml`，训练输出在其 `training/` 子目录。
原始 JSON/JSONL/CSV 不修改。`--audit-only` 仅使用自动清理的临时 HDF5，不创建正式输出。

将下面时间目录替换为导入终端打印的实际路径；所有后续阶段使用同一个快照：

```bash
TRAIN_CONFIG=runs/real_training/manual_tared/MMDD_HHMMSS/run_config.yaml
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training cross-validate --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training evaluate --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training export --config "$TRAIN_CONFIG"
# 同一数据、代码与配置下中断续训：
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train --config "$TRAIN_CONFIG" --resume
```

导入检查 8 条接受记录及原始哈希、完整时序、每条 start/接受记录/清零 sidecar 的一致性、
基线样本均值、至少 40 个不同接收时刻和 0.9 秒覆盖，以及所有清零值等于原始值减对应基线。
哈希涵盖被使用的原始记录及 sidecar。基线必须先于示教，禁止重复扣零、额外轴翻转、
时间戳重写或填充缺帧。完整性验收不等于真实接触质量通过。
同一清零 sidecar 被多条示教复用时，整个导入期间也必须保持同一哈希；此检查不改变上述命令。

训练继续使用冻结的 wide1200 编码器，训练 XY 分支 10000 轮、力反馈高度分支 2000 轮。
探索预处理为 `(1,400,6)`，8 条示教为 `(8,25,6)` 力序列、`(8,25,2)` XY，
共 160 个力反馈窗口。约 56 Hz 的独立 FT 接收被因果保持到 100 Hz，不代表独立 100 Hz 测量。

软件清零不等于跨姿态重力补偿或坐标标定。事后安装确认、探索跟踪误差和编码器质量/越界警告
会保留在报告中。单一海绵不能证明条件泛化，训练或导出成功不授予真机部署资格。

## 本批验收结果

2026-09-15 已完成上述真实数据的审计、正式导入、inspect 和 prepare，未启动正式训练。
结果目录为 `runs/real_training/manual_tared/0915_171709/`，有效窗口 160 个，
探索六通道归一化越界比例均为 0。数据、基线与起点 sidecar 共 27 个源文件的哈希保持不变。
现有编码器文件名所在目录为 wide1200，但检查点实际记录 `epoch=1000`；未重训或替换该编码器。
探索跟踪 RMS 约 6.52 mm、编码器质量及未标定警告仍保留，不能视为部署验收。

本批数据已经 prepare，可从项目根目录直接执行：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train \
  --config runs/real_training/manual_tared/0915_171709/run_config.yaml
```

## 回归验证

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_manual_tared_import \
  tests.real_training.test_airbot_native_training \
  tests.real_training.test_manual_exploration \
  tests.real_training.test_programmed_padding \
  tests.real_training.test_airbot_demonstrations \
  tests.shared.test_run_paths
```

测试仅使用假硬件或读取真实审计样本，不连接真机。
