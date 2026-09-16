# runs 输出目录与运行步骤

输出按七类存放，目录含义、旧路径兼容和清理恢复见 [runs 分类说明](runs_layout.md)。

2026-09-15 起，新的 CLI 输出采用本机时间 `月日_时分秒`，例如 `0915_142205`。
不带年份、微秒或编号后缀。同一个父目录在同秒启动多次时，使用下一个未占用的秒名称，
不等待、不覆盖。名称是目录分配时间，准确采样时间仍以日志时间戳为准。

## 目录规则

- 文件参数：`runs/real_exploration/manual_exploration_001.jsonl` 自动变为
  `runs/real_exploration/0915_142205/manual_exploration_001.jsonl`。
- 目录参数：`runs/sim_data/explore_once` 自动变为
  `runs/sim_data/0915_142205/explore_once/`，其中保存该轮全部文件。
- 同轮日志、传感器 CSV、曲线 PNG 共用分配后的路径；不会每个采样或检查点新建目录。
- 终端 `Output:` 显示实际保存路径。后续输入参数、示教关联、部署绑定必须使用该路径，
  不再使用省略时间目录的旧示例，也不自动查找“最新”文件。
- 显式路径中已有合法 `MMDD_HHMMSS` 目录时不再嵌套，适用于续采、续训和追加分析。
  显式重复文件仍遵循各工具原有的禁止覆盖规则；要开始新一轮，请传不含时间目录的路径。
- `runs/` 已按七类迁移，历史产物内容与哈希不变；`archive/` 仍只读。
  `runs/` 外的输出不自动添加时间层。

以上适用于探索、示教、部署回放及日志、机械臂控制记录、力传感器采集/标定/绘图，
仿真探索、实验输出及分析报告，以及真机导入和训练准备。
`status`、`check`、`preview`、`inspect`、`--audit-only` 不分配新输出；恢复或验证已有数据的
`--output` 实际是输入目录，也不改指向。低层文件写入函数不重定向已确定的路径。

## 真机探索

在项目根目录执行。Python 路径开头有 `/home`，入口使用 `-m scripts.real_training`：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training explore preview
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --execute --time-scale 1 --plot \
  --output runs/real_exploration/manual_exploration_001.jsonl
```

每次新探索可重复使用这条命令，不需要手工递增 `_001`。
例如本轮打印 `Output: .../real_exploration/0915_142205/manual_exploration_001.jsonl`，则 PNG
也保存在 `0915_142205` 内。不要把这个示例时间预先写死在新探索命令里。

现场流程不变：重力补偿拖拽，确认 1 mm 间隙，`h` 固定，新的 `s` 清零并执行探索和回撤，
保持期间准备支撑，输入 `IDLE` 回车交接。无项目绝对关节角范围检查、静止验收和超力停机；
返回目标完成不等于实测回位。仍需实体急停和全程监护。
重力补偿拖拽阶段不执行项目侧速度阈值检查，速度防护仅依赖 SDK/固件；切入 servo 后
恢复 1.2 rad/s 实测超速停止，设备健康、控制权、数据有效性和力流时效检查仍保留。详见
[手动起点探索](real_training/manual_start_exploration.md)。本次路径修改不触发任何真机运动。

## 示教与导入

1. 将探索打印的实际 JSONL 路径填入新程序示教配置的 `exploration_log`，完成现场安装确认。
2. 启动新示教时可传原来的目录名称，程序会在其前增加时间层。
3. 记录本轮打印的实际会话目录；`status` 和恢复同一会话必须传这个完整目录。
   显式传入含 `session.json` 的旧目录仍表示续采，不代表开始新会话。
4. 导入时使用完整探索路径和完整示教目录；新导入生成新的派生数据及训练输出路径。
   原始探索和示教不移动。

```bash
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate run \
  --mode programmed --execute \
  --config configs/real_training/airbot_programmed_manual_demonstrations.json \
  --output runs/real_demonstrations/programmed/manual_start_session_001

# 以下时间仅演示路径格式，必须替换为上面和探索命令实际打印的路径
EXPLORATION_LOG=runs/real_exploration/0915_142205/manual_exploration_001.jsonl
DEMO_SESSION=runs/real_demonstrations/programmed/0915_143000/manual_start_session_001
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate status \
  --mode programmed --output "$DEMO_SESSION"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_manual_start.yaml \
  --exploration "$EXPLORATION_LOG" --session "$DEMO_SESSION" \
  --programmed-hold-last --subtract-recorded-baseline
```

去基线仍是 `ft = raw - baseline`，不是再次扣除清零后的值。人工确认和关联验收规则不变。

## 多阶段训练

新真机导入、单独 `prepare`、仿真 `sanity`、宽范围采集和子集训练会打印
`Run config (use for subsequent stages): .../run_config.yaml`。
快照位于新输出目录的旁边，包含实际 `output_dir`；导入快照也包含新的 `raw_data`。
原 YAML 和编码器、采集来源等输入路径不变。快照不会自动选择或指向下一轮。

真机导入完成后，将 `TRAIN_CONFIG` 设为**实际打印的快照路径**，再执行：

```bash
TRAIN_CONFIG=runs/real_training/0915_150000/run_config.yaml
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training evaluate --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training export --config "$TRAIN_CONFIG"
```

`train --resume` 和交叉验证同样使用快照。不要在导入后重新使用原模板运行 `prepare`，
那会分配另一轮训练输出，且原模板的 `raw_data` 不会被自动改为新导入数据。
已经存在的旧训练可以继续显式使用原配置和原续跑命令。

仿真先运行下面的 `sanity`，然后将 `SIM_CONFIG` 替换为打印的快照：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain sanity \
  --config configs/sim_pretrain/pretrain_paper.yaml
SIM_CONFIG=runs/sim_training/0915_151000/run_config.yaml
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain collect --config "$SIM_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain train --config "$SIM_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain evaluate --config "$SIM_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain export --config "$SIM_CONFIG"
```

`scripts.sim_pretrain.experiments.collect_wide_training` 和 `train_collected_subset` 默认分配
新输出；使用打印的快照继续同一轮。需要继续没有时间层的历史目录时，显式添加 `--resume`。
数据恢复、`--verify-only`、`--verify-pilot`、`--verify-export-only` 仍使用准确的既有目录。
力传感器 `record-unloaded` 不传目录时新建一轮；续采请用 `--directory` 指定已有的会话目录。

## 其他入口与验证

从项目根目录使用模块入口，相关输出参数遵循同一规则：

```bash
python -m scripts.force_sensor read --csv runs/force_sensor/reading.csv
python -m scripts.robot_control capture-pose --output runs/real_deploy/robot_control/pose.json
python -m scripts.sim_pretrain explore-once --output runs/sim_data/explore_once
python -m scripts.real_training --help
python -m scripts.real_deploy --help
/home/wp/miniconda3/envs/clean/bin/python -m unittest tests.shared.test_run_paths
```

机械臂不带 `--execute` 的基础控制命令为离线假硬件预演；本页不是现场安全批准。
已存在根目录脚本的旧文档命令应对应改为 `python -m scripts.<模块>`，不要创建同名旧入口。
