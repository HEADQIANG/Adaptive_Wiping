# runs 七类目录与整理操作

2026-09-16 起，运行产物按用途存储，Python 包名、配置目录名和训练算法不变。

| 目录 | 内容 |
| --- | --- |
| `runs/force_sensor/` | 单独采集的六维力/力矩 CSV、标定及曲线 |
| `runs/sim_data/` | 仿真探索、参数扫描、训练数据集及采集记录 |
| `runs/sim_training/` | 仿真编码—解码训练、模型、评估及潜变量诊断 |
| `runs/real_exploration/` | 真机探索日志、起始姿态及探索曲线 |
| `runs/real_demonstrations/` | `manual/` 人工示教、`programmed/` 程序示教和 `analysis/` 分析 |
| `runs/real_training/` | 示教导入后的派生数据、训练配置、检查点及策略 |
| `runs/real_deploy/` | 真机部署、离线回放及 `robot_control/` 控制/SDK 辅助记录 |

部署和示教会话中的传感器 CSV 保留在各自会话内，以维持时间关联；不归入单独传感器采集。
SDK 的根目录 `logs` 链接指向 `runs/real_deploy/robot_control/sdk_logs`。

## 仿真数据与训练的关联

同一轮的仿真数据放在 `sim_data/<轮次>/`，模型放在 `sim_training/<轮次>/`。
训练目录中的 `dataset.h5` 等数据入口是相对符号链接，只有一份物理数据。
采集器新建数据也使用这个布局；自定义在 `runs/sim_training/` 之外的输出保留原行为。
复制训练目录到另一台机器时，要同时复制对应的数据目录并保留相对层级。

历史模型、HDF5、示教清单和配置快照的字节不修改，以保留原始 SHA256 绑定。
项目读取接口兼容旧 `runs/sim_pretrain`、`runs/real_training/real_robot` 和旧示教目录名。
通用文件浏览器、外部脚本不经过该接口，应使用本页的新路径。
部署仍逐项校验来源哈希，只将旧路径解析到新的同一份文件，不放宽内容校验。

已有 1200 条仿真训练的原始配置从其 `manifest.json` 提取为
`runs/sim_training/pretrain_wide_1200_v1/run_config.json`。复核该模型时使用这个快照，
因为训练数据绑定的是原配置的完整哈希。`configs/sim_pretrain/pretrain_wide_1200.yaml`
是新运行模板，路径已更新，不能冒充旧训练配置。
当前模板中的 `collection.source` 指向待提供的 2000 条源数据目录；该目录整理前就不存在，
如需重新抽样，须先采集或提供完整源数据。读取现有 1200 条数据、模型与重训练无需该目录。

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain evaluate \
  --config runs/sim_training/pretrain_wide_1200_v1/run_config.json \
  --output runs/sim_training/recheck_wide_1200
```

## 日常运行

从项目根目录执行，时间目录仍由程序分配；后续操作使用终端打印的实际完整路径。
采集和训练的详细步骤见 [运行步骤](run_outputs.md)。

```bash
python -m scripts.force_sensor read --csv runs/force_sensor/reading.csv
python -m scripts.sim_pretrain explore-once --output runs/sim_data/explore_once
/home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
# 将 SIM_CONFIG 设置为 sanity 打印的 runs/sim_training/.../run_config.yaml
# 然后依次执行 collect、train、evaluate、export，均传 --config "$SIM_CONFIG"
```

真机探索用 `--output runs/real_exploration/manual_exploration_001.jsonl`；
人工示教用 `--output runs/real_demonstrations/manual/session_001`；
程序示教用 `--output runs/real_demonstrations/programmed/session_001`。
真机训练和部署继续使用 `runs/real_training/` 和 `runs/real_deploy/`。
程序示教模板的 `exploration_log` 是待填写的占位路径；采集前填入本轮完整探索路径，
不能使用已清理的无采样探索，也不自动绑定历史探索。
这里的路径整理不启动硬件、不重新训练，也不切换现有策略所绑定的编码器。

## 不完整记录与恢复

本次清理以实际会话内容判断：探索无采样、部署未完成策略段、程序示教不足 8 条、
示教中未被接受且无结束事件的尝试、仅 prepare 而未训练的孤立目录。
已经完成采样/策略、仅退出交接异常的记录保留。零字节锁文件属于正常控制文件。
完整的失败实验、旧塌缩模型、重训练结果和有效采集数据保留。

被清理的文件从 `runs/` 移到项目内 `.runs_cleanup/<整理时间>/`，可恢复，未永久销毁。
`runs/organization_20260916.json` 记录每个原文件的去向、大小、SHA256 和清理原因。
恢复时先查该清单，将恢复区的文件复制到指定的新分类路径；勿覆盖已有文件。

整理工具的 `plan` 仅盘点；`apply` 执行清单；`verify` 对保留及清理文件逐一校验。
执行整理前停止正在写入 runs 的采集、训练及部署进程。

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.shared.reorganize_runs plan
/home/wp/miniconda3/envs/clean/bin/python -m scripts.shared.reorganize_runs apply
/home/wp/miniconda3/envs/clean/bin/python -m scripts.shared.reorganize_runs verify
/home/wp/miniconda3/envs/clean/bin/python -m unittest tests.shared.test_runs_layout tests.shared.test_run_paths
```

## 2026-09-16 整理结果与验证

原有 999 个文件中保留 927 个，665 个调整位置，72 个（71,028,419 字节，约 67.7 MiB）
移入 `.runs_cleanup/20260916_174054/`。全部原文件在保留位置或恢复位置均通过 SHA256 校验。
恢复区仍占用磁盘空间；本次没有永久销毁原始记录。新增目录导航、迁移清单和原配置快照另计。

4 份真机策略及其训练输入均可读取，人工示教和程序示教部署的离线 `load_setup` 校验通过；
程序示教 004 的 `status` 验证 8/8 条有效。旧、新编码器均输出有限的 5 维编码。
1200 条仿真数据仍为 960/120/120，旧 VAE 复核 MSE 为 0.00447345245629549，与整理前一致。
新采集在临时目录实测通过 `sim_data` 存储、训练链接读取及数据哈希校验。

路径与数据分离测试通过。扩展的 122 项回归中 120 项通过，另 2 项历史物理数值回归失败：
`test_reused_environment_matches_fresh_second_assignment` 触发 100 mm/20° 跟踪保护；
`test_vectorized_checks_match_saved_pilot_dynamics` 与归档 pilot 的数值不同。
在测试进程中恢复旧路径解析、旧采集路径和原配置 output_dir 后，这两项仍以同样方式失败；
本次未修改物理模型或降低断言要求。该结果不影响文件完整性和模型读取验证。
