# 真机示教与离线训练

本模块分为真机采集和离线学习，离线学习不连接机器人，不新增在线强化学习。
在 `clean` 环境安装 `requirements/real_training.txt` 和 PyTorch >= 2.6，再可编辑安装本项目。

```bash
python -m scripts.real_training --help
python -m scripts.real_training smoke-test
python -m scripts.real_training explore --help
python -m scripts.real_training demonstrate --help
python -m scripts.real_training demonstrate preview --mode programmed
python -m scripts.real_training import-airbot --help
python -m scripts.real_training calibrate-data --help
```

操作顺序：按机械臂控制文档检查设备、记录起点；完成传感器与坐标标定；
采集探索和示教；导入/转换数据；检查数据契约；准备、训练、评估并导出策略。
采集使用 SDK 环境和原有 `--execute` 人工确认，不因代码重构降低原有安全要求。
新增的 [程序示教](airbot_programmed_demonstrations.md) 支持启动时低速直达探索起点（需确认整条直达路径无障碍，到位后仍等新的 `s`），以及 8/10/12 mm 下压、
固定深度三段横移，只记录力并沿用探索超限保护，不进行 10 N 力反馈。
不再要求闭环参数；当前已记录操作者的新路径现场确认，preview 应退出 0，随后须执行只读起点检查。
程序示教静止速度判据为 0.1 rad/s；基线不稳时丢弃整段，在总计 10 秒内重新确认静止并采集。
回撤独立允许 10 ms 迟到，顺延节拍、不追赶，总回撤最多额外 1 秒；擦拭段时限不变。
本次配置更新须使用新会话目录，见程序示教文档中的 `direct_start_session_004` 命令。
未确认的配置仍禁止真机执行。
它保存变长原始数据，默认不进入固定 10 秒/25 帧训练流程。
按操作者新增要求，可显式生成[保持末状态补齐到 10 秒的派生训练副本](programmed_hold_last_training.md)，
使用独立配置 `real_training_programmed_hold_last.yaml`，不修改原始记录；下面的 native 命令仍指向旧数据。
未指定 `--mode` 时仍为人工拖拽。
探索时可加 `--plot` 实时查看软件清零后的六轴力/力矩，见
[实时曲线运行步骤](real_exploration_live_plot.md)。

```bash
python -m scripts.real_training inspect --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training prepare --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training cross-validate --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training train --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training evaluate --config configs/real_training/real_training_airbot_native.yaml
python -m scripts.real_training export --config configs/real_training/real_training_airbot_native.yaml
```

当前 native 配置读取归档的示教数据和编码器，新输出位于 `runs/real_training/`。
本次新仿真模型加补齐示教的训练使用 `real_training_programmed_wide1200.yaml`，
完整参数核对和命令见[新预训练模型训练](programmed_wide1200_training.md)，不要混用上面的旧归档配置。
使用新数据时同时修改输入路径和新输出目录，不覆盖已有运行。正式训练轮数保持原配置，
先运行 smoke-test 验证软件，不自动启动耗时正式训练。
native 模型缺少已验证坐标标定时仍不能进入默认 calibrated 部署入口。
当前已空载扣除的模型可使用独立的[同海绵固定安装入口](../real_deploy/fixed_setup.md)，
仅适用于已确认不变的安装和同一海绵，并仍需通过回放与现场运动验收。
已有模型质量问题不会因适配消失，固定条件例外不代表标定或泛化验证通过。

示教原始会话位于 `archive/real_training/raw_data/manual_demonstrations/`；
已有训练结果位于 `archive/real_training/real_training_airbot_native_v1/` 等目录。
详细说明见 [离线训练](real_training.md)、[人工示教](airbot_demonstrations.md)、[标定流程](airbot_calibrated_pipeline.md)。
