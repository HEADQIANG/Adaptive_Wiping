# 真机部署

2026-09-13：新增显式 `--mode fixed-setup` 的[同海绵固定安装入口](fixed_setup.md)，
适配当前 `airbot_native_tared_offline` 模型的在线空载扣除、坐标与滤波。
支持 `preflight / replay / shadow / run`。操作者后续确认运动跟踪/姿态偏差仅记录，
其余列出的现场条件可满足；固定安装配置已记录该批准，仍需有效回放与显式交互启动。
起点/回位核验和其他保护保留，不自动连接真机。完整命令与状态见上述固定安装文档。
下文描述默认 `calibrated` 入口；其标定和模型质量要求保持不变。

2026-09-12：项目 XYZ 工作空间边界已移除，不再要求 `workspace_min_m/max_m`。
关节、速度、跟踪、力/力矩及标定、模型和现场确认门禁不变，完整路径由现场检查。
详见 [边界移除说明](../robot_control/workspace_bounds_removed.md)。

部署分为离线回放、离线预检和现场执行。使用具备 PyTorch 及 SDK 5.2.2 的合适环境，
依赖见 `requirements/real_deploy.txt`；不要用安装依赖来替换已固定的 SDK。

```bash
python -m scripts.real_deploy --help
python -m scripts.real_deploy preflight --config configs/real_deploy/airbot_deployment.json
```

当前预检应报告未完成的标定、模型质量或现场确认，返回非零不代表入口失效。
不得将配置里的安全标志全部改为 true 来绕过检查。

历史导出策略和示教的只读回放，输出至新目录：

```bash
python -m scripts.real_deploy replay \
  --training-config archive/_migration/source_snapshot/configs/real_training_airbot_native.yaml \
  --output runs/real_deploy/historical_replay_001
```

该配置使用重构前快照，读取时解析历史路径，不改动原配置/权重/数据。
回放验证记录的 FT 与模型推理，不是闭环力控或真机验证。

完成独立现场验收、绑定准确的策略哈希与有效标定后，人工执行：

```bash
python -m scripts.real_deploy run --execute \
  --config configs/real_deploy/airbot_deployment.json \
  --output runs/real_deploy/attended_001/events.jsonl
```

全程有人监护急停与支撑。软件不自动回零、回起点或故障恢复；
控制权丢失不重新抢占，断连/idle 不保证承重。
历史回放和部署资料按主实验归档，训练目录内成套部署结果保持原相对结构。
