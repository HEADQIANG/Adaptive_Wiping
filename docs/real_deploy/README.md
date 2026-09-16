# 真机部署

切换策略可使用 `preflight --policy /path/to/policy.pt`，随后 `run --policy /path/to/policy.pt`
并追加原有 `--output ... --execute`。自动识别模式并解析策略绑定的数据，清零人工模型无需手改配置路径。
完整命令和旧策略兼容边界见[命令行数据与策略切换](../path_driven_workflow.md)。

人工清零示教模型使用 [manual-tared 现场起点入口](manual_tared.md)。
墙面使用可选的[垂直擦拭模式](vertical_wiping.md)：追加 `--wiping-mode vertical --wall-direction +x`
（另一侧用 `--wall-direction=-x`），映射为 YZ 轨迹与 X 法向反馈，原默认模式不变。
Y/Z 映射改用 `--wall-direction +y` 或 `--wall-direction=-y`，执行 XZ 轨迹与 Y 法向反馈。
重力补偿拖动，按 h 捕获并保持，z 清零，s 后静止采样 2 秒再执行完整 10 秒轨迹，结束保持，g 切重力补偿退出。
XY 只平移到本次起点，不缩放、不裁剪；不自动下压或回位。
新 workflow 不要求旧确认标志/回放，不执行旧项目运动阈值；
仍验证数据、传感器、控制权、通信和时序，SDK/硬件保护不变。尚未完成真机验证。

2026-09-15 输出规则更新：新结果自动增加 `MMDD_HHMMSS` 时间目录（如 `0915_142205`）。
后续关联、状态查询及续跑使用终端打印的实际路径；多阶段训练使用打印的 `run_config.yaml`。
下文未带时间层的历史路径示例不代表新文件的实际路径，完整新命令见 [runs 输出操作步骤](../run_outputs.md)。

2026-09-13：新增显式 `--mode fixed-setup` 的[同海绵固定安装入口](fixed_setup.md)，
适配当前 `airbot_native_tared_offline` 模型的在线空载扣除、坐标与滤波。
支持 `preflight / replay / shadow / run`。操作者后续确认运动跟踪/姿态偏差仅记录，
其余列出的现场条件可满足；固定安装配置已记录该批准，仍需有效回放与显式交互启动。
起点/回位核验和其他保护保留，不自动连接真机。完整命令与状态见上述固定安装文档。
2026-09-14：已移除 TCP 标定和位姿正反转换。下文默认 `calibrated` 入口仅转换传感器 FT，
使用 `airbot_sensor_calibrated_offline` 模型和 `initial_sdk_position_m` 起点；
传感器标定、模型质量和现场保护保留。新版配置、旧模型来源绑定及测试命令见
[传感器转换步骤](../real_training/airbot_calibrated_pipeline.md)。

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
