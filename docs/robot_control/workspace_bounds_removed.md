# 移除项目 XYZ 工作空间边界

2026-09-12 按用户要求，取消本项目自设的笛卡尔 XYZ 工作空间限制，完整运动路径
由现场人员检查确认。该修改不表示程序已验证路径安全，也不修改 SDK 或固件保护。

## 变更范围

真机探索（force-guarded / air / contact-no-ft）、基础控制、人工示教和策略部署
均不再读取或要求 `workspace_min_m`、`workspace_max_m`，当前配置已删除这两个字段。
旧配置或历史日志中的同名字段会被忽略，不再用于起点、目标或实际反馈的越界检查。
历史日志、旧配置快照和训练数据不重写；示教的 `workspace_force_policy` 保留旧名称
兼容现有配置，但现在只选择力/力矩的 stop 或 record-only 行为，不恢复 XYZ 边界。

非有限位姿、关节角度范围、速度、电流、力/力矩、起点匹配、静止检查、控制权、
设备状态和控制节拍等原有检查不因本次修改而关闭，各入口仍遵循原配置策略。
例如当前探索运行跟踪策略为 record-only，但起点匹配仍严格检查；人工示教若原已
配置力或速度 record-only，也不会被本次修改改为 stop。
air 模式的净空检查、contact-no-ft 模式的压缩预算和运行压缩上界仍保留。

取消后不会再出现基于这两个配置字段的 `SDK end position outside approved workspace`
错误，但不保证任何给定目标都能执行：起点不匹配、关节限位、SDK 逆解失败或其余
保护仍可能拒绝运行。软件没有增加自动避碰、自动回到起点或自动恢复动作。

## 操作步骤

在项目根目录激活真机环境，沿用已经记录的新起点：

```bash
source /home/wp/airbot-venv-5.2/bin/activate
python -m scripts.real_training explore check \
  --config configs/real_training/airbot_exploration.json
```

此命令连接 SDK 做只读起点检查，不执行运动，也不检查实时力。当前配置仍指向
`runs/real_training/real_robot/exploration_start_001.json`，不再因旧 X 上限而拒绝它。
实际机械臂仍需位于记录起点附近，且现场确认海绵离桌 1 mm、全程无不允许的碰撞。

现场检查与起点核对通过后，在有人监护的桌面终端执行，选择尚不存在的日志文件名：

```bash
MPLBACKEND=TkAgg python -m scripts.real_training explore run \
  --config configs/real_training/airbot_exploration.json \
  --execute --plot --time-scale 1 \
  --output runs/real_training/real_robot/exploration_no_xyz_bounds_001.jsonl
```

仍需 `EXPLORE` 确认，检查和约 1 秒软件清零后才开始 4 秒探索，另用 2 秒回撤。
回撤后按提示安全支撑并输入 `IDLE`，关闭曲线窗口不停止机械臂。原始数据和清零后
曲线仍按原规则保存/显示。出现其他错误不要用移除额外保护的方法强行通过。

## 无硬件验证

```bash
/home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_demonstrations \
  tests.robot_control.test_basic_control \
  tests.robot_control.test_hardware_backend
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_deploy.test_airbot_deploy \
  tests.robot_control.test_airbot_calibration.CalibrationTests.test_calibrated_deployment_setup_positive_and_rate_mismatch
```

这些测试只验证软件行为，不连接机器人、串口或 CAN，不代替现场验收。
