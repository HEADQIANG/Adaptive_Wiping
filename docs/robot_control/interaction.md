# 统一启动授权与交互操作

2026-09-14：简化启动口令和基础控制进入模式的重复确认。所有既有安全阈值、设备状态、控制权、配置确认、模式匹配、来源哈希和回放门禁保持原逻辑，不因界面简化而跳过。

## 启动授权

带 `--execute` 的命令表示操作者已经完成现场准备并授权执行，不再额外输入 `DRAG`、`EXPLORE`、`AIR-MOTION`、`CONTACT-NO-FT`、`CONTROL`、`PROGRAMMED`、`DEPLOY`、`FIXED-SETUP` 或 `SHADOW`。仍须有人值守的交互终端；配置、起点、输出文件和运行检查仍可拒绝启动。

运行命令前先托稳机械臂，检查急停、工具、完整路径及服务 `--no-return`。启动警告仍会打印，但不暂停等待输入。不要通过管道预填旧口令或预填后续操作。

| 入口 | 现在何时动作 | 保留的交互 |
| --- | --- | --- |
| 初始位姿 `teach --execute` | 设备检查通过后进入重力补偿 | 静止并支撑后按 `S` 保存并切 idle 退出；Ctrl+C 取消 |
| 人工示教 `run --execute` | 设备检查通过后进入重力补偿 | 每条 `s` 开始、`a` 接受，`q` 退出；退出前托稳 |
| 默认探索 `run --execute`，manual-start | 检查后进入重力补偿拖拽 | `h` 固定、`s` 清零探索；回撤后保持，`IDLE` 退出，无起点/静止/载荷阈值验收 |
| 显式旧探索 force-guarded/air/contact-no-ft | 检查及基线流程通过后执行运动 | 正常结束仍输入 `IDLE` 进行支撑交接 |
| 程序示教 `run --mode programmed --execute` | 检查通过即可能自动直达起点，不等第一个 `s` | 到位后每条 `s` 开始、`a` 接受；最后 `IDLE` |
| 默认标定部署 `run --execute` | 完成门禁和现场检查后执行策略 | 正常结束仍输入 `IDLE` |
| 固定安装部署 `run --mode fixed-setup --execute` | 检查通过即可能自动直达起点，不等 `s` | 到位后 `s` 开始策略、`q` 放弃本次策略；回位后 `IDLE` |
| 固定安装 `shadow --execute` | 检查通过后只读连接，不控制机器人 | `s` 开始推理或 `q` 结束；不增加 idle 操作 |
| 基础控制 | 命令及检查确定进入的模式/运动 | 会话 `g/h/k` 直接切拖拽/保持/键盘；`i/q` 后仍需大写 `Y` 确认支撑 |

基础控制的独立 `idle --execute` 是支撑交接，不是普通启动确认：仍须输入 `IDLE` 才连接并释放控制。独立 `stop --execute` 不再等待 `CONTROL`，但仍经过原有配置、终端、连接和控制权检查，不能当作不受这些条件限制的硬件急停。基础控制会话中的空格停止无需二次确认。

空载标定采集的 `r`、示教开始的 `s`、接受数据的 `a`、保存位姿的 `S` 不变。程序示教和固定安装入口仍清理提前键入的命令，必须在相应提示出现后输入新的操作。

## 当前命令示例

在项目根目录，固定 SDK 5.2.2 环境中执行。下面使用原有示教入口：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python \
  airbot_initial_pose.py teach --execute \
  --output runs/real_deploy/robot_control/real_robot/manual_pose_20260914_003.json \
  --label manual_pose
```

文件路径必须尚不存在。命令执行前先托稳；不再输入 DRAG，看到 `Gravity compensation active` 后才拖动。到达目标并停稳、支撑后按 `S` 保存退出。此次未取消 `--output`，因为 `teach` 仍是位姿记录入口。

基础控制配置完成现场验证后，可启动一次长期会话：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python \
  robot_control.py console --execute \
  --config configs/robot_control/basic_control.json
```

不再输入 CONTROL。按 `g` 拖拽、`h` 保持、`k` 键盘控制；每次按键前先准备好支撑。按 `q` 后再按 `Y` 才正常退出。未填写完成的 `basic_control.json` 仍会拒绝运行。其他入口沿用各自文档命令，仅删除启动口令步骤，不删除 `s/a/IDLE`。

## 来源绑定与验证

此次只变更交互和对应说明，未改安全限值或自动批准模型。相关源码改变后，旧训练来源绑定或固定安装 replay 可能被判为过期；应按部署文档重新审核并生成新的验证产物，不得编辑旧报告绕过门禁。人工示教配置仅同步 `setup_note` 的授权说明；已冻结旧配置的会话不覆盖，继续采集应使用新的会话目录。

以下是无硬件测试，不打开机器人、CAN 或串口：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.robot_control.test_basic_control \
  tests.robot_control.test_airbot_initial_pose \
  tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_demonstrations \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.real_deploy.test_airbot_deploy \
  tests.real_deploy.test_fixed_setup
```

测试覆盖无需启动口令即可进入假连接边界、未授权/无交互终端仍拒绝、模式键无需重复确认、退出支撑确认以及已有采样/接受/安全保护流程。软件测试不代表真机现场验收。
