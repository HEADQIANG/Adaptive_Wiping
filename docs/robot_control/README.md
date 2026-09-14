# 机械臂基础控制

入口 `python -m scripts.robot_control`。固定 AIRBOT Play / `arm-sdk==5.2.2`。
基础控制依赖 NumPy、SciPy 和 SDK，不依赖 PyTorch 或 MuJoCo。

## 先做离线预演

```bash
python -m scripts.robot_control --help
python -m scripts.robot_control console
python -m scripts.robot_control gravity-comp
python -m scripts.robot_control hold
python -m scripts.robot_control keyboard --keys '1+2-xXrR'
python -m scripts.robot_control move-joint --joints .01 0 0 0 0 0
python -m scripts.robot_control move-pose --position .251 0 .3 --quaternion 0 0 0 1
python -m scripts.robot_control capture-pose --output runs/robot_control/preview_001/pose.json
```

除 `inspect` 外，不加 `--execute` 的命令均使用假机器人，不连接设备。
假机器人不是动力学或碰撞仿真，记录明确标为 synthetic；不能用预演轨迹证明真实路径安全。
每次运行自动创建不重名的 JSONL，亦可用 `--log` 指定新文件。
交互终端内的 `console` 和未指定 `--keys` 的 `keyboard` 使用相同按键进行离线交互；
非交互运行时执行有限演示后退出，不等待终端输入。

## 环境与现场准备

在项目根目录、固定 SDK 环境中安装项目依赖：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install -r requirements/robot_control.txt
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install --no-deps -e .
```

服务由现场人员按 [设备启动说明](airbot_initial_pose.md) 启动，必须保留 `--no-return`。
本脚本不启动/停止服务，不清故障，不接管其他客户端，不改变负载补偿参数。
底座固定、末端负载、急停、整条机器人/工具/线缆扫掠路径及人工支撑必须现场核实。

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control inspect
```

`inspect` 只读，不申请控制权、不使能、不切换模式。
`configs/robot_control/basic_control.json` 是故意未填写的真机模板；
需逐项填入现场批准的关节边界、电流、速度、步长、跟踪误差、
运动时长、稳定等待及到位容差，并保存实际确认依据。不得照抄假机器人参数。
2026-09-12 起不再设置或检查项目 XYZ 工作空间边界，旧字段也不会生效；
整臂、工具和线缆路径由现场确认，详见 [边界移除说明](workspace_bounds_removed.md)。
现阶段基础控制用于无接触准备，不提供力传感器超力保护或自动避碰。

## 真机模式和操作

只有配置有效、交互终端可用且输入 `CONTROL` 后，`--execute` 才连接并申请控制。
起始控制器必须是 idle；已运行的任务不会被接管。

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control console \
  --execute --config configs/robot_control/basic_control.json \
  --output runs/robot_control/attended_001/pose.json
```

| 按键 | 行为 |
|---|---|
| `g` 然后 `Y` | 确认支撑后进入重力补偿拖拽 |
| `h` 然后 `Y` | 以实测关节位置进入保持，不自动回零 |
| `k` 然后 `Y` | 确认支撑后开启键盘点动 |
| `1` 至 `6`、`+` / `-` | 选择关节并正/负点动 |
| `x/X`、`y/Y`、`z/Z` | 沿 SDK 参考系 XYZ 正/负平移 |
| `r/R`、`p/P`、`w/W` | 绕 SDK 参考系 XYZ 正/负旋转 |
| `c` | 稳定采样后保存 `--output` 位姿，不覆盖 |
| 空格 | 请求锁存软件停止，结束控制，不自动清除 |
| `i` 然后 `Y` | 确认支撑后进入 idle |
| `q` 然后 `Y` | 确认支撑后进入 idle 并退出 |

`zero-gravity` 是 `gravity-comp` 的别名，没有无补偿零力矩模式。
点动是有限步长，运动中丢弃其他点动输入但持续响应空格；输入暂停不会追加目标。
目标使用六轴 rad，或末端 XYZ（m）＋归一化四元数 xyzw。
`move-pose --frame tcp --calibration <有效标定记录>` 将基座坐标系 TCP 目标转换到 SDK 末端；
默认 `--frame sdk`，键盘平移旋转始终明确采用 SDK 参考系。
目标插值限制速度，逐步检查反馈；关节运动中末端路径只能实时监测，不保证预先无碰撞。

完成目标运动后仍保留有人值守的保持会话，需 `q`、`Y` 安全退出。
Ctrl+C、EOF、故障、通信超时或模式不符将请求软件停止，不盲目 idle、回撤或重新抢控制权。
若停止未确认，立即使用现场硬件安全流程。idle、失去控制权、断电或断连都可能失去支撑。

## 验证

```bash
python -m unittest discover -s tests/robot_control -t . -v
python -m unittest tests.robot_control.test_hardware_backend -v
```

第二条使用已安装 SDK 的类型和假客户端，验证六轴请求不包含夹爪字段、
夹爪目标保持、非法夹爪反馈拒绝切换及控制权检查，不建立设备连接。
SDK 未安装时该组测试会跳过；运行命令与现场启动步骤不变。
本机 SDK 环境已补齐清单中的 SciPy/PyYAML，验收结果见 [软件验收记录](../refactor_acceptance.md)。

上述代码与假 SDK 测试不是硬件验收。实际拖拽、模式切换及运动须现场逐项验收后才用于训练/部署准备。
