# Adaptive Wiping

AIRBOT 擦拭项目：仿真预训练、力传感器测试、真机示教离线训练、策略部署和机械臂基础控制。

| 模块 | 操作步骤 | 命令入口 |
|---|---|---|
| 仿真预训练 | [环境、采集、训练与评估](docs/sim_pretrain/README.md) | `python -m scripts.sim_pretrain --help` |
| 力传感器测试 | [读取、绘图与标定](docs/force_sensor/README.md) | `python -m scripts.force_sensor --help` |
| 真机训练 | [示教、数据整理与离线训练](docs/real_training/README.md) | `python -m scripts.real_training --help` |
| 真机部署 | [回放、预检与部署](docs/real_deploy/README.md) | `python -m scripts.real_deploy --help` |
| 机械臂基础控制 | [拖拽、保持、键盘和目标位姿](docs/robot_control/README.md) | `python -m scripts.robot_control --help` |

## 目录约定

- `scripts/`：上述五个模块及 `shared/` 共用代码；实验脚本在各自 `experiments/` 或 `tools/` 内。
- `configs/`、`docs/`、`tests/`、`requirements/`：按模块分类的配置、文档、测试和依赖。
- `asserts/`：[仿真模型与资源](docs/sim_pretrain/models.md)，集中保存实际加载的机器人、工具、场景、网格、纹理与惯量参考。
- `archive/`：[历史实验索引](archive/README.md)，保留原数据、模型、日志及报告，不原地续写。
- `runs/<模块>/<新运行名>/`：新数据、训练、评估及控制会话输出，不与历史实验混用。
- `logs`：仅供固定 SDK 使用的链接，实际指向 `runs/robot_control/sdk_logs/`。
- `scripts/sim_pretrain/robosuite/`：本地仿真代码分支，完整保留已有适配，不以其他 pip 分支替换。
- DISCOVERSE 已脱离运行依赖并从根目录移除；完整来源及未提交改动保存在归档，见 [移除说明](docs/sim_pretrain/discoverse_removal.md)。

## 环境

在项目根目录运行。现有训练环境：

```bash
conda activate clean
python -m pip install --no-deps -e .
```

无法激活 conda 时使用 `/home/wp/miniconda3/envs/clean/bin/python`。
真机使用独立 `/home/wp/airbot-venv-5.2/bin/python` 和固定 `arm-sdk==5.2.2`，详见控制文档。
安装项目不会启动设备；各模块依赖单独安装，不要求合并训练和 SDK 环境。

## 离线验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -m unittest discover -s tests -t . -v
python -m scripts.real_training smoke-test
python -m scripts.robot_control console
python -m scripts.shared.artifacts verify
```

基础控制默认是假机器人；仅 `inspect` 是只读真机连接。
其余基础控制命令必须显式 `--execute` 才允许连接真机，并要求现场配置与人工确认。
部署配置中的未完成标定和安全确认仍阻止真机执行；软件测试不代表真机验收。

旧命令不再保留。迁移详情见 [结构与归档说明](docs/architecture.md)。
Python 包现名为 `scripts`，已有环境需重新安装，见 [包改名说明](docs/package_rename.md)。

# 真机探索动作
下面按两个终端操作。**涉及真机控制，必须有人监护，提前确认急停和安全支撑。程序不会自动寻找或返回起点。**

### 1. 上电前检查

按厂家规程检查底座、工具、线缆及工作空间，确认传感器连接正常，再上电。切换模式、退出或断电时，机械臂都可能失去支撑。

### 2. 终端 A：启动机械臂服务

如果已有正常服务，确认它带 `--no-return`，不要重复启动。否则运行：

```bash
mkdir -p /home/wp/airbot-logs-5.2
env MALLOC_ARENA_MAX=2 AIRBOT_LOG_DIR=/home/wp/airbot-logs-5.2 \
  airbot-arm --address 127.0.0.1:50051 -i can0 -t airbot_play --no-return
```

保持终端 A 开启。服务报错时停止后续操作，不要用运动命令测试连接。

### 3. 终端 B：激活环境并检查状态

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
source /home/wp/airbot-venv-5.2/bin/activate
python -m scripts.robot_control.airbot_initial_pose inspect
```

确认设备身份正确、没有故障，且 `controller_state` 为 `idle`，再继续。

### 4. 拖拽到起点并记录

下面的文件名必须尚不存在：

```bash
python -m scripts.robot_control.airbot_initial_pose teach \
  --execute \
  --output runs/real_training/real_robot/exploration_start_008.json \
  --label exploration_start \
  --tool-note "Installed KWR75A and sponge"
```

输入 **`DRAG` 并回车**，看到 `Gravity compensation active` 后才缓慢拖动。

将海绵面调整为与桌面平行，现场测量未压缩海绵底面距桌面 **1 mm**；检查下压 10 mm、沿 +Y 横移 50 mm 再返回的完整路径。

**托稳机械臂并保持静止，按 `S`，不用回车。** 约 0.5 秒静止检查通过后，自动保存并退出。确认看到 `Saved initial pose` 和 `Idle confirmed`；退出后不保证保持位置。

### 5. 更新探索配置

打开 `configs/real_training/airbot_exploration.json`，把起点路径改为：

```json
"initial_pose_file": "runs/real_training/real_robot/exploration_start_008.json"
```

同时核对新起点对应的工作空间、运动方向、工具及力限值，不能仅换文件名或随意扩大边界。**当前配置的运行跟踪误差策略是仅记录、不因跟踪误差自动停止，需现场确认。**

### 6. 检查起点是否匹配

确保机械臂实际仍在记录起点附近：

```bash
python -m scripts.real_training explore check \
  --config configs/real_training/airbot_exploration.json
```

看到 `start_matches: true` 才继续。该检查不验证实时力，也不能代替现场间隙和路径检查；不匹配时不会自动回到起点。

### 7. 执行探索并显示清零后曲线

使用新的日志文件名：

```bash
MPLBACKEND=TkAgg python -m scripts.real_training explore run \
  --config configs/real_training/airbot_exploration.json \
  --execute --plot --time-scale 1 \
  --output runs/real_training/real_robot/exploration_tared_plot_001.jsonl
```

确认现场安全后输入 **`EXPLORE` 并回车**。程序检查起点、静止采样清零约 1 秒，再执行下压 2 秒、横移往返 2 秒，以及单独回撤 2 秒。

回撤结束后，按提示安排安全支撑，再输入 **`IDLE` 并回车**。确认 `session_complete` 后才算正常结束。关闭曲线窗口不会停止机械臂；异常时按现场急停规程处理，不盲目重试。