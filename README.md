# Adaptive Wiping

AIRBOT 擦拭项目：仿真采集与编码—解码训练、真机探索、人工示教、离线策略训练和现场部署。
下面是截至 **2026-09-16** 的操作入口，按六个阶段组织。所有命令从项目根目录执行。

## 环境与路径

本机解释器如下；在新终端中重新设置，其他机器替换为对应环境：

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
TRAIN_PY=/home/wp/miniconda3/envs/clean/bin/python
ROBOT_PY=/home/wp/airbot-venv-5.2/bin/python
DEPLOY_PY=/home/wp/miniconda3/envs/clean/bin/python
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`TRAIN_PY` 用于仿真和离线训练；`ROBOT_PY` 用于探索与示教。
`DEPLOY_PY` 必须同时安装 PyTorch、串口依赖和固定的 `arm-sdk==5.2.2`。
本机 `clean` 已具备这些依赖；`airbot-venv-5.2` 未安装 PyTorch，不能直接用于策略部署。

已有环境安装项目和对应依赖：

```bash
"$TRAIN_PY" -m pip install -e scripts/sim_pretrain/robosuite
"$TRAIN_PY" -m pip install -r requirements/sim_pretrain.txt -r requirements/real_training.txt
# PyTorch >= 2.6 按机器选择 CPU/CUDA 版本安装
"$TRAIN_PY" -m pip install --no-deps -e .
"$ROBOT_PY" -m pip install -r requirements/manual_demonstrations.txt
"$ROBOT_PY" -m pip install --no-deps -e .
```

新机器先克隆项目及固定版本的仿真依赖，再应用仓库附带的 AIRBOT 适配：

```bash
git clone --recurse-submodules https://github.com/HEADQIANG/Adaptive_Wiping.git
cd Adaptive_Wiping
python3 -m scripts.shared.robosuite_patch apply
```

已有本地适配时该命令仅校验，不覆盖；依赖恢复与补丁维护见
[robosuite 操作说明](docs/sim_pretrain/robosuite_location.md)。
`runs/`、`archive/` 和清理恢复区不上传 Git，新克隆需要另行采集或复制数据与模型。

| 输出目录 | 用途 |
| --- | --- |
| `runs/force_sensor/` | 单独力/力矩传感数据 |
| `runs/sim_data/` | 仿真轨迹、数据集及采集记录 |
| `runs/sim_training/` | 仿真模型、评估及潜变量诊断 |
| `runs/real_exploration/` | 真机探索日志及曲线 |
| `runs/real_demonstrations/` | 人工/程序示教及分析 |
| `runs/real_training/` | 导入数据、训练配置和策略 |
| `runs/real_deploy/` | 部署日志、曲线及机械臂控制辅助记录 |

新输出自动增加 `MMDD_HHMMSS` 时间目录。后续步骤使用终端 `Output:` 或
`Run config` 打印的实际路径，不自行猜测时间、不自动选择“最新”文件。
仿真训练目录以相对链接读取 `sim_data` 中的数据；复制实验时保留这两个目录的对应层级。
完整规则和整理恢复说明见 [runs 分类](docs/runs_layout.md)、[输出步骤](docs/run_outputs.md)。

## 1. 仿真数据采集

标准流程先做物理检查，再采集。无桌面时可设置 `export MUJOCO_GL=egl`。
先核对 `configs/sim_pretrain/pretrain_paper.yaml` 中的参数范围和采集数量：

```bash
"$TRAIN_PY" -m scripts.sim_pretrain sanity \
  --config configs/sim_pretrain/pretrain_paper.yaml
# 输入上一条命令打印的 run_config.yaml 完整路径
read -r -p 'SIM_CONFIG: ' SIM_CONFIG
"$TRAIN_PY" -m scripts.sim_pretrain collect --config "$SIM_CONFIG"
```

`sanity` 必须通过，才能运行 `collect`；数据集 `complete=true` 后再训练。
物理数据写入 `runs/sim_data/<本轮路径>/`，后续仍使用同一个 `SIM_CONFIG`。
采集动作是 4 秒、100 Hz，保存 400 帧六轴力/力矩及轨迹、接触等诊断信息。
当前力语义包含非接触清零及 Y/Z 方向约定，不能混用旧原始力数据或归一化参数。

仅采一次用于查看曲线：

```bash
"$TRAIN_PY" -m scripts.sim_pretrain explore-once --output runs/sim_data/explore_once
```

现有 1200 条宽范围研究数据在 `runs/sim_data/pretrain_wide_1200_v1/dataset.h5`。
它采用保留有限完整轨迹、记录运动质量的研究协议，与上述严格物理门禁流程不同。
原 2000 条来源目录目前不存在，不可直接重跑子集提取；重建宽范围数据的命令及其自动训练行为见
[宽范围采集](docs/sim_pretrain/wide_training_2000.md)、[1200 条实验](docs/sim_pretrain/wide_training_1200.md)。

## 2. 仿真训练

对第 1 步新采集的数据，常规训练、评估、导出依次运行：

```bash
"$TRAIN_PY" -m scripts.sim_pretrain train --config "$SIM_CONFIG"
"$TRAIN_PY" -m scripts.sim_pretrain evaluate --config "$SIM_CONFIG"
"$TRAIN_PY" -m scripts.sim_pretrain export --config "$SIM_CONFIG"
```

这一入口使用采集配置中的常规 VAE 训练参数。最新的防潜变量塌缩方案使用独立联合重训入口，
默认读取现有 1200 条数据及原训练产物：

```bash
"$TRAIN_PY" -m scripts.sim_pretrain retrain-wide-vae \
  --config configs/sim_pretrain/retrain_wide_1200.yaml
```

该方案采用训练集 PCA 初始化、50 轮确定性预热、100 轮 KL 权重渐增，
比较三个 KL 权重并复验三个种子，每次训练 400 轮。
它自动完成验证选择、测试评估和编码器导出，每次新建输出目录。
换成其他新数据时，先完成其常规训练和导出，再复制重训配置，将 `source_run`
设为那个完整训练目录；不要只指向裸 HDF5。

后续真机重新训练可明确选择已完成的新编码器：

```bash
ENCODER=runs/sim_training/0916_164948/retrain_wide_1200/encoder.pt
# 若刚执行了新一轮联合重训，改成它实际输出的 encoder.pt
```

`encoder.pt` 是下游冻结编码器，`vae_best.pt` 是验证选出的编码—解码模型；
`vae_last.pt` 是最后一轮，不等同于推荐模型。新检查点不能套用旧 VAE 的 evaluate/export。
现有真机策略仍绑定训练时的旧编码器，不能直接替换其内部编码器；换编码器需重新准备数据并训练。
算法、指标和新格式加载方法见 [联合重训说明](docs/sim_pretrain/joint_vae_retraining.md)。

## 3. 真机探索动作

先核对探索/示教配置中的机器人编号、传感器串口、海绵编号和桌面方向。
现场由人员监护，准备急停和机械臂支撑。正常运行的服务无需重复启动；
需要启动时，在独立终端运行并保持开启：

```bash
airbot-arm --address 127.0.0.1:50051 -i can0 -t airbot_play --no-return
```

回到项目终端，使用当前默认的手动起点模式：

```bash
"$TRAIN_PY" -m scripts.real_training explore preview \
  --config configs/real_training/airbot_exploration_manual.json
"$ROBOT_PY" -m scripts.real_training explore check \
  --config configs/real_training/airbot_exploration_manual.json
"$ROBOT_PY" -m scripts.real_training explore run \
  --config configs/real_training/airbot_exploration_manual.json \
  --execute --time-scale 1 --plot \
  --output runs/real_exploration/manual_exploration_001.jsonl
```

操作顺序：

1. 进入重力补偿后拖到起点，确认海绵距桌面约 1 mm、全路径可达且无障碍。
2. 按 `h` 固定本次位置和姿态；按新的 `s` 清零并开始探索，均不用回车。
3. 自动下压 10 mm（2 秒）、前移 50 mm（1 秒）、横向返回（1 秒），再上退 10 mm（2 秒）。
4. 结束保持，安排支撑后输入 `IDLE` 加回车，完成退出交接。

该模式不做项目侧关节角绝对范围、起点匹配、静止或力/力矩阈值停止；
servo 阶段的实测超速、数据有效性、力流时效、设备健康和控制权检查仍保留。
完成命令序列不代表实测回位或接触质量通过。只有完整、原速的探索才能导入训练。
详见 [探索动作与保护边界](docs/real_training/manual_start_exploration.md)。

退出后记录实际探索路径，供第 5 步使用：

```bash
read -r -p 'EXPLORATION_LOG（本轮完整 JSONL）: ' EXPLORATION_LOG
```

## 4. 真机示教

当前主流程为**人工拖拽、软件清零、8 条各 10 秒**的示教。
使用与探索一致的海绵、工具、传感器安装和方向；不要另开程序占用同一传感器串口。

```bash
"$ROBOT_PY" -m scripts.real_training demonstrate preview --mode manual \
  --config configs/real_training/airbot_demonstrations.json
"$ROBOT_PY" -m scripts.real_training demonstrate run --mode manual \
  --config configs/real_training/airbot_demonstrations.json \
  --output runs/real_demonstrations/manual/session_001 --execute
```

在提示后输入按键并回车：`z` 在无接触姿态清零 → `s` 录制 10 秒 →
`a` 接受或 `r` 拒绝。按提示完成 8 条；需再次清零时先脱离接触，再按 `z`。
结束托稳机械臂，`q` 退出并确认 idle；idle 不承重。
实时显示六轴力，退出后自动保存已接受示教的曲线；无桌面可追加 `--no-plot`。

```bash
read -r -p 'DEMO_SESSION（本轮完整会话目录）: ' DEMO_SESSION
"$ROBOT_PY" -m scripts.real_training demonstrate status --mode manual --output "$DEMO_SESSION"
```

应显示 `accepted=8`。续采时重新运行 `run`，将 `--output` 换成这个实际会话目录，
使用相同配置，重新连接后再次清零。`training_ready=false` 不代表采集无效，仍需下一步离线审计。
默认人工示教不因力或运动速度阈值停止，详细交互和记录语义见
[清零人工示教](docs/real_training/manual_tare_live_plot.md)。

程序固定深度示教是另一条流程：需提前绑定完整探索日志，准备起点和现场配置，使用
`--mode programmed` 与 `runs/real_demonstrations/programmed/`；完整命令见
[程序示教](docs/real_training/manual_start_exploration.md#新示教与训练关联)，不要混用人工示教按键。

## 5. 真机训练

使用第 3、4 步的实际路径，以及第 2 步明确选择的 `ENCODER`。
确认探索与人工示教的物理安装一致后，先离线审计，再一键训练：

```bash
"$TRAIN_PY" -m scripts.real_training inspect \
  --demonstrations "$DEMO_SESSION" --exploration "$EXPLORATION_LOG" \
  --encoder "$ENCODER" --confirm-same-setup
"$TRAIN_PY" -m scripts.real_training train \
  --demonstrations "$DEMO_SESSION" --exploration "$EXPLORATION_LOG" \
  --encoder "$ENCODER" --confirm-same-setup \
  --output-dir runs/real_training/from_recordings/training
```

`inspect` 不连接硬件、不启动训练；`train` 自动审计、导入、prepare、训练、评估并导出。
默认 XY 分支 10000 轮，力反馈分支 2000 轮。记录打印的 `run_config.yaml` 和 `policy.pt` 路径。
不传 `--encoder` 会使用对应默认配置里的编码器，不会自动选第 2 步的新模型。

需要续训时只用该轮快照，不再传入新的数据或编码器：

```bash
read -r -p 'TRAIN_CONFIG（本轮 run_config.yaml）: ' TRAIN_CONFIG
"$TRAIN_PY" -m scripts.real_training train --config "$TRAIN_CONFIG" --resume
"$TRAIN_PY" -m scripts.real_training evaluate --config "$TRAIN_CONFIG"
"$TRAIN_PY" -m scripts.real_training export --config "$TRAIN_CONFIG"
```

`--resume` 仅适用于有检查点的中断训练；已经完成的一键训练无需再执行上述步骤。
需要额外留一验证时执行 `cross-validate --config "$TRAIN_CONFIG"`。
程序示教必须显式使用 `--programmed-hold-last`，不能按人工流程隐式补齐。
详细路径覆盖和分阶段命令见 [训练操作说明](docs/path_driven_workflow.md)。

## 6. 真机部署

明确选择第 5 步导出的策略；只选路径，不手工修改其来源哈希：

```bash
read -r -p 'POLICY（实际 policy.pt）: ' POLICY
# 本机已有的 09-16 人工示教策略，可按需使用：
# POLICY=runs/real_training/manual_0916_161723/0916_164643/training/policy.pt
"$DEPLOY_PY" -m scripts.real_deploy preflight --policy "$POLICY"
```

预检通过后，保持第 3 步的服务开启、机械臂处于 idle，在现场交互终端执行水平擦拭：

```bash
"$DEPLOY_PY" -m scripts.real_deploy run --policy "$POLICY" \
  --output runs/real_deploy/selected_policy/events.jsonl --execute
```

垂直墙面采用相同的策略选择。以下为墙在 SDK −Y 方向的 Y/Z 映射，预检与运行方向必须一致：

```bash
"$DEPLOY_PY" -m scripts.real_deploy preflight --policy "$POLICY" \
  --wiping-mode vertical --wall-direction=-y
"$DEPLOY_PY" -m scripts.real_deploy run --policy "$POLICY" \
  --wiping-mode vertical --wall-direction=-y \
  --output runs/real_deploy/selected_policy_vertical_yz/events.jsonl --execute
```

墙在 +Y 侧用 `--wall-direction +y`；X/Z 映射用 `+x` 或 `--wall-direction=-x`。
垂直模式只适用于清零人工示教策略，需按 [墙面姿态说明](docs/real_deploy/vertical_wiping.md)
人工摆正工具，不自动识别墙面或搜索接触。

当前人工策略的现场按键均不用回车：拖到非接触起点 → `h` 固定 → `z` 空载清零 →
`s` 保持 2 秒采集力历史，再执行完整 10 秒轨迹 → 结束保持 → 托住机械臂后 `g` 退出。
无自动抬升或回位。该部署流程不执行旧项目侧力、速度、深度或跟踪阈值拦截，
仍检查模型/数据身份、有限值、传感器时效、控制权、设备健康和周期超时；
现场监护和 SDK/硬件保护不可省略。详见 [现场部署流程](docs/real_deploy/manual_tared.md)。

输出包含 `events.jsonl`、`events.sensor.csv` 和自动生成的 `events.plots/`。
跨机器部署时还需带上策略绑定的 prepared、raw、encoder、示教和探索文件；
只复制 `policy.pt` 不能通过完整来源校验。固定安装/标定策略沿用各自门禁，
本节的 h/z/s/g 流程和垂直模式不自动转授给它们。

## 其他入口与离线验证

- [单独传感器读取、绘图与标定](docs/force_sensor/README.md)
- [机械臂基础控制与服务检查](docs/robot_control/README.md)
- [旧带力保护探索](docs/real_training/airbot_exploration.md)
- [固定安装部署](docs/real_deploy/fixed_setup.md)
- [项目结构、模型资源与历史归档](docs/architecture.md)

```bash
"$TRAIN_PY" -m scripts.shared.assets verify
"$TRAIN_PY" -m unittest tests.shared.test_runs_layout tests.shared.test_run_paths
"$TRAIN_PY" -m unittest discover -s tests/real_training -t .
"$TRAIN_PY" -m unittest discover -s tests/real_deploy -t .
# 含仿真物理回归的完整检查
MUJOCO_GL=egl "$TRAIN_PY" -m unittest discover -s tests -t .
```

这些测试不执行真机运动。部分测试依赖本地历史产物；新克隆需要对应数据才能复核历史结果。
本次 753 项回归的结果及 2 项已知仿真问题见 [发布验证记录](docs/publish_20260916.md)。
