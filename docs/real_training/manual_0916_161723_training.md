# 0916_161723 人工示教训练与部署

2026-09-16 已完成 8 条示教的审计、导入、预处理、正式训练、训练集评估及策略导出。
数据来源为 `runs/real_demonstrations/manual/0916_161723/session_001`。
按记录中的 `normal / normal_exp`、机器人和传感器身份，沿用上一批使用的完整探索
`runs/real_exploration/0915_152957/manual_exploration_001.jsonl`，本次未重新采集探索。
记录中的身份匹配不等于新的物理安装标定；沿用的输入与编码器警告保留在导出模型中。

本次结果目录：`runs/real_training/manual_0916_161723/0916_164643/`。
原始数据、旧模型及默认部署配置未改写。

## 本次结果

- 策略：[policy.pt](../../runs/real_training/manual_0916_161723/0916_164643/training/policy.pt)。
- 配置：[run_config.yaml](../../runs/real_training/manual_0916_161723/0916_164643/training/run_config.yaml)。
- 评估：[evaluation.json](../../runs/real_training/manual_0916_161723/0916_164643/training/final/evaluation.json)。
- 曲线：[predictions.png](../../runs/real_training/manual_0916_161723/0916_164643/training/final/predictions.png)、[training.png](../../runs/real_training/manual_0916_161723/0916_164643/training/final/training.png)。

沿用冻结的 wide1200 编码器（检查点记录 epoch=1000），XY 分支完成 10000 轮、
力反馈分支完成 2000 轮，学习率 0.001，seed=42。8 条示教共 160 个有效反馈窗口，
探索数据六通道归一化越界比例均为 0。导出后的重新加载数值一致性检查通过。

| 训练集重建指标 | RMSE | MAE |
|---|---:|---:|
| XY | 22.4772 mm | 13.7671 mm |
| 法向增量 Δh | 0.2796 mm | 0.2230 mm |
| XY 平均轨迹基线 | 22.4771 mm | 13.7683 mm |
| Δh 恒零基线 | 1.1060 mm | 0.7993 mm |

本次使用全部 8 条示教训练并在同一批记录上评估，未运行额外的留一交叉验证。
同一探索对应相同海绵编码，XY 输出接近这批示教的平均轨迹；训练集误差不是未见场景精度。
本次没有连接机器人；水平模式与 Y/Z 映射 `vertical/-y` 的路径式 preflight 均返回
`offline_inputs_valid=true`、`blockers=[]`、`hardware_connected=false`，训练代码来源无差异。
导出 SHA256：`964c718fb7fe10e5c0cfa0ab66ca69ae8d65b5650516f9b0f0f276a0e8f7260d`。

## 部署命令

从项目根目录、原有 SDK/串口环境的现场终端执行。以下明确选用本次模型：

```bash
DEPLOY_PY=/home/wp/miniconda3/envs/clean/bin/python
POLICY=runs/real_training/manual_0916_161723/0916_164643/training/policy.pt
```

Y/Z 映射，墙位于 SDK −Y 方向：

```bash
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy preflight \
  --policy "$POLICY" --wiping-mode vertical --wall-direction=-y
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy run \
  --policy "$POLICY" --wiping-mode vertical --wall-direction=-y \
  --output runs/real_deploy/manual_0916_161723_yz/events.jsonl --execute
```

墙在 +Y 侧则将两个命令的方向均换为 `--wall-direction +y`。
X/Z 映射使用 `+x` 或 `--wall-direction=-x`。各方向的姿态和映射见
[垂直擦拭步骤](../real_deploy/vertical_wiping.md)。Y/Z 映射的法向反馈取实测 Y，
六轴传感器通道不交换；工具人工摆正后保持固定姿态，在该姿态下重新空载清零。

原水平模式：

```bash
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy preflight --policy "$POLICY"
env -u PYTHONPATH "$DEPLOY_PY" -m scripts.real_deploy run --policy "$POLICY" \
  --output runs/real_deploy/manual_0916_161723_horizontal/events.jsonl --execute
```

沿用 `h` 固定起点和姿态 → `z` 空载清零 → `s` 静止采样 2 秒并执行 10 秒轨迹
→ 结束保持 → 托住机械臂后按 `g` 退出。完整流程见[现场起点部署](../real_deploy/manual_tared.md)。
输出会自动增加时间目录，以终端打印的实际路径为准；没有自动回位或自动接触搜索。

## 训练复现

以下为本次使用的完整训练命令。再次执行会创建新时间目录，不覆盖本次结果：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train \
  --demonstrations runs/real_demonstrations/manual/0916_161723/session_001 \
  --exploration runs/real_exploration/0915_152957/manual_exploration_001.jsonl \
  --confirm-same-setup --output-dir runs/real_training/manual_0916_161723/training
```

换海绵或工具/传感器安装时应使用对应的探索记录，不沿用这条复现命令中的关联。
