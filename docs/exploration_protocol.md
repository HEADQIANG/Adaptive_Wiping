# 探索按压协议与运行步骤

当前仿真两份默认配置已增加每段起止各 0.2 秒平滑加减速，保持下文的阶段时长和位移，
但峰值速度及逐点目标与真机分段匀速协议不同。真机代码未改。
仿真新命令、过渡参数和数据兼容说明见 [平滑探索](sim_pretrain/smooth_exploration.md)。

2026-09-11：仿真、真机 force-guarded / contact-no-ft / air 的默认探索
按压速度由 `0.01 m/s` 改为 `0.005 m/s`，持续 `2 s`，目标下移量为 `10 mm`。
横移不变：`+Y 0.05 m/s × 1 s`，随后 `-Y 0.05 m/s × 1 s`。
总探索时间仍为 `4 s`，`100 Hz` 下仍为 `400` 帧。
真机探索完成后单独用 `2 s` 上退 `10 mm` 回到起点，回撤不计入探索数据。
`--time-scale 1` 对应以上速度和时间；大于 1 仍用于整体慢速调试，不缩短距离。

## 离线验证

在项目根目录运行，无需连接机器人：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_native_training \
  tests.force_sensor.test_kwr75_reader \
  tests.sim_pretrain.test_pretraining
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training explore preview
```

预览关键点应为 `2 s: [0,0,-0.01]`、`3 s: [0,0.05,-0.01]`、
`4 s: [0,0,-0.01]`，按压字段为 `press_speed_m_s=0.005`、`press_duration_s=2.0`。

## 仿真与真机运行

原命令入口不变，仿真两份 `configs/sim_pretrain/pretrain_paper*.yaml` 已同步。
先在配置中设置新的 `output_dir`，再按 [仿真操作步骤](sim_pretrain/README.md)
执行 sanity、采集与训练。自定义配置及历史快照的显式 `press_speed` 不会自动迁移。

真机按 [探索操作步骤](real_training/airbot_exploration.md) 完成现场检查后运行，
保持原力限值、确认口令和异常停止流程，不因减速跳过安全检查。
2026-09-12 已按用户要求移除项目 XYZ 工作空间边界，旧字段不再生效，
详见 [边界移除与运行步骤](robot_control/workspace_bounds_removed.md)。
名义压缩由 `19 mm` 降为 `9 mm`（起始间隙 `1 mm`）；无力接触模式仍须加上
跟踪误差、几何误差和压缩余量。现场已确认的允许压缩量不自动修改。
空中模式的起点至少 `50 mm` 净空和运行至少 `20 mm` 净空要求不变。

历史日志、图表及模型不重写。新真机日志记录按压速度和时长；导入训练时拒绝
缺少这些字段或按压参数不同的日志，避免把旧 `0.01 m/s` 数据标为新协议。
旧数据和旧模型不能视为已验证适用于新动作，须按新协议重新采集并验证。

## 探索前软件清零

force-guarded 模式现在默认在初始静止检查通过后、获取控制权和进入 servo 之前，
采集约 1 秒静止六维力/力矩。按传感器接收时间戳去重求均值，至少需要 20 个
新样本，首末样本覆盖至少 0.9 秒。原静止检查和这段清零时间均不计入 4 秒探索。
air / contact-no-ft 模式不读取传感器，也不执行清零。

原运行命令不变，无需增加 `--tare`：

```bash
# 离线预览，不连接机器人或串口
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training explore preview

# 仅在有人监护、完成现场安全检查的终端中运行，使用未存在的输出文件名
env PYTHONPATH='/media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping' \
  /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --config configs/real_training/airbot_exploration.json \
  --execute --time-scale 1 \
  --output runs/real_training/real_robot/exploration_tared_001.jsonl
```

输入 `EXPLORE` 前确认海绵未接触桌面、起点间隙为已实测的 1 mm。显示
`Software tare` 后继续保持静止，不触碰工具。清零期间持续检查起点、原始载荷
与传感器新鲜度，并验证机器人静止；失败或中断不会启动探索。此时仍是 idle，
软件不保证保持位置，现场支撑不得向力传感器下游的工具施加载荷。
软件不能独立证实非接触，不能靠“清零后为零”证明起点没有预压缩。

日志新增 `tare_start` 和 `tare_complete` 事件。后者保存 `tare_bias_si`、
清零起止时间、去重样本数及用于求均值的原始力样本。每次运行重新测量偏置，
不覆盖配置中的 `sensor_bias_si`，不发送硬件清零命令。

| 每次力读数的字段 | 含义 |
| --- | --- |
| `raw_sensor_wrench_si` | 传感器原始六维读数，不清零 |
| `bias_corrected_sensor_wrench_si` | 原始值减去配置的固定电子偏置，继续用于安全限值检查 |
| `tared_sensor_wrench_si` | 原始值减去本次起点均值，用于相对载荷分析 |

清零后字段覆盖初始读数、探索和回撤的发送前/后读数。单位依次为
`N, N, N, N*m, N*m, N*m`，仍在传感器局部坐标系与传感器原点。
这是固定姿态软件清零，不是跨姿态重力补偿。超力/超力矩判断不使用清零后值，
不会因清零隐藏原始载荷。现有训练导入和离线绘图仍使用原始/固定偏置字段，
不自动切换到新字段，不改变旧模型的数据含义。新增 `run --plot` 可在独立窗口
实时显示清零后字段，步骤见 [探索实时曲线](real_training/real_exploration_live_plot.md)。
