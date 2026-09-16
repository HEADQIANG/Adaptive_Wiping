# 手动起点探索与新示教训练

当前默认 `python -m scripts.real_training explore` 使用 `manual-start`。仅有人监护使用；本流程不做
起点匹配、静止验收或力/力矩阈值停止。力数据仍必须有限、及时。没有 XYZ 边界或
实测压缩量停止；软件停止不是安全认证急停，也不保证承重。

2026-09-15：按要求取消本模式的项目关节角绝对范围检查，配置不再需要
`joint_min_rad/joint_max_rad`，旧字段即使存在也不生效。仍检查关节/位姿数据有限、维度正确和
实测关节速度，SDK/固件自身限制未修改，不能据此认定任意关节角安全。
新日志、preview 和 check 标记 `joint_position_limits_enforced=false`。
旧探索、程序示教和部署的关节范围保护不变；历史故障日志不重写。

2026-09-15 拖拽速度更新：`manual-start` 的重力补偿拖拽阶段，包括按 `h` 时最后一次
参考位姿读取，不再执行项目侧实测关节速度阈值检查；该阶段的速度防护仅依赖 SDK/固件
自带机制，不改其参数、不解除急停。切入 servo 后，保持、清零、探索、回撤及最终保持仍执行
`1.2 rad/s` 实测超速停止和原有命令限速。启动前 idle 状态检查也不变。
拖拽期间仍检查数据有效性、设备健康、控制权、预期模式和 20 ms 力流时效。
preview、check 和新日志标记 `drag_measured_joint_speed_stop_enforced=false`、
`servo_measured_joint_speed_stop_enforced=true`；配置中的速度参数不能删除，其他流程不受影响。
本次命令及按键顺序不变，现场需重新核验拖拽切保持；单元测试不代表真机验收通过。

## 探索操作

在项目根目录执行。先核对 `configs/real_training/airbot_exploration_manual.json`
中的设备、传感器、桌面方向、安装和压缩余量；服务使用 `--no-return`，准备实体急停和支撑。

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training explore preview
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check
# 以下连接并控制真机，必须在现场交互终端执行；自动新建时间目录
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run --execute \
  --config configs/real_training/airbot_exploration_manual.json --time-scale 1 --plot \
  --output runs/real_exploration/manual_exploration_001.jsonl
```

日志和曲线自动保存在 `runs/real_exploration/MMDD_HHMMSS/`，例如 `runs/real_exploration/0915_142205/`。
同一命令可重复使用；以终端 `Output:` 的实际路径为准。历史同名文件不覆盖。
所有新输出及续跑规则见 [runs 输出目录](../run_outputs.md)。

1. `--execute` 后进入重力补偿拖拽。人工确认全路径净空、固定安装和 1 mm 非接触间隙。
2. 准备支撑后按 `h`（不用回车），读取本轮参考位姿并切 servo 保持；不保存独立起点文件。
3. 看到下一提示后，松开工具上的外载荷，按新的 `s`（不用回车）。保持状态下先采约 1 秒基线，
   至少 40 个不同接收时刻、覆盖至少 0.9 秒；不做静止验收，样本不足则中止。
4. 自动执行 4 秒探索：2 秒下压 10 mm，1 秒前移 50 mm，1 秒横向返回。另用 2 秒上退 10 mm。
5. 返回目标发完后保持 servo，偏差仅记录，不保证实测回位。安排安全支撑后输入 `IDLE` 加回车退出。

不接受提前输入的启动键、不自动重复探索。`Ctrl+C`、终端关闭和故障请求软件停止，不自动
回撤或解除急停。关闭曲线只关闭显示。保留关节/位姿数值有效性、0.4 rad/s 命令限速、1.2 rad/s 实测
超速（重力补偿拖拽阶段除外）、20 ms 力流新鲜度，以及运动发送截止和 5 ms 采样迟到保护。
故障日志保存阶段、采样索引和可取得的已读观测；不为补齐故障数据而延迟停止或补读传感器。

清零为软件扣除当次原始均值，不改硬件零点和配置偏置。原始值、配置偏置修正值、本轮均值、
清零后值全部保存；曲线显示清零后值。`motion_complete` 是命令序列完成，`session_complete`
是 idle 交接完成，均不是接触运动或实测回位质量通过。日志仍为传感器局部坐标和传感器原点，
`encoder_ready=false`，不宣称完成跨姿态重力或坐标标定。

旧模式须显式选择，例如 `--mode force-guarded --config configs/real_training/airbot_exploration.json`。
它仍要求起点文件、静止和载荷保护；`air`、`contact-no-ft` 也保留原保护。

## 新示教与训练关联

本节为程序示教的采集前绑定流程。人工拖拽的 8 条实测 10 秒清零示教请改用
[清零人工示教导入与训练](manual_tared_training.md)，独立记录事后条件确认，不补写采集前绑定。

仅配对新采集的 8 条程序示教。完成探索并退出 idle 后，在
`configs/real_training/airbot_programmed_manual_demonstrations.json` 中设置实际
`exploration_log`，核对 `sponge_id/exploration_id`，现场确认同一海绵、工具、传感器安装
和桌面方向后将 `exploration_setup_confirmed` 设为 true。程序示教自己的起点记录和
带力保护配置仍由 `exploration_config` 指向，不能改成 manual 配置；核对自身起点，
并按原程序示教说明确认 `setup_confirmed` 和 `startup_approach.direct_path_confirmed`。
这些 false 默认值不能在未核实现场时直接全部改为 true。

示教会话创建时冻结完整探索日志哈希；每条原始 attempt 和接受记录都包含同一哈希。
导入同时核对机器人编号、末端类型、传感器路径、桌面方向、海绵和探索编号。
旧会话不能补写关联，变更探索必须用新的会话目录。示教自身的起点匹配、静止、载荷和
最终实测回位门禁不变。具体接受/拒绝按键仍见程序示教文档。

```bash
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate preview --mode programmed \
  --config configs/real_training/airbot_programmed_manual_demonstrations.json
/home/wp/airbot-venv-5.2/bin/python -m scripts.real_training demonstrate run --mode programmed --execute \
  --config configs/real_training/airbot_programmed_manual_demonstrations.json \
  --output runs/real_demonstrations/programmed/manual_start_session_001
```

八条新示教完成后才执行以下离线命令。使用独立训练配置和新输出目录，不覆盖旧实验：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training import-airbot \
  --config configs/real_training/real_training_manual_start.yaml \
  --exploration runs/real_exploration/0915_142205/manual_exploration_001.jsonl \
  --session runs/real_demonstrations/programmed/0915_143000/manual_start_session_001 \
  --programmed-hold-last --subtract-recorded-baseline
# 上面时间仅为示例，必须替换为实际采集路径；下面必须使用导入打印的 run_config.yaml
TRAIN_CONFIG=runs/real_training/0915_150000/run_config.yaml
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect \
  --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare \
  --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train \
  --config "$TRAIN_CONFIG"
/home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training evaluate \
  --config "$TRAIN_CONFIG"
```

训练输入仍是原始值减各自当轮基线，并保留扣除前副本验证一致性；不会对已清零值再次扣零。
仅原速 400 帧完整探索可导入；慢速调试、中止、缺失基线、过期或冲突的力流不合格。
新配置仍使用现有 wide1200 编码器，不消除它已有的质量警告，也不自动获得部署资格。
本次代码实现没有采集新真机数据、生成真实派生数据或运行训练。

## 回归命令

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.real_training.test_manual_exploration \
  tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.real_training.test_programmed_padding
```

测试不连接机器人。首次真机模式切换、保持、运动与退出必须另做现场验收。
SDK 环境没有 h5py 时，采集侧可以单独验证，不需要为运行新探索安装训练依赖：

```bash
/home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.real_training.test_manual_exploration.ManualTests \
  tests.real_training.test_airbot_exploration \
  tests.real_training.test_airbot_programmed_demonstrations \
  tests.robot_control.test_hardware_backend \
  tests.real_training.test_exploration_live_plot
```
