# 命令行切换真机数据与策略

在项目根目录执行，使用已安装 `requirements/real_training.txt` 的 Python 环境。
本机可先设置 `WIPING_PY=/home/wp/miniconda3/envs/clean/bin/python`；其他机器换为自己的解释器。
训练、审计、预处理、评估、导出和部署 preflight 均不连接机器人。

## 一条命令训练新的清零人工示教

```bash
WIPING_PY=/home/wp/miniconda3/envs/clean/bin/python
"$WIPING_PY" -m scripts.real_training train \
  --demonstrations /path/to/session_001 \
  --exploration /path/to/manual_exploration.jsonl \
  --confirm-same-setup
```

每次只替换两个数据路径，无须复制或修改 YAML/JSON。`--demonstrations` 也可写为
`--session`，接受会话目录或其中的 `session.json`。路径可以是绝对路径或相对于项目根目录。
必须是一条已完成的探索和一组已接受的 8 条示教，不接受任意 CSV 或父目录中模糊匹配的会话。
AIRBOT 真机数据保留采集记录的海绵编号，不再写死 `normal`；一次训练仍必须使用同一海绵，
探索与任一示教编号不同会拒绝。论文/synthetic profile 的 Normal 约束不变。
`--confirm-same-setup` 是操作者对探索与清零人工示教安装条件一致的明确确认，不是自动推断的标定。

命令读取采集元数据选择已有训练默认配置，然后审计、导入、prepare、train、evaluate、export。
默认仍是 XY 10000 轮、FT 2000 轮；不自动运行额外的 8 折交叉验证。
新输出位于 `runs/real_training/from_recordings/MMDD_HHMMSS/`，包含 `training_inputs/` 中的
`raw.h5`、导入报告以及 `training/` 中的 `run_config.yaml`、预处理、检查点、评估、`policy.pt`。
指定其他输出目录时也会创建专用的同级 `<目录名>_inputs/`，避免不同训练覆盖导入报告。
终端打印实际快照及策略路径；不会覆盖旧结果或修改原始数据。

可追加 `--encoder /path/to/encoder.pt` 切换冻结编码器，
`--output-dir runs/real_training/my_dataset/training` 指定新输出，
`--config /path/to/defaults.yaml` 复用自定义训练默认值。路径参数优先于配置；
本流程始终自动生成独立 raw/output 路径，不沿用默认配置中的旧数据路径。

把 `train` 换成 `inspect` 只审计，不创建正式输出；换成 `prepare` 只完成导入和预处理。
旧 raw 人工示教自动使用 native 原始载荷配置。程序示教必须显式加 `--programmed-hold-last`
授权派生补齐，默认使用现有 wide1200 清零配置；不会静默填充人工示教。
标定原始人工数据需显式传 `--config configs/real_training/real_training_airbot_calibrated.yaml`
和 `--calibration /path/to/calibration.json`，保持原有标定检查。

## 分阶段与续训

后续阶段只使用本次终端打印的快照，不再次传示教/探索路径。

```bash
TRAIN_CONFIG=/path/to/training/run_config.yaml
"$WIPING_PY" -m scripts.real_training train --config "$TRAIN_CONFIG" --resume
"$WIPING_PY" -m scripts.real_training evaluate --config "$TRAIN_CONFIG"
"$WIPING_PY" -m scripts.real_training export --config "$TRAIN_CONFIG"
"$WIPING_PY" -m scripts.real_training cross-validate --config "$TRAIN_CONFIG"
```

`--resume` 仅用于已存在检查点，禁止同时换数据、编码器或输出目录；原有代码/数据/优化器绑定保持。
仅 prepare 后首次训练不加 `--resume`。分阶段 train 保持旧行为，不自动评估或导出。
已有导入 HDF5 可使用 `prepare --config ... --raw-data /path/to/raw.h5 --encoder ... --output-dir ...`；
后续阶段使用 prepare 打印的快照。正式轮数和数据审计要求不因路径参数改变。
显式传入 HDF5/编码器/输出路径时，prepare 还会在实际训练目录保存 `run_config.yaml`，
包括项目外的自定义输出路径，后续不需要重复输入这些覆盖参数。
通过 `--raw-data` 换 HDF5 时会移除旧配置中的采集路径提示，以新 HDF5 的来源元数据为准。

## 按策略路径部署

```bash
POLICY=/path/to/training/policy.pt
"$WIPING_PY" -m scripts.real_deploy preflight --policy "$POLICY"
"$WIPING_PY" -m scripts.real_deploy run --policy "$POLICY" \
  --output runs/real_deploy/selected_policy/events.jsonl --execute
```

`--policy` 也接受包含 `policy.pt` 的训练目录；自动识别 manual-tared、fixed-setup 或 calibrated。
清零人工模型可在 preflight/run 追加 `--wiping-mode vertical --wall-direction +x`
（另一侧用 `--wall-direction=-x`）启用[墙面擦拭](real_deploy/vertical_wiping.md)，默认仍为水平模式。
Y/Z 映射使用 `--wall-direction +y` 或 `--wall-direction=-y`，保留 `--wiping-mode vertical`。
显式指定不匹配的 `--mode` 会拒绝，不会误用控制模式。未传 `--policy` 时旧配置入口不变。
默认清零人工模型保留现场 h/z/s/g 流程，详细设备准备、交互和保护边界见
[manual-tared 操作步骤](real_deploy/manual_tared.md)。run 必须在具备 SDK 5.2.2 和串口权限的
本地有人值守终端执行。此次软件验证不执行任何真机运动。

新策略内嵌训练配置；旧策略从同目录的 `final/run.json` 读取。关联的 prepared、raw、encoder、
示教会话和探索文件仍须存在且哈希匹配，不是只复制一个旧权重即可跨机器独立部署。
新策略单独移动时可以使用内嵌的原始路径，训练目录整套移动时优先校验同目录 prepared。
不会自动搜索“最新”数据或在缺失时回退到默认配置里的旧策略。

manual-tared/fixed-setup 部署时，训练时代码哈希保留为历史来源，当前代码作为本次运行绑定，差异列入 preflight。
训练脚本发生兼容修改不再迫使旧权重重新训练；数据哈希、网络结构、预处理、重编码和有限值检查仍执行，
连接前仍检查本次绑定未变。未带 `--policy` 的旧入口同样采用此规则，仍严格核对配置内的策略和数据哈希。

安装配置仍可用 `--config` 指定，但不再需要为清零人工策略手改其中的 policy、training_config、
collection_session、exploration 及其哈希。fixed-setup 的回放/模型例外审批与 calibrated 的
策略审批哈希、现场标定不自动转授给新策略；这些模式的审批失败必须按原文档验收，不能绕过。

## 无硬件验证

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$WIPING_PY" -m unittest tests.real_training.test_path_workflow \
  tests.real_deploy.test_policy_selection tests.real_training.test_real_training -v
```

本次完整相关回归为训练模块、部署模块、共享输出路径和模型数值一致性测试：

```bash
"$WIPING_PY" -m unittest discover -s tests/real_training -t .
"$WIPING_PY" -m unittest discover -s tests/real_deploy -t .
"$WIPING_PY" -m unittest tests.shared.test_run_paths tests.shared.test_model_parity
```

2026-09-15 验证结果：403 项测试通过验收，其中 2 项环境相关测试跳过；语法编译和
`git diff --check` 通过。环境未安装 Ruff，未运行其静态检查。
现有清零人工真实数据路径审计得到 160 个有效窗口，现有人工策略的路径式离线 preflight 通过。
固定安装策略成功解析，但旧回放报告过期，仍阻止执行；未绕过回放门禁。
本次没有正式重训模型、改写已有权重、连接机械臂或执行真机运动。
