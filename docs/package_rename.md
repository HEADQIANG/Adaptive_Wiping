# Python 包改名为 scripts

业务包已从 `adaptive_wiping` 改为 `scripts`，原目录与旧入口不再保留，不建立兼容别名。
五类模块、`shared`、`experiments` 和 `tools` 的内部分类不变。
项目发行包名称仍为 `adaptive-wiping`，以便更新原有安装，不另建重复发行包。

## 安装与运行

改名后在实际使用的环境中重新安装项目，再重启已导入旧包的 Python 会话。
从项目根目录执行，安装不会连接设备：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m pip install --no-deps --no-build-isolation -e .
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install --no-deps -e .
python -m scripts.sim_pretrain --help
python -m scripts.force_sensor --help
python -m scripts.real_training --help
python -m scripts.real_deploy --help
python -m scripts.robot_control --help
python -m scripts.robot_control console
```

必须以包方式运行，不改为直接执行深层 `.py` 文件。
基础控制默认假机器人、现场配置与确认门禁不变；运行命令详见五类 README。
`asserts/`、`configs/`、`runs/` 和 `archive/` 的位置不变，训练算法、数据格式和权重字段不变。

## 历史来源

改名前的 214 个自有源码、测试、配置及文档文件保存在
`archive/_migration/package_rename_20260911/source/`，逐文件记录见 `archive/_migration/package_rename.json`。
已有 `archive/`、`runs/` 数据和模型中的旧路径、哈希不替换，不把旧结果重新标成新代码产物。
当前文档命令已更新；快照中的旧命令只作历史证据。

同一旧路径可能存在不同时间的源码快照。历史来源核验使用
`read_path(path, historical=True, source_sha256=expected_hash)` 按已记录的 SHA-256 定位版本，
并再次校验文件实际字节；找不到匹配版本时不替换成其他快照。
未指定哈希的历史路径读取维持最早归档的解释；不把新 `scripts/` 路径当成旧源码证据。
需要直接查看改名前版本时使用完整快照路径。
跨改名不绕过来源校验原地续采或续训，新任务写入新的运行目录。

## 验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -m unittest discover -s tests -t . -v
python -m scripts.shared.artifacts verify
python -m scripts.shared.assets verify --sources
```

包身份、旧入口缺失、源码引用、历史多版本选择及全部快照校验见
`tests/shared/test_package_rename.py`；数值对照仍使用重构前的独立模型快照。

## 本次验收

2026-09-11：已更新训练与 SDK 环境的项目安装，两者在项目目录外也能导入 `scripts`，
旧包无法导入；五个帮助入口、依赖隔离和独立传感器环境的读取帮助检查通过。

- 完整回归 365 项，126.321 秒，0 失败/错误；训练环境跳过的 3 项 PyKDL 测试在系统 Python 中单独通过。
- 独立 SDK 环境 21 项基础控制测试通过，仅使用假客户端；项目目录外的假机器人关节点动通过。
- 仿真采集、2 轮训练、评估、导出加载通过；测试 MSE `0.39734765887260437` 与改名前一致。
- 历史 200 轮模型只读评估 MSE `0.016111869364976883` 不变；8 条示教离线回放的流式/批量误差为 `0.0`。
- 固定输入及权重下，VAE 与真实策略各分支的训练态/推理态输出对照通过。
- 全部 3,446 个归档与快照文件、1,965,333,140 字节大小及 SHA-256 一致；55 个模型资源的来源校验通过。
- 当前文档本地链接无失效项，当前代码及运行文档没有旧包导入或旧入口命令；历史路径测试保留旧名称。

记录：[完整回归](../runs/sim_training/scripts_package_smoke_001/regression.log)、
[仿真闭环](../runs/sim_training/scripts_package_smoke_001/smoke_report.json)、
[SDK 测试](../runs/real_deploy/robot_control/scripts_package_verification_001/sdk_tests.log)、
[历史模型评估](../runs/sim_training/scripts_package_evaluation_001/evaluation.json)、
[历史策略回放](../runs/real_deploy/scripts_package_replay_001/report.json)。

本次没有连接或驱动真机。仿真 smoke 使用研究 record-only 接触策略，仅验证软件流程。
部署预检仍返回 2，标定、编码器适用范围/质量、策略哈希绑定和现场安全确认缺失继续阻止执行。
