# robosuite 目录归属

## 从 GitHub 还原当前适配（2026-09-16）

主仓库通过 `.gitmodules` 固定上游 robosuite 提交
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。本机新增的 AIRBOT 注册、控制器、
模型/网格和说明以 `patches/robosuite-airbot.patch` 发布，并附逐文件 SHA256 清单。
不向上游仓库推送项目适配，也不让 Git 子模块指向只有本机存在的提交。

新克隆在项目根目录执行：

```bash
git submodule update --init scripts/sim_pretrain/robosuite
python3 -m scripts.shared.robosuite_patch apply
python3 -m scripts.shared.robosuite_patch verify
```

已还原且哈希匹配时 `apply` 不写文件。部分应用、不同版本或冲突时拒绝覆盖，
先检查自己的修改。无需对已正常工作的本机 robosuite 再执行 submodule update。
适配后子模块显示 modified 是预期状态，完整差异已由主仓库补丁保存。

在已配置的仿真环境中，先安装固定 checkout 自身的依赖，再安装项目固定版本：

```bash
python -m pip install -e scripts/sim_pretrain/robosuite
python -m pip install -r requirements/sim_pretrain.txt
python -m pip install --no-deps -e .
```

这样保留 termcolor、numba 等上游依赖，同时使用项目要求的 MuJoCo 等版本。
PyTorch >= 2.6 按机器另行选择 CPU/CUDA 构建。不要另装一个替代本地 checkout 的 robosuite。

后续有意修改依赖适配后，可重新导出再校验，并把两个补丁文件与相应说明一起提交：

```bash
python3 -m scripts.shared.robosuite_patch export
python3 -m scripts.shared.robosuite_patch verify
python3 -m unittest tests.shared.test_robosuite_patch
```

export 收录嵌套仓库中所有未被忽略的新增文件和相对 HEAD 的修改；导出前检查
`git -C scripts/sim_pretrain/robosuite status --short`，避免将临时输出混入适配。

完整仓库已从根目录移动至 `scripts/sim_pretrain/robosuite/`，保留 `.git`、
已有修改、未跟踪的 AIRBOT 适配、文档和模型文件，未改动外部库内部源码。
Python 的外部库导入名仍为 `robosuite`，不是 `scripts.sim_pretrain.robosuite`。
实际 Python 包位于 `scripts/sim_pretrain/robosuite/robosuite/`。

仿真包在加载时加入新的外部库根路径，控制器配置和源码核验统一使用该位置。
项目使用的模型仍从 `asserts/` 加载，不迁回外部库目录。
业务源码扫描、项目打包和 Ruff 检查排除嵌套的外部仓库；
实验来源仍显式记录实际使用的外部库源码，不把整套外部测试与示例混入业务包。

## 运行与校验

在项目根目录运行，业务命令不变：

```bash
python -m pip install --no-deps -e .
python -m scripts.sim_pretrain --help
python -m scripts.shared.assets verify --sources
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -m unittest tests.sim_pretrain.test_robosuite_location tests.sim_pretrain.test_model_assets -v
```

其他模块的帮助、传感器或基础控制不导入 robosuite，不要求合并环境。
若外部 Python 会话已加载旧 robosuite，需重启会话；不要混用搬迁前后模块实例。
仅需业务功能时按上述模块入口运行，不另外安装其他 pip robosuite 分支。

若 SDK 环境无法联网下载构建依赖，可在项目根目录使用现有 `clean` 环境的
setuptools 构建本地可编辑安装包，再离线安装到 SDK 环境；不升级 SDK 或合并依赖：

```bash
wiping_wheel_dir=$(mktemp -d /tmp/wiping-editable-XXXXXX)
/home/wp/miniconda3/envs/clean/bin/python -c \
  'import sys; from setuptools.build_meta import build_editable; build_editable(sys.argv[1])' \
  "$wiping_wheel_dir"
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m pip install \
  --no-index --no-deps --force-reinstall \
  "$wiping_wheel_dir/adaptive_wiping-0.1.0-0.editable-py3-none-any.whl"
```

以上安装包绑定当前源码目录，仅用于本机，后续项目版本号改变时同步修改文件名。

## 历史与迁移

旧 `robosuite/...` 文件路径通过 `scripts.shared.paths.read_path` 解析到新位置，
`asserts/manifest.json` 和历史实验内嵌路径、哈希保持原样。
这是仍在使用的外部仓库路径映射，不是把当前源码伪装成旧快照；来源哈希仍需核验，
不匹配时不放行。根目录不保留符号链接或副本，旧位置不再作为输出目录。

迁移清单 `archive/_migration/robosuite_move.json` 记录整目录文件移动前后的大小和 SHA-256。
它是移动当时的核验记录，不要求之后生成的字节码/Git 元数据保持固定。
自有代码改动前快照与清单位于 `archive/_migration/robosuite_relocation_20260911/source/`
及 `archive/_migration/robosuite_relocation_sources.json`，历史来源可按原哈希选择对应版本。
模型与算法未变，但源码位置改变会影响来源记录，跨移动不绕过校验续采或续训。

## 本次软件验收（2026-09-11）

- 搬迁时逐文件核验：外部仓库 1,575 个文件、1,322,046,290 字节，大小与 SHA-256 全部一致。
- 全量回归：370 项，367 项通过，3 项因 `clean` 环境缺少 PyKDL 跳过，耗时 164.846 秒。
  这 3 项在系统 Python 中单独运行全部通过。
- SDK 环境基础控制与假 SDK 后端：21 项全部通过，未连接硬件。
- 55 个集中模型资源及来源副本校验通过；历史归档 3,661 个文件、1,966,850,044 字节全部通过。
- 小规模仿真采集、训练、导出和加载通过；测试 MSE 为 `0.39734765887260437`，与搬迁前一致。
  此项仅验证软件链路，不代表接触质量或真机安全验收。
- 新位置导入、打包排除、旧路径来源核验和依赖隔离通过，Ruff `F821,E9` 检查通过。

历史记录路径如下；日志不随 Git 发布，部分旧日志已不在当前工作区，复核请运行下方命令：
`runs/sim_training/robosuite_relocation_verification_001/logs/regression.log`、
`runs/real_deploy/robot_control/robosuite_relocation_verification_001/logs/sdk.log`、
`runs/sim_training/robosuite_relocation_smoke_001/smoke_report.json`。

复核命令（在项目根目录运行；仿真输出目录须换成未使用的新运行名）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -t . -v
/usr/bin/python3 -m unittest tests.robot_control.test_airbot_model_chain -v
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.robot_control.test_basic_control tests.robot_control.test_hardware_backend -v
/home/wp/miniconda3/envs/clean/bin/python -m scripts.shared.artifacts verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  /home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain smoke-test \
  --output runs/sim_training/robosuite_relocation_smoke_002
```
