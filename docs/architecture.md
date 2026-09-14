# 结构与归档说明

五类业务模块与 README 导航一致，`shared` 放共用路径、数据、几何和模型组件。
当前业务包名为 `scripts`，安装更新、历史多版本映射见 [包改名说明](package_rename.md)。
底层 `robot_control.client/adapter/safety` 供示教、探索、部署及基础控制复用。
`shared.encoder/policy` 负责推理，离线训练执行器不进入基础控制或部署的导入链。
SDK 只在连接时加载；仿真模块集中选取本地 robosuite 分支。

## 历史保留

重构前的自有源码、配置及说明保存在 `archive/_migration/source_snapshot/`。
`artifacts.json` 记录每个历史产物的旧路径、新路径、大小和 SHA-256；
`snapshot.json` 记录源码快照校验；`layout.json` 记录代码目录迁移。
旧目录是空目录时才移除，字节码另存，不删除数据或失败实验。

```bash
python -m scripts.shared.artifacts list
python -m scripts.shared.artifacts verify
python -m scripts.shared.artifacts resolve outputs/normal_mu1p2_pretraining_v1/normal/encoder.pt
```

历史文件内的路径和哈希不重写。读取旧产物使用 `shared.paths.read_path`，
需要重构前源码证据时明确传入 `historical=True`；映射到多个分类的混合旧目录会拒绝歧义解析。
归档字节验证不是当前源码或真机有效性验证。归档写入由应用保护拒绝；
研究工具读取旧实验的源码哈希时同样显式选择历史快照，报告区分 historical_evidence 与 current_files，
不会把重构前的源码哈希当作当前实现未改变的证明。
原地续采/续训不得绕过来源校验，新任务写 `runs/<类别>/<运行名>/`。
该保护不是操作系统级只读挂载，外部编辑器和本地第三方程序仍有文件访问权限。

SDK 5.2.2 的 Python 日志路径固定为 `logs/app.log`，不能通过环境变量配置。
根目录 `logs` 因此仅为指向 `runs/robot_control/sdk_logs` 的链接，实际新日志仍归机械臂控制分类。
它不是历史日志目录，也不是旧代码入口；原始历史日志保留在归档中。

## 当前运行文档

各分类 `README.md` 为主操作入口，分类内其他文件是专题说明。
快照中的文档仅作历史证据。每次修改代码后检查命令、配置和运行步骤；
发生变化时更新对应文档，缺失步骤时新增文档，不能仅改代码。

测试从项目根目录使用 `python -m unittest discover -s tests -t . -v`，
单类测试使用 `python -m unittest discover -s tests/robot_control -t . -v`。
新环境按模块 `requirements` 安装，随后 `python -m pip install --no-deps -e .`。
项目使用本地外部资源，需保持源码目录、`configs` 和 `scripts/sim_pretrain/robosuite` 代码目录完整；不提供独立 wheel 资源分发。
外部仓库虽在仿真目录内，仍与自有业务包的打包、源码扫描及检查分离，见 [robosuite 目录说明](sim_pretrain/robosuite_location.md)。
项目仿真模型现集中于 `asserts/`，该目录也必须一并携带；模型清单与校验步骤见
[仿真模型资源](sim_pretrain/models.md)。DISCOVERSE 已脱离运行依赖并完整归档，
旧来源映射另存在 `archive/_migration/discoverse.json`，详见 [移除说明](sim_pretrain/discoverse_removal.md)。

固定权重、输入及随机种子的重构前后模型对照：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest tests.shared.test_model_parity -v
```

该测试只读载入已归档的独立模型定义，与当前实现比较训练态、推理态输出及 VAE 损失，
不恢复旧包入口、不修改快照，也不替代历史数据完整性与现场验收。

本次测试、数值对照、归档校验及现场阻断项见 [软件验收记录](refactor_acceptance.md)。
