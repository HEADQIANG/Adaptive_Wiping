# 仿真模型与资源

项目实际使用的仿真资源统一放在根目录 `asserts/`（按指定拼写，不是 `assets/`）。

| 目录 | 内容 |
|---|---|
| `asserts/robosuite_models/robots/airbot_play/` | AIRBOT Play XML 与所有被引用网格 |
| `asserts/robosuite_models/robots/ur5e/` | UR5e 对照实验 XML 与所有被引用网格 |
| `asserts/robosuite_models/grippers/` | 擦拭工具、接触几何、FT 测量点 |
| `asserts/robosuite_models/bases/` | 支撑底座、无底座安装及底座网格 |
| `asserts/robosuite_models/arenas/` | 桌面与场景 XML |
| `asserts/robosuite_models/textures/` | 桌面、地面与污渍纹理 |
| `asserts/references/discoverse/` | AIRBOT 连杆惯量参考 XML，仅提取惯量，不是独立可运行场景 |
| `asserts/licenses/` | 原来源项目许可说明 |
| `asserts/manifest.json` | 每个复制文件的来源、新路径、大小及 SHA-256 |

XML、网格、纹理按原字节复制，内部相对引用保持完整。MuJoCo 使用 XML 中的材质定义；
OBJ 内上游遗留的外部 MTL 名称不是 MuJoCo 运行依赖，不据此生成或修改材质。
本目录不是整个 robosuite/DISCOVERSE 模型库，不包含项目未使用的机器人和场景。
工具安装、直接腕部几何及桌面位置调整仍由现有仿真代码在内存中生成，不写回资源文件。

## 运行与校验

仿真命令不变，`PretrainingWipe` 与 UR5e 对照实验均从本目录加载。
`shared.assets` 集中解析模型路径；不使用当前工作目录猜测模型位置。
仿真加载时设置 robosuite 已有的模型根路径，不修改外部依赖的内部代码。

在项目根目录、`clean` 环境中运行：

```bash
python -m scripts.shared.assets verify --sources
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -m unittest tests.sim_pretrain.test_model_assets -v
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.sim_pretrain smoke-test --output runs/sim_pretrain/models_smoke_001
```

`verify` 检查集中资源；`--sources` 另检查它们与原来源文件一致，DISCOVERSE 来源通过旧路径映射读取归档。命令只读，不连接设备。
新实验的来源记录包含实际使用的 `asserts/` 文件哈希。
模型数值未改变，但资源与源码路径变化会改变来源记录，不绕过校验继续已有采集或训练。
历史文件、旧源码快照及历史内嵌路径/哈希不改写，仍按原规则只读评估与回放。

`scripts/sim_pretrain/robosuite/` 中的原始模型保留供依赖库自身使用及历史来源核验，项目当前仿真不再从那里加载模型。
根目录 DISCOVERSE 已完整归档并移除，见 [脱离与归档说明](discoverse_removal.md)。
不要删除 `scripts/sim_pretrain/robosuite/`：项目仍依赖其仿真代码和控制器配置；硬件 SDK 的系统 URDF 审计路径不属于仿真资源，不作替换。
迁移项目时一并携带 `asserts/`，不能只安装不含这些资源的独立 wheel。

## 本次验证

2026-09-11：集中保存 55 个资源及许可文件，共 38,700,517 字节，全部与来源文件大小、SHA-256 一致。
新增 4 项模型资源测试通过，覆盖依赖完整性、来源记录及 AIRBOT 支架/桌面安装、UR5e 模型编译；
与原路径加载模型的质量、惯量、关节/执行器参数、网格及 10 步状态逐项完全一致。

完整回归运行 360 项，125.943 秒，0 失败/错误；训练环境跳过的 3 项 PyKDL 测试在系统 Python 中单独通过。
[回归日志](../../runs/sim_pretrain/models_smoke_001/regression.log) 与
[仿真采集、训练、导出加载报告](../../runs/sim_pretrain/models_smoke_001/smoke_report.json) 已保存。
smoke 的测试 MSE 为 `0.39734765887260437`，与集中前小规模验证一致。
它采用研究 record-only 接触策略，仅验证软件流程，不代表接触质量或硬件安全通过。
历史归档另核验 1,250 个文件，大小及哈希全部一致；本次未连接设备。
