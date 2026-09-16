# DISCOVERSE 脱离与归档

项目不再需要根目录 `DISCOVERSE/` 或 DISCOVERSE Python 环境。当前仿真只使用
`scripts/sim_pretrain/robosuite/` 代码、`asserts/` 模型和独立安装的 MuJoCo 等依赖；五个业务入口不变。
配置中的 `discoverse_standard` 是惯量方案名称，不表示导入或运行 DISCOVERSE。

## 保留内容

根目录的整个 DISCOVERSE 工作树已移入
`archive/sim_pretrain/discoverse_source_20260911/source/`，包括 `.git`、全部文件及三处未提交改动：

- `discoverse/doc/usage.md`
- `examples/force_control/joint_impedance_control.py`
- `models/mjcf/manipulator/new_airbot_play/mjx_airbot_play.xml`

采用完整归档后移除根目录的方式，没有丢弃用户修改，没有递归删除历史文件。
因此不释放这份归档占用的磁盘空间，但可以恢复完整源码和工作树。
移动前检查可见进程的命令、工作目录及文件占用，存在使用时拒绝操作，不终止进程。
每个文件移动前后检查大小、SHA-256；清单位于 `archive/_migration/discoverse.json`。
原有历史产物、源码快照和 `asserts/manifest.json` 的旧路径及哈希未改写。

## 来源核验

正常运行不读取归档来源；只有显式历史读取和 `--sources` 校验需要它。
`shared.paths.read_path` 将缺失的原 DISCOVERSE 路径定位到只读归档；
`historical=True` 显式选择归档来源。应用拒绝写入归档。

在项目根目录运行：

```bash
python -m scripts.shared.assets verify --sources
python -m scripts.shared.artifacts verify
python -m scripts.shared.artifacts resolve DISCOVERSE/models/mjcf/manipulator/airbot_play/airbot_play.xml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  python -m unittest tests.shared.test_discoverse_removal tests.sim_pretrain.test_model_assets -v
```

需要查看旧源码或改动时直接读取归档，不在归档内训练、运行 DISCOVERSE 或写 Git 状态。
需要恢复开发时，从归档复制完整 `source/` 到一个新的非归档目录，保留 `.git`；
不要覆盖当前项目文件，也不要恢复旧的业务入口或绕过来源校验续训。
这里的归档用于保留证据，不是当前运行依赖，也不改变任何真机安全门禁。

## 验收记录

2026-09-11：根目录 DISCOVERSE 已移除，完整归档 1,982 个文件、799,083,527 字节，
逐文件移动前后校验通过。合并原有历史清单后，共 3,232 个文件、1,963,818,489 字节，
大小与 SHA-256 全部一致；模型资源的 55 个文件来源校验也通过。

完整回归运行 362 项，95.343 秒，无失败/错误；训练环境跳过的 3 项 PyKDL 测试在系统 Python 中单独通过。
新增隔离测试禁止导入 DISCOVERSE，以及读取原目录或归档源码；模型校验、当前来源记录、仿真初始化与步进仍通过。
旧路径读取、写入拒绝和 AIRBOT/UR5e 模型校验通过。当前文档本地链接检查无失效项。

[完整回归日志](../../runs/sim_training/discoverse_removed_smoke_001/regression.log) 和
[移除后的仿真闭环报告](../../runs/sim_training/discoverse_removed_smoke_001/smoke_report.json) 已保存。
仿真采集、训练、导出与加载通过，测试 MSE `0.39734765887260437` 与移除前一致。
该小规模测试使用研究 record-only 接触策略，只验证软件流程，不证明接触质量或模型部署有效性。
本次没有连接或驱动真机。
