# 分类重构软件验收

2026-09-11 完成软件验收。五类业务入口已切换，旧入口不保留；
训练算法、数据单位、坐标语义和历史权重字段未改变，没有新增在线学习。
当前运行步骤见 [项目导航](../README.md)，归档规则见 [结构说明](architecture.md)。

## 回归结果

| 项目 | 结果 |
|---|---|
| 重构前基线 | 325 项，0 失败/错误，3 项 PyKDL 跳过；91.943 秒 |
| 当前完整回归 | 356 项，0 失败/错误，3 项 PyKDL 跳过；106.255 秒 |
| 系统 Python 的 PyKDL 测试 | 3 项通过，未连接设备 |
| 独立 SDK 环境基础控制 | 21 项通过，使用假客户端与真实 SDK 消息类型，未连接设备 |
| 五个入口与依赖隔离 | 帮助入口、基础控制/标定导入、历史路径与旧入口检查通过 |
| 固定输入数值对照 | VAE、XYDecoder、FTEncoder、HeightFeedback 的训练态/推理态输出及 VAE 损失差异均为 0 |
| 历史文件完整性 | 1,087 个产物 + 163 个源码/配置/文档快照，共 1,250 文件、1,164,734,962 字节；大小及 SHA-256 全部一致 |
| 静态检查 | Ruff F821/E9 通过；当前文档本地链接无失效项 |

主回归使用 `clean` 环境；SDK 环境保持 `arm-sdk==5.2.2`、NumPy 2.2.6，
补装清单中原本缺失的 SciPy 1.15.3 与 PyYAML 6.0.3，没有替换 SDK 或合并环境。
测试日志保存到 [分类验收目录](../runs/real_deploy/robot_control/refactor_verification_001/)。

## 数据与模型流程

- [仿真闭环报告](../runs/sim_training/refactor_smoke_001/smoke_report.json)：MuJoCo 采集 2/1/1 条训练/验证/测试轨迹，训练 2 轮、评估、导出及冻结编码器加载通过。
- 合成示教流程：8 折验证、160 个窗口，各分支 2 轮训练、独立导出加载通过；临时测试数据由测试自动清理，不冒充真实采集。
- [历史模型只读评估](../runs/sim_training/refactor_evaluation_001/evaluation.json)：测试 MSE 为 `0.016111869364976883`，与原记录一致。
- [历史策略离线回放](../runs/real_deploy/refactor_replay_001/report.json)：8 条示教，各 1001 tick，流式/批量预测最大差异为 `0.0`。
- [SDK 环境离线点动日志](../runs/real_deploy/robot_control/refactor_sdk_preview_001.jsonl)：假机器人执行有限关节目标后进入 idle，`hardware_connected=false`。

仿真 smoke 显式采用研究 record-only 接触策略：轨迹被记录不代表接触运动验收通过，
不能据此证明模型质量或真实扫掠路径安全。数值一致性与软件流程通过也不是部署许可。

## 复验命令

在项目根目录运行。结果目录已存在时使用新的运行名，不覆盖本次记录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -t . -v
/usr/bin/python3 -m unittest tests.robot_control.test_airbot_model_chain -v
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/wp/airbot-venv-5.2/bin/python -m unittest \
  tests.robot_control.test_basic_control tests.robot_control.test_hardware_backend -v
/home/wp/miniconda3/envs/clean/bin/python -m scripts.shared.artifacts verify
```

仿真闭环和历史只读评估命令见 [仿真步骤](sim_pretrain/README.md)，
示教闭环见 [真机离线训练步骤](real_training/README.md)，
历史回放和预检命令见 [部署步骤](real_deploy/README.md)。
一次性迁移工具保留在 `archive/_migration/` 仅作证据，不能重跑目录重写。
迁移中断恢复、旧路径解析与禁止写入归档已纳入回归测试。

## 现场待验收

本次没有连接或驱动机械臂，没有打开串口或启动/停止设备服务。
部署预检返回 2 且 `hardware_connected=false`、`executable=false`，仍因以下问题阻断：

- 传感器/TCP 标定及训练坐标对应关系未确认，标定身份与训练数据不一致。
- 探索数据超出编码器范围，来源编码器质量诊断未通过。
- 未绑定经审核的策略 SHA-256，现场调试、急停、服务 no-return、工具/扫掠路径等证据缺失。

基础控制配置中的未确认速度、电流、工作空间与安全参数仍保持未填写状态，拒绝执行。
基础控制不提供力传感器超力保护；实际重力补偿、保持、关节和末端运动必须现场监护验收。
软件停止不能替代硬件急停，停止锁存不自动清除，控制权丢失不抢占，异常不自动恢复。
