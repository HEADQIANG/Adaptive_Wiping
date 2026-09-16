# 传感器标定数据转换（SDK 位姿不变）

2026-09-14：已移除真机 TCP 标定、基座/末端位姿转换及部署 TCP 门槛。
位置、姿态直接使用 SDK 输出；XY 为 SDK 水平轨迹，高度预测为 SDK Z 增量。
同安装、同参考系、近似固定姿态的流程不需要末端到海绵的偏移量。
通用仿真/论文数据中的 TCP 字段仍保留，不会重命名历史原始数据。

## 两条独立流程

- 当前同海绵固定安装实验继续使用 `real_training_programmed_wide1200.yaml`
  和部署 `--mode fixed-setup`：每次非接触基线扣除，不需要本页的传感器标定记录。
- 需要传感器坐标/原点变换时，使用 `airbot_sensor_calibrated_offline`
  profile 和部署 `--mode calibrated`。后者现在仅表示传感器标定，
  不执行 TCP 或基座坐标变换，也不自动允许原生模型进入该模式。

## 传感器记录

配置路径保持 `configs/robot_control/airbot_calibration.json`，格式升级为
`schema_version: 2`、`artifact_kind: airbot_sensor_calibration`。
旧版位姿标定记录不能直接作为新版记录加载。

只保留 `sensor_to_ft_frame`（传感器到目标 FT 坐标的 4x4 刚体变换）
和 `sensor_bias_si`（传感器坐标下的六轴电子零偏）。
力矩仍包含平移力臂项；电子零偏只扣一次，工具重力保留。
空载基线、电子零偏及重力补偿不是同一操作，不应叠加扣除。

记录还需要机器人/传感器身份、标定编号、轴向说明、测量方法、证据路径和 SHA256，
以及单位/轴向、电子零偏、适用安装的真实确认。
已删除 `base_from_sdk`、`end_from_tcp`、`tcp_definition` 和
`tcp_geometry_verified`。不要重新添加这些字段作为运行条件。
当前传感器记录仍未填完整，检查失败是预期行为，不得用单位矩阵或假证据放行。

## 离线命令

从项目根目录执行，使用 clean 环境：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training calibrate-data check
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training calibrate-data --help
```

仅在传感器记录完整、输入为当前协议下的一次真实探索和八条人工 10 秒示教时，
通过 `training --exploration 实际探索路径 --session 实际人工会话路径` 转换，
可先加 `--audit-only` 在临时目录审核。
默认历史探索为 10 mm/s，下压协议与现行 5 mm/s 不兼容，不能直接重用默认输入；
当前程序示教需保留其显式补齐流程，不支持直接传给此人工示教转换器。

转换后的独立配置是 `configs/real_training/real_training_airbot_calibrated.yaml`。
输出包含原始 SDK 位置/四元数、原始接收 FT、转换 FT 和来源哈希；
不生成 `tcp_position` 或 `tcp_quaternion`。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training inspect --config configs/real_training/real_training_airbot_calibrated.yaml
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training prepare --config configs/real_training/real_training_airbot_calibrated.yaml
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training train --config configs/real_training/real_training_airbot_calibrated.yaml
```

新数据、prepared、训练输出均使用未占用目录，不覆盖或改写旧模型元数据。
新任务探索仍通过 `calibrate-data exploration --exploration 实际探索路径 --output 新NPZ路径`
生成，经核对的六轴变换和输入契约必须与训练一致。

## 部署

`configs/real_deploy/airbot_deployment.json` 将起点改为
`initial_sdk_position_m`，与固定的 `sdk_end_orientation_xyzw` 一起使用。
部署直接比较 SDK 起点、按 SDK 坐标发送目标，不做位姿正反转换。
传感器校验、模型质量、关节、载荷、速度、调度、实际起点和现场确认保护均保留。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_deploy preflight --mode calibrated --config configs/real_deploy/airbot_deployment.json
```

此命令纯离线。固定安装的现场操作仍见
[fixed_setup.md](../real_deploy/fixed_setup.md)，不因移除 TCP 而自动启动真机。

## 旧产物与验证

原始日志、归档模型和既有回放不修改。旧元数据中的 TCP 未标定描述作为历史来源保留。
代码来源已变化，旧检查点可能不能续训/重新导出，旧 fixed-setup 模型的源码绑定和
回放不能直接作为当前执行依据；须在独立运行中生成当前代码绑定的产物，再重新审核
配置并回放。仅重新运行旧回放命令不能绕过训练源码不一致，不修改哈希或关闭校验。
本次不自动重新训练、不替换策略、不执行任何硬件命令。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m unittest tests.robot_control.test_airbot_calibration tests.real_deploy.test_airbot_deploy tests.real_deploy.test_fixed_setup tests.real_training.test_real_training tests.real_training.test_airbot_native_training tests.real_training.test_programmed_padding -v
```

标定正向测试使用临时数值夹具，不能作为实物传感器标定证据。
归档集成测试仅在测试进程内恢复旧探索协议，不放宽生产导入检查。

2026-09-14 验证：上述测试加基础控制、初始位姿、人工/程序示教、探索和空载拟合，
共 263 项通过。当前 wide1200 数据重算与既有 prepared 数值逐项相同；临时离线回放
8 条示教、160 次高度预测，流式/批量最大差为 0。没有连接硬件。
传感器模板 check 退出 2，仅列传感器字段/证据缺失，不再要求 TCP。
旧 fixed-setup 策略 preflight 退出 2，原因是源码绑定变化；没有绕过此保护。
