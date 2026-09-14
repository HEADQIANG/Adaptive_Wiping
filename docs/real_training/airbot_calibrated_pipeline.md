# AIRBOT 标定数据、训练与部署流程

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

新增入口均为离线转换，不连接机械臂、不采集标定值，也不认证标定的物理正确性。
现有原生坐标系模型和日志保留不变。`configs/robot_control/airbot_calibration.json` 是未完成的
标定记录模板，实测参数为 null，确认项为 false；不能填单位矩阵、零偏零值或改
确认项来绕过缺失的物理测量。

论文 III-B 使用相对于 base link 的绝对位置，III-C 组合离线 XY 轨迹与在线高度
反馈，因此不能只把 SDK 末端坐标的标签改为 TCP / base_link。转换后的数据仍
需通过编码器输入分布和独立验证检查；论文的网络配置不等于真机部署认证。

## 现场需要提供的测量

由熟悉安装结构的现场人员填写标定记录，提供实际测量方法及证据文件：

| 字段 | 物理含义 |
| --- | --- |
| `base_from_sdk` | SDK 参考系到 `base_link` 的 4x4 刚体变换 |
| `end_from_tcp` | 选定 TCP 坐标到 SDK 末端坐标的 4x4 刚体变换 |
| `sensor_to_ft_frame` | 传感器原点坐标到仿真 `ft_frame` 坐标的 4x4 刚体变换 |
| `sensor_bias_si` | 独立测量的电子零偏，单位 N、N*m；不包含工具重力 |
| `tcp_definition` | 实际使用的 TCP 点、轴及安装定义 |
| `method_notes` | 测量方法、安装状态和适用范围，说明为何适用于已有采集日志 |
| `evidence` | 非空证据文件列表，每项包含 `path` 和该文件的 `sha256` |

`evidence.path` 相对标定 JSON 所在目录解析，其他命令的路径从项目根目录解析。
检查单位、轴向、右手系、位置单位 m、四元数 xyzw，以及标定是否适用于采集时
的原有安装。工具安装改变后不得套用旧标定。仅在真实检查完成后填写对应确认项。
保存证据文件后可用 `sha256sum 证据文件路径` 获取其哈希。

力转换为 `F_ft = R F_sensor`，力矩转换为
`T_ft = R T_sensor + p × F_ft`，其中电子零偏先在传感器坐标中扣除。
不会扣除工具重力、自动接触清零、拟合坐标变换或为接近仿真范围而裁剪数值。
位置和姿态均按实际 SDK 末端姿态及 TCP 安装变换处理；训练和部署共用实现。

## 标定记录检查与转换

从项目根目录运行，使用已配置的 `clean` 环境：

```bash
conda activate clean
python -m scripts.real_training.airbot_calibrated_data check
python -m scripts.real_training.airbot_calibrated_data training --audit-only
python -m scripts.real_training.airbot_calibrated_data training
```

默认输入为第 007 次完整探索和 `session_record_only_002` 的 8 条已接受示教。
另选输入时使用 `--exploration`、`--session`；使用 `--calibration` 指定实测标定。
当前模板检查应退出 2，并且不生成训练数据。转换先验证原始文件哈希、数据覆盖、
四元数、时间顺序、20 ms 数据年龄及间隔，再执行物理变换和因果保持采样。

输出为 `runs/real_training/real_training/airbot_calibrated_v1/calibrated_resampled.h5` 及同名前缀
的 `.import.json` 审计报告。文件禁止覆盖，`--audit-only` 只在临时目录验证。
HDF5 的 `ft_time` 和 `ft` 是 100 Hz 的处理网格，不是 100 Hz 独立传感器测量。
原接收时间、原传感器数据和转换后的接收数据分别保存在 `received_ft_time`、
`received_sensor_ft`、`received_calibrated_ft`；实际接收率保存在
`received_ft_hz` 及根元数据 `sampling` 中。约 56 Hz 接收率这一论文差异不会被隐藏。
原始 SDK 位姿也保留，新增 `tcp_position` 和 `tcp_quaternion` 才是转换后的位姿。

## 训练与新任务探索

```bash
python -m scripts.real_training inspect --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_training prepare --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_training train --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_training evaluate --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_training cross-validate --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_training export --config configs/real_training/real_training_airbot_calibrated.yaml
python -m scripts.real_deploy replay --training-config configs/real_training/real_training_airbot_calibrated.yaml \
  --output runs/real_training/real_training_airbot_calibrated_v1/deployment_replay
python -m scripts.real_training.airbot_calibrated_data exploration --audit-only
python -m scripts.real_training.airbot_calibrated_data exploration
```

标定后训练目录独立为 `runs/real_training/real_training_airbot_calibrated_v1`，不会替换原生
坐标系模型。保持论文下游网络、10000/2000 轮及学习率 .001；标定不会自动修复
已有编码器的潜变量塌缩，也不保证验证误差变好。训练后的诊断仍须如实保留。

`exploration` 转换已完成的 4 秒真实探索为部署使用的 NPZ。实际新任务需通过
`--exploration` 传入当前海绵的新日志，用 `--output` 指定新文件，不能把旧探索
自动当作当前海绵数据。NPZ 保存 400 帧、标定哈希、源日志哈希和原始接收序列，
部署会重算变换和采样并检查一致性，不接受只有坐标标签而无转换证据的文件。

## 接入部署

在 `configs/real_deploy/airbot_deployment.json` 中指定审核后的 `policy`、`policy_sha256`、
`prepared_data`、`exploration`、`calibration_record` 和一致的 `calibration_id`。
四个坐标变换/零偏字段可保持 null，由指定的标定记录在内存中填入；显式给出时
必须与记录一致。策略中的标定文件哈希、实际数值和输入定义也必须一致。
现场工作空间、速度、电流、力、力矩限制及安全确认仍需单独实测/批准。

```bash
python -m scripts.real_deploy preflight --config configs/real_deploy/airbot_deployment.json
python -m scripts.real_deploy run --config configs/real_deploy/airbot_deployment.json \
  --execute --output runs/real_training/real_robot/policy_run_001.jsonl
python -m unittest discover -s tests -t . -p 'test_airbot_calibration.py' -v
python -m unittest discover -s tests -t . -p 'test_airbot_deploy.py' -v
```

`preflight` 是离线检查。现场启动服务、手动摆位、急停与支撑要求和 `DEPLOY` /
`IDLE` 确认流程见 [airbot_native_training.md](airbot_native_training.md)。校验失败时
不得删警告或改变数据来源来放行运动；只有物理验证和模型质量问题解决后才可执行。

`replay` 同时支持原生模型和通过本转换器生成的已标定数据；后者直接使用已转换
的 FT 和 TCP，不再二次变换。接机前必须先通过 160 次流式/批量预测一致性检查。
策略或探索文件在读取、人工确认期间发生改变时会在连接前拒绝执行；标定记录及
证据也在确认后重新检查。

## 本轮验证结果

2026-09-11：全量 300 项测试通过，耗时 87.639 秒，无跳过测试。新增 14 项测试
覆盖标定缺失/变更、虚构来源标记、非刚体变换、位姿组合、力矩力臂、因果采样、
转换不覆盖、接收流保留、部署参数冲突、部署探索重算及已标定流式回放。
所有正向标定测试只使用带 `UNIT_TEST_ONLY` 标签的临时数值夹具，结束后删除；
不构成任何现场标定或硬件成功证据。

验证记录位于 `archive/real_training/airbot_calibrated_pipeline_verification_v1`：
`regression.json` 保存全量测试结果，`training_audit/summary.json` 复核原始日志、
冻结编码器和九组已完成模型，`verification.json` 记录当前源文件哈希与未解决项。
原生模型的权重、配置和训练来源保持不变；本轮未生成正式的已标定数据、模型或
新任务探索文件。模板检查和 `training --audit-only` 均按预期在缺少标定时退出 2。
