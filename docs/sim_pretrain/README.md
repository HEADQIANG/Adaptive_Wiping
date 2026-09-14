# 仿真预训练

与真机 003 的幅值搜索：原 90 组及扩展增益后的新增 135 组诊断结果，见 [幅值对照搜索](match_real_amplitude.md)。
新增 128 组分层随机覆盖诊断（含逐时刻包络、六维联合容差和独立运动验收）也见上述文档。
用户授权取消运动筛选后的 [2000 条采集与 1000 轮训练](wide_training_2000.md) 使用独立实验入口。
当前计划已改为 [保留 1200 条并训练](wide_training_1200.md)，新旧目录分别保留。
没有找到六通道幅值和运动同时合格的组合，默认参数未因此改变。

当前仿真已启用每段起止各 0.2 秒平滑加减速，保持 4 秒总时长和原位移，
允许峰值速度提高；真机轨迹未改。命令及兼容说明见 [平滑探索](smooth_exploration.md)。

当前配置已按海绵质量和几何尺寸修正工具转动惯量，运行命令不变；
假设、历史兼容性和验证步骤见 [工具惯量说明](sponge_inertia.md)。

只运行一条并同步显示 MuJoCo 画面与清零后的六维曲线：见 [单次探索操作步骤](explore_once.md)。
入口 `python -m scripts.sim_pretrain explore-once --mu 1.2 --stiffness 500.25 --width 0.16`，
支持手动设置接触参数，不执行批量采集或训练。

2026-09-12：生产仿真探索在初始稳定、无接触时读取六维基线，输出
`ft = (ft_raw - initial_bias) * [1,-1,-1,1,-1,-1]`，扣除起点静态重力载荷及初始偏置，
并按用户要求反转 Fy、Fz、Ty、Tz 输出方向；Fx、Tx 不变。
初始状态相对读数为零；运动后的惯性、姿态变化和接触响应仍保留，
不强行把所有悬空帧设为零，也不是跨姿态动态重力补偿。
起点有接触会拒绝清零。HDF5 新增 `ft_raw`，保留原始读数，
并以 `ft_processing=initial_noncontact_tare_yz_flip_v2` 标记新数据含义。
旧数据及模型不覆盖、不与新数据混用；真机采集不受此修改影响。

运行步骤：先将配置的 `output_dir` 改为一个未使用的新目录，再依次执行
下方 `sanity`、`collect`、`train`、`evaluate`、`export` 命令，不能复用旧 sanity。
回归检查在 clean 环境运行：

```bash
python -m unittest tests.sim_pretrain.test_pretraining
```

2026-09-11 测力轴更新：两份当前配置启用 `ft_frame_profile: kwr52_left_v1`，
按零位盖板朝左、厂家图 30 度关系对齐传感器方向。只旋转测力坐标系，
不改变海绵或测力原点；输出改用新的 `*_kwr52_left_v1` 目录，旧 FT 数据和模型不可混用。
运行命令不变，轴向验证和适用边界见 [KWR52 测力轴说明](kwr52_sensor_axes.md)。

2026-09-11 更新：仿真与真机探索均改为 `0.005 m/s` 按压 `2 s`，
共下压 `10 mm`；横移仍为 `0.05 m/s`、两个方向各 `1 s`，共 `4 s / 400` 帧。
两份 `pretrain_paper*.yaml` 已同步，下面运行命令不变。
旧配置快照、数据和模型仍对应旧速度，不能与新协议混用或原地续采；
新采集需使用新输出目录并重新执行 sanity。离线检查步骤见
[探索协议操作说明](../exploration_protocol.md)。

在项目根目录使用 `clean` 环境。保留本地 robosuite 分支依赖，并单独安装匹配环境的 PyTorch >= 2.6。
实际加载的仿真模型统一位于 `asserts/`，目录清单与校验命令见 [模型资源说明](models.md)。
robosuite 仓库现位于 `scripts/sim_pretrain/robosuite/`，见 [目录与校验说明](robosuite_location.md)。

```bash
python -m pip install -r requirements/sim_pretrain.txt
python -m pip install --no-deps -e .
python -m scripts.sim_pretrain --help
python -m scripts.sim_pretrain visualize-wiping --help
python -m scripts.sim_pretrain sanity --config configs/sim_pretrain/pretrain_paper.yaml
```

严格采集要求 sanity 通过，不得篡改验收报告。现有接触运动仍可能不达标；
研究采集是否保留运动不合格轨迹，必须显式选择原有研究选项，不能与严格验收混淆。

```bash
python -m scripts.sim_pretrain collect --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain train --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain evaluate --config configs/sim_pretrain/pretrain_paper.yaml
python -m scripts.sim_pretrain export --config configs/sim_pretrain/pretrain_paper.yaml
```

默认输出由配置的 `output_dir` 指定在 `runs/sim_pretrain/` 下；新实验使用新名称。
小规模全流程软件验证可运行：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.sim_pretrain smoke-test --output runs/sim_pretrain/smoke_001
```

该测试仅采集 2/1/1 条真实仿真轨迹并训练 2 轮，显式采用研究 record-only 接触策略，
验收未通过的完整轨迹也会保留；结果不代表接触运动或模型质量达标。
正式配置和严格验收规则不被该测试修改。
数据格式仍为每条 `400 x 6` FT，冻结编码器输出 5 维，训练和滤波算法不变。
可用 `collect-control-comparison`、`collect-normal-parallel`、`continue-pretraining` 等子命令查看原研究参数。
跨重构不原地续采续训；新运行产生的检查点可以按原规则续训。

历史模型只读评估到新目录：

```bash
python -m scripts.sim_pretrain evaluate \
  --config archive/sim_pretrain/normal_mu1p2_pretraining_v1/normal/config.json \
  --output runs/sim_pretrain/historical_evaluation_001
```

该入口适用于原预训练格式；独立续训/修复格式按其专题工具读取，不把不同格式混用。
更多参数见 [预训练专题](pretraining.md)、[普通模式研究采集](normal_mu1p2_pretraining.md)。
