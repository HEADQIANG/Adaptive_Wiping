# 改为 1200 条并训练

本次用户将原 2000 条计划缩减为总计 1200 条。原任务达到 1200 条后停止，
保留源目录 `runs/sim_pretrain/pretrain_wide_2000_v1`，不删除原始数据。
取源数据按原划分/索引顺序的前 1200 条完整有限记录，再以 seed=20260912
随机打乱并划分为 train=960、validation=120、test=120，无交叉重复。
它们是原分层随机分配的子集，不宣称重新构造了 1200 点完整分层采样。
源数据此前尚未训练，划分在任何训练开始前固定；预处理仅拟合新训练集。

范围、动作和取消运动筛选的规则不变：mu=[0,1.2]，stiffness=[10,10000]，
width=[0.001,1.2]，gain=[2000,10000]；mu 均匀，其余对数均匀。
4 秒、100 Hz、0.2 秒平滑段，保留执行器限幅但不据此筛选。

确认原采集任务已停止、源目录至少有 1200 条完整记录后，在项目根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.sim_pretrain.experiments.train_collected_subset \
  --config configs/sim_pretrain/pretrain_wide_1200.yaml
```

新输出目录 `runs/sim_pretrain/pretrain_wide_1200_v1`。
命令验证源数据并复制原始轨迹，不重新仿真，不修改原目录的数据。
`source_mapping.json`、HDF5 source_split/source_index 和源 SHA256 记录逐条来源。
已生成数据再次执行时核对哈希，不重新划分；中断训练从每 200 轮的恢复点继续。

训练 1000 轮，每 200 轮在 `epoch_0200/` 到 `epoch_1000/` 保存 `vae.pt`、
`training.png`、`reconstruction.png`、`validation.json`、`history.json`。
最终保存 `vae_last.pt`、`vae_best.pt`、`evaluation.json` 和测试重建图。
原采集完整数可能因停止信号到达时已在写入而略多于 1200；训练严格只使用 1200 条。

检查命令：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.sim_pretrain.test_collected_subset tests.sim_pretrain.test_wide_training
```

## 本次完成结果

源采集任务恰好在 1200 条完整记录时停止，停止说明保存在源目录 `stop.json`。
新数据集按 960/120/120 固定划分，全部字段与对应源记录逐条一致且三个划分无重复。
原运动指标仍记录：13 条满足原运动门槛、54 条出现过限幅；所有 1200 条均保留。

1000 轮已完成，200/400/600/800/1000 轮的模型、优化器状态、图表和验证指标全部存在。
最终训练 MSE=0.00457321，验证 MSE=0.00537134，测试 MSE=0.00447345，
均为预处理归一化空间的误差，不是 N 或 N*m。
测试均值基线 MSE=0.00444912；固定潜变量 MSE=0.00447346，打乱潜变量 MSE=0.00447346。
测试 KL=1.2536e-5，按方差 1e-4 门槛的活跃潜变量维数为 0/5，collapse_warning=true。
因此训练过程完成，但模型发生潜变量塌缩，尚不能视为学到了有效的海绵属性表示；
没有因该结果擅自更换模型、beta、采样策略或追加训练。

最终测试物理单位 RMSE（Fx,Fy,Fz,Tx,Ty,Tz）：
`[2.17943, 1.37812, 8.63949, 0.117007, 0.128532, 0.018016]`，前三项 N，后三项 N*m。
详细结果见新目录 `evaluation.json`；训练与重建图见 `epoch_1000/`。
38 项回归测试通过；源/目标数据 SHA256、1200 条唯一来源、1000 轮有限损失记录，
以及五个检查点的数据来源与轮次均已核对。
