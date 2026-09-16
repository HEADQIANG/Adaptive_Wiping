# 宽范围 2000 条采集及 1000 轮训练

用户明确取消本次实验的运动验收筛选：不因姿态、位移、接触比例、断触或限幅剔除轨迹。
不运行正式 sanity 门槛；原正式采集入口及门槛保持不变。
物理执行器限幅仍保留，所有运动验收指标仍记录。`valid` 仅表示数据完整且有限，
并非运动合格。无接触但完整有限的数据也保留；异常或 NaN 不伪造、不随机补样。

项目根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.sim_pretrain.experiments.collect_wide_training \
  --config configs/sim_pretrain/pretrain_wide_2000.yaml
```

范围：mu=[0,1.2]，stiffness=[10,10000]，width=[0.001,1.2]，gain=[2000,10000]。
沿用分层随机采样：mu 均匀，其余对数均匀，每条参数恒定；各划分独立 seed。
width 是接触阻抗过渡参数，不是海绵尺寸。gain 随机化还会同步改变 kd=2*sqrt(gain)。
动作保持 4 秒、100 Hz、0.2 秒平滑段及原目标位移。

共 2000 条：训练 1600、验证 200、测试 200。4 个独立仿真工作进程，主进程单独写 HDF5。
训练现有 SpongeVAE 1000 轮，CPU 单线程、batch=32、lr=1e-4、beta=0.06。
预处理仅拟合训练集，每轮统计训练和验证损失；测试集仅最终评估使用。

输出：`runs/sim_training/pretrain_wide_2000_v1/`。

- `manifest.json`、`assignments.json`：配置、代码来源和固定参数分配。
- `dataset.h5`：完整六维数据、原始数据、清零值、位姿、每条 gain/材料参数及诊断。
- `collection.json`、`progress.json`：采集结果和阶段状态。
- `history.json`、`vae_best.pt`、`vae_last.pt`：训练记录、验证最优和最终模型。
- `epoch_0200/`、`epoch_0400/`、`epoch_0600/`、`epoch_0800/`、`epoch_1000/`：
  `vae.pt`（包含优化器与 CPU RNG 状态）、损失曲线、验证样本重建图、验证指标。
- `evaluation.json`、`training.png`、`reconstruction.png`：最终测试评估及图像。

同一命令再次运行时，来源不变则继续缺失的数据条目，不重采已提交的条目。
训练从最近的 `vae_resume.pt`（每 200 轮提交）恢复；最后一个区间可能重算。
来源或配置变化时拒绝混用数据，须指定新配置及新的 output_dir，不删除旧数据。
训练完成后再运行只核验数据并重新评估，不重新训练。

回归检查：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.sim_pretrain.test_wide_training tests.sim_pretrain.test_pretraining \
  tests.sim_pretrain.test_match_real_amplitude tests.sim_pretrain.test_explore_once
```
