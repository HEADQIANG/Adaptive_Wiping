# 1200 条数据的编码器—解码器联合重训

## 2026-09-16 完成结果

输出目录：`runs/sim_training/0916_164948/retrain_wide_1200/`。
三个 beta 候选均完成 400 轮，验证集选中最终 beta=1e-5；seed43/44 随后各独立
完成 400 轮，三个种子均通过验证和历史测试集验收。

| seed | 按验证选择的轮次 | 测试 MSE | 固定编码 MSE | 打乱编码平均 MSE |
| --- | ---: | ---: | ---: | ---: |
| 42，导出 | 152 | 0.0003220033 | 0.0043769237 | 0.0082910049 |
| 43 | 205 | 0.0003222163 | 0.0043770359 | 0.0082470048 |
| 44 | 237 | 0.0003219695 | 0.0043768547 | 0.0082719381 |

seed42 相对原模型测试 MSE=0.0044734525 改善 **92.80%**，相对均值模板改善 **92.76%**。
活跃潜变量从 0/5 变成 5/5，测试潜变量方差约 0.566–1.046。
固定编码使误差变成约 13.6 倍，20 次打乱平均约 25.7 倍，潜变量已被有效利用。

六通道物理 RMSE：Fx=0.71490 N、Fy=0.50129 N、Fz=1.71737 N、Tx=0.04232 N*m、
Ty=0.06152 N*m、Tz=0.006834 N*m。原 Fz RMSE 为 8.63949 N。
反向阶段仍较弱，MSE=0.00082794，Fz RMSE=2.56954 N；部分返回瞬态峰值仍被平滑。

新编码的独立属性探针由验证集选择 MLP，测试 R2：摩擦 0.9664、stiffness_direct
0.7795、width 0.2610；三项相对常数预测的配对 RMSE 改善区间均严格大于零。
width 仍较弱。材料标签只用于独立探针，没有进入 VAE 梯度训练。

PCA 初始化的验证 MSE=0.0004030614，导出模型为 0.0004037281：本次解决了原模型
塌缩并在随机 VAE 训练中保持有效编码，没有进一步超过该确定性初始化的重建误差。

推荐后续使用的模型：

- `runs/sim_training/0916_164948/retrain_wide_1200/encoder.pt`
- `runs/sim_training/0916_164948/retrain_wide_1200/vae_best.pt`

`vae_last.pt` 保留 seed42 的第 400 轮模型，不是验证选择的推荐模型。
原始 FT frame、单位、清零/YZ 方向约定原样导出；冻结加载结果与训练编码器完全一致。
已有下游策略使用旧编码空间，需用新编码器重新准备数据并训练，当前未切换其配置。

## 协议与操作步骤

沿用原数据、960/120/120 固定划分及训练集归一化，使用独立入口。原数据、原模型和
已有下游配置均保留；新模型用于后续独立的下游重训，不能直接替换旧策略内的编码器。

保留 `SpongeVAE` 编码、ReLU 和共享逐帧解码结构，Dropout 从 0.1 改为 0。
仅从训练集拟合 PCA 初始编码和解码参数，并用逐通道平移保持 ReLU 初始激活。
编码器、解码器均参与梯度训练，无材料参数标签进入 VAE 损失。

固定协议：前 50 轮用均值编码、beta=0 预热；随后开始后验采样，100 轮内线性增加
KL 权重；总计 400 轮，batch32，Adam 学习率从 1e-4 余弦下降至 1e-5。
seed42 对比最终 beta=1e-5/3e-5/1e-4，只在第 150–400 轮按验证 MSE 选择，
不把 epoch0 当作训练结果。选定 beta 后 seed43/44 独立重训，全部选择锁定后才评估
历史测试集。三个种子均须超过均值模板至少 20%，固定/打乱编码误差均须增大至少 10%。
导出预先指定的 seed42，不能按测试集选种子。

在项目根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.sim_pretrain.experiments.retrain_wide_vae \
  --config configs/sim_pretrain/retrain_wide_1200.yaml
```

统一入口也可运行 `python -m scripts.sim_pretrain retrain-wide-vae --config configs/sim_pretrain/retrain_wide_1200.yaml`。

目录整理后，`source_run` 仍指向完整的 `runs/sim_training/<轮次>/`，其中
`dataset.h5` 通过相对链接读取 `runs/sim_data/<轮次>/dataset.h5`。
启动时按链接解析后的实际文件查找来源哈希，并与原检查点绑定的哈希比较；
2026-09-16 修复了链接路径与哈希清单键不一致导致的启动错误。
运行命令不变，复制源实验时须同时保留数据和训练两类目录，不修改原数据或来源哈希。

终端打印含时间层的实际输出目录。每次新建目录，不覆盖或续训历史实验。
输出 `run_config.json`、`manifest.json`、`selection.json`、`locked_selections.json`、
每候选/种子的训练日志、50 轮检查点、最优与最终模型，以及六通道测试重建图。
最终 `vae_best.pt` 是选定 seed42 的完整模型，`vae_last.pt` 是该次训练最后一轮，
`encoder.pt` 是通过验收的冻结编码器。`report.json` / `report.md` 保存完整对照。

`properties/` 中的原编码、新编码、PCA5 属性探针独立训练，仅用于报告，不影响 VAE
模型选择。测试集已在历史诊断中使用，本次不是新的盲测；仿真运动质量和真机迁移仍需
独立验证。关闭 Dropout、初始化、预热和 KL 权重一起改变，因此重训结果不用于声称
每项设置的独立因果效果。

加载新模型（将路径改为终端输出的实际目录）：

```python
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.sim_pretrain.experiments.retrain_wide_vae import load_model

encoder = FrozenSpongeEncoder("runs/sim_training/MMDD_HHMMSS/retrain_wide_1200/encoder.pt")
model, metadata = load_model("runs/sim_training/MMDD_HHMMSS/retrain_wide_1200/vae_best.pt")
# encoder.encode(raw_ft): [N,400,6] 原单位 FT -> [N,5] 均值编码。
```

新检查点使用独立格式，评估时使用上述加载器，不使用旧 `learning.load_checkpoint`。
测力 frame、单位和清零约定从源检查点/HDF5 继承并原样导出。

软件验证：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.sim_pretrain.test_joint_vae_retraining -v
```
