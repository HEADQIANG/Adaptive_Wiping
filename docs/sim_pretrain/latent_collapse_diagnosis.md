# 当前 VAE 潜变量塌缩分段诊断

## 2026-09-16 完成结果

当前证据支持：主要诱因是本数据及损失归约尺度下固定 `beta=0.06` 的 KL 约束过强，
使后验表示和解码器共同退化。原逐帧编码层保留了大量信息，原解码网络结构也具备
良好重建能力，不能简单归因于编码维数不足或解码器容量不足。

分段报告：`runs/sim_training/0916_161237/latent_collapse_diagnosis/report.json`。
原架构配对报告：`runs/sim_training/0916_162016/original_vae_beta_diagnosis/report.json`。
两个目录均有 `report.md`、曲线及独立检查点。

| 分段对照 | 测试归一化 MSE | 解释 |
| --- | ---: | --- |
| 原模型 | 0.00447345 | 接近均值模板 0.00444912 |
| 固定原最终均值编码，重拟合线性解码器 | 0.00401831 | 仅改善约 9.68%；对均值基线的配对改善区间跨零 |
| 固定原最终均值编码，重训原结构解码器 | 0.00348111–0.00356363 | 三个种子改善约 20–22%，仍明显受编码限制 |
| 固定原逐帧投影，重新 PCA5 压缩及线性解码 | 0.00037569 | 大量可重建信息在后验压缩之前仍存在 |
| 完整输入 PCA5 及线性解码 | 0.00031567 | 仅用训练集拟合，较均值模板改善 92.9% |
| PCA5 输入，重训原结构解码器 | 0.00034623–0.00043654 | 原解码结构在有效编码下可良好重建 |

上述同结构解码器重训使用标准化的固定编码、关闭 dropout 并改善优化设置，
不能把它单独当作原训练方式的复现。原均值和 logvar 共十维读出 MSE=0.00243218，
说明 logvar 还残留信息；这是十维辅助探针，不是现有五维导出接口的效果。

原后验均值的跨样本方差约 1.2e-7 至 2.2e-7，而后验采样噪声方差约 1。
解析最优的随机潜变量线性解码期望 MSE=0.00444912，也接近均值模板。

原架构配对复核保留原始 Dropout(0.1)，各 seed 内初始权重和随机序列相同，
只有 beta 不同。以下为固定 200 轮后的结果，三个 seed 全部列入范围：

| beta | 测试 MSE 范围 | 打乱编码后 MSE 范围 |
| --- | ---: | ---: |
| 0 | 0.00059935–0.00076137 | 0.00731924–0.00750382 |
| 0.0001 | 0.00067811–0.00080036 | 0.00706639–0.00730471 |
| 0.06 | 0.00448416–0.00450714 | 0.00448456–0.00450762 |

因此即使从有用的表示开始，原 beta 也会导致潜变量利用近乎消失；较小 beta 则保留
重建信息。低 beta 的最终误差仍高于这批有用初始模型，九个原架构实验的验证最优
均为 epoch0，所以不能称为已经完成模型修复，也不能将 beta=0 当作合格 VAE。

属性探针只更新独立预测器，验证集选择 MLP 后的测试 R2：

| 固定输入 | 摩擦 | stiffness_direct | width |
| --- | ---: | ---: | ---: |
| 原均值编码 | 0.2440 | 0.0444 | -0.0015 |
| 完整输入 PCA5 | 0.9707 | 0.7162 | 0.2902 |

原编码只有摩擦的 95% 配对 RMSE 改善区间严格大于零；其余两项本次未检出稳定改善。
PCA5 三项均检出改善。探针失败不证明不存在其他可解码信息，历史测试集也不是新盲测。

共完成 6 组冻结编码解码器实验、9 组线性解码 VAE 配对和 9 组原架构 VAE 配对，
另有线性及属性探针。原模型和数据哈希不变；只增加诊断脚本、测试与本文，不切换模型。

## 操作步骤

本工具分析已有 `pretrain_wide_1200_v1`，保持原 960/120/120 划分和原预处理，
仅生成独立离线诊断结果，不修改原模型、采集数据或部署配置。新输出自动增加时间目录。

在项目根目录，使用已有 clean 环境：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.sim_pretrain.experiments.diagnose_latent_collapse \
  --source runs/sim_training/pretrain_wide_1200_v1 \
  --output runs/sim_training/latent_collapse_diagnosis
```

使用终端打印的实际输出目录；不覆盖、不续跑已有诊断。CPU 单线程运行，逐阶段写
`status.json`、训练日志、检查点与结果。查看 `report.md` / `report.json`，以及
`reconstruction_errors.png`、`reconstruction_*.png`、`paired_beta.png`。

诊断顺序：

1. 冻结原均值编码，训练集标准化后拟合线性解码器；正则强度只按验证误差选择。
   标准化保留微小但非零变化，不把小方差编码截断为常数。
2. 对比原编码的均值和 logvar 共十维、冻结原逐帧投影再做 PCA5，以及完整输入 PCA5。
   所有 PCA、均值、缩放、回归仅在训练集拟合。十维结果仅为辅助诊断，不能当作五维模型。
3. 计算原后验均值方差/采样噪声方差，解析求解随机采样下最优线性解码器的期望误差。
4. 冻结原编码/PCA5，分别训练原结构的解码器，禁用 dropout，400 轮，Adam 1e-3，
   batch32，种子 42/43/44，按验证 MSE 保存最佳模型。这个对照同时改善输入尺度、
   优化方式和去除 dropout，不能单独归因于网络容量。
5. 相同 PCA 初始化、原编码器结构、全轨迹线性解码器，配对测试固定 beta=0/0.0001/0.06。
   全部采用随机潜变量采样，200 轮、Adam 1e-4、batch32；各 beta 使用相同初始权重、
   批次顺序和随机数种子。三个种子全部报告最佳和最终结果，最佳选择允许初始 epoch0。
   该对照只在此初始化和解码架构下隔离 beta 影响，不等于复现原模型的全部训练因果。
6. 冻结原编码与 PCA5，复用现有属性探针：ridge/MLP 只按验证集选取，报告测试集材料
   参数 R2 和 1000 次配对 bootstrap 误差改善区间。参数为仿真标签，不是实物标定量。

已有测试集曾被查看，本实验属于历史数据诊断，不能宣称新的盲测性能。全部原文件哈希
在实验前后核对，不产生用于自动替换的 `encoder.pt`。预测器训练仅作用于独立诊断模型。

软件检查（合成数据，不代表仿真模型效果）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.sim_pretrain.test_latent_collapse_diagnosis -v
```

## 原版网络的 beta 配对复核

第一阶段完成后，使用打印的实际诊断目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.sim_pretrain.experiments.diagnose_original_vae_beta \
  --diagnosis runs/sim_training/0916_161237/latent_collapse_diagnosis \
  --output runs/sim_training/original_vae_beta_diagnosis
```

该复核使用原 `SpongeVAE`，保留 ReLU、共享逐帧输出层及训练时 Dropout(0.1)。
以第一阶段训练集拟合的 PCA 编码器和同结构解码器构造有用的初始模型，坐标转换仅使用
训练集。每个 seed 内初始权重、批次顺序、后验采样及 dropout 随机数一致，仅改变固定
beta=0/0.0001/0.06，分别训练 200 轮、Adam1e-4、batch32。

报告保存最佳和最终模型；对比训练退化时看最后 200 轮的结果，不能把未训练的 epoch0
最佳检查点误写成训练改善。检查三种 beta 的初始验证预测完全相同，并核对输入哈希。
这能检验原架构内 beta 的条件因果影响，但仍不是从随机初始化重训或真机部署验证。
