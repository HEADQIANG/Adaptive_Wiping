# Contact Load And Cartesian Control Experiment

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

This is an isolated simulation experiment. Production configuration, robot assets,
torque limits, contact mapping, exploration targets and acceptance thresholds are
unchanged. No training data, hardware control or production-gate override is allowed.

## Protocol

1. `contact_breakaway.py` copies the current sponge and table into a two-slide-joint
   fixture. Orientation and X are mechanically constrained. Compare native contact,
   only friction changed, and the full current mapping at normal loads 1 and 3 N.
   The force ramp follows 2 s settling and lasts at most 3 s. Detect whole-tool slip
   at >5 mm/s and >0.25 mm for 50 ms; continue until reaching 50 mm/s for 50 ms
   or the ramp ends. Clone all compiled MuJoCo option fields from the robot scene.
   Stop on >30 mm penetration. Unreachable loads
   are reported, not forced through the sponge. This fixture is not robot acceptance.
2. `contact_control_experiment.py` uses unchanged production IK to prepare, then
   switches to a candidate only for exploration and unloading. Goals stay absolute
   world poses. Cartesian twist feedforward is a difference of nominal pose goals,
   never a difference of achieved poses. There is no force target or load-triggered
   lifting in the exploration controller.
3. Compare stock OSC150 plus nominal twist FF and three preset physical Cartesian
   impedance controllers. The latter use robosuite's OSC state/goal interface and
   orientation-error helper with tau=J.T@(K*pose_error+D*(twist_ref-twist))+bias.
   Physical gains: XY K=2000/6000/12000 N/m, D=45/80/120 Ns/m; all use Z K=150 N/m,
   D=20 Ns/m. Rotation K=80/120/120 Nm/rad, D=2/2.5/2.5 Nms/rad. OSC gains are
   inertia-scaled and are not the same physical units. No integration/windup term.
4. Candidate free-space must pass the existing 1 mm RMS / 1 degree gate. Then run
   mu=[0,.9,3.5] crossed with k=[.5,1000], width=.02. A failed free-space candidate
   receives only one explicitly diagnostic mu=.9 contact. A candidate passing all
   six contacts is frozen for all 27 grid cases and a sensor check. No per-material
   gain selection. Stop at the first full passing candidate.
5. Record normal loads as well as motion. Passing geometric contact alone does
   not establish meaningful pressing or calibrated sponge response. Even a full
   motion pass is not automatic approval to collect or replace production.

The experiment retains the aggregate link6 inertia and original sponge inertia.
The isolated assay also reports the uniform-box inertia for comparison but never
substitutes it. Any inertia repair or change of exploration protocol needs a
separate, explicitly documented decision.

## Commands

From the project root, using new output directories:

```bash
conda activate clean
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -p 'test_cartesian_experiment.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.contact_breakaway --output runs/sim_pretrain/contact_breakaway_v2
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.contact_control_experiment --output runs/sim_pretrain/contact_cartesian_v2 --gif
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.sim_pretrain.experiments.analyze_contact_experiment --rig runs/sim_pretrain/contact_breakaway_v2 --robot runs/sim_pretrain/contact_cartesian_v2
```

Use `/home/wp/miniconda3/envs/clean/bin/python` if activation is unavailable.
`--candidate baseline|osc_ff|cart_2000|cart_6000|cart_12000` runs one variant;
default `all` follows the ordered list above. Exit 2 means no full candidate pass.
`--gif` exports actual mu=.9 rollout playback using the existing renderer. Omit
it when EGL is unavailable. Reports are written incrementally; partial traces
and exceptions are retained. A completed command is not an acceptance pass.

`archive/sim_pretrain/contact_breakaway_v1` is an initial diagnostic using different solver
options and only a 5 mm/s onset criterion. Do not use it for the robot comparison;
v2 copies the robot's Euler/elliptic solver options and also measures 50 mm/s.
Soft-contact slip onset is operationally defined, not an ideal static-friction
coefficient measurement; do not assume onset force equals mu times normal load.

`archive/sim_pretrain/contact_cartesian_v1` retains the failed initial interface run: the
baseline completed, but candidates received an empty arm action because the
experimental composite used arm-keyed grippers instead of controller-part-keyed
grippers. v2 reuses the original composite's gripper mapping. The regression
test now issues a real environment command after preparation/reset and checks
the six-dimensional arm action slice. v1 candidate errors are not control results.

The analysis command requires both runs to finish and refuses an existing
`analysis/` directory. It is for the default `all` run with the current
`direct_wrist_v3` config and its saved production mu=.9 baseline, not a standalone
`--candidate` output. It re-scores saved robot traces, verifies nominal twist
against the frozen 400-step exploration, checks source/configuration hashes and
reproduction of the production mu=.9 baseline, and writes figures and GIF pixel
checks. It also reports 50 ms speed-confirmation load ranges, not steady drag.

Quasistatic budgets transfer each mapped high-stiffness rig ramp's measured
WORLD wrench to the FT site at recorded mu=.9 robot postures (2, 3 and 4 s).
Moments are shifted from FT to TCP before computing `tau=gravity-J.T@wrench`;
the mapping is cross-checked with MuJoCo `mj_applyFT`. These are hypothetical
loads held in the same world axes, NOT a reconstruction of tilted contact.
Velocity and acceleration are zero. Passing a sampled budget does not establish
dynamic stability, inertial margin, actual contact distribution or reachability.

Regression tests cover the physical impedance equation, stock OSC plus nominal
twist feedforward, actual environment commands, reset, reference independence,
preserved torque limits/fixture solver options and FT-to-TCP wrench balance.

## Results: 2026-09-08

已完成预设实验，但没有候选通过完整接触验收，不能切换生产控制器或采集。
有效结果为 `archive/sim_pretrain/contact_breakaway_v2/report.json` 与
`archive/sim_pretrain/contact_cartesian_v2/report.json`，分析与图像在后者的 `analysis/`。
v1 保留故障记录，不覆盖。未改生产配置、工具/机器人资产、惯量、材料映射、
探索速度、力矩限幅或验收阈值；普通可视化命令仍运行生产 IK+FF 基线。
完整37项回归测试通过，包括4项本次实验测试；测试通过不代表接触验收通过。

### 隔离接触

20 组中14组完成加载和速度测试，6组 `mapped/k=.5` 在加载阶段超过30 mm
穿透边界而停止。原生接触有效滑动摩擦约.03；仅把海绵设为mu0仍受桌面
摩擦混合影响，不等于当前priority映射下的有效mu0。

下表为当前映射、k1000/width.02。1/3 N是静态加载目标，滑动时实测法向力
可以波动。起滑为>5 mm/s且位移>.25 mm保持50 ms的工程判据。

| mu | 目标法向力 N | 起滑施力 N | 达到50 mm/s时施力 N | 随后50 ms法向力范围 N | 该窗平均阻力 N | FT最大X力矩 Nm |
|---|---|---|---|---|---|---|
| .9 | 1 | .918 | .981 | 1.005-1.085 | .950 | .0294 |
| .9 | 3 | 2.723 | 2.808 | 3.082-3.141 | 2.800 | .0825 |
| 3.5 | 1 | 3.575 | 3.848 | 0-1.337 | 3.344 | .1447 |
| 3.5 | 3 | 9.825 | 11.218 | 0-3.417 | 10.316 | .3695 |

这证明工具并非完全不能整体移动，但**高摩擦固定姿态夹具也出现法向力归零**。
表中50 mm/s是上升力斜坡中的短时速度，不是稳定恒载擦拭通过证据。
仅改摩擦、保留原生软接触时，mu3.5也存在载荷波动，不能只怪当前刚度映射。
当前映射k1000静态1/3 N时穿透约4.148/6.340 mm；k.5加载失败不能被动态
机器人轨迹中瞬时出现1-3 N反驳，两者的加载历史、运动约束和阻尼响应不同。

### 机器人验收

共30条完整机器人轨迹：5条自由空间、25条接触。25条均卸载，无力矩饱和。
代表点为mu[0,.9,3.5]与k[.5,1000]组合，width固定.02。

| 方案 | 自由空间RMS mm | 最大姿态 度 | 代表接触通过 |
|---|---|---|---|
| baseline IK+FF | .452535 | .542584 | 1/6 |
| osc_ff | 1.984012 | 1.636824 | 自由空间失败；诊断0/1 |
| cart_2000 | .245250 | .256388 | 3/6 |
| cart_6000 | .244491 | .240820 | 2/6 |
| cart_12000 | .246691 | .247987 | 2/6 |

mu=.9/k1000/width.02单点对比：

| 方案 | 正向/反向 mm | 最大姿态 度 | 正向/反向接触比例 | 结果 |
|---|---|---|---|---|
| baseline | 4.787 / -2.281 | 7.843 | 100% / 100% | 行程、误差、姿态失败 |
| osc_ff | 1.044 / -.190 | .053 | 100% / 100% | 行程、Y误差失败 |
| cart_2000 | 48.930 / 47.820 | .181 | 83% / 89% | 接触连续性失败 |
| cart_6000 | 49.869 / 49.697 | .179 | 78% / 87% | 接触连续性失败 |
| cart_12000 | 50.119 / 50.121 | .204 | 73% / 69% | 接触连续性失败 |

cart_6000的mu3.5/k1000行程48.651/46.490 mm、姿态.350度，也通过全部
位移/姿态指标，但接触仅64%/81%，最长正向断触60 ms，因此失败。
所有候选至少有一个代表点失败，按预设流程未启动27点扩展或候选传感器专项。

cart_6000在mu.9的两段平均法向力2.233/2.305 N，高于基线1.712/1.509 N，
并非平均载荷接近零获得滑动；但反复断触依然不合格。不能用平均载荷掩盖
瞬时离面。继续把横向K加到12000虽然减小Y误差，却恶化接触连续性。

### 核查与下一步

`analysis/verification.json` 已重算30条轨迹验收，核对名义前馈完全遵循
固定探索指令，生产mu.9位置/FT/关节轨迹逐值复现，源和配置哈希仍匹配。
60组采样姿态/夹具载荷准静态核算均未超限，最高占限幅35.639%，FT/TCP
力矩映射与MuJoCo广义力一致。这不是动态可行性证明，也不是厂家力矩认证。
五份GIF各81帧，排除标题后像素有变化；比较图与cart_6000回放关键帧已检查。

保留当前实验结论：横向/姿态独立增益能显著改善整体运动，但仍需解决法向
振荡和断触。下一轮建议以cart_6000作为诊断参考，冻结XY/姿态与探索动作，
先验证法向阻尼和接触求解响应，不再继续堆高横向增益。若改恒力控制，将
改变原固定Z探索协议，需单独确定与论文复现的边界，不应静默替换。

另有未解决的物理真实性问题：原海绵30 g，却继承三个.01 kg m2主惯量；
同质量120x50x30 mm均匀盒子的估计为[8.5e-6,3.825e-5,4.225e-5] kg m2。
本轮只记录异常，没有把估算值当成标定值替换，也没有证明这是断触主因。
link6仍保留拆除可视壳体之前的聚合惯量。
