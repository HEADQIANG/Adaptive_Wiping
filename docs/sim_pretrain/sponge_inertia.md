# 海绵工具转动惯量修正

当前两份 `pretrain_paper*.yaml` 启用 `simulation.tool_inertia_profile: uniform_box_v1`。
保持已有海绵质量 0.03 kg、质心和 120 x 50 x 30 mm 几何不变，按均匀长方体
计算质心主惯量，将 `[0.01, 0.01, 0.01]` 改为
`[0.0000085, 0.00003825, 0.00004225] kg*m^2`。

计算读取现有箱体几何半尺寸，使用 `Ixx=m*(hy^2+hz^2)/3`，其他轴类推。
只修改环境加载时的 XML 副本，不修改 asserts 资源或机器人六个连杆的惯量。
碰撞几何、传感器原点/轴向、初始清零、输出 Y/Z 反转、探索速度、控制增益和
力矩限幅均保持不变。支架模式先修正海绵惯量，再按原逻辑合并背板惯量。

这是现有模型的几何估算，不是实物标定，也未凭空增加传感器、连接件的质量。
惯量定义于质心，不是测力原点；不应将平行轴项重复加到质心惯量上。
更改会影响完整闭环轨迹，不能承诺所有通道峰值都降低或已经匹配真机。

## 对照结果

在内存中固定 `mu=0.5, stiffness=100.25, width=0.16, gain=300` 对比新旧惯量，
不保存新训练轨迹。33 项相关回归测试通过，目标轨迹逐值一致。

| 指标 | legacy | uniform_box_v1 |
| --- | --- | --- |
| 2.01 s Tx / N*m | 0.026057 | 0.018734 |
| 2.01 s Ty / N*m | 0.044167 | 0.026345 |
| 2.00-2.10 s 最大绝对 Tz / N*m | 0.007499 | 0.011983 |
| 全程位置 RMS / mm | 8.297 | 10.012 |
| 最大姿态误差 / deg | 1.511 | 9.832 |
| 2 s 绝对 Fz / N | 1.561 | 1.367 |

Tx、Ty 起步尖峰减小，但 Tz 短时峰值和姿态跟踪变差。
该参数组合未通过接触运动验收，不可把惯量修正当作控制器已调好。
本次不额外调节增益、阻尼、前馈或轨迹来掩盖这些变化。

## 运行步骤

单次双窗口命令不变，在项目根目录执行：

```bash
MPLBACKEND=TkAgg /home/wp/miniconda3/envs/clean/bin/python -m scripts.sim_pretrain explore-once \
  --mu 0.5 --stiffness 100.25 --width 0.16 --gain 300
```

每次使用自动生成的新目录。参数可以手动修改，但比较惯量效果时必须固定其他参数。
完成后关闭两个窗口退出；加 `--close-after-run` 保存后自动退出。
只有当前配置自动启用新惯量；自定义配置须显式添加 `tool_inertia_profile: uniform_box_v1`。
缺少字段或设为 `legacy` 时保留旧惯量，供历史配置解释和独立对照使用。
完整配置随单条 NPZ 元数据保存。旧数据、模型和 sanity 不修改，不得混入新惯量数据；
正式批量采集须更换配置中的 `output_dir`，重新执行 sanity，通过后再 collect/train。

回归检查：

```bash
/home/wp/miniconda3/envs/clean/bin/python -m unittest \
  tests.sim_pretrain.test_sponge_inertia tests.sim_pretrain.test_pretraining \
  tests.sim_pretrain.test_direct_wrist tests.sim_pretrain.test_tool_mount \
  tests.sim_pretrain.test_sensor_axes tests.sim_pretrain.test_explore_once
```
