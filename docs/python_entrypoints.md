# Python 模块入口

2026-09-16：当前项目统一使用 `python -m scripts.<模块>`，在项目根目录执行。
根目录不再提供 `sim_pretrain.py`、`real_training.py` 等启动文件。
六阶段完整命令见 [README](../README.md)，输出路径规则见 [runs 输出步骤](run_outputs.md)。

## 入口与环境

| 用途 | 命令入口 |
| --- | --- |
| 仿真采集与训练 | `python -m scripts.sim_pretrain` |
| 单独传感器采集与绘图 | `python -m scripts.force_sensor` |
| 真机探索、示教与离线训练 | `python -m scripts.real_training` |
| 策略预检与部署 | `python -m scripts.real_deploy` |
| 机械臂基础控制 | `python -m scripts.robot_control` |
| 起点姿态记录 | `python -m scripts.robot_control.airbot_initial_pose` |

模块入口不切换解释器，也不启动机器人服务。仿真和离线训练使用 `clean`；
探索、示教和基础控制使用固定 `arm-sdk==5.2.2` 的 SDK 环境。
部署解释器须同时具备 PyTorch 和 SDK 5.2.2，本机使用 `clean`。
实际任务的依赖安装与服务步骤见 README；查看帮助不连接硬件。

```bash
cd /media/wp/新加卷/yuelk_project/claen_wipe/Adaptive_Wiping
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training --help
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control --help
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python \
  -m scripts.robot_control.airbot_initial_pose --help
```

从项目根目录使用模块入口无需可编辑安装即可找到源码；实际任务仍需对应依赖。
`env -u PYTHONPATH` 是本机环境隔离措施，入口不会自动清理环境变量。
从其他目录运行时，先切换回项目根目录，避免配置和输出相对路径产生歧义。
不要把包内 `scripts/a/b.py` 当作根目录启动文件，相对导入可能失败。

## 子命令与验证

子命令继续跟在模块名之后。以下只展示帮助，不执行部署或采集：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_deploy preflight --help
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python \
  -m scripts.real_training demonstrate --mode manual --help
```

入口回归覆盖未安装项目时的模块帮助、非法命令拒绝、延迟导入、子命令参数和返回码传递。
测试只使用替身命令，不连接设备：

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m unittest tests.shared.test_module_entrypoints -v
```

现场启动、按键与退出规则以 [README](../README.md) 及对应模式文档为准，
模块入口不改变 `--execute` 或设备控制权限的含义。
