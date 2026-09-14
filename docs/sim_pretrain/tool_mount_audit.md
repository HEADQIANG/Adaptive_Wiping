# AIRBOT 末端安装坐标核对

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

日期：2026-09-08。用户目标：机械臂末端法兰 -> 工具连接件/背板 -> 海绵 -> 桌面。

后续更新：用户已选择先做几何自洽的工程仿真，`bridge_v1` 安装结构已实现，
详见 [tool_mount_bridge.md](tool_mount_bridge.md)。下面的“尚未修改”描述的是此前
真实法兰审计当时的状态；真实法兰尺寸仍未获得，工程安装不得冒称厂家标定。

## 结论与实施边界

当前普通版 DISCOVERSE 来源并未提供独立、可核验的裸法兰坐标。
不能把现有 `endpoint`、适配层 `right_hand` 或可见夹爪壳体底面直接称为真实法兰。
因此本次完成来源及几何核对，**未修改生产模型安装偏移、质量、工具或控制器**。
还需要用户实际 AIRBOT 硬件版本和对应末端法兰尺寸图/URDF/CAD，才能实施有真实来源的安装关系。

仅增加一根任意长度的可视连接杆可以填补画面空隙，但不是对真实法兰的核对，
也不等价于完成连接件质量、碰撞、传感器坐标和 TCP 的适配。本次没有这样处理。

## 当前普通版来源

以下是当时审计使用的上游来源路径，保留作证据，不是当前模型加载入口。
现用模型及惯量参考已集中到 `asserts/`，见 [模型资源说明](models.md)。
根目录 DISCOVERSE 已移除；以下 `DISCOVERSE/` 前缀可通过历史路径解析命令定位归档，见 [移除说明](discoverse_removal.md)。
以下旧 `robosuite/` 前缀现映射到 `scripts/sim_pretrain/robosuite/`，见 [目录说明](robosuite_location.md)。

- `DISCOVERSE/models/mjcf/manipulator/airbot_play/airbot_play.xml`
- `DISCOVERSE/models/urdf/airbot_play_v3_gripper.urdf`
- `DISCOVERSE/models/meshes/airbot_play/link6.obj`
- 适配文件 `robosuite/robosuite/models/assets/robots/airbot_play/robot.xml`

普通版 MJCF 的 link6 下同时存在 link6 外观、相机支架、左右手指 body 和 endpoint。
当前 robosuite 适配没有移植两个手指 body，但保留了 link6 和相机支架网格。
所以“手指已经移除”不代表剩余模型已经是单独裸法兰模型。

普通版 MJCF 的 endpoint：`pos=[0,0,0.015]`，`euler=[0,-pi/2,0]`。
配套 v3 gripper URDF 的 `joint_custom_end`：`xyz=[0,0,0.02]`，`rpy=[0,-pi/2,0]`。
这些是夹爪端参考定义，且位置还存在 5 mm 差异；不能无依据地选一个作法兰。

当前适配的 `right_hand` 只采用了位置 `[0,0,0.015]`，姿态是单位旋转。
它不是原 endpoint 完整位姿变换的复制，也没有独立的厂家法兰定位证据。

## 当前几何为何悬空

在 link6 局部坐标中，目前固定链为：

```text
link6
  right_hand: z = +0.015 m，单位旋转
    wiping_gripper: 再平移 z = +0.015 m，并绕 z 旋转 -90 度
      海绵外观中心: link6 z = +0.030 m
      海绵背面:     link6 z = +0.015 m
      海绵前表面/TCP: link6 z = +0.045 m
```

海绵外观是 120×50×30 mm 的盒子；具体接触几何还有小球等局部细节，
不能将外观盒表面当作所有碰撞几何的精确包络。

通过 MuJoCo 载入 link6 OBJ，并把编译时重心/主轴变换还原到原始 link6 坐标后，
7143 个顶点的包围范围为：

```text
x: [-0.0285,  +0.0285] m
y: [-0.0690,  +0.0690] m
z: [-0.19595, -0.05140] m
```

已核对 robosuite 与普通版 DISCOVERSE 的 link6 OBJ SHA256 相同。
网格最前方与海绵背面在轴向上的间距为 `0.015-(-0.0514)=0.0664 m`，约 66.4 mm。
这个数值解释了可见间隙，但 **网格包围盒极值不能证明安装平面或法兰中心**。

绿色 link6 碰撞盒的中心 z=-0.07 m，半厚度 0.015 m。它同样不是法兰定位依据。

## 为什么不能直接借用 95.5 mm

另一套文件确实提供明确的连接链：

- `DISCOVERSE/models/mjcf/manipulator/airbot_play_force/_play_force.urdf`
- `DISCOVERSE/models/mjcf/manipulator/airbot_play_force/airbot_play_peg.xml`

其中 `arm_connect_joint` 从 link6 到 `eef_connect_base_link` 的固定平移为
`[0,0,0.0955]` m，URDF 分别提供 link6、连接件、g2_base_link 的几何及惯性。

但是该版本 joint6 相对 link5 的平移为零；当前普通版为 `[0,0.23645,0]`。
link6 的质量聚合、连杆网格、其余连杆参数也不相同。这是另一套建模约定，
不能把 0.0955 m 直接应用在当前 link6 原点上，也不能只移植它的几何而保留
当前聚合惯量后声称已经获得一致的裸法兰模型。

## 获得实物来源后的步骤

1. 确认硬件版本及工具安装方案，定位真实裸法兰相对于所用 link6 坐标的完整变换。
2. 识别原模型已包含的夹爪/连接件质量和几何，避免既保留旧总成又重复添加新总成。
3. 明确定义 flange、adapter/backplate、sponge 的固定安装链；将传感器放在与实物一致的位置。
4. 依据尺寸和质量创建一致的外观、碰撞及惯性，设置擦拭面 TCP；不通过视觉平移掩盖坐标错误。
5. 检查相邻构件贴合、穿插、自碰撞、工具对桌接触以及中心/偏心载荷的六维传感器响应。
6. 更换输出目录，重新跑自由空间和接触验收；安装几何改变后旧轨迹、门禁和数据不能直接沿用。

## 当前查看命令

以下仍显示未修改的当前模型，用于对照，不表示安装关系已修好：

```bash
conda activate clean
python -m scripts.sim_pretrain.experiments.visualize_wiping --mu 0 --stiffness 1000 --width 0.02
```

本次只新增核对文档，无生产代码改动，因此没有以运行旧测试冒充新安装验证。
