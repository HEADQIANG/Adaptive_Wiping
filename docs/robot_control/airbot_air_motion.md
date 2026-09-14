# AIRBOT 空中探索运动验证

2026-09-12：项目 XYZ 工作空间边界已移除，旧边界字段不再生效。
下方历史记录中的 XYZ 范围不代表当前软件限制；空中起点及运行净空检查仍保留，
详见 [边界移除说明](workspace_bounds_removed.md)。

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

用于当前选择：**工具全程不接触桌面或障碍物**，当前轨迹下移 10 mm（0.005 m/s，2 s）、沿 +Y
移动 50 mm 再沿 -Y 返回，姿态固定。无六维力读取，不打开传感器串口，不产生零力占位数据。
默认 100 Hz、4 秒探索，之后另用 2 秒上退 10 mm 回到空中起点。
`--time-scale 5` 只把时间放慢 5 倍，不缩短运动距离。

新增入口为 `--mode air`，默认使用 `configs/robot_control/airbot_air_motion.json`。
原来的 `force-guarded` 和 `contact-no-ft` 配置、确认要求和接触检查保持不变。
此模式不是绕过接触检查的方法，不允许从此前离桌约 1 mm 的起点执行。
软件尚未进行真机运动验收；本次实施不启动机械臂运动。

## 1. 重新记录空中起点

保持已正常运行的 SDK 5.2.2 服务，确认其启动参数有 `--no-return`。
按现场规程检查急停、负载、安全支撑及全臂、工具和线缆的活动空间。然后在项目根目录运行：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose teach \
  --execute --output runs/robot_control/real_robot/air_motion_start_001.json \
  --label air_motion_start --tool-note "28 mm sponge; elevated non-contact motion start"
```

安全支撑机械臂后输入 `DRAG`，待显示重力补偿已激活，再手动移到空中姿态。
保留待验证的工具朝向，避开关节极限/奇异姿态。确认沿整个 50 mm 横移范围，
工具最低部位对下方最高障碍物的起始净空**至少 50 mm**，并且侧面和整臂扫过的区域也无碰撞。
不要只根据 SDK Z 坐标或海绵厚度推算净空，需考虑实际安装件、工具尺寸和姿态误差。

静止且已安全支撑后按 `S` 或 `s`，无需回车，检查通过即保存并自动退出。
不再需要 `Q`；退出 idle 不保证保持位置，按键前就要托稳机械臂；
脚本会重新检查起点，偏差超限则拒绝运动，不自动回起点或自动从接触位置抬升。
保留旧 `contact_motion_start_001.json`，不覆盖它，也不要仅修改旧文件标签冒充已经抬高。
若重录空中起点，选新文件名并同步修改空中配置的 `initial_pose_file`。

## 2. 确认空中配置

若退出拖动后产生小幅位置变化，且现场确认机械臂已稳定、不会继续下沉，可只读记录新起点：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.robot_control.airbot_initial_pose capture-idle \
  --output runs/robot_control/real_robot/air_motion_start_idle_001.json \
  --label air_motion_start --tool-note "28 mm sponge; settled idle air start"
```

此命令不获取控制权、不切换模式、不发送运动；要求整个采样期间为 idle 且通过原静止检查，
不覆盖已有文件。采样通过不保证将来保持位置，也不证明 SDK 数据有硬件时间戳。
确认新轨迹仍在已批准空间内、起始净空仍至少 50 mm 后，才将 `initial_pose_file` 指向新文件，
执行下文只读检查。起点降低时不能直接沿用原净空下界，需现场重新核对。

2026-09-10 已保存上述 idle 新起点，操作人再次确认当前位置沿整个横移路径的工具最低处
净空仍至少 50 mm。运行配置已切换至 `air_motion_start_idle_001.json`，确认编号更新为
`air-motion-20260910-idle-start-confirmed`，净空下界仍为 0.05 m。
新轨迹 401 个目标点均在既有工作空间内，旧起点文件保留；运动前仍需实时检查。

`configs/robot_control/airbot_air_motion.json` 已使用用户提供的机械臂序列号和 `expected_eef_type="NULL"`。
空中起点文件的标签必须为 `air_motion_start`，序列号和末端类型必须与在线状态匹配。
NULL 末端复用已加入的六轴 servo 请求，不查询不存在的夹爪、不发送夹爪字段。

需要填写/核实的字段：

- `air_start_clearance_m`：以米计、考虑工具几何/测量/姿态误差后的起始净空下界，必须至少 0.050。
  下界应覆盖整个横移路径下方的最高障碍物，而非只测起点正下方。
- `air_path_clear_confirmed`：现场确认整个机器人、工具和线缆的运动空间无遮挡后设 true。
- `table_normal_sdk` / `slide_direction_sdk`：SDK 坐标中的上法向及横移正方向，模板为 +Z/+Y，
  需要与现场核对；不是沿工具自身局部坐标移动。
- `joint_min_rad/max_rad`：六轴保守关节界限；完整运动空间由现场检查，无项目 XYZ 边界。
- `joint_current_limits`：六轴经厂家规定/现场负载确认的电流限制；默认关节限速为 0.2 rad/s。
  软件不虚构这些参数，空中运行也不等于可取消电流或关节限制。
- `calibration_id` / `sdk_frame_and_tool_note`：此次坐标和净空检查记录编号、实际工具/SDK 末端说明。
- `calibration_confirmed`：上述现场检查完成后设 true；此处不要求力传感器标定。

空中配置不需要接触间隙、海绵压缩量、接触几何误差或力阈值。
若配置仍带 `initial_gap_m`，脚本会拒绝，防止把旧接触配置误用为空中配置。
没有实际确认的 `null`/`false` 不要仅为通过程序而填写。

2026-09-10 现场操作人已确认整个空中运动路径无碰撞，起点工具最低处对下方障碍物
净空至少 50 mm。配置据此设置 `air_path_clear_confirmed=true`、
`air_start_clearance_m=0.05`（操作人声明的下界，非传感器测量）。此确认不代表其他
标定和限制已完成。当时 `calibration_confirmed` 保持 false，工作空间待填写。

操作人随后确认所述全部现场条件安全，配置据此记录确认编号
`air-motion-20260910-operator-confirmed`、SDK +Z 向上/+Y 横移说明，并将
`calibration_confirmed` 设为 true。已确认的局部 SDK 末端边界为
`workspace_min_m=[0.14955, -0.00705, 0.24650]`、
`workspace_max_m=[0.16955, 0.06295, 0.28650]`（米），来源为原空中起点轨迹外扩约 10 mm。
这些是现场操作人确认，不是传感器测量，也不覆盖实时起点检查。
重新记录起点或改变工具、环境后必须重新核对边界和净空。运行命令不变，先执行只读检查；
若出现 `Start mismatch`，不得跳过检查或手改原记录，应重新处理实际起点。

操作人随后选择使用默认电流参数：`joint_current_limits=[8, 8, 8, 8, 8, 8]`，
与本机 `arm-sdk==5.2.2` 的 `ArmControlOptions().eff` 及
[官方示例](https://docs.airbots.online/airbot-play/sdk/api/examples.html)一致。
这是采用 SDK 默认阈值，不代表对当前负载完成了独立安全验证；SDK 接受的 0 至 20
数值范围也不是电机额定安全范围。不修改 MIT `torque` 参数，不提高关节限速。

关节范围已按操作人提供的参数截图换算为弧度写入配置：J1 [-180, 120]、
J2 [-170, 10]、J3 [-5, 180]、J4 [-172.5, 172.5]、J5 [-105, 105]、
J6 [-172, 172]（此处单位为度）。使用前须核实截图适用于当前型号及 SDK 关节零位定义。
截图最大速度为 J1-J3 180 度/秒、J4-J6 360 度/秒，不作为首次验证速度；
仍保留 `joint_speed_limit_rad_s=0.2`（约 11.46 度/秒）。截图未提供电流限制。
运行命令不变，仍须先完成配置并通过只读 `check --mode air`。

50 mm 起始净空扣除 10 mm 名义下移后还剩 40 mm；运行中根据 SDK 位移和声明的净空下界
估计剩余净空，若小于 20 mm 则中止。位置跟踪误差阈值保留为 5 mm，起点位置容差为 1 mm。
这些是工程检查值，不是机器人安全额定值。估计值不是距离传感器反馈或碰撞检测，
无法感知突然进入的物体、SDK 不可见的工具变形或未计入的安装偏差；独立急停和现场监护仍必需。

## 3. 预览、只读检查和运行

离线预览，不连接任何硬件，也不要求起点/配置已填好：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore preview --mode air
```

记录空中起点、填写配置后，只读检查：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore check --mode air
```

通过现场检查后可先低速验证相同的完整距离：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --mode air --execute --time-scale 5 --output runs/robot_control/real_robot/air_motion_slow_001.jsonl
```

现场批准原速度时使用以下命令，目标时间与仿真一致：

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m scripts.real_training explore run \
  --mode air --execute --time-scale 1 --output runs/robot_control/real_robot/air_motion_001.jsonl
```

必须在交互终端输入独立确认词 `AIR-MOTION`。`EXPLORE` 或 `CONTACT-NO-FT` 不会启用空中动作。
四秒探索结束时还在下移 10 mm 的高度，随后进行单独的上退阶段；回到空中起点后 servo 仍激活，
安排安全支撑并输入 `IDLE` 才正常释放控制。不要关终端、强杀或断电来替代正常交接。

保留电机错误、控制权、关节位置/速度、姿态和实时节拍检查。
异常、Ctrl+C、SIGTERM 或终端 EOF 请求软件停止，不自动回零、抢占控制权或盲目回撤。
软件停止不保证防坠或立即生效；现场物理停止和支撑流程始终优先。

日志标为 `mode=air`、`contact_expected=false`、`force_monitoring=false`，力数据为 null，
压缩量估计为 null；另记录 `estimated_air_clearance_m`。它不是接触探索训练数据，
固定 `encoder_ready=false`。`nominal_protocol_match` 仅说明目标轨迹/时间，不说明接触物理一致。

## 4. 无硬件测试

```bash
env -u PYTHONPATH /home/wp/airbot-venv-5.2/bin/python -m unittest discover -s tests -t . -p 'test_airbot*.py' -v
/home/wp/miniconda3/envs/clean/bin/python -m unittest discover -s tests -v
```

测试使用假机械臂，不发 CAN、不开串口。原仿真源码和历史训练产物保持不变。

2026-09-10 验证：SDK 环境 60 项 AIRBOT 测试全部通过；`clean` 全量 230 项，
228 项通过、2 项 SDK 类型测试跳过（均已在 SDK 环境通过），耗时 82.278 s。
测试覆盖完整位移/时间、无传感器导入、空中确认词、拒绝接触起点、净空边界、模式隔离和停止。
离线预览及未确认配置在连接前拒绝执行已验证。此前测试未执行真机动作。
之后已保存空中起点 `archive/robot_control/real_robot/air_motion_start_001.json`；退出拖动后的只读检查
曾发现约 1.33 mm 起点偏差，超过 1 mm 门限，启动前仍需核对，不能据此认定起点检查已通过。
