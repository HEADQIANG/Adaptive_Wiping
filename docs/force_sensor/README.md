# 力传感器测试

纯传感器读取不需要机械臂 SDK、PyTorch 或 MuJoCo。使用现有 `.venv-kwr75` 或其他独立环境：

```bash
python -m pip install -r requirements/force_sensor.txt
python -m pip install --no-deps -e .
python -m scripts.force_sensor --help
python -m scripts.force_sensor read --help
```

确认串口对应正确传感器、没有其他进程占用之后，人工运行：

```bash
python -m scripts.force_sensor read --port /dev/ttyUSB0 --secs 60 \
  --csv runs/force_sensor/test_001/raw.csv
python -m scripts.force_sensor plot runs/force_sensor/test_001/raw.csv --mode raw --show
```

如需去零须先确认工具卸载、零点语义合适，再显式添加 `--tare`；不可在接触时去零。
实时绘图使用 `read --plot`，独立显示见 `live --help`，峰值分析见 `peaks --help`。
历史原始 CSV 和配套图片集中于 `archive/force_sensor/historical_logs/`。

卸载姿态标定用 `capture-unloaded`、`record-unloaded`、`fit-unloaded` 子命令。
这些工具可能读取机器人状态，需要 SDK 环境和人工无接触确认；不等同于纯串口测试。
新标定目录使用 `runs/force_sensor/`，不向归档的标定会话追加。
详细参数见 [传感器专题](kwr75_reader.md)、[卸载姿态采集](unloaded_pose_capture.md)。
