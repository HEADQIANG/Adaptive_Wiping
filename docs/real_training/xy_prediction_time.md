# X、Y 预测轨迹时间曲线

在项目根目录执行，使用 `clean` 环境。只读取已有预测数据，不训练、不连接机器人，
不修改原有 `predictions.png`、检查点或评估数据。

```bash
env -u PYTHONPATH /home/wp/miniconda3/envs/clean/bin/python \
  -m scripts.real_training.tools.plot_xy_prediction_time \
  --predictions runs/real_training/training_programmed_wide1200_v1/final/predictions.npz \
  --output-dir runs/real_training/training_programmed_wide1200_v1/final
```

输出 `predictions_x_time.png` 和 `predictions_y_time.png`，重复执行会更新这两张图。
每张图按行排列 Episode 1 至 8，分别对比示教、网络预测和训练均值。
横轴是原训练协议的 25 个采样时刻：0.4、0.8、…、10 秒，不是从零开始重新编号。
纵轴为基座坐标系绝对位置，单位从米换算为毫米，没有减去起点；每张图的各示教共享坐标范围。
均值采用绿色虚线，预测采用橙色实线，以便识别两者重合。

这些是包含末状态补齐段的训练集离线曲线，不是独立测试或真机闭环执行结果。
运行上述命令后，检查两张图的 8 个子图、坐标单位和图例是否完整。
