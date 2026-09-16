# 部署日志中的高度增量差值

从项目根目录离线运行；不连接机械臂，不覆盖原日志或已有图表：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_deploy.plot_delta_h_error \
  --events runs/real_deploy/manual_tared_run/0916_141944/events.jsonl \
  --output runs/real_deploy/manual_tared_run/0916_141944/delta_h_error
```

输出 `delta_h_error.png`、逐次预测的 `delta_h_error.csv` 及指标和输入哈希 `summary.json`。
重绘需指定未存在的目录，例如 `delta_h_error_v2`。部署启动命令和自动生成的两张力图不变。

差值定义为 `预测Δh − [Z(实测锚点时刻+0.4秒) − Z(实测锚点时刻)]`。
使用模型当时的实测 Z 锚点和 SDK 单调时间戳，未来 Z 从相邻实测位置线性插值得到；
不把计划时间当作实测时间，不外推，不跨越超过 20 ms 的位置采样间隙。
图的横轴保留日志中按 s 后的计划预测时刻，CSV 同时保留实际锚点/端点时刻。

本次为 25 个预测点，首个预测在第 2 秒，最后一个在第 11.6 秒。
实测运动由模型自己下发的指令驱动，因此这是闭环预测与执行的差异，不能冒充独立示教数据上的泛化误差。

当前训练取 0.4、0.8、…、10.0 秒的示教点，五帧历史的第一个标签为 `Z(2.4)-Z(2.0)`；
0～2 秒的大幅下降不在高度网络的预测标签中，但该阶段的力进入第一组历史输入。
当前部署前 2 秒静止采样，之后执行完整 XY 轨迹，Z 由反馈增量生成；并不重放示教的起始下压轨迹。
