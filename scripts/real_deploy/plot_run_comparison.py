"""Offline comparison of completed runtime-origin and fixed-setup deployments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from scripts.shared.common import file_digest
from scripts.shared.real_preprocessing import CausalFTFilter, finite_array
from scripts.shared.run_paths import new_output
from scripts.real_deploy.wiping_frame import WipingFrame


COLORS = ("#087e8b", "#bd4937")
AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
PREDICTION_TICKS = np.arange(200, 1000, 40)
STATIONARY_INITIALIZATION = "hold_pose_2s_then_policy_10s"
PHASES = (("initialization", 0, 200), ("first_feedback", 200, 240),
          ("later_feedback", 240, 1000))
PLOTS = ("position_xyz.png", "position_relative_xyz.png", "force.png",
         "torque.png", "wrench_raw.png", "height_feedback.png")


def close_enough(actual, expected, label, *, atol=1e-10):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise ValueError(f"{label} mismatch")


def array(value, shape, label):
    result = finite_array(value, label)
    if result.shape != shape:
        raise ValueError(f"Expected {label} shape {shape}, got {result.shape}")
    return result


def load_run(path, *, policy_only=False):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        rows = []
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if policy_only and rows[-1].get("event") == "policy_complete":
                    break

    def one(event):
        matches = [row for row in rows if row.get("event") == event]
        if len(matches) != 1:
            raise ValueError(f"Expected one {event}")
        return matches[0]

    header = one("session_start")
    if header.get("shadow") is not False:
        raise ValueError("Expected real motion session, not shadow")
    if any(row.get("event") in ("fault", "aborted", "stop_error") for row in rows):
        raise ValueError("Expected completed deployment without faults")
    one("policy_complete")
    if not policy_only:
        one("session_complete")
    spec = header["spec"]
    frame = WipingFrame.from_settings(spec)
    if "wiping_frame" in spec and spec["wiping_frame"] != frame.metadata():
        raise ValueError("Session wiping frame mismatch")
    final_tick = 1000
    if spec.get("workflow") == "runtime_start_h_z_s_g_v1":
        kind = "manual"
        pose_key, receive_key = "state", "causal_sensor_receive_perf_s"
        tare = one("tare_complete")
        baseline = tare["tare_bias_si"]
        reference = one("reference_captured")
        initialization = reference["initialization"]
        if initialization == STATIONARY_INITIALIZATION:
            final_tick = 1200
        elif initialization != "hold_z_first_2s_no_automatic_press":
            raise ValueError("Unsupported runtime initialization")
        initial = reference["runtime_reference_pose"]["sdk_end_position_m"]
        if "wiping_frame" in reference and reference["wiping_frame"] != frame.metadata():
            raise ValueError("Captured wiping frame mismatch")
        if frame.mode == "vertical" and (initialization != STATIONARY_INITIALIZATION
                                         or reference.get("wiping_frame") != frame.metadata()):
            raise ValueError("Vertical logs require stationary runtime initialization and captured wiping frame")
    elif spec.get("mode") == "fixed_setup_tared_v1":
        kind = "fixed"
        pose_key, receive_key = "pose", "sensor_receive_perf_s"
        tare = one("baseline_complete")
        baseline = tare["bias_si"]
        initialization = spec["initialization"]
        if initialization != "nominal_10mm_during_first_2s":
            raise ValueError("Unsupported fixed-setup initialization")
        initial = None
    else:
        raise ValueError("Unsupported deployment workflow")
    if frame.mode == "vertical" and kind != "manual":
        raise ValueError("Vertical logs require runtime-start manual wiping")
    samples = [row for row in rows if row.get("event") == "sample" and row.get("phase") == "policy"]
    count = final_tick + 1
    prediction_ticks = np.arange(200, final_tick, 40)
    if [row["tick"] for row in samples] != list(range(count)):
        raise ValueError(f"Expected contiguous policy ticks 0..{final_tick}")
    run = {key: array([row[key] for row in samples], (count, width), key)
           for key, width in (("raw_ft", 6), ("tared_ft", 6), ("filtered_ft", 6), ("target_sdk_m", 3))}
    due = array([row["due_perf_s"] for row in samples], (count,), "due time")
    run["time"] = due - due[0]
    close_enough(run["time"], np.arange(count) / 100, "100 Hz time grid", atol=1e-8)
    run["pose_time"] = array([row[pose_key]["host_monotonic_s"] for row in samples],
                              (count,), "pose time") - due[0]
    if np.any(np.diff(run["pose_time"]) <= 0):
        raise ValueError("Pose timestamps must increase")
    run["position"] = array([row[pose_key]["sdk_end_position_m"] for row in samples],
                             (count, 3), "position")
    receive = array([row[receive_key] for row in samples], (count,), "sensor receive time")
    if np.any(np.diff(receive) < 0) or np.any(due - receive < -1e-8) or np.any(due - receive > 0.02000001):
        raise ValueError("Invalid causal sensor timestamps")
    run["baseline"] = array(baseline, (6,), "baseline")
    close_enough(run["raw_ft"] - run["baseline"], run["tared_ft"], "baseline subtraction", atol=1e-12)
    close_enough(CausalFTFilter().process(run["tared_ft"]), run["filtered_ft"], "online filter")
    predicted = [i for i, row in enumerate(samples) if row.get("delta_h_m") is not None]
    if predicted != prediction_ticks.tolist():
        raise ValueError(f"Expected {len(prediction_ticks)} height predictions at ticks 200..{final_tick - 40}")
    run["delta_h"] = array([samples[i]["delta_h_m"] for i in predicted], (len(predicted),), "height prediction")
    axis, sign = frame.normal_axis, frame.normal_sign
    close_enough(run["target_sdk_m"][prediction_ticks + 40, axis],
                 run["position"][prediction_ticks, axis] + sign * run["delta_h"], "next height endpoint")
    initial = run["target_sdk_m"][0] if initial is None else array(initial, (3,), "reference position")
    expected_normal = np.full(201, initial[axis]) if kind == "manual" else initial[axis] - np.arange(201) * 0.00005
    close_enough(run["target_sdk_m"][:201, axis], expected_normal, "initialization normal position")
    if initialization == STATIONARY_INITIALIZATION:
        close_enough(run["target_sdk_m"][:201], np.tile(initial, (201, 1)), "stationary initialization XYZ")
        if ([row.get("control_phase") for row in samples] != ["history_hold"] * 200 + ["motion"] * 1001
                or [row.get("motion_tick") for row in samples] != [None] * 200 + list(range(1001))):
            raise ValueError("Invalid stationary hold / motion timeline")
    run.update(path=path, name=path.parent.name, kind=kind, header=header,
               normal_axis=axis, normal_sign=sign, wiping_frame=frame.metadata(),
               final_tick=final_tick, prediction_ticks=prediction_ticks,
               initialization=initialization, initial=np.array(initial),
               tare_claims={key: tare[key] for key in ("noncontact_operator_confirmed", "stationary_validated") if key in tare},
               sensor_age_ms=(due - receive) * 1000)
    return run


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {"count": len(values), "first": float(values[0]), "last": float(values[-1]),
            "min": float(values.min()), "max": float(values.max()),
            "span": float(np.ptp(values)), "end_minus_start": float(values[-1] - values[0]),
            "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
            "median_abs": float(np.median(np.abs(values)))}


def metrics(run):
    phases = {}
    final_tick, prediction_ticks = run["final_tick"], run["prediction_ticks"]
    for name, start, end in (("whole_policy", 0, final_tick), *PHASES[:2], ("later_feedback", 240, final_tick)):
        selected = slice(start, end + 1)
        predictions = (prediction_ticks >= start) & (prediction_ticks < end)
        phases[name] = {
            "time_s": [start / 100, end / 100],
            "target_xyz_mm": {axis: stats(run["target_sdk_m"][selected, i] * 1000) for i, axis in enumerate("XYZ")},
            "measured_xyz_mm": {axis: stats(run["position"][selected, i] * 1000) for i, axis in enumerate("XYZ")},
            "tared_ft": {axis: stats(run["tared_ft"][selected, i]) for i, axis in enumerate(AXES)},
            "filtered_ft": {axis: stats(run["filtered_ft"][selected, i]) for i, axis in enumerate(AXES)},
            "prediction_delta_mm": stats(run["delta_h"][predictions] * 1000),
        }
    spec = run["header"]["spec"]
    predicted_sum = float(run["delta_h"].sum() * 1000)
    axis, sign = run["normal_axis"], run["normal_sign"]
    axis_name = "xyz"[axis]
    anchor_sum = float(sign * np.sum(run["position"][prediction_ticks, axis] - run["target_sdk_m"][prediction_ticks, axis]) * 1000)
    target_change = float(sign * (run["target_sdk_m"][-1, axis] - run["target_sdk_m"][200, axis]) * 1000)
    close_enough(predicted_sum + anchor_sum, target_change, "height change decomposition", atol=1e-6)
    return {"name": run["name"], "kind": run["kind"], "initialization": run["initialization"],
            "wiping_frame": run["wiping_frame"],
            "duration_s": final_tick / 100, "sample_count": final_tick + 1,
            "prediction_count": len(prediction_ticks),
            "reference_xyz_mm": (run["initial"] * 1000).tolist(),
            "baseline_si": run["baseline"].tolist(), "phases": phases,
            "logged_tare_claims_not_independent_measurements": run["tare_claims"],
            "height_decomposition_mm": {"sum_predicted_delta": predicted_sum,
                                         "sum_measured_reanchor_offset": anchor_sum,
                                         "target_change_feedback_s": target_change,
                                         "feedback_time_s": [2, final_tick / 100],
                                         **({"target_change_2_to_10_s": target_change} if final_tick == 1000 else {})},
            "sensor_age_ms": stats(run["sensor_age_ms"]),
            "policy": spec.get("policy"), "policy_sha256": spec.get("policy_sha256"),
            "training_config": spec.get("training_config"), "collection_session": spec.get("collection_session"),
            "prediction_table": [
                {"prediction_time_s": int(tick) / 100, "endpoint_time_s": int(tick + 40) / 100,
                 "delta_h_mm": float(delta * 1000),
                 f"anchor_measured_{axis_name}_mm": float(run["position"][tick, axis] * 1000),
                 f"endpoint_target_{axis_name}_mm": float(run["target_sdk_m"][tick + 40, axis] * 1000)}
                for tick, delta in zip(prediction_ticks, run["delta_h"])]}


def decorate(fig, axes, runs, title, subtitle, handles):
    end_s = max(run["final_tick"] for run in runs) / 100
    fig.suptitle(title, fontsize=17, y=0.986)
    fig.text(0.5, 0.949, subtitle, ha="center", fontsize=10)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.930),
               ncol=2, frameon=False, fontsize=10)
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Time from policy start (s)")
        ax.set_xlim(0, end_s + 0.035)
        ax.set_xticks(np.arange(0, end_s + 1))
        ax.axvspan(0, 2, color="#808080", alpha=0.08)
        ax.axvspan(2, 2.4, color="#d8b93d", alpha=0.14)
        ax.axvline(2, color="#666666", ls=":", lw=1)
        ax.axvline(2.4, color="#888888", ls=":", lw=0.8)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.text(0.5, 0.026, "Gray: 0-2 s initialization. Yellow: first feedback segment, 2-2.4 s.\n"
             "Policy phase only. No time stretching or lag correction. Sensor-local axes; not normal contact force.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.11, top=0.81, hspace=0.42, wspace=0.25)


def legend(runs, first, second, *, second_style="--", first_alpha=1.0):
    return [Line2D([], [], color=color, lw=2, ls=style, alpha=alpha, label=f"{run['label']} | {name}")
            for run, color in zip(runs, COLORS)
            for name, style, alpha in ((first, "-", first_alpha), (second, second_style, 1.0))]


def make_plots(runs, output):
    for relative, name in ((False, PLOTS[0]), (True, PLOTS[1])):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        for channel, ax in enumerate(axes):
            for run, color in zip(runs, COLORS):
                # Use one reference per run for BOTH command and feedback, retaining their initial offset.
                offset = run["initial"][channel] if relative else 0
                ax.plot(run["pose_time"], (run["position"][:, channel] - offset) * 1000, color=color, lw=1.9)
                ax.plot(run["time"], (run["target_sdk_m"][:, channel] - offset) * 1000, color=color, ls="--", lw=1.5)
            ax.set_ylabel(f"{'Relative' if relative else 'SDK'} {'XYZ'[channel]} (mm)")
        decorate(fig, axes, runs, "X / Y / Z trajectories" + (": relative to each start" if relative else ": absolute SDK coordinates"),
                 "Separate start references; not distance to the table" if relative else "Measured timestamps and scheduled target timestamps are shown separately",
                 legend(runs, "measured", "target"))
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
    for start, name, title in ((0, PLOTS[2], "Force"), (3, PLOTS[3], "Torque")):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        for channel, ax in zip(range(start, start + 3), axes):
            for run, color in zip(runs, COLORS):
                ax.plot(run["time"], run["tared_ft"][:, channel], color=color, alpha=0.33, lw=0.9)
                ax.plot(run["time"], run["filtered_ft"][:, channel], color=color, lw=2)
            ax.set_ylabel(f"{AXES[channel]} ({'N' if start == 0 else 'N m'})")
        decorate(fig, axes, runs, title + ": tared input and online filtered input",
                 "100 Hz causal held samples; online filter: order 2, cutoff 1 Hz; each run's own baseline",
                 legend(runs, "tared, unfiltered", "online filtered", second_style="-", first_alpha=0.33))
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for channel in range(6):
        ax = axes[channel % 3, channel // 3]
        for run, color in zip(runs, COLORS):
            ax.plot(run["time"], run["raw_ft"][:, channel], color=color, lw=1.3)
            ax.axhline(run["baseline"][channel], color=color, ls="--", lw=1.3)
        ax.set_ylabel(f"Raw {AXES[channel]} ({'N' if channel < 3 else 'N m'})")
    decorate(fig, axes, runs, "Raw wrench and the subtracted baseline",
             "Raw policy samples include static load; taring while pressed can also remove existing contact load",
             legend(runs, "raw policy input", "tare baseline"))
    fig.savefig(output / PLOTS[4], dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for run, color in zip(runs, COLORS):
        axes[0].plot(run["pose_time"], (run["position"][:, 2] - run["initial"][2]) * 1000, color=color, lw=1.8)
        axes[0].plot(run["time"], (run["target_sdk_m"][:, 2] - run["initial"][2]) * 1000, color=color, ls="--", lw=1.5)
        t = run["prediction_ticks"] / 100
        dh = run["delta_h"] * 1000
        axes[2].scatter(t, dh, color=color, marker="o", s=27, zorder=3)
        axes[2].hlines(dh, t, t + 0.4, color=color, alpha=0.65, lw=1)
        axes[2].scatter(t + 0.4, dh, facecolors="none", edgecolors=color, marker="s", s=25, zorder=3)
    # Separate Z scales avoid hiding the small manual adjustment behind the old 10 mm press.
    for run, color in zip(runs, COLORS):
        anchor = run["target_sdk_m"][200, 2]
        axes[1].plot(run["pose_time"][200:], (run["position"][200:, 2] - anchor) * 1000, color=color, lw=1.8)
        axes[1].plot(run["time"][200:], (run["target_sdk_m"][200:, 2] - anchor) * 1000, color=color, ls="--", lw=1.5)
    axes[0].set_ylabel("Z - start Z (mm)")
    axes[1].set_ylabel("Z - target Z at 2 s (mm)")
    axes[2].set_ylabel("Predicted delta Z (mm)")
    axes[2].axhline(0, color="#777777", lw=0.7)
    axes[2].text(0.01, 0.08, "Filled circle: prediction time\nOpen square: endpoint time (+0.4 s)",
                 transform=axes[2].transAxes, fontsize=9)
    decorate(fig, axes, runs, "Height feedback: initialization versus learned adjustments",
             "Middle panel: each run's 2 s target is zero; initial tracking offset is retained",
             legend(runs, "measured / prediction", "target / endpoint"))
    end_s = max(run["final_tick"] for run in runs) / 100
    axes[1].set_xlim(2, end_s + 0.035)
    axes[1].set_xticks(np.arange(2, end_s + 1))
    fig.savefig(output / PLOTS[5], dpi=170)
    plt.close(fig)


def analysis_text(summary):
    manual, fixed = summary["runs"]
    stationary = manual["initialization"] == STATIONARY_INITIALIZATION
    m, f = manual["phases"]["whole_policy"], fixed["phases"]["whole_policy"]
    later_m = manual["phases"]["later_feedback"]["prediction_delta_mm"]
    later_f = fixed["phases"]["later_feedback"]["prediction_delta_mm"]
    tare_note = {
        "loaded": "操作者确认本次在接触或受压状态下按 z 去皮。原有接触载荷可能一起被扣进基线。",
        "unloaded": "操作者声明本次去皮时离开表面；这是现场声明，不是传感器独立验证。",
        "unknown": "本次去皮时是否受压未确认；不能仅凭去皮后读数判断绝对接触载荷。",
    }[summary["manual_tare_contact"]["state"]]
    lines = [
        "# 本次轨迹与旧固定安装流程对比", "",
        f"本次：`{manual['name']}`，完整记录 {manual['duration_s']:g} 秒、{manual['sample_count']} 个策略周期和 {manual['prediction_count']} 次高度预测；"
        f"参考：`{fixed['name']}`，{fixed['duration_s']:g} 秒、{fixed['sample_count']} 个策略周期和 {fixed['prediction_count']} 次高度预测。", "",
        "## 结论", "",
        f"本次按 h 捕获的初始 Z 为 **{manual['reference_xyz_mm'][2]:.3f} mm**。它是 SDK 坐标，",
        "不是海绵底面离桌高度，也不是压缩量。" + (
            "前 2 秒 XYZ 目标均保持起点并采集力历史，随后执行完整 10 秒轨迹。" if stationary else
            "前 2 秒目标 Z 保持不变，XY 仍执行网络轨迹。"),
        "这条流程没有自动寻找桌面、目标法向力控制或脱离接触后主动重新压紧的逻辑，",
        "所以离开桌面不会必然触发向下运动；之后按模型增量和实测高度生成目标。", "",
        f"旧流程起始目标 Z 为 **{fixed['reference_xyz_mm'][2]:.3f} mm**，前 2 秒固定下压 10 mm，",
        f"到 **{fixed['reference_xyz_mm'][2] - 10:.3f} mm**。这段是预编程初始化，不是网络学出的主动接触能力。",
        "因此，旧结果起初明显按压的首要原因是流程不同，不能将整段 Z 跨度用于评价网络优劣。", "",
        f"本次预测增量范围为 **{m['prediction_delta_mm']['min']:.3f}～{m['prediction_delta_mm']['max']:.3f} mm**；",
        f"目标 Z 总跨度 **{m['target_xyz_mm']['Z']['span']:.3f} mm**，实测 Z 跨度 **{m['measured_xyz_mm']['Z']['span']:.3f} mm**。",
        "高度确实在预测、发送和变化，不是锁死不动，但变化远小于横向运动且正负调整相互抵消。", "",
        "## 去皮条件的重要影响", "", tare_note,
        "日志中的 noncontact_operator_confirmed 是旧程序写入的流程字段，不是实际接触检测；",
        "若与操作者此次补充的信息冲突，报告保留双方来源，不把日志字段当作物理事实。", "",
        "去皮计算为 `输入 = 原始六轴载荷 - 本次基线`。在受压状态去皮后，接近零仅表示",
        "接近去皮时的载荷；载荷减小或脱离接触时输入可能转向另一符号，并不自动等价于应当下压。",
        "这也偏离了训练约定的空载基线扣除；但仅凭两次运行不能量化它对预测的独立贡献。",
        f"两次基线 Fz 分别为 **{manual['baseline_si'][2]:.3f} N** 和 **{fixed['baseline_si'][2]:.3f} N**。",
        "原始数据同时包含静载和接触等影响，差值不能直接解释为海绵法向接触力。", "",
        "## 整段统计", "",
        "| 指标 | 本次 manual | 旧 fixed |", "|---|---:|---:|",
    ]
    for axis in "XYZ":
        lines.append(f"| 目标 {axis} 跨度 / mm | {m['target_xyz_mm'][axis]['span']:.3f} | {f['target_xyz_mm'][axis]['span']:.3f} |")
        lines.append(f"| 实测 {axis} 跨度 / mm | {m['measured_xyz_mm'][axis]['span']:.3f} | {f['measured_xyz_mm'][axis]['span']:.3f} |")
    lines.extend([
        f"| 预测增量绝对值中位数 / mm | {m['prediction_delta_mm']['median_abs']:.4f} | {f['prediction_delta_mm']['median_abs']:.4f} |",
        f"| 2.4 秒后预测增量绝对值中位数 / mm | {later_m['median_abs']:.4f} | {later_f['median_abs']:.4f} |",
        "", "## 分阶段高度对比", "",
        "| 阶段 | 本次目标 Z 变化 / mm | 旧目标 Z 变化 / mm | 本次实测 Z 变化 / mm | 旧实测 Z 变化 / mm |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in (("initialization", "0–2 s 初始化"), ("first_feedback", "2–2.4 s 首次反馈"),
                        ("later_feedback", f"后续反馈：本次 2.4–{manual['duration_s']:g} s；参考 2.4–{fixed['duration_s']:g} s")):
        a, b = manual["phases"][name], fixed["phases"][name]
        lines.append(f"| {label} | {a['target_xyz_mm']['Z']['end_minus_start']:.3f} | {b['target_xyz_mm']['Z']['end_minus_start']:.3f} | "
                     f"{a['measured_xyz_mm']['Z']['end_minus_start']:.3f} | {b['measured_xyz_mm']['Z']['end_minus_start']:.3f} |")
    lines.extend([
        "", "阶段高度变化采用该阶段边界 tick 的末值减初值；相邻阶段共享边界，预测次数按左闭右开归属。",
        "实测位置仍保留各自实际读取时间，不把实测值伪装成严格位于计划时刻。", "",
        "## 高度增量如何变成运动", "",
        "2.0 秒积累完五帧、间隔 0.4 秒的滤波力历史后开始预测；"
        f"本次预测时刻为 2.0、2.4、…、{manual['duration_s'] - 0.4:g} 秒；参考最后预测为 {fixed['duration_s'] - 0.4:g} 秒。",
        "每次 `下一端点 Z = 预测时实测 Z + delta_h`，对应 0.4 秒后的目标，段内再按 100 Hz 插值。",
        "每段都重新锚定实测 Z，所以总位移不等于简单累加 delta_h，跟随偏差也会影响后续端点。",
        "本工具已逐个核对日志中下一端点与实测锚点加预测增量的等式。", "",
        "对第 2 秒至各自结束的所有反馈段，可精确分解：",
        "`目标 Z 总变化 = 预测增量之和 + 各次(实测 Z - 当段起始目标 Z)之和`。", "",
        f"本次为 **{manual['height_decomposition_mm']['sum_predicted_delta']:+.3f} + "
        f"({manual['height_decomposition_mm']['sum_measured_reanchor_offset']:+.3f}) = "
        f"{manual['height_decomposition_mm']['target_change_feedback_s']:+.3f} mm**。",
        f"旧流程对应为 **{fixed['height_decomposition_mm']['sum_predicted_delta']:+.3f} + "
        f"({fixed['height_decomposition_mm']['sum_measured_reanchor_offset']:+.3f}) = "
        f"{fixed['height_decomposition_mm']['target_change_feedback_s']:+.3f} mm**。",
        "这些偏移来自记录中的实测锚点，不是额外下发的补偿命令，也不能直接当作机械臂静态定位误差。", "",
        f"旧流程第一次预测为 **{fixed['prediction_table'][0]['delta_h_mm']:.3f} mm**，本次为 **{manual['prediction_table'][0]['delta_h_mm']:.3f} mm**。",
        f"去掉首次预测后，旧流程增量范围仅为 **{later_f['min']:.3f}～{later_f['max']:.3f} mm**，",
        "旧模型后续也主要是小调整；不能把它的固定 10 mm 下压误认为后续网络输出一直更强。", "",
        "## 力与力矩统计", "",
        "下表为各自基线扣除后的在线二阶 1 Hz 因果低通数据，保留传感器局部轴和符号。", "",
        "| 分量 | 单位 | 本次最小 / 最大 | 旧流程最小 / 最大 |", "|---|---|---:|---:|",
    ])
    for i, axis in enumerate(AXES):
        a, b = m["filtered_ft"][axis], f["filtered_ft"][axis]
        lines.append(f"| {axis} | {'N' if i < 3 else 'N·m'} | {a['min']:.4f} / {a['max']:.4f} | {b['min']:.4f} / {b['max']:.4f} |")
    lines.extend([
        "", "## 可比性和不能得出的结论", "",
        f"- 两次模型权重 SHA256 {'不同' if manual['policy_sha256'] != fixed['policy_sha256'] else '相同'}。",
        f"- 本次训练配置：`{manual['training_config']}`；示教来源：`{manual['collection_session']}`。",
        f"- 旧训练配置：`{fixed['training_config']}`；示教来源：`{fixed['collection_session']}`。",
        "- 起点、姿态、横向覆盖范围、初始化和去皮条件不同，不是只更换一个因素的对照实验。",
        "- 本次更大的 XY 跨度与不足 1 mm 的 Z 调整也使高度变化在肉眼观察中不明显。",
        "- 没有桌面高度、工具几何及接触真值，不报告离桌距离、压缩量、绝对法向力或擦净率。",
        "- 目标与实测采用不同时间戳；观测发生在本轮目标发送之前，不能直接将同一行差值视为严格跟踪误差。",
        "- 这份比较能解释初始按压为何不同，不能据此证明旧模型泛化、闭环稳定性或擦拭效果更好。",
        "- 不应在当前已经受压的起点直接补上旧流程的 10 mm 下压。若设计接触建立，需要另行确认间隙、载荷和安全边界。",
        "", "## 图表与数据口径", "",
        "仅使用策略阶段 100 Hz 因果保持值，不混入启动接近、回撤、退出保持，也不声称使用 CSV 的全部原始帧。",
        "各自策略开始为 t=0，无时间拉伸、相位移动、时延补偿或额外平滑。",
        "相对轨迹中，实测与目标统一减去该次参考起点，保留两者起始偏差。",
        "原始载荷图为策略采用的 raw_ft，与全速串口 CSV 并非同一采样密度。", "",
    ])
    descriptions = ("绝对 X/Y/Z", "相对起点 X/Y/Z", "去皮及滤波力", "去皮及滤波力矩", "原始六轴及去皮基线", "高度初始化与反馈放大")
    lines.extend(f"- [{description}]({name})" for description, name in zip(descriptions, PLOTS))
    lines.extend(["", "输入哈希、模型来源、详细阶段指标及每次高度预测见 [summary.json](summary.json)。",
                  "本分析不连接硬件，不修改历史数据、控制策略或初始化行为。", ""])
    return "\n".join(lines)


def generate(events, reference_events, output, *, tare_contact="unknown"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if tare_contact not in ("loaded", "unloaded", "unknown"):
        raise ValueError("Unknown tare contact annotation")
    paths = [Path(events).resolve(), Path(reference_events).resolve()]
    hashes = {str(path): file_digest(path) for path in paths}
    runs = [load_run(path) for path in paths]
    if [run["kind"] for run in runs] != ["manual", "fixed"]:
        raise ValueError("Expected manual --events and fixed --reference-events")
    if any(run["wiping_frame"]["wiping_mode"] != "horizontal" for run in runs):
        raise ValueError("This historical table comparison requires horizontal runs; use plot_inference or plot_delta_h_error for vertical runs")
    for run, label in zip(runs, ("Manual", "Fixed")):
        run["label"] = f"{label}: {run['name']}"
    summary = {
        "schema_version": 1, "scope": "completed_manual_vs_fixed_policy_only",
        "inputs_sha256": hashes, "plots": list(PLOTS),
        "time_alignment": "each_policy_start_no_lag_or_phase_correction",
        "position_units": "mm; SDK coordinates, not table clearance or compression",
        "wrench_units": ["N"] * 3 + ["N m"] * 3,
        "filter": {"sample_hz": 100, "order": 2, "cutoff_hz": 1},
        "manual_tare_contact": {"state": tare_contact, "source": "operator CLI annotation; not sensor inference"},
        "prediction_interval_s": 0.4,
        "phase_boundary_rule": "state endpoints shared; prediction times left-inclusive right-exclusive",
        "runs": [metrics(run) for run in runs],
    }
    output.mkdir(parents=True, exist_ok=False)
    make_plots(runs, output)
    if any(file_digest(path) != hashes[str(path)] for path in paths):
        raise RuntimeError("Input changed during plotting; no completed report written")
    with (output / "analysis.md").open("x", encoding="utf-8") as stream:
        stream.write(analysis_text(summary))
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="Completed runtime-origin events.jsonl")
    parser.add_argument("--reference-events", required=True, type=Path, help="Completed fixed-setup events.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="New output directory, never overwritten")
    parser.add_argument("--tare-contact", choices=("loaded", "unloaded", "unknown"), default="unknown",
                        help="Operator annotation for manual-run taring; never inferred from the force curve")
    args = parser.parse_args(argv)
    output = new_output(args.output)
    generate(args.events, args.reference_events, output, tare_contact=args.tare_contact)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
