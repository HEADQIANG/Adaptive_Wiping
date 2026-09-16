"""Compare logged delta-h commands with subsequent measured motion, offline only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scripts.real_deploy.plot_run_comparison import load_run
from scripts.shared.common import file_digest
from scripts.shared.run_paths import new_output


def generate(events, output):
    events, output = Path(events).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    fingerprint = file_digest(events)
    run = load_run(events)
    ticks = run["prediction_ticks"]
    anchor_time = run["pose_time"][ticks]
    endpoint_time = anchor_time + 0.4
    pose_time = run["pose_time"]
    axis, sign = run["normal_axis"], run["normal_sign"]
    axis_name = "xyz"[axis]
    position_normal = run["position"][:, axis]
    if endpoint_time[0] < pose_time[0] or endpoint_time[-1] > pose_time[-1]:
        raise ValueError("Measured poses do not cover every exact 0.4s endpoint; extrapolation refused")
    right = np.searchsorted(pose_time, endpoint_time, side="left")
    left = np.maximum(0, right - 1)
    gaps = pose_time[right] - pose_time[left]
    if np.any(gaps > 0.02 + 1e-9):
        raise ValueError("Measured endpoint interpolation crosses a pose gap over 20ms")
    anchor_normal = position_normal[ticks]
    endpoint_normal = np.interp(endpoint_time, pose_time, position_normal)
    measured_delta = sign * (endpoint_normal - anchor_normal)
    predicted_delta = run["delta_h"]
    error_mm = (predicted_delta - measured_delta) * 1000
    if not np.isfinite(error_mm).all():
        raise ValueError("Non-finite delta-h error")
    # Compare in the model's normal direction, including the wall sign.
    np.testing.assert_allclose(error_mm, sign * (anchor_normal + sign * predicted_delta - endpoint_normal) * 1000,
                               atol=1e-10, rtol=0)
    worst = int(np.argmax(np.abs(error_mm)))
    times = run["time"][ticks]
    report = {
        "scope": "closed_loop_predicted_increment_minus_subsequent_measured_increment",
        "not_independent_demonstration_prediction_error": True,
        "source_events": str(events), "source_sha256": fingerprint,
        "policy": run["header"]["spec"].get("policy"),
        "policy_sha256": run["header"]["spec"].get("policy_sha256"),
        "count": len(ticks), "horizon_s": 0.4,
        "wiping_frame": run["wiping_frame"],
        "error_definition": f"predicted_delta_h - ({sign}) * [measured_{axis_name.upper()}(anchor_pose_time + 0.4s) - measured_{axis_name.upper()}(anchor_pose_time)]",
        "alignment": "recorded prediction anchor pose; linear interpolation of measured SDK poses at exact +0.4s; no extrapolation",
        "max_endpoint_bracket_s": float(gaps.max()),
        "mae_mm": float(np.mean(np.abs(error_mm))),
        "rmse_mm": float(np.sqrt(np.mean(error_mm ** 2))),
        "bias_mm": float(np.mean(error_mm)),
        "max_absolute_error_mm": float(np.max(np.abs(error_mm))),
        "worst_prediction_time_s": float(times[worst]),
        "worst_predicted_delta_h_mm": float(predicted_delta[worst] * 1000),
        "worst_measured_delta_h_mm": float(measured_delta[worst] * 1000),
        "interpretation": "Measured motion was driven by the model commands; this measures execution discrepancy, not model accuracy on independent demonstration labels.",
    }
    if file_digest(events) != fingerprint:
        raise RuntimeError("Input log changed while reading")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "delta_h_error.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["prediction_time_s", "anchor_pose_time_s", "endpoint_pose_time_s",
                         f"anchor_{axis_name}_m", f"endpoint_{axis_name}_m", "predicted_delta_h_mm", "measured_delta_h_mm",
                         "error_mm", "absolute_error_mm", "endpoint_bracket_s"])
        writer.writerows(zip(times, anchor_time, endpoint_time, anchor_normal, endpoint_normal,
                             predicted_delta * 1000, measured_delta * 1000, error_mm,
                             np.abs(error_mm), gaps))
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = FontProperties(fname=font_path) if font_path.is_file() else FontProperties()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle(f"{events.parent.name}：法向增量预测与随后实测变化（SDK {axis_name.upper()}）\n闭环执行差异，非独立示教测试误差",
                 fontproperties=font, fontsize=16)
    axes[0].plot(times, predicted_delta * 1000, "o-", color="#2864ad", ms=5, lw=1.8, label="预测 Δh")
    axes[0].plot(times, measured_delta * 1000, "s-", color="#dc7a2c", ms=5, lw=1.8, label="随后 0.4 秒实测 Δh")
    axes[0].set_ylabel("模型法向增量（mm）", fontproperties=font)
    axes[0].legend(prop=font, loc="lower left")
    axes[1].plot(times, error_mm, "o-", color="#9a4560", ms=5, lw=1.8, label="预测 − 实测")
    axes[1].fill_between(times, 0, error_mm, color="#9a4560", alpha=0.12)
    axes[1].set_ylabel("增量差值（mm）", fontproperties=font)
    axes[1].set_xlabel("按 s 后的预测时刻（秒）", fontproperties=font)
    axes[1].legend(prop=font, loc="lower left")
    axes[1].text(0.02, 0.97,
                 f"MAE = {report['mae_mm']:.4f} mm    RMSE = {report['rmse_mm']:.4f} mm\n"
                 f"最大绝对差值 = {report['max_absolute_error_mm']:.4f} mm（{times[worst]:.1f} 秒）",
                 transform=axes[1].transAxes, va="top", fontproperties=font, fontsize=11,
                 bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9})
    for ax in axes:
        ax.axhline(0, color="#555555", ls="--", lw=1)
        ax.axvspan(0, 2, color="#778899", alpha=0.10)
        ax.axvline(2, color="#778899", ls=":", lw=1)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, run["time"][-1])
        ax.set_xticks(np.arange(0, run["time"][-1] + 1))
    axes[0].text(0.7, 0.9, "静止采样\n尚无预测", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontproperties=font, color="#5a6570")
    fig.savefig(output / "delta_h_error.png", dpi=180)
    plt.close(fig)
    if file_digest(events) != fingerprint:
        raise RuntimeError("Input log changed during plotting")
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = new_output(args.output)
    report = generate(args.events, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(output / "delta_h_error.png")


if __name__ == "__main__":
    main()
