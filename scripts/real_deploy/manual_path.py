"""Audit original network XY coordinates without scaling, translation or retiming."""

import numpy as np

from scripts.shared.real_preprocessing import finite_array

TRANSFORM = "original_xy_v1"
XY_LIMITS = np.array([0.005, 0.05])


def plan_path(xy, initial, speed_limit):
    xy, initial = finite_array(xy, "XY prediction"), finite_array(initial, "initial SDK position")
    if xy.shape != (25, 2) or initial.shape != (3,):
        raise ValueError("Expected 25 XY points and one SDK XYZ start")
    if not 0 < speed_limit <= 0.05:
        raise ValueError("Invalid historical speed reference")
    vertical_speed = (0.003 + 0.005) / 0.4
    candidate = xy.copy()
    speed = np.linalg.norm(np.diff(np.vstack((initial[:2], candidate)), axis=0), axis=1) / 0.4
    original_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.4
    report = {
        "transform": TRANSFORM, "uniform_scale": 1.0, "translation_sdk_m": [0.0, 0.0],
        "candidate_start_sdk_m": initial.tolist(), "horizon_s": 10.0, "waypoint_dt_s": 0.4,
        "time_scaled": False, "runtime_clipping": False, "original_xy_preserved_exactly": True,
        "cartesian_speed_policy": "record-only", "cartesian_speed_stop_enforced": False,
        "original_xy_span_m": np.ptp(xy, axis=0).tolist(),
        "candidate_xy_span_m": np.ptp(candidate, axis=0).tolist(),
        "candidate_max_abs_offset_m": np.abs(candidate - initial[:2]).max(axis=0).tolist(),
        "original_max_adjacent_xy_speed_m_s": float(original_speed.max()),
        "candidate_max_xy_speed_including_entry_m_s": float(speed.max()),
        "candidate_max_3d_speed_with_vertical_budget_m_s": float(np.hypot(speed.max(), vertical_speed)),
        "xy_limits_m": XY_LIMITS.tolist(), "max_cartesian_speed_m_s": speed_limit,
        "xy_transform_changes_coverage": False, "hardware_ready": False,
    }
    return candidate, report


class CandidatePolicy:
    """Freeze the exact original XY path; feedback inference and filtering are unchanged."""

    def __init__(self, policy, embedding, initial, speed_limit):
        self.original = policy
        self.xy, self.path_report = plan_path(policy.predict_xy(embedding)[0], initial, speed_limit)
        self.embedding = np.asarray(embedding).copy()

    def predict_xy(self, embedding):
        if not np.array_equal(embedding, self.embedding):
            raise ValueError("Candidate path is bound to one reviewed exploration embedding")
        return self.xy[None].copy()

    def make_ft_filter(self, hz):
        return self.original.make_ft_filter(hz)

    def predict_delta_h(self, embedding, history):
        return self.original.predict_delta_h(embedding, history)


def plot_path(output, original, candidate, initial):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, values, title in (
        (axes[0], (original - initial[:2]) * 1000, "Original network XY"),
        (axes[1], (candidate - initial[:2]) * 1000, "Deployment XY (unchanged)"),
    ):
        ax.plot(values[:, 0], values[:, 1], ".-", color="#127c72", linewidth=1.5)
        ax.scatter(*values[0], color="#bf4141", label="First waypoint", zorder=3)
        ax.add_patch(Rectangle((-5, -50), 10, 100, fill=False, edgecolor="#555555", linestyle="--", label="Old bounds (reference)"))
        ax.set(title=title, xlabel="SDK X offset (mm)", ylabel="SDK Y offset (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].plot([0, (candidate[0, 0] - initial[0]) * 1000],
                 [0, (candidate[0, 1] - initial[1]) * 1000], ":", color="#bf4141")
    bounds = np.vstack((initial[:2], candidate))
    lo, hi = (bounds.min(axis=0) - initial[:2]) * 1000 - 5, (bounds.max(axis=0) - initial[:2]) * 1000 + 5
    axes[1].add_patch(Rectangle(lo, *(hi - lo), fill=False, edgecolor="#127c72", linestyle=":"))
    axes[2].plot(np.arange(2, 26) * 0.4, np.linalg.norm(np.diff(original, axis=0), axis=1) / 0.4,
                 label="Original adjacent XY", color="#bf4141")
    axes[2].plot(np.arange(1, 26) * 0.4,
                 np.linalg.norm(np.diff(np.vstack((initial[:2], candidate)), axis=0), axis=1) / 0.4,
                 label="Unchanged XY incl. entry", color="#127c72", linestyle=":")
    axes[2].axhline(0.05, color="#555555", linestyle="--", label="Old speed cap (reference)")
    axes[2].set(title="Segment XY speed", xlabel="Waypoint time (s)", ylabel="Speed (m/s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    fig.suptitle("Original XY preserved; speed recorded, not limited here; no hardware validation", fontsize=11)
    fig.savefig(output / "path_review.png", dpi=150)
    plt.close(fig)
