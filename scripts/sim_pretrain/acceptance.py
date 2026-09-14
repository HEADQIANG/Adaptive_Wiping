"""Engineering contact-motion gate, distinct from free-space tracking."""

import numpy as np

CONTACT_LIMITS = {
    "travel_min_m": 0.045,
    "travel_max_m": 0.055,
    "y_error_max_m": 0.005,
    "return_error_max_m": 0.005,
    "x_error_max_m": 0.003,
    "orientation_max_deg": 2.0,
    "contact_fraction_min": 0.95,
    "contact_gap_max_s": 0.05,
    "saturation_fraction_max": 0.01,
}


def contact_motion_acceptance(data):
    required = {
        "position": (400, 3),
        "target_position": (400, 3),
        "orientation_error": (400,),
        "contact_count": (400,),
        "saturated": (400, 6),
        "time": (400,),
    }
    for key, shape in required.items():
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid contact acceptance field: {key}")
    if not np.allclose(data["time"], np.arange(1, 401) / 100, rtol=0, atol=1e-8):
        raise ValueError("Contact acceptance requires the 100 Hz exploration clock")
    pos = np.asarray(data["position"])
    error = pos - data["target_position"]
    metrics = {
        "forward_travel_m": float(pos[299, 1] - pos[199, 1]),
        "reverse_travel_m": float(pos[299, 1] - pos[399, 1]),
        "return_error_m": float(abs(pos[399, 1] - pos[199, 1])),
        "y_error_max_m": float(np.abs(error[199:, 1]).max()),
        "x_error_max_m": float(np.abs(error[199:, 0]).max()),
        "orientation_max_deg": float(np.rad2deg(data["orientation_error"]).max()),
        "saturation_fraction": float(np.any(data["saturated"], axis=1).mean()),
        "z_error_max_m": float(np.abs(error[:, 2]).max()),
        "slide_z_range_m": float(np.ptp(pos[199:, 2])),
    }
    checks = {}
    for name in ("forward", "reverse"):
        travel = metrics[f"{name}_travel_m"]
        checks[f"{name}_travel"] = (
            CONTACT_LIMITS["travel_min_m"] <= travel <= CONTACT_LIMITS["travel_max_m"]
        )
    for name in ("return_error", "y_error_max", "x_error_max"):
        limit = "return_error_max_m" if name == "return_error" else f"{name}_m"
        checks[name] = metrics[f"{name}_m"] <= CONTACT_LIMITS[limit]
    checks["orientation"] = metrics["orientation_max_deg"] <= CONTACT_LIMITS["orientation_max_deg"]
    checks["saturation"] = (
        metrics["saturation_fraction"] <= CONTACT_LIMITS["saturation_fraction_max"]
    )
    for name, section in (("forward", slice(200, 300)), ("reverse", slice(300, 400))):
        contact = np.asarray(data["contact_count"])[section] > 0
        longest = current = 0
        for present in contact:
            current = 0 if present else current + 1
            longest = max(longest, current)
        fraction, gap = float(contact.mean()), longest / 100
        metrics[f"{name}_contact_fraction"] = fraction
        metrics[f"{name}_contact_gap_s"] = gap
        checks[f"{name}_contact"] = fraction >= CONTACT_LIMITS["contact_fraction_min"]
        checks[f"{name}_contact_gap"] = gap <= CONTACT_LIMITS["contact_gap_max_s"]
    return {
        "version": 1,
        "passed": bool(all(checks.values())),
        "limits": dict(CONTACT_LIMITS),
        "metrics": metrics,
        "checks": {key: bool(value) for key, value in checks.items()},
        "failed_checks": [key for key, value in checks.items() if not value],
    }
