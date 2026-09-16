"""Offline static-load diagnostic fit; never approves production calibration."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.force_sensor.tools.record_unloaded_session import (
    pose_files,
    session_status,
)
from scripts.shared.common import file_digest, write_json
from scripts.shared.paths import ROOT


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def condition(u):
    singular = np.linalg.svd(np.column_stack([np.ones(len(u)), u]), compute_uv=False)
    full = len(singular) == 4 and singular[-1] > singular[0] * 1e-8
    return {
        "full_rank": bool(full),
        "condition": float(singular[0] / singular[-1]) if full else None,
        "singular_values": singular.tolist(),
    }


def groups(u, degrees=10.0):
    """Connected near-orientation groups prevent repeated-pose validation leakage."""
    distance = np.rad2deg(np.arccos(np.clip(u @ u.T, -1, 1)))
    remaining, result = set(range(len(u))), []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            near = {j for j in remaining if distance[i, j] < degrees}
            stack.extend(sorted(near))
            remaining -= near
        result.append(sorted(group))
    return result


def fit(u, wrench):
    if len(u) < 4 or not condition(u)["full_rank"]:
        raise ValueError("Insufficient orientation rank")
    force = wrench[:, :3]
    a, b = u - u.mean(0), force - force.mean(0)
    candidates = []
    # Both global force signs are diagnostic hypotheses, not axis verification.
    for sign in (1, -1):
        left, singular, right = np.linalg.svd(a.T @ (sign * b))
        d = np.array([1.0, 1.0, np.linalg.det(left @ right)])
        row_rotation = left @ np.diag(d) @ right
        weight = float(np.dot(singular, d) / np.sum(a**2))
        bias = force.mean(0) - sign * weight * u.mean(0) @ row_rotation
        prediction = bias + sign * weight * u @ row_rotation
        candidates.append(
            (float(np.mean((prediction - force) ** 2)), sign, weight, bias, row_rotation)
        )
    _, sign, weight, bias, row_rotation = min(candidates, key=lambda v: v[0])
    gravity = sign * weight * u @ row_rotation
    design = np.concatenate([np.column_stack([-skew(v), np.eye(3)]) for v in gravity], axis=0)
    torque_params = np.linalg.lstsq(design, wrench[:, 3:].reshape(-1), rcond=None)[0]
    affine = np.linalg.lstsq(np.column_stack([np.ones(len(u)), u]), force, rcond=None)[0]
    return {
        "force_bias_N": bias.tolist(),
        "force_sign_hypothesis": sign,
        "sensed_weight_N": weight,
        "mass_equivalent_kg": weight / 9.81,
        "candidate_sensor_from_sdk_rotation": row_rotation.T.tolist(),
        "candidate_com_sensor_m": torque_params[:3].tolist(),
        "torque_bias_Nm": torque_params[3:].tolist(),
        "affine_force_coefficients_diagnostic_only": affine.tolist(),
        "affine_linear_singular_values": np.linalg.svd(affine[1:], compute_uv=False).tolist(),
    }


def predict(model, u):
    gravity = (
        model["force_sign_hypothesis"]
        * model["sensed_weight_N"]
        * u
        @ np.asarray(model["candidate_sensor_from_sdk_rotation"]).T
    )
    force = gravity + model["force_bias_N"]
    torque = np.cross(model["candidate_com_sensor_m"], gravity) + model["torque_bias_Nm"]
    return np.column_stack([force, torque])


def metrics(target, prediction):
    error = prediction - target
    return {
        "channel_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        "torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
    }


def analyze(u, wrench, names):
    model = fit(u, wrench)
    prediction = predict(model, u)
    clustered = groups(u)
    folds = []
    for held in clustered:
        train = [i for i in range(len(u)) if i not in held]
        c = condition(u[train])
        fold = {"held_out": [names[i] for i in held], "train_condition": c}
        if c["full_rank"]:
            local = fit(u[train], wrench[train])
            fold["metrics"] = metrics(wrench[held], predict(local, u[held]))
            fold["mass_equivalent_kg"] = local["mass_equivalent_kg"]
            fold["force_bias_N"] = local["force_bias_N"]
        folds.append(fold)
    repeat_angle = float(np.rad2deg(np.arccos(np.clip(u[0] @ u[-1], -1, 1))))
    observed_delta = wrench[-1] - wrench[0]
    modeled_delta = prediction[-1] - prediction[0]
    return {
        "model_candidate_not_approved": model,
        "orientation_design": condition(u),
        "orientation_groups_10deg_connected": [[names[i] for i in g] for g in clustered],
        "training_fit": metrics(wrench, prediction),
        "leave_orientation_group_out": folds,
        "per_pose_residual_si": (wrench - prediction).tolist(),
        "first_last_repeat": {
            "gravity_direction_difference_deg": repeat_angle,
            "observed_delta_si": observed_delta.tolist(),
            "model_expected_delta_si": modeled_delta.tolist(),
            "unexplained_force_delta_norm_N": float(
                np.linalg.norm((observed_delta - modeled_delta)[:3])
            ),
        },
        "production_approved": False,
        "hardware_ready": False,
        "assumptions": [
            "SDK +Z is vertical; base tilt not independently measured",
            "Operator confirms unchanged mount, no contact and no manual tare",
            "Rigid tool; constant electronic bias and center of mass",
            "No production transform, TCP or bias file is changed",
        ],
    }


def run(directory, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new report path")
    files = pose_files(directory)
    hashes = {str(p): file_digest(p) for p in files}
    status = session_status(directory)
    u, wrench = [], []
    for name in status["accepted"]:
        record = json.loads((Path(directory) / name).read_text())
        rows = record["rows"]
        rotation = Rotation.from_quat([r["pose"]["sdk_end_orientation_xyzw"] for r in rows]).mean()
        u.append(rotation.inv().apply([0.0, 0.0, -1.0]))
        _, indices = np.unique([r["sensor_receive_perf_s"] for r in rows], return_index=True)
        wrench.append(np.array([r["raw_sensor_wrench_si"] for r in rows])[indices].mean(0))
    report = analyze(np.array(u), np.array(wrench), status["accepted"])
    report.update(
        record_status=status,
        input_hashes=hashes,
        source_hashes={
            str(p): file_digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("capture_unloaded_pose.py"),
                Path(__file__).with_name("record_unloaded_session.py"),
            )
        },
    )
    if any(file_digest(p) != sha for p, sha in hashes.items()):
        raise RuntimeError("Input records changed during analysis")
    report["inputs_unchanged"] = True
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "runs/force_sensor/unloaded_calibration"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/force_sensor/unloaded_calibration_fit_v1.json")
    )
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    print(json.dumps(run(args.directory, args.output), indent=2))
