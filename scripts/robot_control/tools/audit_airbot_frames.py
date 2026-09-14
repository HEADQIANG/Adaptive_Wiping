"""Compare printed startup DH candidates with recorded poses; never connect hardware."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dh(path):
    text = Path(path).read_text()
    values = {}
    for name in ("alpha", "a", "d", "theta_offset"):
        matches = re.findall(r"\b" + name + r": ([^\n]+)", text)
        arrays = [np.array([float(x) for x in row.split()]) for row in matches]
        if not arrays or any(x.shape != (7,) or not np.isfinite(x).all() for x in arrays):
            raise ValueError(f"Missing or invalid seven-element {name}")
        if any(not np.array_equal(x, arrays[0]) for x in arrays):
            raise ValueError(f"Inconsistent printed {name}")
        values[name] = arrays[0]
    blocks = re.findall(r"end_convert:\s*\n((?:[^\n]+\n){4})", text)
    matrices = [
        np.array([[float(v) for v in row.split()] for row in b.splitlines()]) for b in blocks
    ]
    if not matrices or any(m.shape != (4, 4) or not np.isfinite(m).all() for m in matrices):
        raise ValueError("Invalid end_convert")
    if any(not np.array_equal(m, matrices[0]) for m in matrices):
        raise ValueError("Inconsistent end_convert")
    end = matrices[0]
    if not (
        np.allclose(end[3], [0, 0, 0, 1])
        and np.allclose(end[:3, :3].T @ end[:3, :3], np.eye(3))
        and np.isclose(np.linalg.det(end[:3, :3]), 1)
    ):
        raise ValueError("end_convert is not rigid")
    return values, end


def fk(q, dh, convention, end, placement):
    transform = np.eye(4)
    # Both common DH conventions are hypotheses, not a replacement robot model.
    for alpha, a, d, theta in zip(
        dh["alpha"], dh["a"], dh["d"], dh["theta_offset"] + np.r_[q, 0.0]
    ):
        x = np.eye(4)
        x[:3, :3] = Rotation.from_rotvec([alpha, 0, 0]).as_matrix()
        x[0, 3] = a
        z = np.eye(4)
        z[:3, :3] = Rotation.from_rotvec([0, 0, theta]).as_matrix()
        z[2, 3] = d
        transform = transform @ (z @ x if convention == "standard" else x @ z)
    return (
        transform
        @ {"none": np.eye(4), "right": end, "right_inverse": np.linalg.inv(end)}[placement]
    )


def compare(poses, dh, end):
    results = []
    for convention in ("standard", "modified"):
        for placement in ("none", "right", "right_inverse"):
            position, angle = [], []
            for pose in poses:
                q = np.asarray(pose["joint_position_rad"], dtype=float)
                p = np.asarray(pose["sdk_end_position_m"], dtype=float)
                quat = np.asarray(pose["sdk_end_orientation_xyzw"], dtype=float)
                if (
                    q.shape != (6,)
                    or p.shape != (3,)
                    or quat.shape != (4,)
                    or not all(np.isfinite(x).all() for x in (q, p, quat))
                    or not np.isclose(np.linalg.norm(quat), 1, atol=1e-3)
                ):
                    raise ValueError("Invalid recorded pose")
                predicted = fk(q, dh, convention, end, placement)
                position.append(np.linalg.norm(predicted[:3, 3] - p) * 1000)
                relative = predicted[:3, :3].T @ Rotation.from_quat(quat).as_matrix()
                angle.append(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
            results.append(
                {
                    "dh_convention": convention,
                    "end_conversion": placement,
                    "position_rms_mm": float(np.sqrt(np.mean(np.square(position)))),
                    "position_max_mm": float(max(position)),
                    "orientation_rms_deg": float(np.sqrt(np.mean(np.square(angle)))),
                    "orientation_max_deg": float(max(angle)),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-log", type=Path, required=True)
    parser.add_argument(
        "--session",
        type=Path,
        default=Path(
            "archive/real_training/raw_data/manual_demonstrations/session_record_only_002"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dh, end = read_dh(args.startup_log)
    hashes = {str(args.startup_log): digest(args.startup_log)}
    episodes = []
    for index in range(1, 9):
        meta_path = args.session / f"demo_{index:02}.json"
        meta = json.loads(meta_path.read_text())
        path = args.session / meta["raw_file"]
        if meta.get("accepted") is not True or digest(path) != meta["sha256"]:
            raise ValueError(f"Unaccepted or changed source: {meta_path}")
        hashes[str(meta_path)], hashes[str(path)] = digest(meta_path), digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows[0]["event"] != "start" or rows[-1]["event"] != "finished":
            raise ValueError("Incomplete recording")
        poses = [r["pose"] for r in rows if r.get("event") == "sample"][::20]
        if not poses:
            raise ValueError("No recorded poses")
        episodes.append(
            {"demo": index, "samples": len(poses), "candidates": compare(poses, dh, end)}
        )
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError("Source changed during audit")
    report = {
        "hardware_connected": False,
        "physical_calibration_verified": False,
        "scope": "Printed-DH hypotheses only; no fitted transforms or production changes",
        "limitations": [
            "Startup values are rounded",
            "Joint and pose reads are not atomic",
            "Numerical agreement cannot establish physical flange location",
            "Startup log postdates demonstrations; historical settings not proven",
        ],
        "source_sha256": hashes,
        "script_sha256": digest(__file__),
        "dh": {k: v.tolist() for k, v in dh.items()},
        "end_convert": end.tolist(),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "episodes": len(episodes),
                "samples": sum(x["samples"] for x in episodes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
