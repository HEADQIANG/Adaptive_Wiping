"""Offline KDL/URDF comparison. Report model frames, never physical calibration."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import PyKDL as kdl
from scipy.spatial.transform import Rotation

from scripts.robot_control.tools.audit_airbot_frames import digest, fk, read_dh


def vector(node, key, default="0 0 0"):
    value = np.array([float(x) for x in node.get(key, default).split()])
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {key}")
    return value


def chain_from_urdf(path, tip):
    root = ET.parse(path).getroot()
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        if child in by_child:
            raise ValueError("Duplicate child link")
        by_child[child] = joint
    joints, seen = [], set()
    current = tip
    while current != "base_link":
        if current in seen or current not in by_child:
            raise ValueError("Invalid chain")
        seen.add(current)
        joint = by_child[current]
        joints.append(joint)
        current = joint.find("parent").get("link")
    chain = kdl.Chain()
    for joint in reversed(joints):
        origin = joint.find("origin")
        origin = origin if origin is not None else ET.Element("origin")
        frame = kdl.Frame(
            kdl.Rotation.RPY(*vector(origin, "rpy")), kdl.Vector(*vector(origin, "xyz"))
        )
        kind, name = joint.get("type"), joint.get("name")
        if kind == "fixed":
            kjoint = kdl.Joint(name, kdl.Joint.Fixed)
        elif kind in ("revolute", "continuous"):
            axis = joint.find("axis")
            if axis is None:
                raise ValueError("Explicit joint axis required")
            v = vector(axis, "xyz")
            if not np.isclose(np.linalg.norm(v), 1):
                raise ValueError("Nonunit axis")
            # KDL expresses the joint axis in its parent frame; URDF uses joint frame.
            kjoint = kdl.Joint(name, frame.p, frame.M * kdl.Vector(*v), kdl.Joint.RotAxis)
        else:
            raise ValueError(f"Unsupported joint type: {kind}")
        chain.addSegment(kdl.Segment(joint.find("child").get("link"), kjoint, frame))
    return chain


def solve(chain, q):
    if chain.getNrOfJoints() != len(q):
        raise ValueError("Joint count mismatch")
    angles = kdl.JntArray(len(q))
    for i, value in enumerate(q):
        angles[i] = float(value)
    frame = kdl.Frame()
    if kdl.ChainFkSolverPos_recursive(chain).JntToCart(angles, frame) < 0:
        raise ValueError("KDL FK failed")
    result = np.eye(4)
    result[:3, :3] = [[frame.M[i, j] for j in range(3)] for i in range(3)]
    result[:3, 3] = [frame.p[i] for i in range(3)]
    return result


def metrics(matrices, expected):
    pos = [np.linalg.norm(t[:3, 3] - expected[:3, 3]) * 1000 for t in matrices]
    ang = [
        np.rad2deg(Rotation.from_matrix(expected[:3, :3].T @ t[:3, :3]).magnitude())
        for t in matrices
    ]
    return {
        "position_rms_mm": float(np.sqrt(np.mean(np.square(pos)))),
        "position_max_mm": float(max(pos)),
        "orientation_rms_deg": float(np.sqrt(np.mean(np.square(ang)))),
        "orientation_max_deg": float(max(ang)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--startup-log", type=Path, required=True)
    p.add_argument(
        "--urdf",
        type=Path,
        default=Path("/usr/share/airbot_controllers/asset/play_arm/urdf/play.urdf"),
    )
    p.add_argument(
        "--session",
        type=Path,
        default=Path(
            "archive/real_training/raw_data/manual_demonstrations/session_record_only_002"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {str(x): digest(x) for x in (args.startup_log, args.urdf)}
    dh, end = read_dh(args.startup_log)
    model = ET.parse(args.urdf).getroot()
    fixed = model.find("joint[@name='arm_connnect_joint']/origin")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", vector(fixed, "rpy")).as_matrix()
    transform[:3, 3] = vector(fixed, "xyz")
    candidate = np.linalg.inv(end) @ transform
    base_delta = vector(model.find("joint[@name='joint1']/origin"), "xyz") - np.array(
        [0, 0, dh["d"][0]]
    )
    chain = chain_from_urdf(args.urdf, "eef_connect_base_link")
    relative, adjusted = [], []
    for i in range(1, 9):
        meta_path = args.session / f"demo_{i:02}.json"
        meta = json.loads(meta_path.read_text())
        raw = args.session / meta["raw_file"]
        if meta.get("accepted") is not True or digest(raw) != meta["sha256"]:
            raise ValueError("Invalid demonstration binding")
        hashes[str(raw)], hashes[str(meta_path)] = digest(raw), digest(meta_path)
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        poses = [row["pose"] for row in rows if row.get("event") == "sample"][::20]
        for pose in poses:
            q = pose["joint_position_rad"]
            sdk = fk(q, dh, "modified", end, "right")
            urdf = solve(chain, q)
            relative.append(np.linalg.inv(sdk) @ urdf)
            urdf[:3, 3] -= base_delta
            adjusted.append(np.linalg.inv(sdk) @ urdf)
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError("Input changed")
    result = {
        "hardware_connected": False,
        "physical_calibration_verified": False,
        "samples": len(relative),
        "source_sha256": hashes,
        "script_sha256": digest(__file__),
        "dh_script_sha256": digest(Path(__file__).with_name("audit_airbot_frames.py")),
        "candidate_sdk_from_model_connector": candidate.tolist(),
        "urdf_minus_dh_base_translation_m": base_delta.tolist(),
        "unadjusted_model_error": metrics(relative, candidate),
        "after_explicit_base_difference_model_error": metrics(adjusted, candidate),
        "limitations": [
            "No fitted transform; explicit model constants only",
            "Connector model origin is not proven to be the measured physical mounting plane",
            "Sensor mounting rotation and electronic bias remain unknown",
            "Base difference is diagnostic, not a calibrated base correction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
