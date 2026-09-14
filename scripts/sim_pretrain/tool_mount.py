"""Documented fixed sponge mounts for the ordinary AIRBOT wrist."""

import xml.etree.ElementTree as ET

import numpy as np

ALUMINUM_DENSITY = 2700.0
# In right_hand coordinates, whose origin is link6 z=+0.015 m.
BRIDGE_BOXES = (
    ("interface", (0, 0, -0.0664), (0.025, 0.065, 0.001)),
    ("rail_left", (0, -0.052, -0.0337), (0.004, 0.004, 0.0317)),
    ("rail_right", (0, 0.052, -0.0337), (0.004, 0.004, 0.0317)),
)
BACKPLATE_POSITION = np.array([0.0, 0.0, -0.016])
BACKPLATE_SIZE = np.array([0.06, 0.025, 0.001])

# At q=0 the legacy tool axes are right, down, forward. KWR52 X is
# 30 degrees below left, so tool_from_sensor is Rz(+150 degrees), not Rz(30).
KWR52_TOOL_FROM_SENSOR_QUAT = np.array([np.cos(5 * np.pi / 12), 0, 0, np.sin(5 * np.pi / 12)])


def orient_ft_frame(worldbody, gripper_prefix, profile="legacy"):
    if profile not in ("legacy", "kwr52_left_v1"):
        raise ValueError("Unknown FT frame profile")
    if profile == "legacy":
        return
    tool = worldbody.find(f".//body[@name='{gripper_prefix}wiping_gripper']")
    sensor = None if tool is None else tool.find(f"site[@name='{gripper_prefix}ft_frame']")
    if sensor is None or not np.allclose(
        np.fromstring(tool.get("quat", "1 0 0 0"), sep=" "), [0.707107, 0, 0, -0.707107]
    ):
        raise ValueError("KWR52 alignment requires the audited AIRBOT wiping tool orientation")
    sensor.set("quat", vector(KWR52_TOOL_FROM_SENSOR_QUAT))


def vector(values):
    return " ".join(f"{x:.12g}" for x in values)


def box_mass_inertia(size):
    size = np.asarray(size)
    mass = float(8 * np.prod(size) * ALUMINUM_DENSITY)
    inertia = mass / 3 * (np.sum(size**2) - size**2)
    return mass, np.diag(inertia)


def combine_inertias(parts):
    mass = sum(p[0] for p in parts)
    com = sum(m * np.asarray(pos) for m, pos, _ in parts) / mass
    tensor = np.zeros((3, 3))
    for m, pos, inertia in parts:
        offset = np.asarray(pos) - com
        tensor += inertia + m * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return mass, com, tensor


def inertial_element(parts):
    mass, com, tensor = combine_inertias(parts)
    full = [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]
    return ET.Element("inertial", pos=vector(com), mass=str(mass), fullinertia=vector(full))


def add_box(body, name, position, size):
    for suffix, group, color, enabled in (
        ("visual", "1", ".65 .68 .72 1", "0"),
        ("collision", "0", ".65 .2 .15 1", "1"),
    ):
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_{suffix}",
            type="box",
            pos=vector(position),
            size=vector(size),
            rgba=color,
            group=group,
            contype=enabled,
            conaffinity=enabled,
            mass="0",
        )


def install_tool_mount(worldbody, robot_prefix, gripper_prefix, profile="bridge_v1"):
    if profile not in ("bridge_v1", "compact_v2", "direct_wrist_v3"):
        raise ValueError("Unknown engineering mount profile")
    hand = worldbody.find(f".//body[@name='{robot_prefix}right_hand']")
    tool = None if hand is None else hand.find(f"body[@name='{gripper_prefix}wiping_gripper']")
    if tool is None:
        raise ValueError("Expected the ordinary AIRBOT fixed WipingGripper hierarchy")
    if not np.allclose(np.fromstring(hand.get("pos", ""), sep=" "), [0, 0, 0.015]):
        raise ValueError("Engineering mount requires the audited right_hand offset")
    old_inertia = tool.find("inertial")
    sensor = tool.find("site[@name='" + gripper_prefix + "ft_frame']")
    if old_inertia is None or sensor is None or old_inertia.get("quat") is not None:
        raise ValueError("Unexpected tool inertial or sensor convention")
    if not np.allclose(np.fromstring(tool.get("pos", ""), sep=" "), [0, 0, 0.015]):
        raise ValueError("Engineering mount requires the audited tool offset")
    if profile == "direct_wrist_v3":
        from scripts.sim_pretrain.wrist import WRIST_FACE_Z

        # Hand is +15 mm in link6, and the sponge's rear face is -15 mm in tool.
        tool.set("pos", vector([0, 0, WRIST_FACE_Z]))
        sensor.set("pos", "0 0 -0.015")
        return
    # Keep the pre-existing sponge inertia assumption; add the backing plate exactly.
    old_mass = float(old_inertia.get("mass"))
    old_com = np.fromstring(old_inertia.get("pos"), sep=" ")
    old_tensor = np.diag(np.fromstring(old_inertia.get("diaginertia"), sep=" "))
    plate_mass, plate_tensor = box_mass_inertia(BACKPLATE_SIZE)
    combined = inertial_element(
        [(old_mass, old_com, old_tensor), (plate_mass, BACKPLATE_POSITION, plate_tensor)]
    )
    tool.remove(old_inertia)
    tool.insert(0, combined)
    add_box(tool, gripper_prefix + "mount_backplate", BACKPLATE_POSITION, BACKPLATE_SIZE)
    sensor.set("pos", "0 0 -0.017")
    if profile == "compact_v2":
        # Backplate support face: .015 - .0494 - .017 = -.0514 in link6.
        tool.set("pos", "0 0 -0.0494")
    else:
        parts = []
        adapter = ET.Element("body", name=gripper_prefix + "mount_adapter", pos="0 0 0")
        for name, pos, size in BRIDGE_BOXES:
            mass, inertia = box_mass_inertia(size)
            parts.append((mass, pos, inertia))
            add_box(adapter, gripper_prefix + "mount_" + name, pos, size)
        adapter.insert(0, inertial_element(parts))
        hand.remove(tool)
        hand.append(adapter)
        adapter.append(tool)


def place_base_on_table(worldbody, prefix, base_xy, table_offset, table_size):
    from scipy.spatial.transform import Rotation

    root = worldbody.find(f".//body[@name='{prefix}base']")
    box = root.find("geom[@type='box']") if root is not None else None
    if box is None:
        raise ValueError("Missing audited AIRBOT base mounting box")
    position = np.fromstring(box.get("pos"), sep=" ")
    size = np.fromstring(box.get("size"), sep=" ")
    rotation = Rotation.from_euler(
        "xyz", np.fromstring(box.get("euler", "0 0 0"), sep=" ")
    ).as_matrix()
    extent = np.abs(rotation) @ size
    base_xy = np.asarray(base_xy, dtype=float)
    table_offset, table_size = np.asarray(table_offset), np.asarray(table_size)
    low = base_xy + position[:2] - extent[:2]
    high = base_xy + position[:2] + extent[:2]
    if np.any(low < table_offset[:2] - table_size[:2] / 2) or np.any(
        high > table_offset[:2] + table_size[:2] / 2
    ):
        raise ValueError("Robot mounting footprint is outside the tabletop")
    bottom = position[2] - extent[2]
    root.set("pos", vector([*base_xy, table_offset[2] - bottom]))
    ET.SubElement(
        root,
        "site",
        name=prefix + "table_mount_plane",
        pos=vector([0, 0, bottom]),
        size="0.001",
        rgba="0 0 0 0",
        group="2",
    )
