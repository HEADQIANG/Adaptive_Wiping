"""Scoped visual/collision simplification, not a bare-wrist inertial model."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.sim_pretrain.tool_mount import vector

WRIST_FACE_Z = -0.0795
WRIST_REAR_Z = -0.19595
WRIST_RADIUS = 0.0285


def install_direct_wrist(worldbody, asset, prefix):
    link = worldbody.find(f".//body[@name='{prefix}link6']")
    mesh = asset.find(f"mesh[@name='{prefix}airbot_link6']")
    visual = None if link is None else link.find(f"geom[@mesh='{prefix}airbot_link6']")
    camera = None if link is None else link.find(f"geom[@mesh='{prefix}airbot_camera_stand']")
    collision = None if link is None else link.find("geom[@type='box']")
    if any(node is None for node in (mesh, visual, camera, collision)) or mesh.get("file") is None:
        raise ValueError("Direct wrist requires the audited ordinary AIRBOT mesh and housing")
    if not np.allclose(np.fromstring(collision.get("pos", ""), sep=" "), [0, 0, -0.07]):
        raise ValueError("Unexpected ordinary AIRBOT housing collision offset")

    # MuJoCo parses the source OBJ; restore its compiled vertices to link6 coordinates.
    root = ET.Element("mujoco")
    ET.SubElement(root, "asset").append(copy.deepcopy(mesh))
    ET.SubElement(ET.SubElement(root, "worldbody"), "geom", type="mesh", mesh=mesh.get("name"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices = model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
    faces = model.mesh_face
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))
    )
    count, labels = connected_components(graph, directed=False)
    maxima = np.full(count, -np.inf)
    np.maximum.at(maxima, labels, vertices[:, 2])
    # Keep complete disconnected components; never slice triangles or stretch the wrist.
    keep = maxima[labels] <= WRIST_FACE_Z + 1e-8
    retained = vertices[keep]
    if (
        not len(retained)
        or keep.all()
        or not np.allclose(
            retained.min(axis=0), [-WRIST_RADIUS, -WRIST_RADIUS, WRIST_REAR_Z], atol=1e-7
        )
        or not np.allclose(
            retained.max(axis=0), [WRIST_RADIUS, WRIST_RADIUS, WRIST_FACE_Z], atol=1e-7
        )
    ):
        raise ValueError("Source wrist envelope differs from the audited geometry")
    remap = np.full(len(vertices), -1, dtype=int)
    remap[keep] = np.arange(len(retained))
    retained_faces = remap[faces[keep[faces].all(axis=1)]]
    mesh.attrib.clear()
    mesh.attrib.update(
        name=prefix + "airbot_link6",
        vertex=vector(retained.ravel()),
        face=" ".join(str(i) for i in retained_faces.ravel()),
    )

    # Reuse registered geom names so robosuite's instance/ID mappings remain valid.
    # The camera bracket's visual slot becomes a thin circular end cap.
    material, name = visual.get("material"), camera.get("name")
    camera.attrib.clear()
    camera.attrib.update(
        name=name,
        type="cylinder",
        size=vector([WRIST_RADIUS, 0.0001]),
        pos=vector([0, 0, WRIST_FACE_Z - 0.0001]),
        material=material,
        group="1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    name, rgba = collision.get("name"), collision.get("rgba")
    collision.attrib.clear()
    collision.attrib.update(
        name=name,
        type="cylinder",
        size=vector([WRIST_RADIUS, (WRIST_FACE_Z - WRIST_REAR_Z) / 2]),
        pos=vector([0, 0, (WRIST_FACE_Z + WRIST_REAR_Z) / 2]),
        group="0",
        contype="1",
        conaffinity="1",
        mass="0",
        rgba=rgba,
    )
