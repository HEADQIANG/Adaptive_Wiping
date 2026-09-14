"""Independent static sensor check; never modifies the exploration simulation."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def sensor_load_check(env):
    # A welded clone isolates the sensor convention from servo acceleration.
    root = ET.fromstring(env.model.get_xml())
    world = root.find("worldbody")
    for parent in world.iter():
        for child in list(parent):
            if child.tag in ("joint", "freejoint"):
                parent.remove(child)
    for name in ("actuator", "equality", "keyframe", "tendon"):
        node = root.find(name)
        if node is not None:
            root.remove(node)
    for geom in world.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper = env.robot.gripper["right"]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, gripper.naming_prefix + "ft_frame")
    body = model.site_bodyid[site]
    rotation = data.site_xmat[site].reshape(3, 3).copy()
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, gripper.important_sensors[key])
        for key in ("force_ee", "torque_ee")
    ]
    if any(i < 0 for i in ids) or any(model.sensor_dim[i] != 3 for i in ids):
        raise RuntimeError("Expected local three-axis force and torque sensors")

    def wrench():
        return np.concatenate(
            [data.sensordata[model.sensor_adr[i] : model.sensor_adr[i] + 3] for i in ids]
        )

    baseline = wrench()
    rows = []
    for force_local, offset_local in (
        ([0, 0, 1], [0, 0, 0]),
        ([0, 0, 1], [0.03, 0, 0]),
        ([0.7, -0.4, 0.2], [0.01, 0.02, 0]),
    ):
        force_local, offset_local = np.asarray(force_local), np.asarray(offset_local)
        force_world = rotation @ force_local
        application = data.site_xpos[site] + rotation @ offset_local
        data.xfrc_applied[:] = 0
        data.xfrc_applied[body, :3] = force_world
        data.xfrc_applied[body, 3:] = np.cross(application - data.xipos[body], force_world)
        mujoco.mj_forward(model, data)
        measured = wrench() - baseline
        # MuJoCo reports parent-on-child support wrench, opposite the applied external load.
        expected = -np.r_[force_local, np.cross(offset_local, force_local)]
        rows.append(
            {
                "force_local_n": force_local.tolist(),
                "offset_local_m": offset_local.tolist(),
                "measured_delta": measured.tolist(),
                "expected_delta": expected.tolist(),
                "passed": bool(np.allclose(measured, expected, atol=1e-7, rtol=1e-6)),
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "cases": rows,
        "method": "welded clone; central and eccentric external loads; no changes to exploration state",
    }
