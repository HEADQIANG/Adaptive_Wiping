"""Restore published link inertials in the pretraining XML copy only."""

import copy
import xml.etree.ElementTree as ET

import numpy as np

from scripts.shared.common import DISCOVERSE_INERTIA_SOURCE


def set_sponge_inertia(worldbody, prefix, profile="legacy"):
    """Estimate the existing sponge as a uniform box, before adding mount masses."""
    if profile == "legacy":
        return
    if profile != "uniform_box_v1":
        raise ValueError("Unknown tool_inertia_profile")
    tool = worldbody.find(f".//body[@name='{prefix}wiping_gripper']")
    inertial = None if tool is None else tool.find("inertial")
    geom = None if tool is None else tool.find(f"geom[@name='{prefix}wiping_surface_vis']")
    if inertial is None or geom is None or geom.get("type") != "box":
        raise ValueError("Uniform sponge inertia requires the existing box and inertial")
    if any(key in node.attrib for node in (geom, inertial)
           for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Uniform sponge inertia requires body-aligned principal axes")
    size = np.fromstring(geom.get("size", ""), sep=" ")
    center = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
    com = np.fromstring(inertial.get("pos", ""), sep=" ")
    mass = float(inertial.get("mass", "nan"))
    if (size.shape != (3,) or center.shape != (3,) or com.shape != (3,)
            or not all(np.isfinite(v).all() for v in (size, center, com))
            or not np.isfinite(mass) or mass <= 0 or np.any(size <= 0)
            or not np.allclose(center, com, rtol=0, atol=1e-12)):
        raise ValueError("Uniform sponge inertia requires positive mass/size and centered COM")
    # MuJoCo box sizes are half-extents; these moments are about the unchanged COM.
    moments = mass / 3 * (np.sum(size**2) - size**2)
    inertial.attrib.pop("fullinertia", None)
    inertial.set("diaginertia", " ".join(f"{value:.12g}" for value in moments))


def restore_discoverse_inertia(worldbody, prefix):
    reference = ET.parse(DISCOVERSE_INERTIA_SOURCE).getroot()
    replacements = []
    for index in range(1, 7):
        name = f"link{index}"
        original = reference.find(f".//body[@name='{name}']/inertial")
        target = worldbody.find(f".//body[@name='{prefix}{name}']")
        joint = worldbody.find(f".//joint[@name='{prefix}joint{index}']")
        if original is None or target is None or joint is None or target.find("inertial") is None:
            raise ValueError(f"Missing reference or target inertial/joint for {name}")
        if not {"pos", "mass", "diaginertia"}.issubset(original.attrib):
            raise ValueError(f"Incomplete DISCOVERSE inertial for {name}")
        replacements.append((original, target, joint))
    for original, target, joint in replacements:
        target.remove(target.find("inertial"))
        target.insert(0, copy.deepcopy(original))
        joint.set("armature", "0")
