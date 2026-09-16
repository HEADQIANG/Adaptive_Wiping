"""Isolated fixed-orientation sponge force-ramp assay, not exploration data."""

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scripts.shared.common import (
    file_digest,
    load_config,
    provenance,
    write_json,
)
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.experiments.diagnose_contact import contact_result
from scripts.sim_pretrain.simulation import PretrainingWipe, contact_parameters
from scripts.sim_pretrain.tool_mount import vector


def build_rig(env, mode, parameters):
    if mode not in ("native", "friction_only", "mapped"):
        raise ValueError("Unknown contact assay mode")
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    table = copy.deepcopy(env.model.worldbody.find(".//geom[@name='table_collision']"))
    table.set("pos", "0 0 -0.025")
    table.set("size", ".25 .4 .025")
    world.append(table)
    prefix = env.robot.gripper["right"].naming_prefix
    tool = copy.deepcopy(env.model.worldbody.find(f".//body[@name='{prefix}wiping_gripper']"))
    rotation = env.goal_rotation
    tool.set("pos", "0 0 0.017")
    tool.set("quat", vector(np.roll(rotation.as_quat(), 1)))
    for name, axis in (("slide_y", [0, 1, 0]), ("slide_z", [0, 0, 1])):
        ET.SubElement(
            tool,
            "joint",
            name=name,
            type="slide",
            axis=vector(rotation.inv().apply(axis)),
            damping="0",
        )
    world.append(tool)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    source_option = env.sim.model._model.opt
    for name in dir(source_option):
        if not name.startswith("_"):
            value = getattr(source_option, name)
            if isinstance(value, np.ndarray):
                getattr(model.opt, name)[:] = value
            else:
                setattr(model.opt, name, value)
    data = mujoco.MjData(model)
    tool_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, env.sim.model.geom_id2name(i))
        for i in env.tool_ids
    ]
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    if mode == "friction_only":
        model.geom_friction[tool_ids, 0] = parameters[0]
    elif mode == "mapped":
        friction, solref, solimp = contact_parameters(*parameters)
        model.geom_friction[tool_ids] = friction
        model.geom_solref[tool_ids] = solref
        model.geom_solimp[tool_ids] = solimp
        model.geom_priority[tool_ids] = 1
        model.geom_priority[table_id] = 0
    mujoco.mj_forward(model, data)
    return model, data, tool_ids, table_id


def force_ramp(env, mode, parameters, normal_load):
    model, data, tool_ids, table_id = build_rig(env, mode, parameters)
    mass = float(model.body_mass.sum())
    prefix = env.robot.gripper["right"].naming_prefix
    ft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "ft_frame")
    rows, effective = [], None
    slip_start = None
    breakaway = None
    speed_start = None
    speed_reached = None
    y0 = 0
    for step in range(2500):
        t = step * 0.002
        if step == 1000:
            y0 = float(data.qpos[0])
        mu = effective["friction"][0] if effective else max(parameters[0], 0.03)
        ramp = max(0, t - 2) / 3
        tangential = (1.5 * mu * normal_load + 0.2) * ramp
        data.qfrc_applied[:] = [tangential, -normal_load + mass * 9.81 - 2 * data.qvel[1]]
        if t < 2:
            data.qfrc_applied[0] = -2 * data.qvel[0]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts = contact_result(model, data, tool_ids, table_id, data.site_xpos[ft], np.arange(2))
        if len(contacts["rows"]) and effective is None:
            contact = data.contact[int(contacts["rows"][0, 0])]
            effective = {
                "friction": contact.friction.tolist(),
                "solref": contact.solref.tolist(),
                "solimp": contact.solimp.tolist(),
                "condim": int(contact.dim),
            }
        rows.append(
            np.r_[
                data.time,
                data.qpos,
                data.qvel,
                tangential,
                normal_load,
                contacts["wrench"],
                contacts["normal_sum"],
            ]
        )
        if not np.isfinite(data.qpos).all() or data.qpos[1] < -0.031:
            return {
                "completed": False,
                "reason": "normal load exceeds 30 mm penetration envelope",
                "effective_contact": effective,
            }, np.asarray(rows)
        slipping = t >= 2 and data.qvel[0] > 0.005 and data.qpos[0] - y0 > 0.00025
        if slipping:
            if slip_start is None:
                slip_start = len(rows) - 1
            if breakaway is None and len(rows) - 1 - slip_start >= 25:
                breakaway = rows[slip_start]
        else:
            slip_start = None
        if t >= 2 and data.qvel[0] >= 0.05:
            if speed_start is None:
                speed_start = len(rows) - 1
            if len(rows) - 1 - speed_start >= 25:
                speed_reached = rows[speed_start]
                break
        else:
            speed_start = None
    trace = np.asarray(rows)
    settled = trace[(trace[:, 0] >= 1.8) & (trace[:, 0] <= 2)]
    normal_mean = float(settled[:, -1].mean()) if len(settled) else None
    row = {
        "completed": True,
        "normal_load_n": normal_load,
        "normal_achieved_n": normal_mean,
        "normal_settled": bool(
            normal_mean is not None and abs(normal_mean - normal_load) <= 0.1 * normal_load
        ),
        "effective_contact": effective,
        "breakaway_detected": breakaway is not None,
        "wiping_speed_reached": speed_reached is not None,
        "columns": [
            "time",
            "q_y",
            "q_z",
            "v_y",
            "v_z",
            "applied_fy",
            "normal_target",
            "contact_fx",
            "contact_fy",
            "contact_fz",
            "contact_mx_ft",
            "contact_my_ft",
            "contact_mz_ft",
            "normal_sum",
        ],
    }
    if breakaway is not None:
        row.update(
            breakaway_applied_fy_n=float(breakaway[5]),
            breakaway_contact_wrench_world_ft=breakaway[7:13].tolist(),
            breakaway_y_m=float(breakaway[1] - y0),
            breakaway_normal_n=float(breakaway[-1]),
        )
    if speed_reached is not None:
        row.update(
            wiping_speed_applied_fy_n=float(speed_reached[5]),
            wiping_speed_contact_wrench_world_ft=speed_reached[7:13].tolist(),
            wiping_speed_actual_m_s=float(speed_reached[3]),
        )
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from scripts.shared.run_paths import new_output

    args.output = new_output(args.output)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    report = {
        "config": cfg,
        "provenance": provenance(),
        "script_sha256": file_digest(__file__),
        "scope": "2-DOF isolated sponge; orientation fixed by fixture; not a robot gate",
        "cases": [],
        "protocol": "2 s force settle, up to 3 s tangential force ramp; slip >5 mm/s and 0.25 mm for 50 ms",
    }
    write_json(out / "report.json", report)
    env = PretrainingWipe(cfg, 300)
    try:
        bid = env.sim.model.geom_bodyid[env.tool_ids[0]]
        uniform = 0.03 / 12 * np.array([0.05**2 + 0.03**2, 0.12**2 + 0.03**2, 0.12**2 + 0.05**2])
        report["inertia_audit"] = {
            "tool_mass_kg": float(env.sim.model.body_mass[bid]),
            "inherited_tool_principal_inertia": env.sim.model.body_inertia[bid].tolist(),
            "uniform_120x50x30mm_box_inertia": uniform.tolist(),
            "note": "Uniform box is a diagnostic comparison, not a calibrated replacement; no inertia changed",
        }
        opt = env.sim.model._model.opt
        report["physics_options"] = {
            key: getattr(opt, key).tolist()
            if isinstance(getattr(opt, key), np.ndarray)
            else getattr(opt, key)
            for key in dir(opt)
            if not key.startswith("_")
        }
        cases = [("native", (0, 1000, 0.02))]
        cases += [
            (mode, (mu, 1000, 0.02)) for mode in ("friction_only", "mapped") for mu in (0, 0.9, 3.5)
        ]
        cases += [("mapped", (mu, 0.5, 0.02)) for mu in (0, 0.9, 3.5)]
        for mode, parameters in cases:
            for normal in (1.0, 3.0):
                index = len(report["cases"])
                row, trace = force_ramp(env, mode, parameters, normal)
                row.update(mode=mode, parameters=list(parameters), normal_target_n=normal)
                np.savez_compressed(out / f"rig_{index:02d}.npz", trace=trace)
                report["cases"].append(row)
                write_json(out / "report.json", report)
                print(
                    mode,
                    parameters,
                    normal,
                    row.get("breakaway_applied_fy_n", row.get("reason")),
                    flush=True,
                )
    finally:
        env.close()
    report["source_unchanged"] = provenance() == report["provenance"]
    write_json(out / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
