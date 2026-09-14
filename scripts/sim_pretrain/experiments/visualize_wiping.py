"""Replay a fresh, unchanged pretraining rollout in MuJoCo; never collect training data."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from scripts.shared.common import load_config
from scripts.shared.paths import ROOT
from scripts.sim_pretrain.simulation import PretrainingWipe, rollout_metrics


class RecordedWipe(PretrainingWipe):
    def __init__(self, *args, **kwargs):
        self.recording = False
        self.frames = []
        super().__init__(*args, **kwargs)

    def prepare(self, extra_height=0.0):
        self.recording = False
        start = super().prepare(extra_height)
        self.frames = []
        self.time_zero = self.sim.data.time
        self.recording = True
        self.capture(start)
        return start

    def command(self, position, rotation=None, check_contacts=False):
        super().command(position, rotation, check_contacts)
        if self.recording:
            self.capture(position)

    def capture(self, target):
        model, data = self.sim.model._model, self.sim.data._data
        force = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            if self.table_id in (c.geom1, c.geom2) and (
                c.geom1 in self.tool_ids or c.geom2 in self.tool_ids
            ):
                local = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, local)
                sign = 1 if c.geom2 in self.tool_ids else -1
                force += sign * c.frame.reshape(3, 3).T @ local[:3]
        self.frames.append(
            dict(
                time=float(data.time - self.time_zero),
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                ctrl=data.ctrl.copy(),
                act=data.act.copy(),
                position=data.site_xpos[self.site_id].copy(),
                target=np.asarray(target).copy(),
                contact_force_world=force,
            )
        )


def restore_frame(model, data, frame):
    # Replay never integrates or writes back into the live simulation state.
    data.qpos[:] = frame["qpos"]
    data.qvel[:] = frame["qvel"]
    data.ctrl[:] = frame["ctrl"]
    data.act[:] = frame["act"]
    data.time = frame["time"]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def configure_view(camera, option, collisions, tabletop=False, closeup=False):
    camera.lookat[:] = [0.15, 0.0, 1.08] if tabletop else [-0.17, 0.0, 0.99]
    camera.distance = 1.25 if tabletop else 1.15
    camera.azimuth = 135
    camera.elevation = -25 if tabletop else -22
    if closeup:
        camera.lookat[:] = [0.31 if tabletop else 0, 0.025, 0.965]
        camera.distance = 0.42
        camera.elevation = -15
    option.geomgroup[0] = int(collisions)
    option.geomgroup[1] = 1
    option.sitegroup[:] = 0


def decorate(scene, frames, index):
    def geom(kind, position, color, end=None):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            kind,
            np.array([0.003, 0.003, 0.003]),
            np.asarray(position),
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.emission = 0.5
        if end is not None:
            width = 0.004 if kind == mujoco.mjtGeom.mjGEOM_ARROW else 2.0
            mujoco.mjv_connector(g, kind, width, np.asarray(position), np.asarray(end))
        scene.ngeom += 1

    current = frames[index]
    # Decimated past paths keep viewer geometry bounded without changing the recorded trajectory.
    for key, color in (("target", [1, 0.2, 0.15, 1]), ("position", [0, 0.65, 0.9, 1])):
        points = [f[key] for f in frames[: index + 1 : 4]]
        points.append(current[key])
        for start, end in zip(points, points[1:]):
            if np.linalg.norm(end - start) > 1e-7:
                geom(mujoco.mjtGeom.mjGEOM_LINE, start, color, end)
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, current[key], color)
    force = current["contact_force_world"]
    length = np.linalg.norm(force)
    if length > 1e-5:
        origin = current["position"] + np.array([0, -0.10, 0.04])
        geom(mujoco.mjtGeom.mjGEOM_LINE, current["position"], [0.6, 0.6, 0.6, 1], origin)
        end = origin + force * min(0.02, 0.12 / length)
        geom(mujoco.mjtGeom.mjGEOM_ARROW, origin, [1, 0.7, 0.05, 1], end)


def show_viewer(model, frames, args):
    import mujoco.viewer

    state = {"paused": False, "restart": False}
    lock = threading.Lock()

    def key_callback(key):
        with lock:
            if key == 32:
                state["paused"] = not state["paused"]
            elif key in (82, 114):
                state["restart"] = True

    data = mujoco.MjData(model)
    restore_frame(model, data, frames[0])
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            configure_view(viewer.cam, viewer.opt, args.collisions, args.tabletop, args.closeup)
        elapsed = 0.0
        previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            with lock:
                if state["restart"]:
                    elapsed = 0.0
                    state["restart"] = False
                elif not state["paused"]:
                    elapsed += (now - previous) * args.speed
            previous = now
            if elapsed >= 4.8:
                if args.once:
                    break
                elapsed = 0.0
            index = min(400, int(min(elapsed, 4.0) * 100 + 1e-8))
            with viewer.lock():
                restore_frame(model, data, frames[index])
                viewer.user_scn.ngeom = 0
                decorate(viewer.user_scn, frames, index)
            viewer.sync()
            time.sleep(0.01)


def export_gif(model, frames, args):
    from PIL import Image, ImageDraw

    path = Path(args.gif).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultOption(option)
    configure_view(camera, option, args.collisions, args.tabletop, args.closeup)
    images = []
    palette = None
    fps = 20
    with mujoco.Renderer(model, height=480, width=640, max_geom=2000) as renderer:
        for wall_time in np.arange(0, 4.8 / args.speed, 1 / fps):
            index = min(400, int(min(wall_time * args.speed, 4.0) * 100 + 1e-8))
            restore_frame(model, data, frames[index])
            renderer.update_scene(data, camera=camera, scene_option=option)
            decorate(renderer.scene, frames, index)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            phase = "PRESS" if index < 200 else "SLIDE +Y" if index < 300 else "SLIDE -Y"
            draw.rectangle((0, 0, 640, 25), fill=(245, 245, 245))
            draw.text(
                (10, 6),
                f"{phase}   t={frames[index]['time']:.2f}s   mu={args.mu:g}   k={args.stiffness:g}",
                fill=(20, 20, 20),
            )
            if palette is None:
                palette = im.quantize(colors=240)
                colors = palette.getpalette()[:720]
                # Reserve overlay colors; an adaptive GIF palette can erase thin colored arrows.
                colors += [
                    255,
                    255,
                    0,
                    255,
                    180,
                    0,
                    255,
                    60,
                    40,
                    255,
                    0,
                    0,
                    0,
                    180,
                    230,
                    0,
                    255,
                    255,
                    255,
                    255,
                    255,
                    0,
                    0,
                    0,
                ] * 2
                palette.putpalette(colors)
            images.append(im.quantize(palette=palette, dither=Image.Dither.NONE))
    with path.open("xb") as stream:
        images[0].save(
            stream,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0,
            disposal=2,
        )
    print(f"GIF: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sim_pretrain/pretrain_paper.yaml"))
    parser.add_argument("--mu", type=float, default=1.75)
    parser.add_argument("--stiffness", type=float, default=500.25)
    parser.add_argument("--width", type=float, default=0.16)
    parser.add_argument("--gain", type=float, default=300)
    parser.add_argument(
        "--speed", type=float, default=0.5, help="Playback speed only; physics is unchanged"
    )
    parser.add_argument("--collisions", action="store_true")
    parser.add_argument(
        "--closeup", action="store_true", help="Frame the compact tool mounting area"
    )
    parser.add_argument("--once", action="store_true", help="Close viewer after one playback")
    parser.add_argument("--gif", help="Optionally export a 640x480 animated GIF; refuses overwrite")
    parser.add_argument(
        "--no-viewer", action="store_true", help="GIF export without an interactive window"
    )
    args = parser.parse_args()
    if not all(
        np.isfinite(x) for x in (args.mu, args.stiffness, args.width, args.gain, args.speed)
    ):
        parser.error("Parameters must be finite")
    if args.mu < 0 or min(args.stiffness, args.width, args.gain) <= 0 or args.gain > 1000:
        parser.error("Require mu >= 0, positive stiffness/width/gain, and gain <= 1000")
    if not 0.1 <= args.speed <= 4:
        parser.error("Playback speed must be between 0.1 and 4")
    if args.no_viewer and not args.gif:
        parser.error("--no-viewer requires --gif")
    if args.gif and Path(args.gif).exists():
        parser.error("GIF already exists; choose a new path")
    cfg = load_config(args.config)
    args.tabletop = cfg["simulation"].get("base_mount", "stand") == "tabletop"
    print(
        f"Simulating current model: mu={args.mu}, k={args.stiffness}, width={args.width}, gain={args.gain}",
        flush=True,
    )
    print(
        "Preparation and 4-second exploration are computed before playback. No training data or gate writes.",
        flush=True,
    )
    env = RecordedWipe(cfg, args.gain, (args.mu, args.stiffness, args.width))
    try:
        rollout = env.rollout()
        if len(env.frames) != 401:
            raise RuntimeError("Expected start pose plus 400 recorded frames")
        print(json.dumps(rollout_metrics(rollout), indent=2), flush=True)
        print(
            "Red: target; cyan: actual TCP; yellow: resultant contact force on tool (scaled).",
            flush=True,
        )
        if args.gif:
            export_gif(env.sim.model._model, env.frames, args)
        if not args.no_viewer:
            print(
                "Space: pause/resume; R: restart. Default: repeat at half speed. Close window to exit.",
                flush=True,
            )
            show_viewer(env.sim.model._model, env.frames, args)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
