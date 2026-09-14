"""Attended motion state machine. Fake and SDK backends share the same guards."""

import copy
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .client import StateError, vector
from .safety import check_bounds, positive, quaternion_angle

DT = 0.01
FLAGS = (
    "commissioning_verified",
    "physical_estop_verified",
    "server_no_return_verified",
    "tool_and_swept_path_verified",
    "sdk_frame_verified",
    "payload_verified",
)


def validate_config(cfg, *, hardware=False):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported robot-control configuration")
    if hardware and any(cfg.get(k) is not True for k in FLAGS):
        raise ValueError(
            "On-site confirmations missing: "
            + ", ".join(k for k in FLAGS if cfg.get(k) is not True)
        )
    if hardware and cfg.get("source_kind") != "real":
        raise ValueError("A synthetic configuration cannot command hardware")
    if not isinstance(cfg.get("robot_sn"), str) or not cfg["robot_sn"].strip():
        raise ValueError("Set the approved robot serial number")
    for key, n in (
        ("joint_min_rad", 6),
        ("joint_max_rad", 6),
        ("joint_current_limits", 6),
    ):
        vector(cfg.get(key, []), n, key)
    if any(a >= b for a, b in zip(cfg["joint_min_rad"], cfg["joint_max_rad"])):
        raise ValueError("Invalid joint_min_rad/joint_max_rad")
    for value in cfg["joint_current_limits"]:
        positive(value, "joint_current_limits", 20)
    caps = {
        "joint_speed_limit_rad_s": 0.5,
        "measured_joint_speed_stop_rad_s": 1.0,
        "max_cartesian_speed_m_s": 0.05,
        "max_angular_speed_rad_s": 0.3,
        "joint_step_rad": 0.05,
        "translation_step_m": 0.005,
        "rotation_step_rad": 0.05,
        "max_joint_tracking_error_rad": 0.15,
        "max_tracking_error_m": 0.01,
        "orientation_error_limit_rad": 0.15,
        "max_motion_duration_s": 30,
        "input_timeout_s": 0.5,
        "settle_timeout_s": 5,
        "target_joint_tolerance_rad": 0.02,
        "target_position_tolerance_m": 0.002,
        "target_orientation_tolerance_rad": 0.02,
    }
    for key, cap in caps.items():
        positive(cfg.get(key), key, cap)
    if cfg["measured_joint_speed_stop_rad_s"] < cfg["joint_speed_limit_rad_s"]:
        raise ValueError("Measured speed threshold must be >= commanded speed limit")
    if cfg.get("expected_eef_type") != "NULL":
        positive(cfg.get("eef_current_limit"), "eef_current_limit", 20)
    return cfg


def fake_config():
    cfg = dict(
        schema_version=1,
        source_kind="synthetic",
        robot_sn="FAKE_OFFLINE",
        expected_eef_type="NULL",
        joint_min_rad=[-3.0] * 6,
        joint_max_rad=[3.0] * 6,
        joint_current_limits=[1.0] * 6,
        joint_speed_limit_rad_s=0.1,
        measured_joint_speed_stop_rad_s=0.2,
        max_cartesian_speed_m_s=0.01,
        max_angular_speed_rad_s=0.1,
        joint_step_rad=0.01,
        translation_step_m=0.001,
        rotation_step_rad=0.01,
        max_joint_tracking_error_rad=0.1,
        max_tracking_error_m=0.005,
        orientation_error_limit_rad=0.1,
        max_motion_duration_s=30.0,
        input_timeout_s=0.2,
        settle_timeout_s=2.0,
        target_joint_tolerance_rad=0.005,
        target_position_tolerance_m=0.001,
        target_orientation_tolerance_rad=0.01,
    )
    cfg.update({k: False for k in FLAGS})
    return cfg


class FakeRobot:
    """Deterministic software backend, deliberately NOT a dynamics or collision model."""

    hardware = False

    def __init__(self):
        self.mode, self.lease, self.stopped = "idle", False, False
        self.state = dict(
            joint_position_rad=[0.0] * 6,
            joint_velocity_rad_s=[0.0] * 6,
            sdk_end_position_m=[0.25, 0.0, 0.3],
            sdk_end_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        self.calls = []

    def read(self, expected=None):
        if expected is not None and expected != self.mode:
            raise StateError("Unexpected fake controller")
        return copy.deepcopy(self.state)

    def owned(self):
        if not self.lease:
            raise StateError("Control lost; no reacquisition")

    def acquire(self):
        if self.lease or self.mode != "idle":
            raise StateError("Robot is already controlled")
        self.lease = True
        self.calls.append("acquire")

    def switch(self, mode):
        self.owned()
        if self.stopped:
            raise StateError("Stop is latched")
        self.mode = mode
        self.calls.append(mode)

    def send_joint(self, joints):
        self.owned()
        if self.stopped or self.mode != "servo":
            raise StateError("Joint command requires active servo")
        self.state["joint_position_rad"] = list(joints)
        self.calls.append("joint")

    def send(self, position, quaternion):
        self.owned()
        if self.stopped or self.mode != "servo":
            raise StateError("Pose command requires active servo")
        self.state["sdk_end_position_m"] = list(position)
        self.state["sdk_end_orientation_xyzw"] = list(quaternion)
        self.calls.append("pose")

    def abort(self):
        self.owned()
        self.stopped = True
        self.calls.append("stop")

    def idle(self):
        self.switch("idle")


class MotionSession:
    def __init__(
        self, robot, cfg, *, emit=lambda event: None, clock=time.monotonic, sleep=time.sleep
    ):
        self.robot, self.cfg = robot, validate_config(cfg, hardware=robot.hardware)
        self.emit, self.clock, self.sleep = emit, clock, sleep
        self.mode, self.acquired, self.faulted = "idle", False, False
        self.hold_target = None
        self.joint_axis = 0

    def event(self, event_name, **fields):
        self.emit(
            dict(
                event=event_name,
                monotonic_s=self.clock(),
                mode=self.mode,
                source_kind="real" if self.robot.hardware else "synthetic",
                **fields,
            )
        )

    def read(self):
        if self.acquired:
            self.robot.owned()
        state = self.robot.read(self.mode)
        for key, size in (
            ("joint_position_rad", 6),
            ("joint_velocity_rad_s", 6),
            ("sdk_end_position_m", 3),
            ("sdk_end_orientation_xyzw", 4),
        ):
            vector(state[key], size, key)
        if abs(np.linalg.norm(state["sdk_end_orientation_xyzw"]) - 1) > 0.001:
            raise StateError("Invalid measured quaternion")
        check_bounds(state, self.cfg)
        if (
            max(map(abs, state["joint_velocity_rad_s"]))
            > self.cfg["measured_joint_speed_stop_rad_s"]
        ):
            raise StateError("Measured joint speed exceeded configured stop threshold")
        return state

    def begin(self):
        self.read()
        self.robot.acquire()
        self.acquired = True
        self.read()
        self.event("acquired")

    def stationary(self):
        from .airbot_initial_pose import (
            SAMPLE_COUNT,
            SAMPLE_INTERVAL_S,
            validate_stationary,
        )

        samples = []
        for i in range(SAMPLE_COUNT):
            if i:
                self.sleep(SAMPLE_INTERVAL_S)
            samples.append(self.maintain())
        validate_stationary(samples)
        return samples

    def switch(self, mode):
        if self.faulted:
            raise StateError("Session stopped; no automatic recovery")
        mode = "gravity_comp" if mode in ("gravity-comp", "zero-gravity") else mode
        if mode not in ("servo", "gravity_comp", "idle"):
            raise ValueError(f"Unsupported controller: {mode}")
        state = self.stationary()[-1]
        self.hold_target = None
        self.robot.switch(mode)
        self.mode = mode
        after = self.read()
        if (
            max(
                abs(a - b) for a, b in zip(state["joint_position_rad"], after["joint_position_rad"])
            )
            > self.cfg["target_joint_tolerance_rad"]
        ):
            raise StateError("Robot moved during controller transition")
        if mode == "servo":
            self.hold_target = ("joint", after["joint_position_rad"])
            self.robot.send_joint(after["joint_position_rad"])
        self.event("mode-confirmed", controller=mode)

    def hold(self):
        if self.faulted:
            raise StateError("Session stopped; no automatic recovery")
        if self.mode != "servo":
            self.switch("servo")
        else:
            state = self.read()
            self.hold_target = ("joint", state["joint_position_rad"])
            self.robot.send_joint(state["joint_position_rad"])
            self.event("hold", joint_position_rad=state["joint_position_rad"])

    def tracking(self, state, kind, target, *, final=False):
        c = self.cfg
        if kind == "joint":
            error = max(abs(a - b) for a, b in zip(state["joint_position_rad"], target))
            return (
                error
                <= c["target_joint_tolerance_rad" if final else "max_joint_tracking_error_rad"]
            )
        pos, quat = target
        return (
            math.dist(state["sdk_end_position_m"], pos)
            <= c["target_position_tolerance_m" if final else "max_tracking_error_m"]
            and quaternion_angle(state["sdk_end_orientation_xyzw"], quat)
            <= c["target_orientation_tolerance_rad" if final else "orientation_error_limit_rad"]
        )

    def maintain(self):
        state = self.read()
        if self.mode == "servo" and self.hold_target is not None:
            kind, target = self.hold_target
            if not self.tracking(state, kind, target):
                raise StateError("Position holding error exceeded threshold")
            self.send(kind, target)
        return state

    def send(self, kind, target):
        if self.faulted:
            raise StateError("No targets allowed after stop")
        if kind == "joint":
            self.robot.send_joint(target)
        else:
            self.robot.send(*target)

    def plan(self, kind, target, state):
        cfg = self.cfg
        if kind == "joint":
            target = vector(target, 6, "joint target")
            if any(
                not lo <= q <= hi
                for q, lo, hi in zip(target, cfg["joint_min_rad"], cfg["joint_max_rad"])
            ):
                raise ValueError("Joint target outside approved limits")
            start = np.asarray(state["joint_position_rad"])
            duration = 1.875 * max(abs(np.asarray(target) - start)) / cfg["joint_speed_limit_rad_s"]
        elif kind == "pose":
            position, quaternion = target
            position = vector(position, 3, "position target")
            quaternion = vector(quaternion, 4, "quaternion xyzw")
            if abs(np.linalg.norm(quaternion) - 1) > 1e-6:
                raise ValueError("Target quaternion must be normalized")
            start = np.asarray(state["sdk_end_position_m"])
            duration = 1.875 * max(
                math.dist(start, position) / cfg["max_cartesian_speed_m_s"],
                quaternion_angle(state["sdk_end_orientation_xyzw"], quaternion)
                / cfg["max_angular_speed_rad_s"],
            )
            rotation = Slerp(
                [0.0, 1.0], Rotation.from_quat([state["sdk_end_orientation_xyzw"], quaternion])
            )
        else:
            raise ValueError("Unknown motion type")
        if duration > cfg["max_motion_duration_s"]:
            raise ValueError(
                "Target requires too long a motion; choose a nearer intermediate target"
            )
        count = max(1, math.ceil(duration / DT))
        for tick in range(1, count + 1):
            t = tick / count
            alpha = 10 * t**3 - 15 * t**4 + 6 * t**5
            if kind == "joint":
                yield (start + alpha * (np.asarray(target) - start)).tolist()
            else:
                yield (
                    (start + alpha * (np.asarray(position) - start)).tolist(),
                    rotation(alpha).as_quat().tolist(),
                )

    def move(self, kind, target, *, stop_requested=lambda: False):
        if self.faulted:
            raise StateError("Session stopped")
        start = self.read()
        trajectory = list(self.plan(kind, target, start))
        if self.mode != "servo":
            self.switch("servo")
            start = self.read()
            trajectory = list(self.plan(kind, target, start))
        previous = (
            start["joint_position_rad"]
            if kind == "joint"
            else (start["sdk_end_position_m"], start["sdk_end_orientation_xyzw"])
        )
        deadline = self.clock()
        for point in trajectory:
            if stop_requested():
                self.stop()
                return False
            state = self.read()
            if not self.tracking(state, kind, previous):
                raise StateError("Motion tracking error exceeded threshold")
            if self.robot.hardware and self.clock() - deadline > 0.05:
                raise StateError("Control tick missed; no catch-up burst")
            self.send(kind, point)
            previous = point
            self.event("target", kind=kind, target=point, state=state)
            deadline += DT
            self.sleep(max(0.0, deadline - self.clock()))
        until = self.clock() + self.cfg["settle_timeout_s"]
        while True:
            if stop_requested():
                self.stop()
                return False
            state = self.read()
            if not self.tracking(state, kind, previous):
                raise StateError("Final target tracking error exceeded threshold")
            if self.tracking(state, kind, previous, final=True):
                break
            if self.clock() > until:
                raise StateError("Target did not settle before deadline")
            self.send(kind, previous)
            self.sleep(DT)
        self.hold_target = (kind, previous)
        self.event("target-reached", kind=kind, state=state)
        return True

    def jog(self, key, *, stop_requested=lambda: False):
        if not key:
            return
        if self.faulted:
            raise StateError("Session stopped")
        if key == " ":
            self.stop()
            return
        if key in "123456":
            self.joint_axis = int(key) - 1
            return
        if self.mode != "servo":
            raise StateError("Keyboard motion requires explicit servo/hold entry")
        state = self.read()
        if key in "+-":
            q = list(state["joint_position_rad"])
            q[self.joint_axis] += self.cfg["joint_step_rad"] * (1 if key == "+" else -1)
            return self.move("joint", q, stop_requested=stop_requested)
        if key in "xXyYzZ":
            p = list(state["sdk_end_position_m"])
            p["xyz".index(key.lower())] += self.cfg["translation_step_m"] * (
                1 if key.islower() else -1
            )
            return self.move(
                "pose", (p, state["sdk_end_orientation_xyzw"]), stop_requested=stop_requested
            )
        if key in "rRpPwW":
            rotvec = np.zeros(3)
            rotvec["rpw".index(key.lower())] = self.cfg["rotation_step_rad"] * (
                1 if key.islower() else -1
            )
            # Extrinsic rotations about the SDK reference axes.
            q = (
                Rotation.from_rotvec(rotvec) * Rotation.from_quat(state["sdk_end_orientation_xyzw"])
            ).as_quat()
            return self.move(
                "pose", (state["sdk_end_position_m"], q), stop_requested=stop_requested
            )
        if key == " ":
            self.stop()

    def stop(self):
        self.hold_target = None
        self.faulted = True
        if not self.acquired:
            return
        self.robot.abort()
        self.event("software-stop-acknowledged", hardware_estop_required_if_unsafe=True)

    def finish(self, *, supported):
        if not supported:
            raise StateError("Support confirmation required before idle")
        if self.acquired and not self.faulted:
            self.hold_target = None
            self.robot.idle()
            self.mode = "idle"
            self.read()
            self.event("idle-confirmed", position_holding=False)
