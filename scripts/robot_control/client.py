"""Pinned AIRBOT SDK access; no automatic control-lease acquisition."""

import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version

SDK_VERSION = "5.2.2"


class StateError(RuntimeError):
    """Robot state is unavailable, invalid, or no longer under our control."""


def vector(values, length, name):
    result = [float(v) for v in values]
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise StateError(f"Invalid {name}: expected {length} finite values")
    return result


def health(client, controller=None):
    state = client.get_service_state()
    if state is None or not state.valid or not state.service_state:
        raise StateError("Service state missing, stale, or unavailable")
    if controller is not None and state.controller_state != controller:
        raise StateError(f"Expected {controller}, got {state.controller_state!r}")
    motors = client.get_arm_motor_state()
    if motors is None or len(motors.error_ids) != 6:
        raise StateError("Missing six-joint motor status")
    # AIRBOT-Play-Hardware MotorState::error_id defines both 0x00 and 0x01 as no error.
    # See docs/robot_control/airbot_initial_pose.md section 9 for the pinned upstream definition.
    if any(code not in (0x00, 0x01) for code in motors.error_ids):
        raise StateError(f"Motor fault: {motors.error_ids}")
    result = asdict(state)
    result["motor_status_codes"] = list(motors.error_ids)
    return result


def snapshot(client, controller=None):
    started = time.monotonic()
    state = health(client, controller)
    joints = client.get_arm_joint_state()
    pose = client.get_end_pose()
    if joints is None or pose is None:
        raise StateError("Joint state or end pose unavailable")
    angles = vector(joints.angles, 6, "joint angles")
    velocities = vector(joints.velocities, 6, "joint velocities")
    position = vector(pose.position, 3, "end position")
    quaternion = vector(pose.orientation, 4, "quaternion xyzw")
    if abs(math.sqrt(sum(v * v for v in quaternion)) - 1.0) > 0.01:
        raise StateError("End-pose quaternion is not normalized")
    elapsed = time.monotonic() - started
    if elapsed > 0.5:
        raise StateError(f"State read too slow: {elapsed:.3f}s")
    return {
        "host_time_utc": datetime.now(timezone.utc).isoformat(),
        "host_monotonic_s": time.monotonic(),
        "read_duration_s": elapsed,
        "joint_position_rad": angles,
        "joint_velocity_rad_s": velocities,
        "sdk_end_position_m": position,
        "sdk_end_orientation_xyzw": quaternion,
        "service_state": state,
    }


def open_client(host, port):
    installed = version("arm-sdk")
    if installed != SDK_VERSION:
        raise RuntimeError(f"Requires arm-sdk {SDK_VERSION}, found {installed}")
    from arm_sdk import AirbotClient

    class TeachingClient(AirbotClient):
        # The 5.2.2 decorator otherwise reacquires control during mode switches.
        def _try_auto_acquire(self):
            return False

        def has_control(self):
            with self._lease_mu:
                return (
                    self._lease_id is not None
                    and self._lease_expire_unix_ms is not None
                    and self._lease_expire_unix_ms > time.time() * 1000
                )

    return TeachingClient(host=host, port=port)


def wait_controller(client, expected):
    deadline = time.monotonic() + 2.0
    while True:
        state = health(client)
        if state["controller_state"] == expected:
            return
        if time.monotonic() >= deadline:
            raise StateError(f"Controller did not become {expected}: {state}")
        time.sleep(0.05)
