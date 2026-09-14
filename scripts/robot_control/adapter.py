"""Shared bounded SDK adapter for acquisition, deployment and basic control."""

import math

from .client import StateError, health, snapshot, wait_controller


class DeadlineStub:
    """Bound SDK 5.2.2's otherwise unbounded unary RPCs without editing the SDK."""

    def __init__(self, stub):
        self.stub = stub

    def __getattr__(self, name):
        call = getattr(self.stub, name)

        def bounded(*args, **kwargs):
            kwargs.setdefault("timeout", 1.5 if name == "SwitchController" else 0.05)
            return call(*args, **kwargs)

        return bounded


class Robot:
    def __init__(self, client, cfg):
        from arm_sdk import ArmControlOptions, CartesianPose, Controller
        from arm_sdk.utilities import _GRIPPER_SPECS

        self.client, self.cfg = client, cfg
        self.pose_type, self.controllers = CartesianPose, Controller
        firmware = client.get_firmware_info()
        if firmware is None or firmware.arm_sn != cfg["robot_sn"]:
            raise StateError("Connected robot serial number differs from approved setup")
        expected_eef = cfg.get("expected_eef_type")
        if expected_eef is not None and firmware.eef_type != expected_eef:
            raise StateError("Connected end-effector type differs from approved setup")
        self.arm_only = expected_eef == "NULL" and firmware.eef_type == "NULL"
        if self.arm_only:
            self.eef_pos = None
            self.options = ArmControlOptions(
                blocking=False,
                sampling_time=0.01,
                eff=cfg["joint_current_limits"],
                eef_pos=0.0,
                eef_eff=0.0,
            )
            return
        eef = client.get_eef_joint_state()
        if eef is None or not math.isfinite(eef.eef_pos):
            raise StateError(
                "Cannot preserve gripper position; check server end-effector configuration"
            )
        spec = _GRIPPER_SPECS.get(firmware.eef_type)
        if spec is not None and not spec[0] <= eef.eef_pos <= spec[1]:
            raise StateError(
                "Gripper feedback outside SDK bounds; refusing a silently clamped target"
            )
        self.eef_pos = eef.eef_pos
        self.options = ArmControlOptions(
            blocking=False,
            sampling_time=0.01,
            eff=cfg["joint_current_limits"],
            eef_pos=eef.eef_pos,
            eef_eff=cfg["eef_current_limit"],
        )

    def read(self, controller):
        return snapshot(self.client, controller)

    def acquire(self):
        if not self.client.acquire_control():
            raise StateError("Control acquisition refused")

    def owned(self):
        if not self.client.has_control():
            raise StateError("Control lost; no automatic reacquisition or recovery")

    def enter(self):
        self.owned()
        health(self.client, "idle")
        if self.arm_only:
            firmware = self.client.get_firmware_info()
            if (
                firmware is None
                or firmware.eef_type != "NULL"
                or firmware.arm_sn != self.cfg["robot_sn"]
            ):
                raise StateError("NULL end-effector identity changed during preflight")
        else:
            eef = self.client.get_eef_joint_state()
            if (
                eef is None
                or not math.isfinite(eef.eef_pos)
                or abs(eef.eef_pos - self.eef_pos) > 0.001
            ):
                raise StateError("Gripper position changed during preflight")
        if not self.client.set_arm_speed([self.cfg["joint_speed_limit_rad_s"]] * 6):
            raise StateError("Joint speed setting rejected")
        if not self.client.switch_controller(self.controllers.servo_control):
            raise StateError("Servo entry failed")
        wait_controller(self.client, "servo")

    def send(self, position, quaternion):
        self.owned()
        target = self.pose_type(position=position, orientation=quaternion)
        if self.arm_only:
            # Pinned 5.2.2 servo request, with no EEF fields. The public wrapper
            # unconditionally injects gripper speed and warns every cycle for NULL.
            from arm_sdk._internal.airbot_proto import fsm_service_pb2 as pb

            if self.client._current_controller != self.controllers.servo_control:
                raise StateError("Arm-only Cartesian requests require servo control")
            request = pb.MoveEndPoseRequest(
                arm_dof=6,
                target=pb.CartPos(position=target.position, orientation=target.orientation),
                vel=[self.cfg["joint_speed_limit_rad_s"]] * 6,
                eff=self.options.eff,
                timeout_ms=20,
                blocking=False,
            )
            response = self.client._stub.move_end_pose(
                request, metadata=self.client._md, timeout=0.05
            )
            if response.message != "OK":
                raise StateError(f"Arm-only Cartesian servo target rejected: {response.message}")
            return
        if not self.client.move_end_pose(target, self.options, timeout_ms=20):
            raise StateError("Cartesian servo target rejected")
        if self.options.eef_pos != self.eef_pos:
            raise StateError("SDK clamped gripper target; verify gripper configuration")

    def abort(self):
        # Never clear a stop, reacquire a lost lease, or initiate a blind retract.
        if not self.client.has_control() or not self.client.set_arm_emergency_stop(True):
            raise StateError("Software stop NOT acknowledged; use physical emergency stop NOW")

    def idle(self):
        self.owned()
        if not self.client.switch_controller(self.controllers.idle):
            raise StateError("Idle request rejected")
        wait_controller(self.client, "idle")
