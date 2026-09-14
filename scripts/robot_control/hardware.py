"""AIRBOT 5.2.2 backend for the basic attended control session."""

import math

from .adapter import Robot
from .client import StateError, health, wait_controller


class HardwareRobot(Robot):
    hardware = True

    def switch(self, mode):
        self.owned()
        state = health(self.client)
        if state["controller_state"] not in ("idle", "servo", "gravity_comp"):
            raise StateError("Refusing to take over an unrelated controller")
        firmware = self.client.get_firmware_info()
        if (
            firmware is None
            or firmware.arm_sn != self.cfg["robot_sn"]
            or firmware.eef_type != self.cfg["expected_eef_type"]
        ):
            raise StateError("Robot/tool identity changed")
        if not self.arm_only:
            eef = self.client.get_eef_joint_state()
            if (
                eef is None
                or not math.isfinite(eef.eef_pos)
                or abs(eef.eef_pos - self.eef_pos) > 0.001
            ):
                raise StateError("Gripper position changed; no gripper motion permitted")
        if mode == "gravity_comp":
            accepted = self.client.enter_gravity_compensation_mode()
        elif mode == "servo":
            if not self.client.set_arm_speed([self.cfg["joint_speed_limit_rad_s"]] * 6):
                raise StateError("Speed limit rejected")
            accepted = self.client.switch_controller(self.controllers.servo_control)
        elif mode == "idle":
            accepted = self.client.switch_controller(self.controllers.idle)
        else:
            raise ValueError("Unsupported controller")
        if not accepted:
            raise StateError(f"Mode switch rejected: {mode}")
        wait_controller(self.client, mode)
        self.owned()

    def send_joint(self, joints):
        self.owned()
        if self.client._current_controller != self.controllers.servo_control:
            raise StateError("Joint targets require servo control")
        if self.arm_only:
            from arm_sdk._internal.airbot_proto import fsm_service_pb2 as pb

            request = pb.MovejointRequest(
                arm_dof=6,
                pos=joints,
                vel=[self.cfg["joint_speed_limit_rad_s"]] * 6,
                eff=self.options.eff,
                timeout_ms=20,
                blocking=False,
            )
            response = self.client._stub.move_joint(request, metadata=self.client._md, timeout=0.05)
            if response.message != "OK":
                raise StateError("Arm-only joint target rejected: " + response.message)
        elif not self.client.move_joint(joints, self.options, timeout_ms=20):
            raise StateError("Joint target rejected")
        if not self.arm_only and self.options.eef_pos != self.eef_pos:
            raise StateError("SDK clamped the preserved gripper target")
