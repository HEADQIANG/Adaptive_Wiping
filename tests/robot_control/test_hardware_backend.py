"""Pinned SDK request checks with mocked clients; never connect to hardware."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.robot_control.client import StateError
from scripts.robot_control.hardware import HardwareRobot
from scripts.robot_control.motion import fake_config


@unittest.skipUnless(importlib.util.find_spec("arm_sdk"), "Requires SDK types, no hardware")
class HardwareBackendTests(unittest.TestCase):
    def make_robot(self, eef_type="NULL"):
        from arm_sdk import Controller

        cfg = fake_config()
        cfg.update(robot_sn="fake", expected_eef_type=eef_type, eef_current_limit=0.1)
        client = Mock()
        client.get_firmware_info.return_value = SimpleNamespace(arm_sn="fake", eef_type=eef_type)
        client.get_eef_joint_state.return_value = SimpleNamespace(eef_pos=0.03)
        client.has_control.return_value = True
        client._current_controller = Controller.servo_control
        client._stub.move_joint.return_value = SimpleNamespace(message="OK")
        return HardwareRobot(client, cfg), client

    def test_arm_only_joint_request_has_no_gripper_fields(self):
        robot, client = self.make_robot()
        robot.send_joint([0.01] * 6)
        call = client._stub.move_joint.call_args
        request = call.args[0]
        self.assertEqual(request.arm_dof, 6)
        self.assertEqual(list(request.pos), [0.01] * 6)
        self.assertEqual(list(request.vel), [robot.cfg["joint_speed_limit_rad_s"]] * 6)
        self.assertEqual(list(request.eff), robot.cfg["joint_current_limits"])
        self.assertEqual(request.timeout_ms, 20)
        self.assertFalse(request.blocking)
        self.assertEqual(call.kwargs["timeout"], 0.05)
        self.assertFalse(
            {"eef_pos", "eef_vel", "eef_eff"} & {field.name for field, _ in request.ListFields()}
        )
        client.get_eef_joint_state.assert_not_called()
        client.move_joint.assert_not_called()
        client.move_eef.assert_not_called()
        client._stub.move_joint.return_value.message = "rejected"
        with self.assertRaisesRegex(StateError, "rejected"):
            robot.send_joint([0.01] * 6)

    def test_joint_target_requires_lease_and_servo(self):
        from arm_sdk import Controller

        robot, client = self.make_robot()
        client.has_control.return_value = False
        with self.assertRaisesRegex(StateError, "Control lost"):
            robot.send_joint([0.01] * 6)
        client.has_control.return_value = True
        client._current_controller = Controller.idle
        with self.assertRaisesRegex(StateError, "require servo"):
            robot.send_joint([0.01] * 6)
        client._stub.move_joint.assert_not_called()
        client.acquire_control.assert_not_called()

    def test_gripper_joint_target_is_preserved_and_clamp_rejected(self):
        robot, client = self.make_robot("G2")
        robot.send_joint([0.01] * 6)
        args, kwargs = client.move_joint.call_args
        self.assertEqual(args[1].eef_pos, 0.03)
        self.assertFalse(args[1].blocking)
        self.assertEqual(kwargs, {"timeout_ms": 20})
        client.move_eef.assert_not_called()

        def clamp(*args, **kwargs):
            args[1].eef_pos = 0.02
            return True

        client.move_joint.side_effect = clamp
        with self.assertRaisesRegex(StateError, "clamped"):
            robot.send_joint([0.01] * 6)

    def test_mode_switch_rejects_invalid_gripper_feedback_before_request(self):
        for value in (None, float("nan"), float("inf"), 0.04):
            with self.subTest(value=value):
                robot, client = self.make_robot("G2")
                client.get_eef_joint_state.return_value = (
                    None if value is None else SimpleNamespace(eef_pos=value)
                )
                with patch(
                    "scripts.robot_control.hardware.health",
                    return_value={"controller_state": "idle"},
                ):
                    with self.assertRaisesRegex(StateError, "Gripper"):
                        robot.switch("gravity_comp")
                client.enter_gravity_compensation_mode.assert_not_called()
                client.switch_controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
