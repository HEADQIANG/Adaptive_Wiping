"""Velocity-reference extension used only by the pretraining environment."""

import numpy as np
from robosuite.controllers.parts.generic.joint_pos import JointPositionController


class JointPositionFeedforward(JointPositionController):
    def __init__(self, *, velocity_limit, **kwargs):
        super().__init__(**kwargs)
        if (
            self.input_type != "absolute"
            or self.impedance_mode != "fixed"
            or self.interpolator is not None
        ):
            raise ValueError(
                "Feedforward requires fixed-gain absolute joint goals without interpolation"
            )
        if not np.isfinite(velocity_limit) or velocity_limit <= 0 or self.control_freq <= 0:
            raise ValueError("Feedforward requires positive velocity limit and control frequency")
        self.velocity_limit = float(velocity_limit)
        self.goal_velocity = np.zeros(self.control_dim)
        self._previous_goal = None

    def set_goal(self, action, set_qpos=None):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.control_dim,) or not np.isfinite(action).all():
            raise ValueError("Expected finite absolute joint goals")
        super().set_goal(action, set_qpos)
        if self._previous_goal is None:
            self.goal_velocity[:] = 0
        else:
            self.goal_velocity = (self.goal_qpos - self._previous_goal) * self.control_freq
            # Preserve the joint-velocity direction, following the existing IK speed-limit convention.
            peak = np.max(np.abs(self.goal_velocity))
            if peak > self.velocity_limit:
                self.goal_velocity *= self.velocity_limit / peak
        self._previous_goal = self.goal_qpos.copy()

    def run_controller(self):
        torque = super().run_controller()
        correction = self.kd * self.goal_velocity
        if self.use_torque_compensation:
            correction = self.mass_matrix @ correction
        self.torques = torque + correction
        # FixedBaseRobot applies the unchanged actuator limits after this return.
        return self.torques

    def seed_reference(self, joint_position):
        """Start a new continuous nominal trajectory without a stale FF difference."""
        joint_position = np.asarray(joint_position, dtype=float)
        if joint_position.shape != (self.control_dim,) or not np.isfinite(joint_position).all():
            raise ValueError("Expected finite reference seed")
        self._previous_goal = joint_position.copy()
        self.goal_velocity[:] = 0

    def reset_goal(self):
        super().reset_goal()
        self.goal_velocity[:] = 0
        self._previous_goal = None
