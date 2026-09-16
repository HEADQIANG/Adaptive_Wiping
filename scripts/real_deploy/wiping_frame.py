"""Axis-aligned table/wall motion mappings; wrench channels stay sensor-local."""

from dataclasses import dataclass

import numpy as np

WALL_DIRECTIONS = ("+x", "-x", "+y", "-y")


@dataclass(frozen=True)
class WipingFrame:
    mode: str = "horizontal"
    wall_direction: str | None = None

    def __post_init__(self):
        if self.mode not in ("horizontal", "vertical"):
            raise ValueError("Unknown wiping mode")
        if self.mode == "vertical" and self.wall_direction not in WALL_DIRECTIONS:
            raise ValueError("Vertical wiping requires --wall-direction=+x, -x, +y or -y")
        if self.mode == "horizontal" and self.wall_direction is not None:
            raise ValueError("--wall-direction is only valid with --wiping-mode vertical")

    @property
    def normal_axis(self):
        return "xy".index(self.wall_direction[-1]) if self.mode == "vertical" else 2

    @property
    def normal_sign(self):
        # Negative learned delta-h presses into the surface in either mode.
        return -1 if self.wall_direction in ("+x", "+y") else 1

    @property
    def tangent_axes(self):
        return ((2, 1), (0, 2), (0, 1))[self.normal_axis]

    @property
    def tangent_signs(self):
        if self.normal_axis == 0:
            return (-self.normal_sign, 1)
        if self.normal_axis == 1:
            return (1, -self.normal_sign)
        return (1, 1)

    @property
    def tool_rotation_axis_angle(self):
        """World-axis rotation from the table tool pose; applied manually before h."""
        if self.normal_axis == 0:
            return "Y", 90 * self.normal_sign
        if self.normal_axis == 1:
            return "X", -90 * self.normal_sign
        return "Z", 0

    @property
    def rotation(self):
        rotation = np.zeros((3, 3))
        rotation[list(self.tangent_axes), [0, 1]] = self.tangent_signs
        rotation[self.normal_axis, 2] = self.normal_sign
        return rotation

    def endpoint(self, tangent, normal):
        point = np.empty(3)
        point[list(self.tangent_axes)] = tangent
        point[self.normal_axis] = normal
        return point

    def translate_path(self, original, start):
        return (original - original[0]) * self.tangent_signs + start[list(self.tangent_axes)]

    def metadata(self):
        return {"wiping_mode": self.mode, "wall_direction": self.wall_direction,
                "normal_axis_sdk": "XYZ"[self.normal_axis], "delta_h_to_sdk_sign": self.normal_sign,
                "tangent_axes_sdk": ["XYZ"[i] for i in self.tangent_axes],
                "tangent_signs": list(self.tangent_signs),
                "task_rotation_to_sdk": self.rotation.tolist(),
                "ft_mapping": "sensor_local_unchanged_after_onsite_tare"}

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.get("wiping_mode", "horizontal"), settings.get("wall_direction"))
