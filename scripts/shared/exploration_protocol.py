"""Dependency-free counterpart of the frozen simulation displacement law."""

import math

SAMPLE_HZ = 100
DURATION_S = 4.0
INITIAL_GAP_M = 0.001
PRESS_SPEED_M_S = 0.005
PRESS_DURATION_S = 2.0
PRESS_DEPTH_M = PRESS_SPEED_M_S * PRESS_DURATION_S


def exploration_offset(t, press_speed=PRESS_SPEED_M_S, slide_speed=0.05):
    """Metres in the table frame: -Z press, +Y slide, -Y return."""
    if not all(math.isfinite(x) for x in (t, press_speed, slide_speed)):
        raise ValueError("Exploration inputs must be finite")
    if not 0 <= t <= DURATION_S or min(press_speed, slide_speed) < 0:
        raise ValueError("Expected time in [0, 4] and nonnegative speeds")
    return [0.0, slide_speed * (max(t - 2, 0) if t <= 3 else 4 - t), -press_speed * min(t, PRESS_DURATION_S)]
