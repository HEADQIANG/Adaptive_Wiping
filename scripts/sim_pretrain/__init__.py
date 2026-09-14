"""Simulation and pretraining using the local robosuite fork."""

import sys

from scripts.shared.paths import ROBOSUITE_REPO

# The vendor fork is an explicit external dependency, not an application import shortcut.
_vendor = str(ROBOSUITE_REPO)
if _vendor not in sys.path:
    sys.path.insert(0, _vendor)
