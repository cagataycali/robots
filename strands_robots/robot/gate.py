"""The operator gate: a real body never moves because of a default.

A command from the developer's own code acts directly. A command from an agent tool or a dashboard
goes through ``gate``: a simulated body passes, a real body passes only when an operator already said
yes or the ``Robot(..., confirm=callable)`` answers True for these targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..types import Action, Observation

if TYPE_CHECKING:
    from .robot import Robot


def gate(robot: Robot, action: Action, *, confirmed: bool = False) -> np.ndarray:
    """Coerce ``action`` and, for a real body, require the operator's yes; otherwise raise ``PermissionError``."""
    targets = robot.coerce(action)
    if not robot.is_real or confirmed:
        return targets
    if robot.confirm is None:
        raise PermissionError(
            f"{robot.id} is a real robot and this command came through an agent or dashboard: pass "
            f"confirm=<callable(action) -> bool> to Robot(...) to allow it"
        )
    if not robot.confirm(targets):
        raise PermissionError(f"{robot.id}: the operator gate refused this action")
    return targets


def confirm_act(robot: Robot, action: Action, *, steps: int | None = None, confirmed: bool = False) -> Observation:
    """``act`` through the operator gate: what an agent tool and the dashboard call."""
    return robot.act(gate(robot, action, confirmed=confirmed), steps=steps)
