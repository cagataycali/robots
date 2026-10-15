"""The safety chain every control write from an agent or a dashboard passes: range -> safe -> interlocks
-> validators -> operator gate. The developer's own ``robot.act`` skips it (they are the operator); the
e-stop bypasses it entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...types import Action
from .validators import JointLimitValidator, Request, RequestRefused, Validator, outside_safe

if TYPE_CHECKING:
    from ..robot import Robot


class SafetyChain:
    """``check(robot, action)`` returns the targets to write, or raises ``RequestRefused``."""

    def __init__(self, validators: tuple[Validator, ...] = ()) -> None:
        self.validators = validators

    def check(self, robot: Robot, action: Action, *, confirmed: bool = False, source: str = "agent") -> np.ndarray:
        from ..gate import operator_gate

        robot.estop_latch.check(robot.id)
        request = Request(robot.action_keys, robot.coerce(action, clip=False), confirmed, source)
        JointLimitValidator()(robot, request)
        soft = outside_safe(robot, request)
        if soft and not confirmed:
            if robot.confirm is None or not robot.confirm(request.targets):
                raise RequestRefused(f"{robot.id}: {'; '.join(soft)}: needs the operator's yes (confirm=)")
            confirmed = True  # the operator was asked once, for exactly these targets
        for v in (*robot.interlocks, *self.validators, *robot.validators):
            v(robot, request)
        operator_gate(robot, request.targets, confirmed=confirmed)
        return request.targets
