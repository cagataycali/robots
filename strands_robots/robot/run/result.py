"""What a run measured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...types import Observation


@dataclass(frozen=True)
class RunResult:
    """What ``Robot.run`` measured: how many policy steps ran, over how long, and why it stopped."""

    steps: int
    seconds: float
    hz_target: float
    hz_achieved: float
    stopped_by: Literal["steps", "stop_when", "stopped", "fault", "estop"]
    recorded: str | None = None
    final: Observation | None = None
    fault: str | None = None  # why the policy could not go on (stopped_by == "fault")
