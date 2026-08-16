"""Behavior tests for the accepted ``n_steps`` domain of a hardware task.

``strands_robots.hardware_robot.Robot`` bounds every rollout by two ANDed
conditions: elapsed wall-clock time against ``duration``, and the applied-action
count against the optional ``n_steps`` cap
(``n_steps is None or step_count < n_steps``). ``duration`` has been refused up
front since it became the effective horizon; the step cap beside it in the same
signature was read straight into that comparison. These tests pin that a cap the
loop cannot count against is refused at the public entry point instead of being
spent on the arm:

    - ``0`` / negative / ``nan`` / ``False`` make the comparison false on its
      first evaluation, so the task reported ``status="success"`` and "Policy
      rollout completed: 0 steps" for a rollout that never queried the policy
      and never commanded a servo - the same false ``completed`` an unusable
      ``duration`` used to produce;
    - ``inf`` is never false, so the requested cap silently vanished and the
      rollout ran to the ``duration`` budget instead;
    - ``True`` read as a silent cap of one, and a float cap applied a count the
      caller never named (``2.7`` stopped after three applied actions);
    - a non-numeric cap reached the comparison intact and surfaced a bare
      ``TypeError`` naming a comparison internal ("'<' not supported between
      instances of 'int' and 'str'") rather than the parameter;
    - the refusal happens before the policy is initialized and before the arm is
      commanded, so a cap that cannot be honored costs no inference and no write;
    - ``_execute_task_sync`` refuses too: it is the chokepoint the agent-tool
      ``execute`` action and the mesh ``execute`` dispatch reach directly, so a
      peer-supplied cap is bounded by the same rule;
    - ``None`` stays the documented "no cap" spelling, and every cap that IS
      accepted still stops the rollout at exactly that many applied actions;
    - the accepted domain matches the simulation's rollout horizon
      (``SimEngine._resolve_horizon``), so the same cap cannot be refused for a
      digital twin and accepted for the arm it mirrors.

No serial/USB hardware is touched: the driver is an in-memory fake and the
policy is a structural stub.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.simulation.base import SimEngine
from tests.test_hardware_control_loop_rate_guard import _FakeArm

# Caps the loop cannot count against. ``0`` / negative / ``nan`` / ``False``
# make ``step_count < n_steps`` false immediately (a task that commands
# nothing); ``inf`` never makes it false (the cap vanishes); ``True`` is a
# silent cap of one; a float applies a count nobody named; the rest are not
# counts the comparison can be made against at all. ``np.int64`` is refused for
# parity with the simulation horizon, which requires a true ``int``.
UNUSABLE_STEP_CAPS: list[Any] = [
    0,
    -1,
    -5,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    2.7,
    3.0,
    "10",
    [4],
    {},
    np.int64(3),
    np.float64(4.0),
]

# Caps that are honorable: a positive ``int``. ``None`` is the documented
# "no cap" spelling and is covered by its own test rather than this list.
USABLE_STEP_CAPS: list[int] = [1, 2, 4, 25]


class _CountingPolicy:
    """Structural stand-in for a policy that records every inference."""

    supports_rtc = False
    execution_horizon = 1

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, seed: int | None = None) -> None:
        return None

    def set_control_frequency(self, hz: float) -> None:
        return None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        return None

    async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
        self.calls += 1
        return [{"j0.pos": 0.1}]


@pytest.fixture
def hw() -> Any:
    """A ``Robot`` wired to an in-memory arm, with the connect path stubbed.

    ``policy_inits`` records every policy initialization so a test can assert a
    refused cap never got that far, and ``robot.sent_actions`` records every
    command that reached the arm. The control frequency is high so a rollout
    bounded by ``duration`` alone finishes many steps inside a short budget.
    """
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = "test_arm"
    robot.action_horizon = 1
    robot.data_config = None
    robot.control_frequency = 200.0
    robot.action_sleep_time = 1.0 / 200.0
    robot._task_state = RobotTaskState()
    robot._executor = ThreadPoolExecutor(max_workers=1)
    robot._shutdown_event = threading.Event()
    robot._stop_requested = threading.Event()
    robot._task_admission = threading.Lock()
    robot._task_claimed = False
    robot.mesh = None
    robot.peer_id = None
    robot.robot = _FakeArm()
    robot.policy_inits = []  # type: ignore[attr-defined]

    async def _connected() -> tuple[bool, str]:
        return (True, "")

    async def _ready() -> bool:
        return True

    def _init_policy(policy: Any) -> Any:
        robot.policy_inits.append(policy)  # type: ignore[attr-defined]
        return _ready()

    def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
        return None

    robot._connect_robot = _connected  # type: ignore[method-assign]
    robot._initialize_policy = _init_policy  # type: ignore[method-assign]
    robot._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
    try:
        yield robot
    finally:
        robot._shutdown_event.set()
        robot._task_state.status = TaskStatus.STOPPED
        robot._executor.shutdown(wait=False)


def _text(result: dict[str, Any]) -> str:
    """The text of a tool-shaped result."""
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _sim_refuses(cap: Any) -> bool:
    """Whether the simulation rollout horizon refuses ``cap``.

    ``SimEngine._resolve_horizon`` is the sim's ``n_steps`` chokepoint; it
    validates the effective horizon before converting it into a duration.
    """
    _duration, _steps, error = SimEngine._resolve_horizon(cap, None, 50.0, 10.0, "run_policy")
    return error is not None


class TestUnusableStepCapRefused:
    """Every entry point taking a step cap refuses one it cannot count against."""

    @pytest.mark.parametrize("cap", UNUSABLE_STEP_CAPS)
    def test_run_policy_refuses(self, hw: Any, cap: Any):
        """``run_policy`` errors naming the parameter, not a comparison."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps=cap)

        assert result["status"] == "error"
        assert "n_steps" in _text(result)
        assert "run_policy" in _text(result)

    @pytest.mark.parametrize("cap", UNUSABLE_STEP_CAPS)
    def test_the_shared_chokepoint_refuses(self, hw: Any, cap: Any):
        """``_execute_task_sync`` refuses on its own.

        The agent-tool ``execute`` action and the mesh ``execute`` dispatch call
        it directly rather than through ``run_policy``, so a peer-supplied cap
        must be bounded here too.
        """
        result = hw._execute_task_sync("probe", policy_port=9000, n_steps=cap)

        assert result["status"] == "error"
        assert "n_steps" in _text(result)

    def test_the_message_never_names_a_comparison_internal(self, hw: Any):
        """A non-numeric cap is reported as a caller error, not a ``TypeError``."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps="10")

        assert "n_steps must be a positive integer" in _text(result)
        assert "'<' not supported" not in _text(result)

    def test_a_refused_cap_is_not_reported_as_a_completed_rollout(self, hw: Any):
        """The false ``completed`` this guard exists to remove."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps=0)

        assert result["status"] == "error"
        assert "completed" not in _text(result)
        assert hw._task_state.step_count == 0


class TestRefusalPrecedesTheArm:
    """A cap that cannot be honored costs no inference and no servo write."""

    @pytest.mark.parametrize("cap", UNUSABLE_STEP_CAPS)
    def test_no_action_reaches_the_arm(self, hw: Any, cap: Any):
        """The refusal lands before the control loop commands the bus."""
        hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps=cap)

        assert hw.robot.sent_actions == []

    def test_no_policy_is_initialized_and_no_inference_is_spent(self, hw: Any):
        """Bring-up is not reached, so the checkpoint is never queried."""
        policy = _CountingPolicy()

        hw.run_policy(policy_object=policy, instruction="probe", n_steps=-5)

        assert hw.policy_inits == []
        assert policy.calls == 0

    def test_the_bus_claim_is_released_for_the_next_rollout(self, hw: Any):
        """A refused cap must not take the single command bus away."""
        hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps=0)

        accepted = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", n_steps=2)

        assert accepted["status"] == "success"


class TestUsableStepCapAccepted:
    """Every cap that can be counted against still runs, and stops at it."""

    @pytest.mark.parametrize("cap", USABLE_STEP_CAPS)
    def test_the_rollout_stops_at_exactly_that_many_actions(self, hw: Any, cap: int):
        """The cap is honored, not merely accepted."""
        policy = _CountingPolicy()

        result = hw.run_policy(policy_object=policy, instruction="probe", duration=30.0, n_steps=cap)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == cap
        assert hw._task_state.step_count == cap

    def test_none_requests_no_cap_and_duration_bounds_the_rollout(self, hw: Any):
        """``None`` is the documented "no cap" spelling and stays accepted."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", duration=0.2, n_steps=None)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) > 1

    def test_the_default_is_no_cap(self, hw: Any):
        """Omitting the parameter behaves exactly as ``None``."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", duration=0.2)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) > 1

    def test_the_chokepoint_accepts_a_usable_cap(self, hw: Any):
        """The refusal at ``_execute_task_sync`` is not a blanket rejection."""
        result = hw._execute_task_sync("probe", policy_object=_CountingPolicy(), n_steps=3)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == 3


class TestDomainMatchesSimulation:
    """The arm and its digital twin agree on every step cap."""

    @pytest.mark.parametrize("cap", [*UNUSABLE_STEP_CAPS, *USABLE_STEP_CAPS])
    def test_hardware_and_simulation_agree(self, hw: Any, cap: Any):
        """Neither surface may accept a cap the other refuses."""
        hardware_refuses = hw._n_steps_error(cap, "run_policy") is not None

        assert hardware_refuses is _sim_refuses(cap)

    def test_both_read_none_as_no_cap(self, hw: Any):
        """``None`` is a request for no cap on both surfaces, not a bad value."""
        assert hw._n_steps_error(None, "run_policy") is None
        assert not _sim_refuses(None)

    def test_the_refusal_wording_is_the_shared_domain(self, hw: Any):
        """One domain, so the two surfaces report a bad cap identically."""
        hardware = _text(hw._n_steps_error(0, "run_policy"))
        _duration, _steps, sim_error = SimEngine._resolve_horizon(0, None, 50.0, 10.0, "run_policy")

        assert sim_error is not None
        assert hardware == _text(sim_error)


class TestTheStepCapGuardIsNotTheDurationGuard:
    """The two horizon conditions are ANDed, so each is validated on its own."""

    def test_a_usable_cap_with_an_unusable_budget_is_still_refused(self, hw: Any):
        """``duration`` remains the effective horizon and keeps its own domain."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", duration=0, n_steps=4)

        assert result["status"] == "error"
        assert "duration" in _text(result)

    def test_a_usable_budget_with_an_unusable_cap_is_refused(self, hw: Any):
        """A cap is not excused by a budget that could have bounded the rollout."""
        result = hw.run_policy(policy_object=_CountingPolicy(), instruction="probe", duration=5.0, n_steps=0)

        assert result["status"] == "error"
        assert "n_steps" in _text(result)
