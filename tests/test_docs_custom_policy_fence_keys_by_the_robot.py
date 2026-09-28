"""The custom-policy walkthrough's ``HoldPolicy`` must act on the keys it is given.

``SimEngine.run_policy`` calls ``policy.set_robot_state_keys(robot_action_keys)``
before the rollout, and ``PolicyRunner`` refuses a policy whose first three
actions resolve to no actuator on the robot (``"the robot has not moved"``). The
"Swap the policy, keep the robot" fence on ``docs/learn/policies/index.md`` is
the reader's first policy and runs on ``so101`` in the same fence, so the action
it returns must be keyed by the names ``set_robot_state_keys`` received, not by
literals no arm carries.

This executes the fence's class with ``register_policy`` stubbed, hands it the
SO-101's action keys the way the runtime does (read from the live simulation,
not typed), and grades the first action.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots import Robot

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PAGE = _REPO_ROOT / "docs" / "learn" / "policies" / "index.md"
_PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
_CLASS = "HoldPolicy"
_PROVIDER = "hold"
_ROBOT = "so101"


def _policy_fence() -> str:
    """The one fence on the page that defines the reader's own policy."""
    fences = [f for f in _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8")) if f"class {_CLASS}(Policy)" in f]
    assert len(fences) == 1, f"docs/learn/policies/index.md defines {_CLASS} in {len(fences)} fences; expected one"
    return fences[0]


def _policy_class() -> type:
    """The class from the fence, with registration and the rollout stubbed out."""
    fence = _policy_fence()
    definition = fence.split(f'register_policy("{_PROVIDER}"', 1)[0]
    assert definition != fence, f"the fence no longer registers {_CLASS} as {_PROVIDER!r}"
    namespace: dict[str, object] = {}
    registered: list[str] = []
    exec(  # noqa: S102 - the docs fence is the artefact under test
        definition.replace(
            "from strands_robots.policies import Policy, register_policy",
            "from strands_robots.policies import Policy",
        )
        + f'\nregister_policy("{_PROVIDER}", lambda: {_CLASS}, aliases=["freeze"])\n',
        {"register_policy": lambda name, *a, **k: registered.append(name)},
        namespace,
    )
    assert registered == [_PROVIDER]
    return namespace[_CLASS]  # type: ignore[return-value]


@pytest.fixture(scope="module")
def robot_keys() -> list[str]:
    """The action keys the runtime hands the policy for the robot the fence runs on."""
    assert f'sim.add_robot("{_ROBOT}")' in _policy_fence(), f"the fence no longer runs on {_ROBOT}"
    sim = Robot(_ROBOT, mode="sim")
    try:
        return list(sim.robot_action_keys(_ROBOT))
    finally:
        sim.destroy()


def test_the_policy_action_is_keyed_by_the_keys_the_runtime_gave_it(robot_keys: list[str]) -> None:
    assert robot_keys, f"{_ROBOT} reports no action keys; the rollout would have nothing to command"
    policy = _policy_class()(angle=0.5)
    policy.set_robot_state_keys(list(robot_keys))

    actions = asyncio.run(policy.get_actions({}, "do something"))

    assert isinstance(actions, list) and actions, "get_actions returns a non-empty list of dicts"
    unresolved = sorted(set(actions[0]) - set(robot_keys))
    assert not unresolved, f"first action names keys the {_ROBOT} has no actuator for: {unresolved}"
    assert set(actions[0]) == set(robot_keys), "the hold pose leaves a joint uncommanded"
    assert all(isinstance(v, float) for v in actions[0].values())


def test_the_policy_declares_it_reads_no_instruction() -> None:
    """The page's next paragraph hangs on ``reads_instruction = False``; the class says so."""
    policy_cls = _policy_class()
    assert policy_cls.reads_instruction is False
    assert policy_cls().requires_images is False
    assert policy_cls().provider_name == _PROVIDER
