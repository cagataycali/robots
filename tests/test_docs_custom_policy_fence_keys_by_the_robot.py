"""The custom-policy walkthrough's ``MyPolicy`` must act on the keys it is given.

``SimEngine.run_policy`` calls ``policy.set_robot_state_keys(robot_action_keys)``
before the rollout, and ``PolicyRunner`` refuses a policy whose first three
actions resolve to no actuator on the robot (``"the robot has not moved"``). The
``docs/reference/policies/custom-policies.md`` opening fence is the reader's first policy
and is run on ``Robot("so100")`` two fences later, so the action it returns must
be keyed by the names ``set_robot_state_keys`` received, not by literals no arm
carries.

This executes the fence's class with ``register_policy`` stubbed, hands it the
SO-100's action keys the way the runtime does, and grades the first action.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PAGE = _REPO_ROOT / "docs" / "reference" / "policies" / "custom-policies.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SO100_KEYS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _my_policy_class() -> type:
    fence = _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8"))[0]
    assert "class MyPolicy(Policy)" in fence, "the walkthrough's first fence defines MyPolicy"
    namespace: dict[str, object] = {}
    registered: list[str] = []
    exec(  # noqa: S102 - the docs fence is the artefact under test
        fence.replace(
            "from strands_robots.policies import Policy, register_policy",
            "from strands_robots.policies import Policy",
        ),
        {"register_policy": lambda name, *a, **k: registered.append(name)},
        namespace,
    )
    assert registered == ["my_provider"]
    return namespace["MyPolicy"]  # type: ignore[return-value]


def test_my_policy_action_is_keyed_by_the_keys_the_runtime_gave_it() -> None:
    policy = _my_policy_class()()
    policy.set_robot_state_keys(list(_SO100_KEYS))

    actions = asyncio.run(policy.get_actions({}, "do something"))

    assert isinstance(actions, list) and actions, "get_actions returns a non-empty list of dicts"
    unresolved = sorted(set(actions[0]) - set(_SO100_KEYS))
    assert not unresolved, f"first action names keys the so100 has no actuator for: {unresolved}"
    assert all(isinstance(v, float) for v in actions[0].values())
