"""The moveit2.md Quickstart must send panda_arm a start state it can plan from.

The page's Install section brings the reference sidecar up on planning group
``panda_arm`` and the Quickstart dials it with ``observation.state`` as the
start configuration. The sidecar reads that vector in order onto the group's
active joints and refuses one with fewer values than the group plans over
(``_start_state`` in ``server/zmq_node.py``: ``joint_state carries 6 values but
planning group 'panda_arm' plans over 7 joints``), which the client raises as
``RuntimeError``. The page's own In-simulation note adds that the Panda's zero
pose is a start state in collision, so the values must be the model's home
keyframe, not zeros.

This reads every ``get_actions_sync`` fence on the page whose policy names
``planning_group="panda_arm"`` and grades the state vector it sends.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots

_PAGE = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "reference" / "policies" / "moveit2.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
# moveit_resources_panda_moveit_config: panda_arm is panda_joint1..panda_joint7.
_PANDA_ARM_JOINTS = 7
# The Panda's home keyframe, the start state the page's In-simulation fence uses.
_PANDA_HOME = (0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _as_list(node: ast.expr) -> list[float]:
    """Evaluate a literal list, allowing the ``[x] * n`` repeat spelling."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _as_list(node.left) * int(ast.literal_eval(node.right))
    return [float(v) for v in ast.literal_eval(node)]


def _panda_arm_start_states() -> list[list[float]]:
    states: list[list[float]] = []
    for fence in _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8")):
        try:
            tree = ast.parse(fence)
        except SyntaxError:
            continue  # prose placeholders are not fences under test
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        groups = [_keyword(c, "planning_group") for c in calls if _call_name(c) in {"create_policy", "MoveIt2Policy"}]
        if not any(isinstance(g, ast.Constant) and g.value == "panda_arm" for g in groups):
            continue
        for call in calls:
            if _call_name(call) != "get_actions_sync":
                continue
            obs = _keyword(call, "observation_dict")
            if not isinstance(obs, ast.Dict):
                continue
            for key, value in zip(obs.keys, obs.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "observation.state":
                    states.append(_as_list(value))
    return states


def test_page_has_a_panda_arm_quickstart() -> None:
    assert _panda_arm_start_states(), "moveit2.md has no panda_arm get_actions_sync fence to grade"


def test_quickstart_start_state_is_as_wide_as_panda_arm() -> None:
    for state in _panda_arm_start_states():
        assert len(state) == _PANDA_ARM_JOINTS, (
            f"Quickstart fence sends {len(state)} values to panda_arm, which plans {_PANDA_ARM_JOINTS} joints; "
            "the sidecar refuses the start state (start_state_error) and the client raises RuntimeError"
        )


def test_quickstart_start_state_is_the_panda_home_keyframe() -> None:
    for state in _panda_arm_start_states():
        assert tuple(state) == _PANDA_HOME, (
            f"Quickstart fence starts panda_arm at {state}; the model's zero pose is a start state in "
            f"collision the planner refuses, the page's own sim fence starts from {list(_PANDA_HOME)}"
        )
