"""Every refusal on the customer path names the bad value AND the fix.

``tests/simulation/mujoco/test_error_paths.py`` locks that the tool surface
returns ``status="error"`` instead of raising. This file locks the next layer of
the contract, measured from the customer's seat (``Robot("so100")``, the README
robot): the error text must quote the value the caller passed and say what to
pass instead - a range, the valid literals, a did-you-mean, or the action that
lists the valid names. An error that only says "invalid" costs the model one
extra turn per motion; the "first ten minutes" experiment measured that turn.

Each row is one wrong thing a first-day user actually did. Add a row when a
refusal is fixed; never loosen ``_FIX_MARKERS``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytest.importorskip("mujoco")

# One of these must appear in every refusal: it is the part that tells the
# caller what to do next.
_FIX_MARKERS = (
    "must be",
    "must contain",
    "Did you mean",
    "Available",
    "Use action=",
    "Check units",
    "range [",
    "see ``",
)

# (label, call, fragments of the offending value that must be quoted back)
_ROWS: list[tuple[str, dict[str, Any], tuple[str, ...]]] = [
    (
        "move_to tol negative",
        dict(action="move_to", robot_name="so100", position=[0, -0.28, 0.2], tol=-0.01),
        ("-0.01",),
    ),
    (
        "move_to max_steps zero",
        dict(action="move_to", robot_name="so100", position=[0, -0.28, 0.2], max_steps=0),
        ("max_steps", "0"),
    ),
    (
        "move_to 2-d target",
        dict(action="move_to", robot_name="so100", position=[0, -0.28]),
        ("'position'", "3", "got 2"),
    ),
    ("move_to target 5 m away", dict(action="move_to", robot_name="so100", position=[0, -0.28, 5.0]), ("5.0", "5 m")),
    ("move_to unknown robot", dict(action="move_to", robot_name="nope", position=[0, -0.28, 0.2]), ("'nope'",)),
    ("rotate_wrist out of range", dict(action="rotate_wrist", robot_name="so100", target_yaw=99.0), ("99.0",)),
    ("set_gripper typo state", dict(action="set_gripper", robot_name="so100", state="opne"), ("'opne'", '"open"')),
    ("set_gripper steps zero", dict(action="set_gripper", robot_name="so100", state="open", steps=0), ("steps", "0")),
    ("set_timestep zero", dict(action="set_timestep", timestep=0), ("timestep", "0")),
    ("step negative", dict(action="step", n_steps=-1), ("-1",)),
    ("apply_force inf", dict(action="apply_force", body_name="so100/Upper_Arm", force=[float("inf"), 0, 0]), ("inf",)),
    ("get_body_state unknown body", dict(action="get_body_state", body_name="nope"), ("'nope'",)),
    (
        "set_joint_positions joint typo",
        dict(action="set_joint_positions", robot_name="so100", positions={"Rotationn": 0.1}),
        ("'Rotationn'", "so100/Rotation"),
    ),
]


def _text(result: dict[str, Any]) -> str:
    return " ".join(str(c.get("text", "")) for c in result.get("content", []) if isinstance(c, dict))


@pytest.fixture(scope="module")
def so100():
    from strands_robots import Robot

    return Robot("so100")


@pytest.mark.parametrize(("label", "call", "quoted"), _ROWS, ids=[row[0] for row in _ROWS])
def test_the_refusal_quotes_the_value_and_names_the_fix(so100, label, call, quoted) -> None:
    result = so100(**call)
    assert result["status"] == "error", f"{label}: expected a refusal, got {result}"
    text = _text(result)
    for fragment in quoted:
        assert fragment in text, f"{label}: the value {fragment!r} is not quoted back in: {text}"
    assert any(marker in text for marker in _FIX_MARKERS), f"{label}: no fix in the refusal: {text}"


def test_the_refusal_leaves_the_state_unchanged(so100) -> None:
    so100(action="reset")
    before = so100.get_robot_state("so100")["content"][1]["json"]["state"]
    for _label, call, _quoted in _ROWS:
        if call["action"] in {"set_timestep", "step"}:
            continue
        so100(**call)
    after = so100.get_robot_state("so100")["content"][1]["json"]["state"]
    assert after == before, "a refused call wrote to the simulation"


def test_unknown_robot_name_is_quoted_with_the_list_hint() -> None:
    from strands_robots import Robot

    with pytest.raises(ValueError, match=re.escape("'so1000'")) as info:
        Robot("so1000")
    assert "list_robots" in str(info.value)
