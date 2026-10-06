"""Repro: Robot(name)'s auto-derived sim tool name bypasses _tool_name_error().

Target: strands-labs/robots v0.5.3
Rotation: readme_quickstart
Class:   Error UX + Footgun (silent-wrong until the first LLM call)

Finding
=======

``strands_robots/robot.py:244-284`` defines ``_tool_name_error``, which refuses
a tool name that contains anything outside ``[A-Za-z0-9_-]`` or exceeds 64
characters -- "the constraints a provider puts on a tool name" -- and names
the remedy ("use letters, digits, '_' or '-'").

``strands_robots/robot.py:827-829`` on main calls that validator -- but ONLY
against the explicit ``tool_name=`` kwarg::

    tool_name_reason = _tool_name_error(tool_name)
    if tool_name_reason is not None:
        raise ValueError(tool_name_reason)

The sim-mode path at line 880 then derives the tool name from the RAW ``name``
the caller typed, not from ``canonical`` (the string ``resolve_name`` returned
after stripping whitespace and casefolding)::

    sim = cast("Simulation", create_simulation(backend,
              tool_name=tool_name or f"{name}_sim", **kwargs))

Because ``resolve_name`` normalises the input before the registry lookup, a
copy-paste typo (a stray leading or trailing space, a sentence-case robot id)
sails past the "Unknown robot" guard AND past the explicit-only tool-name
gate.

Impact
======

::

    Robot("so100 ")           -> tool_name = "so100 _sim"   (SPACE in middle)
    Robot(" so100")           -> tool_name = " so100_sim"   (leading SPACE)

Each one builds a Robot ``status="success"`` and ``Agent(tools=[robot])``
accepts it. The failure surfaces on the FIRST model call, where Bedrock
Converse rejects the toolSpec with::

    ValidationException: Member must satisfy regular expression pattern:
    [a-zA-Z][a-zA-Z0-9_]{0,63}

...naming a request-body slot the caller never set. The user is left to
reverse-engineer the problem from a server-side pattern error.

Fix (on this branch)
====================

Screen the derived tool name against the SAME validator the explicit path uses
(``_tool_name_error``), when and only when the caller did not supply an
explicit ``tool_name=``. The design that preserves user aliases
(``Robot("h1")`` keeps ``tool_name='h1_sim'`` rather than
``'unitree_h1_sim'``, pinned by
``tests/test_robot_factory.py::TestRobotNamePreservesUserInput``) is
unchanged; only invalid characters are refused.

Running this file
=================

On main (before the fix), ``Robot("so100 ")`` returns an engine with
``tool_name='so100 _sim'`` and the assertion at the bottom fails.
On this branch, ``Robot("so100 ")`` raises ``ValueError`` naming the problem
and the character set the user may use.
"""

from __future__ import annotations

import re

from strands_robots import Robot

# The same pattern the project uses to decide if a tool name is acceptable
# (strands_robots/robot.py:240):
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\Z")
_TOOL_NAME_MAX_LEN = 64


def _tool_name_is_valid(tool_name: str) -> bool:
    return bool(_TOOL_NAME_PATTERN.match(tool_name)) and len(tool_name) <= _TOOL_NAME_MAX_LEN


if __name__ == "__main__":
    # --- The three mistakes any copy-paste user will make from README.md ---
    typo_cases = ["so100 ", " so100", "so100\t"]

    saw_invalid = False
    for raw in typo_cases:
        try:
            robot = Robot(raw)
            tool_name = robot.tool_name
            valid = _tool_name_is_valid(tool_name)
            marker = "OK (valid)" if valid else "DEFECT (invalid tool_name derived silently)"
            print(f"{raw!r:<16} -> tool_name={tool_name!r:<22} valid={valid}  [{marker}]")
            if not valid:
                saw_invalid = True
        except ValueError as e:
            print(f"{raw!r:<16} -> REFUSED: {str(e)[:160]}")

    # --- The design-preserving cases must still succeed ---
    for canonical_input in ["so100", "h1"]:
        robot = Robot(canonical_input, mesh=False)
        tool_name = robot.tool_name
        valid = _tool_name_is_valid(tool_name)
        assert valid, f"Regression: Robot({canonical_input!r}) now produces invalid {tool_name!r}"
        print(f"{canonical_input!r:<16} -> tool_name={tool_name!r} valid=True [OK]")

    assert not saw_invalid, (
        "Defect reproduces: at least one Robot(<copy-paste typo>) built with "
        "an invalid auto-derived tool_name. The sim-mode path at robot.py:880 "
        "interpolates the raw ``name`` into ``f\"{name}_sim\"`` without "
        "running the derived string through ``_tool_name_error`` -- the "
        "validator at line 244-284 is only called on the explicit "
        "``tool_name=`` kwarg. Fix: screen the derived name with "
        "``_tool_name_error`` after the explicit screen (14 LOC)."
    )
    print("\nOK: no silent-invalid tool_names.")
