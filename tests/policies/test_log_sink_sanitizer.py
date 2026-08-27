# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A value read out of a live observation cannot split the record that quotes it.

Eight ``py/log-injection`` alerts are open across three policy providers and
issue #2853 has the census. What the alerts agree on is the taint path: an
observation dict's own key names, a camera key, a language instruction and a
joint-state object all reach a logging sink. What they do not distinguish is
whether the value can still carry a ``\r`` or ``\n`` *by the time it is
interpolated* - and that is the only thing that splits a record for a
line-oriented consumer, which is the defect the rule is named for.

Measured on the tree before this module existed, driving each sink with a payload
whose name carried a line feed:

===== ================================= ==============================================
alert sink                              raw break in ``LogRecord.getMessage()``
===== ================================= ==============================================
715   ``_to_lerobot_observation``       yes - bare ``%s`` on the camera key
716   ``_build_observation_batch``      yes - bare ``%s`` on the instruction slice
717   ``_resolve_camera_targets``       yes - bare ``%s`` on the camera name
949   ``_resolve_state_order``          no - the keys arrive inside a list, and
                                        ``repr`` escapes a ``str`` element
714   ``_collect_state_values``         no - same list interpolation
710   ``_apply_world_update``           no - ``%r`` over a list of keys
711   cuRobo ``_extract_joint_state``   no - ``tolist()`` runs before the sink, so
                                        ``%r`` renders a list
712   MoveIt2 ``_extract_joint_state``  no - same
===== ================================= ==============================================

So three of the eight could forge a record and five could not. The five are not
safe *by construction* though - they are safe by the shape the value happens to
arrive in. Interpolate the same keys as ``', '.join(scalar_keys)`` for
readability, or hand ``%r`` a list holding one object with a multi-line
``__repr__`` instead of a list of strings, and the property is gone with nothing
at the sink to notice. That last case is not hypothetical: it is what cells
:func:`test_curobo_joint_state_object_with_a_multiline_repr_cannot_split_the_record`
and its MoveIt2 sibling drive, and both fail on the pre-change tree.

Hence the grading here is split in three, and each cell says which kind it is:

* **Forging cells** - drive a sink with a payload that *did* split the record
  before, and pin that it no longer does. These fail on the pre-change tree.
* **Property cells** - drive a sink whose escape was incidental, and pin the
  property directly so it is stated at the sink rather than inferred from the
  caller's formatting choice. These pass either way; what they defend is the
  next refactor of the message, not this change.
* **The wiring cell** - hold the set of sanitized sinks to exactly the eight the
  census names, keyed by the function that owns each one because a line number is
  the part that goes stale. Removing a wrapper fails it. Finding a *new* sink is
  CodeQL's job, not this file's.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import strands_robots.policies as policies_pkg
from strands_robots.policies._log_safety import sanitize_log_value
from strands_robots.policies.curobo.policy import CuroboPolicy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.moveit2.policy import MoveIt2Policy

# A payload shaped like the thing an operator would see in a forged line: the
# second half looks like a record this process never wrote.
FORGED = "wrist\nWARNING:root:actuators disabled"


def _stub(**attributes: object) -> Any:
    """Return a stand-in ``self`` for driving one sink in isolation.

    Each sink under test reads a handful of attributes and nothing else, so an
    unbound call against a namespace keeps the cell on its one message instead of
    standing up a provider with a loaded model and its optional backend. Typed
    ``Any`` for the same reason the tree casts a deliberate stand-in elsewhere.
    """
    return SimpleNamespace(**attributes)


def _rendered(caplog: pytest.LogCaptureFixture) -> str:
    """Return the last captured record's fully interpolated message."""
    assert caplog.records, "the sink under test emitted no record"
    return caplog.records[-1].getMessage()


def _has_raw_break(text: str) -> bool:
    """True when ``text`` can start a new line in a line-oriented log consumer."""
    return "\n" in text or "\r" in text


class MultilineRepr:
    """An ``observation.state`` element whose ``repr`` spans two lines.

    A joint-state object is not obliged to have a single-line ``repr``, and a
    container's ``repr`` escapes only its ``str`` elements - it calls ``repr``
    on everything else and passes the result through verbatim. So this is the
    shape that defeats the incidental escape at the two ``%r`` state sinks.
    """

    def __repr__(self) -> str:
        """Return a two-line representation."""
        return "JointState(\nWARNING:root:actuators disabled)"


# --------------------------------------------------------------------------
# The helper's own contract.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("a\nb", "a\\nb"),
        ("a\rb", "a\\rb"),
        ("a\r\nb", "a\\r\\nb"),
        ("a\n\nb", "a\\n\\nb"),
        ("\n", "\\n"),
    ],
    ids=["lf", "cr", "crlf", "two-lf", "lf-only"],
)
def test_every_line_break_spelling_becomes_its_visible_escape(payload: str, expected: str) -> None:
    """Each break is replaced in place, and a CRLF pair keeps its order."""
    assert sanitize_log_value(payload) == expected
    assert not _has_raw_break(sanitize_log_value(payload))


def test_a_non_string_is_rendered_before_it_is_escaped() -> None:
    """A caller may hand over any object; ``str`` runs first."""
    assert sanitize_log_value([1, 2]) == "[1, 2]"
    assert sanitize_log_value(MultilineRepr()) == "JointState(\\nWARNING:root:actuators disabled)"


def test_nothing_but_the_two_break_characters_is_touched() -> None:
    """The punctuation these diagnostics are made of survives verbatim.

    An over-broad filter would corrupt the diagnosis the message exists for: a
    key list, a joint-state repr and a remedy sentence are mostly brackets,
    quotes, commas and tabs.
    """
    intact = "['shoulder_pan', 'gripper'] -> set_robot_state_keys(...) 50% \t|;$`"
    assert sanitize_log_value(intact) == intact


def test_a_payload_that_already_reads_as_an_escape_is_left_alone() -> None:
    """Escaping is idempotent on already-escaped text, so the five sinks that
    were escaped incidentally render byte-identically after this change."""
    already = "['shoulder\\npan', 'gripper']"
    assert sanitize_log_value(already) == already


# --------------------------------------------------------------------------
# Forging cells: these fail on the pre-change tree.
# --------------------------------------------------------------------------


def test_an_unmatched_camera_name_cannot_split_the_record(caplog: pytest.LogCaptureFixture) -> None:
    """Alert 717. The camera name reached a bare ``%s``, so a line feed in it
    put half of the warning on a line of its own."""
    policy = _stub(
        _policy_image_keys=lambda: ["observation.images.top"],
        camera_key_map=None,
        strict_keys=False,
        positional_fallback_used=False,
    )
    with caplog.at_level(logging.WARNING):
        LerobotLocalPolicy._resolve_camera_targets(policy, [FORGED])

    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    # The name is still readable, and still names the offending camera.
    assert "wrist\\nWARNING:root:actuators disabled" in rendered


def test_curobo_joint_state_object_with_a_multiline_repr_cannot_split_the_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alert 711. ``%r`` over a list escapes ``str`` elements only, so one
    element with a multi-line ``repr`` put the break back on the wire."""
    policy = _stub()
    with caplog.at_level(logging.WARNING):
        result = CuroboPolicy._extract_joint_state(policy, {"observation.state": [MultilineRepr()]})

    assert result is None, "the degraded-extraction path is the one under test"
    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    assert "JointState(\\nWARNING:root:actuators disabled)" in rendered


def test_moveit2_joint_state_object_with_a_multiline_repr_cannot_split_the_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alert 712. Same shape as the cuRobo sink, and the same fix, so a
    provider that grows a third copy of this message is graded here too."""
    policy = _stub()
    with caplog.at_level(logging.WARNING):
        result = MoveIt2Policy._extract_joint_state(policy, {"observation.state": [MultilineRepr()]})

    assert result is None, "the degraded-extraction path is the one under test"
    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    assert "JointState(\\nWARNING:root:actuators disabled)" in rendered


# --------------------------------------------------------------------------
# Property cells: the escape was incidental at these sinks; state it there.
# --------------------------------------------------------------------------


def test_a_state_key_mismatch_names_the_observation_keys_on_one_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alert 949. Escaped before this change only because the keys arrive as a
    list; pinned here so a reflowed message cannot lose the property quietly."""
    policy = _stub(
        robot_state_keys=["joint_0"],
        strict_keys=False,
        generic_state_keys_used=False,
        _state_key_mismatch_warned=False,
    )
    observation = {FORGED: 0.0, "gripper": 1.0}
    with caplog.at_level(logging.WARNING):
        LerobotLocalPolicy._resolve_state_order(policy, observation, list(observation))

    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    assert "gripper" in rendered, "the observed keys are still named"


def test_missing_state_keys_name_the_absent_keys_on_one_record(caplog: pytest.LogCaptureFixture) -> None:
    """Alert 714. Same list interpolation, same reason to state it at the sink."""
    configured = [FORGED, "gripper"]
    policy = _stub(
        robot_state_keys=configured,
        strict_keys=False,
        missing_state_keys_used=False,
        _state_missing_keys_warned=False,
    )
    with caplog.at_level(logging.WARNING):
        LerobotLocalPolicy._collect_state_values(policy, {"gripper": 1.0}, configured)

    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    assert "1 configured robot_state_keys are not present" in rendered


def test_an_ignored_world_update_names_its_keys_on_one_record(caplog: pytest.LogCaptureFixture) -> None:
    """Alert 710. The planner-shim warning quotes the caller's own scene keys."""
    policy = _stub(_motion_planner=SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        CuroboPolicy._apply_world_update(policy, {FORGED: {}})

    rendered = _rendered(caplog)
    assert not _has_raw_break(rendered), f"record still splits: {rendered!r}"
    assert "world_update=" in rendered


# --------------------------------------------------------------------------
# The wiring cell.
# --------------------------------------------------------------------------

# Every sink issue #2853 counted, keyed by the function that owns it, mapped to
# the argument expressions that must pass through the sanitizer. ``tokens.shape``
# is deliberately absent from the 716 entry: it is a tensor shape the policy
# computed, not anything the observation supplied, and wrapping it would state a
# provenance it does not have.
_SANITIZED_SINKS: dict[str, dict[str, list[str]]] = {
    "lerobot_local/policy.py": {
        "LerobotLocalPolicy._resolve_state_order": ["msg"],
        "LerobotLocalPolicy._collect_state_values": ["msg"],
        "LerobotLocalPolicy._to_lerobot_observation": ["feat", "k"],
        "LerobotLocalPolicy._build_observation_batch": ["instruction[:50]"],
        "LerobotLocalPolicy._resolve_camera_targets": ["cam", "feat"],
    },
    "curobo/policy.py": {
        "CuroboPolicy._apply_world_update": ["repr(shown)"],
        "CuroboPolicy._extract_joint_state": ["e", "repr(state)"],
    },
    "moveit2/policy.py": {
        "MoveIt2Policy._extract_joint_state": ["e", "repr(state)"],
    },
}

_PACKAGE_DIR = Path(policies_pkg.__file__).parent


def _sanitized_arguments(relative_path: str) -> dict[str, list[str]]:
    """Map ``Class.method`` -> sorted ``sanitize_log_value`` argument sources."""
    tree = ast.parse((_PACKAGE_DIR / relative_path).read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, [*scope, child.name])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "sanitize_log_value"
                and scope
            ):
                found.setdefault(".".join(scope), []).append(ast.unparse(child.args[0]))
            visit(child, scope)

    visit(tree, [])
    return {qualname: sorted(args) for qualname, args in found.items()}


@pytest.mark.parametrize("relative_path", sorted(_SANITIZED_SINKS), ids=lambda p: p.split("/")[0])
def test_exactly_the_census_sinks_are_sanitized(relative_path: str) -> None:
    """The sanitized set is the eight the census names - no more, no fewer.

    A wrapper removed from one of them fails here as well as re-opening its
    alert, and a wrapper added somewhere new has to be added to the census in
    the same diff, which is where the argument for it gets written down.
    """
    assert _sanitized_arguments(relative_path) == _SANITIZED_SINKS[relative_path]
