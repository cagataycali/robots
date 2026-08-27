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
import itertools
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import strands_robots.policies as policies_pkg
import strands_robots.policies._log_safety as log_safety
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


# The two spellings ``py/log-injection`` accepts as the first argument of the
# ``.replace`` call it treats as a barrier. Stated here rather than read from the
# module under test so these cells are an independent oracle for the rule's own
# condition, which is:
#
#     this.getFunction().(DataFlow::AttrRead).getAttributeName() = "replace" and
#     this.getArg(0).asExpr().(StringLiteral).getText() in ["\r\n", "\n"]
_BARRIER_FIRST_ARGUMENTS = frozenset({"\r\n", "\n"})


def _returned_replace_chain() -> list[ast.Call]:
    """Return the ``.replace`` calls the escape's returned expression is built from.

    Outermost first, walking inward through each receiver, and empty when the
    escape does not return such a chain at all - which is the shape a loop leaves
    behind, the returned name carrying text no call on the path produced. Anchored
    on the ``return`` rather than on the function body because a ``.replace`` whose
    result is discarded escapes nothing the caller receives.
    """
    module = ast.parse(Path(log_safety.__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "sanitize_log_value"
    )
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1, "sanitize_log_value returns from exactly one place"
    chain: list[ast.Call] = []
    node: ast.expr | None = returns[0].value
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "replace":
        chain.append(node)
        node = node.func.value
    return chain


def _literal_first_argument(call: ast.Call) -> str | None:
    """Return the call's first argument when it is a string literal, else ``None``."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def test_the_escape_is_written_as_a_literal_replace_call() -> None:
    r"""The break characters are passed as literals, not read from a table.

    Helper contract. Two spellings escape the break identically and only one of
    them is recognised as a barrier by the scanner that reports these sinks:
    ``py/log-injection``'s ``ReplaceLineBreaksSanitizer`` holds for a ``.replace``
    call whose first argument is a *string literal* equal to ``"\r\n"`` or ``"\n"``,
    so a loop over a table of pairs closes the defect while reading as no barrier
    at all. That is not a property a reader can see from the rendered output, which
    is why it is asserted here rather than left to the next person who tidies the
    function into a loop.
    """
    chain = _returned_replace_chain()
    assert chain, "the escape must return a chain of .replace calls, not a name a loop rebound"
    literals = {_literal_first_argument(call) for call in chain}
    assert None not in literals, (
        "every .replace in the escape must name its break as a string literal: a spelling "
        "read from a constant escapes the break just as well and is recognised as no barrier"
    )
    assert literals & _BARRIER_FIRST_ARGUMENTS, (
        f"the escape replaces {sorted(spelling for spelling in literals if spelling)!r}, none of "
        f"which is one of the {sorted(_BARRIER_FIRST_ARGUMENTS)!r} the scanner recognises"
    )


def test_a_lone_carriage_return_is_not_one_of_the_barrier_spellings() -> None:
    r"""Premise: the accepted set is those two spellings, and ``"\r"`` is not one.

    Which is why the chain carries a link the barrier does not need. A payload can
    arrive with a bare carriage return, and an escape covering only what the rule
    reads would leave it on the wire.
    """
    assert _BARRIER_FIRST_ARGUMENTS == {"\r\n", "\n"}
    assert "\r" not in _BARRIER_FIRST_ARGUMENTS


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4], ids=lambda n: f"len-{n}")
def test_the_chain_renders_exactly_what_the_table_loop_rendered(length: int) -> None:
    r"""Over-reach control: rewriting the escape's shape changed none of its output.

    Exhaustive over every string of this length drawn from the two break
    characters, a backslash and an ordinary letter - the alphabet that can tell the
    two forms apart if anything can - compared against the table loop the chain
    replaced.
    """

    def table_loop(text: str) -> str:
        for raw, visible in (("\r", "\\r"), ("\n", "\\n")):
            text = text.replace(raw, visible)
        return text

    for letters in itertools.product("\r\na\\", repeat=length):
        payload = "".join(letters)
        assert sanitize_log_value(payload) == table_loop(payload), repr(payload)
        assert not _has_raw_break(sanitize_log_value(payload))


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
