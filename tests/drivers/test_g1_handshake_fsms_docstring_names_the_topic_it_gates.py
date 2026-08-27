"""Regression: the FSM-set docstrings do not name a topic ``send_action`` does not write.

Two docstrings in this driver claimed :data:`HANDSHAKE_FSMS` gates writes to
``rt/armsdk``:

* ``strands_robots/tools/g1/_g1_common.py`` on the ``HANDSHAKE_FSMS`` constant
  itself: "so :meth:`~strands_robots.drivers.g1.G1Driver.send_action` checks
  membership before writing ``rt/armsdk``".
* ``strands_robots/drivers/g1.py`` on ``_check_motion_gates``: ":data:`HANDSHAKE_FSMS`
  covers arm-SDK writes (``rt/armsdk``)".

Neither is what the shipped driver does.  Since #2767 landed (2026-08-24)
``G1Driver.send_action`` publishes on ``rt/lowcmd`` -- ``_TOPIC_LOWCMD`` is
`"rt/lowcmd"` and ``_pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)`` is the one
write site.  The control loop wired by #2779 writes the same topic on every
step and on stop.  The reader following ``HANDSHAKE_FSMS`` from either docstring
was pointed at a topic string this file does not appear in and could not confirm
by grep, and the two mentions of the real topic in the same module
(``send_action``'s own docstring at line 635 and its scope classification at
line 657) already say "``rt/lowcmd``" -- so the module contradicted itself.

The confusion is worth naming: the arm-SDK-*shape* (the client that talks
:data:`HANDSHAKE_FSMS` at the FSM level) and the arm-SDK-*topic* (``rt/armsdk``,
which the ``g1_tools`` client for issue #358 will write) are not the same
thing.  ``HANDSHAKE_FSMS`` is a set of FSM ids on the motion-switcher rail; the
G1 driver's write for that shape lands on ``rt/lowcmd`` today, and the
``rt/armsdk`` topic name appears in this package only inside the SDK error
table (code 7400 says "``rt/armsdk`` topic is occupied", which is the
firmware's own text and out of scope for this rule) and inside forward-looking
docstrings for the issue #358 tools that do not exist yet.

## Scope

This test grades the two named docstrings only.  The rule is *not* a blanket
ban on the ``rt/armsdk`` substring (which would fire on the SDK error table
for firmware response 7400 and on forward-looking references to the
``g1_tools`` client of issue #358); it is that where either docstring names a
topic in the context of a currently-gated write, ``rt/lowcmd`` must appear.
A docstring that mentions ``rt/armsdk`` alongside ``rt/lowcmd`` (naming
future work explicitly) satisfies the rule; naming only ``rt/armsdk`` does
not.

## Fires on the defect, passes on the fix

Both defect cells fail on ``main @ 5ded625b`` (where ``rt/armsdk`` is the only
topic named beside :data:`HANDSHAKE_FSMS`) and pass after the fix (where
``rt/lowcmd`` is named as today's write and ``rt/armsdk`` is optional
context).  The two premise cells fail vacuously if a future refactor renames
the constant or the write topic, at which point the pin has to move with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots
from strands_robots.tools.g1 import _g1_common

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _read(rel: str) -> str:
    return (_PACKAGE_DIR / rel).read_text(encoding="utf-8")


def _handshake_docstring_in_common() -> str:
    """The ``#:``-prefixed docstring above ``HANDSHAKE_FSMS`` in ``_g1_common``.

    Sphinx renders ``#:`` lines that precede a module-level assignment as the
    docstring for the assigned name, so the reader who follows
    :data:`HANDSHAKE_FSMS` sees exactly this block.  The scan collects the
    contiguous run of ``#:`` lines immediately above the assignment.
    """
    text = _read("tools/g1/_g1_common.py")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("HANDSHAKE_FSMS"):
            # Walk backwards to the top of the contiguous ``#:`` block.
            top = idx
            while top > 0 and lines[top - 1].lstrip().startswith("#:"):
                top -= 1
            return "\n".join(lines[top:idx])
    raise AssertionError("HANDSHAKE_FSMS assignment not found in _g1_common.py")


def _check_motion_gates_docstring() -> str:
    """The docstring on ``G1Driver._check_motion_gates``.

    Extracted textually so the test does not have to import ``G1Driver`` (which
    binds a DDS interface at construction time).  The scan captures from the
    ``def _check_motion_gates`` line to the closing ``\"\"\"`` that follows the
    first ``\"\"\"`` opening on the next line.
    """
    text = _read("drivers/g1.py")
    match = re.search(
        r'def _check_motion_gates\(.*?\n\s+"""(.*?)"""',
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "_check_motion_gates docstring not found in drivers/g1.py"
    return match.group(1)


# ---------------------------------------------------------------------------
# Premise cells -- exist so a rename can never make the defect cells vacuous.
# ---------------------------------------------------------------------------


def test_premise_handshake_fsms_still_declared_in_g1_common() -> None:
    """The constant this file grades is at the location it grades."""
    assert isinstance(_g1_common.HANDSHAKE_FSMS, frozenset)
    assert _g1_common.HANDSHAKE_FSMS == frozenset({500, 501, 801})


def test_premise_send_action_writes_rt_lowcmd() -> None:
    """The topic the docstrings must name is the one the driver writes.

    Read out of the module source (not out of an import that would touch the
    SDK).  If ``_TOPIC_LOWCMD`` moves or is renamed, this cell fires before the
    defect cell it supports so a reader is not left grading the wrong site.
    """
    text = _read("drivers/g1.py")
    assert '_TOPIC_LOWCMD = "rt/lowcmd"' in text
    # And the one write site in ``send_action`` reads that constant.
    assert "_pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)" in text


# ---------------------------------------------------------------------------
# Defect cells -- both fire on main, both pass after the fix.
# ---------------------------------------------------------------------------


def test_handshake_fsms_docstring_does_not_name_a_topic_send_action_does_not_write() -> None:
    """The ``HANDSHAKE_FSMS`` docstring in ``_g1_common`` cites ``rt/lowcmd`` where it names a topic.

    The defect was a docstring that said :meth:`send_action` writes ``rt/armsdk``.
    The fix does not have to strip every mention of ``rt/armsdk`` - it may
    legitimately distinguish today (``rt/lowcmd``) from future work
    (``rt/armsdk``, issue #358) - but a docstring that mentions ``rt/armsdk``
    without also naming ``rt/lowcmd`` in the same block is naming only the
    wrong topic.  The rule is *not* "must never say ``rt/armsdk``"; it is
    "if the reader is told a topic name in the ``send_action`` neighbourhood,
    ``rt/lowcmd`` must be one of them."
    """
    doc = _handshake_docstring_in_common()
    if "send_action" not in doc:
        return  # the docstring may have been rewritten to omit send_action entirely
    if "rt/armsdk" in doc:
        assert "rt/lowcmd" in doc, (
            "HANDSHAKE_FSMS docstring names rt/armsdk in a send_action context "
            "without also naming rt/lowcmd; the shipped driver writes rt/lowcmd (#2767) "
            "and the reader must be told which one send_action writes today."
        )


def test_check_motion_gates_docstring_does_not_conflate_the_arm_shape_with_the_armsdk_topic() -> None:
    """The ``_check_motion_gates`` docstring must not tie ``HANDSHAKE_FSMS`` to ``rt/armsdk`` alone.

    The original phrasing was ":data:`HANDSHAKE_FSMS` covers arm-SDK writes
    (``rt/armsdk``)".  ``HANDSHAKE_FSMS`` gates the FSM shape, and the write
    that shape produces today lands on ``rt/lowcmd``; the ``rt/armsdk`` topic
    is future work for issue #358's ``g1_tools`` client.  A docstring naming
    ``rt/armsdk`` without also naming ``rt/lowcmd`` reads as an authoritative
    statement of what ``send_action`` does today, which is wrong.
    """
    doc = _check_motion_gates_docstring()
    if "rt/armsdk" in doc:
        assert "rt/lowcmd" in doc, (
            "_check_motion_gates docstring names rt/armsdk without naming rt/lowcmd; "
            "the driver's arm write goes to rt/lowcmd today (#2767) and the reader "
            "must be able to see that from this docstring."
        )


# ---------------------------------------------------------------------------
# Scope-boundary cell -- keeps the rule from creeping.
# ---------------------------------------------------------------------------


def test_scope_the_sdk_error_table_still_names_rt_armsdk_because_that_is_the_sdks_text() -> None:
    """The error-code table quotes SDK text: ``rt/armsdk topic is occupied``.

    That is not a claim about what this driver writes; it is the firmware's
    own error string for response code 7400 and must survive unchanged so a
    real error surfaces with the same text the SDK sends.  If the rule ever
    grew to blanket-ban the substring, it would fire here and this file's
    scope would need re-stating.
    """
    text = _read("tools/g1/_g1_common.py")
    assert "rt/armsdk topic is occupied" in text
