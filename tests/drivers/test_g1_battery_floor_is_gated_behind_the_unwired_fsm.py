"""The G1's battery floor is masked by the un-wired FSM gate.

:class:`G1Driver.__init__` sets ``_fsm_id = None`` and nothing else writes it:
the value the arm-SDK gate compares against arrives from a motion-switcher API
that has not been wired (harness#361 PR-C; #2765 carries the wire-side
decision). On a driver whose every other field is exactly what a healthy G1
produces, ``_check_motion_gates`` returns ``FSM id unknown`` -- which is
correct, and which means every guard behind it is unreachable through the
verb surface. The battery floor is the guard the caller is asked to think
about (a constructor parameter, named in :meth:`get_status`), so its
unreachability is the case worth pinning.

Two contracts are stated here as literals so a change to either fires this
file, not a distant one:

1. On a fully-healthy driver *except* for ``_fsm_id``, ``send_action`` -- the
   verb an agent reaches -- refuses with ``FSM id unknown``, not with a
   battery-under-floor message. Set the pack to 1.0% against a 15.0% floor:
   still ``FSM id unknown``, still no mention of ``battery``.

2. ``_fsm_id`` has exactly one assignment in :mod:`strands_robots.drivers.g1`
   and it is the ``None`` initialiser. The day a motion-switcher decoder
   writes ``_fsm_id`` for the first time, this test fires -- which is the
   correct signal, because on the same day the reachability assertion in (1)
   must flip (battery floor becomes reachable), and this file must be
   replaced by one that grades the new reachability instead of the current
   unreachability.

Deliberately out of scope: whether the ordering (FSM before battery) is the
right ordering. :meth:`_check_motion_gates`'s docstring names it as a
deliberate choice -- the caller has already been told the FSM if the FSM is
the reason -- and this test would still pass under either ordering, because
today the FSM gate is the one that fires. What this test grades is that the
current arrangement leaves the battery floor unreachable at all, so the day
the FSM gate opens the battery floor is graded on its next call, not
discovered by a caller.
"""

from __future__ import annotations

import ast
import inspect

import strands_robots.drivers.g1 as g1_module
from strands_robots.drivers.g1 import G1Driver

# ``_CRITICAL_PCT`` is far below the ``_HEALTHY_FLOOR_PCT`` so an ordering
# mistake -- battery checked before FSM -- would flip the refusal text and
# fire the reachability assertion below. Literal values, not derivations from
# the module's own constants, so a rename in the module cannot silently make
# this test read a different comparison than the reader thinks it reads.
_CRITICAL_PCT = 1.0
_HEALTHY_FLOOR_PCT = 15.0

# ``_HEALTHY_MODE_MACHINE`` is what the ``rt/lowstate`` decoder produces on a
# healthy G1 (uint8 layout id echoed by the vendor); ``_check_motion_gates``
# only refuses on ``mode_machine`` when it is ``None``, so any populated value
# gets the gate past that check and to the ``_fsm_id`` check this file grades.
_HEALTHY_MODE_MACHINE = 9


# A healthy pack shape the mesh's battery sensor decoder produces. The keys
# match :mod:`strands_robots.mesh.sensors` -- ``pct`` is the only field
# ``_check_motion_gates`` reads, the others are here so the shape is exactly
# what production carries and a decoder change that dropped ``pct`` would
# still be visible.
def _pack(pct: float) -> dict[str, float | bool | int]:
    return {
        "pct": pct,
        "charging": False,
        "current": 0.0,
        "cycle": 0,
        "t": 0.0,
    }


def test_send_action_refuses_with_fsm_unknown_not_battery_on_a_healthy_otherwise_driver() -> None:
    """The battery-floor guard is unreachable through :meth:`send_action`.

    Every field the arm-SDK gate reads is what a healthy G1 produces --
    ``_connected=True`` from a completed connect, ``_mode_machine`` from a
    real ``rt/lowstate`` decode, ``_battery`` from a real ``rt/lf/bmsstate``
    decode -- except ``_fsm_id``, which is left at its ``None`` initialiser
    because nothing in the driver writes it. The pack is set to a critical
    value against a configured floor: if the battery floor were reachable,
    the refusal would name the battery; it names the FSM instead.

    The day :meth:`_check_motion_gates` reads ``_fsm_id`` from a real
    motion-switcher source, this test fires -- which is the correct signal
    to replace this file with one that grades the reachable battery floor.
    """
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
    )
    driver._connected = True
    driver._mode_machine = _HEALTHY_MODE_MACHINE
    driver._battery = _pack(_CRITICAL_PCT)
    # ``_fsm_id`` is deliberately left at its ``None`` initialiser: this is
    # exactly the state the driver produces without a caller injecting
    # anything, and it is the state today's motion-switcher wire (absent)
    # keeps it in.
    assert driver._fsm_id is None

    result = driver.send_action({"any": 0.0})

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # The refusal names the un-wired FSM source, not the battery. Both parts
    # are stated: an ordering mistake that put the battery check first would
    # trip the ``"battery" not in text`` assertion.
    assert "FSM id unknown" in text
    assert "motion-switcher" in text
    assert "battery" not in text
    # And the critical percentage does not appear anywhere in the refusal --
    # the caller learns about the FSM, and the pack reading is not surfaced
    # because it never became relevant to the refusal.
    assert f"{_CRITICAL_PCT}" not in text


def test_fsm_id_has_exactly_one_assignment_in_the_driver_module() -> None:
    """``_fsm_id`` has one write site and it is the ``None`` initialiser.

    :meth:`_check_motion_gates` refuses with ``FSM id unknown`` on every
    connected driver because :attr:`_fsm_id` has no producer -- the
    motion-switcher decoder that would write it is the exact deferral
    harness#361 PR-C and #2765 name.

    The reachability assertion above rests on this: if ``_fsm_id`` were
    written from a real source, the arm-SDK gate would open and the battery
    floor's reachability would flip. Grading the write-count here means the
    day a new writer lands, both this test and the reachability one fire in
    the same PR -- and the fix is not to silence them but to replace this
    file with one that grades the new reachability directly.

    The assertion counts assignments in the shipped source, parsed with
    :mod:`ast` rather than :mod:`re`, so a comment that mentions
    ``self._fsm_id = 500`` does not read as a write. Attribute-assignment
    targets on ``self`` are what the gate observes; anything else (a
    function-local ``fsm_id = ...``, a mock's ``.``-set outside the module)
    is not what this test grades.
    """
    source = inspect.getsource(g1_module)
    tree = ast.parse(source)

    write_sites: list[tuple[int, str, ast.expr | None]] = []
    for node in ast.walk(tree):
        # ``self._fsm_id = ...`` and ``self._fsm_id: int | None = ...``. Both
        # forms are attribute stores; the annotated form is what the
        # initialiser uses today. The value node is recorded separately so
        # the assertion below can check the assigned value directly rather
        # than by substring-matching a snippet that also carries the type
        # annotation (``int | None``) and would read ``None`` even when the
        # value was mutated to a non-``None`` literal.
        if isinstance(node, ast.Assign):
            targets = node.targets
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "_fsm_id"
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                # Record the line so a failure names the site, not just a
                # count -- helpful if a new writer lands and the fix is to
                # replace this test, not to remove the writer.
                snippet = ast.get_source_segment(source, node) or ""
                write_sites.append((node.lineno, snippet.splitlines()[0], value))

    # Exactly one write site -- the ``None`` initialiser at the top of
    # ``__init__``. Stated as a length-1 assertion with the sites listed so
    # a failure shows what the second writer is.
    assert len(write_sites) == 1, (
        "expected exactly one ``self._fsm_id = ...`` in "
        f"strands_robots.drivers.g1; found: {write_sites}. "
        "The battery-floor reachability contract in this file rests on "
        "``_fsm_id`` having no producer; if a real writer lands, replace "
        "this test file with one that grades the new reachability directly."
    )
    line_no, snippet, value = write_sites[0]
    # The one write site's assigned value is the ``None`` literal. Grade the
    # ``ast`` value node (a ``Constant`` whose ``value is None``) rather than
    # substring-matching the snippet: the snippet also carries the type
    # annotation ``int | None`` and reads ``None`` even for a mutation like
    # ``self._fsm_id: int | None = 501``, which is exactly the silent-gate-
    # open this test must refuse.
    assert isinstance(value, ast.Constant) and value.value is None, (
        f"the single ``self._fsm_id`` assignment at line {line_no} is "
        f"{snippet!r}; its assigned value must be the ``None`` literal so "
        "the FSM gate stays shut and the battery-floor reachability contract "
        "in this file remains graded. A non-``None`` default would open the "
        "gate silently."
    )
