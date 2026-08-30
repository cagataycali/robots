"""The DDS-max-buffer envelope lookup tools name what the neon subscribe verb admits.

The neon bundle's DDS subscribe verb
(``cagataycali/neon-the-g1/tools/g1_dds.py::g1_dds_subscribe``) takes a
``max_buffer`` argument that names the ``collections.deque`` ``maxlen``
on the per-topic subscription handle
(``cagataycali/neon-the-g1/tools/_dds_engine.py::_SubHandle.buffer``).
The neon verb signature names ``max_buffer: int = 20`` -- the value
tuned for the agent-facing snapshot verb's token budget on one
:func:`g1_dds_read` payload -- and forwards it verbatim to
:func:`collections.deque` which admits ``maxlen=0`` at the Python
level (drops every message on arrival, empty read) but raises
``ValueError`` on a negative value.  The
:mod:`strands_robots.tools.g1.g1_dds_max_buffer_envelope` module
snapshots that observed range into module-level constants and exposes
two agent-facing verbs -
:func:`g1_list_dds_max_buffer_envelope` (name the whole envelope)
and :func:`g1_max_buffer_admits` (decide one query) - so a caller can
decide the refusal decidably before a future DDS-subscribe write path
is attempted.

The tests here fix that contract without pulling the SDK: the module
is loadable on a host without ``unitree_sdk2py`` and without a
``collections`` submodule import at load time (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
observed range surfaces here as a shape change rather than as a
diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The :func:`collections.deque` constructor's own answer at wire
  time.  The envelope is the neon bundle's observed range, not the
  deque's compile-time ``Py_ssize_t`` ceiling (which is an
  implementation detail of the CPython deque C layer and not part of
  the ``g1_dds_subscribe`` Python surface).  A driver-side wrapper
  for the DDS subscribe that lands later will re-check the envelope
  at wire time and its refusal string will surface the same
  module-local :data:`_REFUSAL_TEXT` the admits-verb quotes today.
* The live DDS state.  Whether the ``ChannelSubscriber`` singleton is
  currently subscribed to the requested topic, whether the IDL type
  resolves, whether the bus is silent: those are live driver-instance
  reads and belong on a future ``g1_dds_liveness`` verb; the envelope
  surfaces only the numeric bound decision.

One property this file explicitly refuses to pin: the ``7404``
motion-FSM refusal code from
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`.  That code is
the driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refusal on ``rt/lowcmd`` writes and its decoded text reads
``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a locomotion FSM
remedy.  The DDS subscription handle sits on the SDK-owned reader
thread in-process and never touches ``rt/lowcmd``; the DDS-subscribe
handle ships no distinct rc for a bounds-violated buffer argument,
and the refusal text this module surfaces is module-local so a
planner reading a buffer refusal sees a remedy on the same surface
the write belongs on, not a re-borrowed motion FSM code.  Cells
below pin only the module-local text; a re-borrowing of ``7404``
would fail
``test_the_refusal_text_names_the_buffer_envelope_not_the_motion_fsm``.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_dds_max_buffer_envelope import (
    _MAX_BUFFER_MAX,
    _MAX_BUFFER_MIN,
    _MAX_BUFFER_NEON_DEFAULT,
    _REFUSAL_TEXT,
    g1_list_dds_max_buffer_envelope,
    g1_max_buffer_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload."""
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358); a
    module that pulled a submodule at import time would break every
    headless CI runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dds_max_buffer_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dds_max_buffer_envelope imports "
        f"pulled SDK submodules: {leaked}. The rule for this package "
        f"is that the SDK loads only inside function bodies (refs "
        f"strands-labs/robots#358)."
    )


def test_the_import_pulls_no_collections_submodule_new() -> None:
    """The tool module load pulls no fresh ``collections`` submodule.

    ``collections`` (top-level) may already be in ``sys.modules`` from
    an earlier stdlib import (pytest, typing, etc.), so the cell
    checks the *delta*: a fresh submodule newly imported by this
    module's load path is a rule violation; an already-loaded
    ``collections`` from an unrelated pathway is not.  Pinned so a
    future edit that reaches into ``collections.deque`` at import
    time (for a compile-time ``maxlen`` ceiling, say) fails this cell
    first.
    """
    before = set(sys.modules)
    sys.modules.pop("strands_robots.tools.g1.g1_dds_max_buffer_envelope", None)
    importlib.import_module("strands_robots.tools.g1.g1_dds_max_buffer_envelope")
    after = set(sys.modules)
    added = after - before
    collections_added = {n for n in added if n.startswith("collections.")}
    assert collections_added == set(), (
        f"strands_robots.tools.g1.g1_dds_max_buffer_envelope imports "
        f"newly pulled collections submodules: {collections_added}. "
        f"The envelope port is numeric-only; the deque construction "
        f"belongs inside the driver-side wrapper that lands later, "
        f"refs strands-labs/robots#358."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """The envelope bounds are integers with min <= max.

    An inverted min/max pair (min > max) would reject every integer
    value; a non-integer bound would let a caller passing ``20.5``
    slip through the type-refusal path.
    """
    assert isinstance(_MAX_BUFFER_MIN, int) and not isinstance(_MAX_BUFFER_MIN, bool), (
        f"_MAX_BUFFER_MIN is not a plain int: {_MAX_BUFFER_MIN!r}"
    )
    assert isinstance(_MAX_BUFFER_MAX, int) and not isinstance(_MAX_BUFFER_MAX, bool), (
        f"_MAX_BUFFER_MAX is not a plain int: {_MAX_BUFFER_MAX!r}"
    )
    assert _MAX_BUFFER_MIN <= _MAX_BUFFER_MAX, (
        f"max-buffer bounds inverted: min={_MAX_BUFFER_MIN} > "
        f"max={_MAX_BUFFER_MAX}. g1_max_buffer_admits would refuse "
        f"every value."
    )


def test_the_neon_default_sits_inside_the_envelope() -> None:
    """The neon-tuned default is inside the observed clamp pair.

    The neon bundle's ``g1_dds_subscribe`` names
    ``max_buffer: int = 20`` -- the agent-facing token-budgeted
    value.  Surfacing the default at the envelope layer without also
    keeping it inside the clamp pair would hand a caller a
    "neon-observed" value the admits verb would then refuse.
    """
    assert isinstance(_MAX_BUFFER_NEON_DEFAULT, int) and not isinstance(_MAX_BUFFER_NEON_DEFAULT, bool), (
        f"_MAX_BUFFER_NEON_DEFAULT is not a plain int: {_MAX_BUFFER_NEON_DEFAULT!r}"
    )
    assert _MAX_BUFFER_MIN <= _MAX_BUFFER_NEON_DEFAULT <= _MAX_BUFFER_MAX, (
        f"neon default {_MAX_BUFFER_NEON_DEFAULT} is outside "
        f"[{_MAX_BUFFER_MIN}, {_MAX_BUFFER_MAX}]. A caller pinning "
        f"the neon-observed value would then be refused by "
        f"g1_max_buffer_admits."
    )


def test_the_envelope_matches_the_neon_observed_range() -> None:
    """The bounds match the neon-observed ``[1, 10000]`` range.

    The neon bundle's parser refuses ``max_buffer=0`` implicitly (an
    empty-read subscription) and :func:`collections.deque` refuses a
    negative :func:`~collections.deque` explicitly with
    ``ValueError``, so the floor of ``1`` is the smallest useful
    subscribe window.  The ceiling of ``10000`` names the practical
    RSS bound on a G1 :class:`PointCloud2_` window that fits inside
    the process a mesh peer allocates.  Pinning the numbers here
    surfaces a drift in either direction: a widen to ``[1, 100000]``
    (a change in the mesh RSS bound) or a narrow to ``[1, 1000]``
    (a caller-side field-note correction) would fail this cell
    first.
    """
    assert _MAX_BUFFER_MIN == 1
    assert _MAX_BUFFER_MAX == 10000
    assert _MAX_BUFFER_NEON_DEFAULT == 20


def test_the_refusal_text_names_the_buffer_envelope_not_the_motion_fsm() -> None:
    """The refusal text is module-local, not a re-borrowed motion FSM code.

    The G1 driver's :meth:`_check_motion_gates` refuses locomotion
    writes with rc=``7404`` whose text reads ``"Invalid FSM id -
    need FSM in {500, 501, 801}"``.  The DDS subscription handle
    sits on the SDK-owned reader thread in-process and never touches
    ``rt/lowcmd``; the DDS-subscribe path ships no distinct rc for a
    bounds-violated buffer argument.
    """
    assert isinstance(_REFUSAL_TEXT, str) and _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT is not a non-empty string: {_REFUSAL_TEXT!r}"
    )
    assert "max_buffer" in _REFUSAL_TEXT, f"_REFUSAL_TEXT does not name the max_buffer dimension: {_REFUSAL_TEXT!r}."
    fsm_text = ERR_CODES[7404]
    assert _REFUSAL_TEXT != fsm_text, (
        f"_REFUSAL_TEXT re-borrows the motion-FSM ``7404`` text "
        f"{fsm_text!r}. The DDS subscription handle sits on the "
        f"SDK-owned reader thread in-process and never touches "
        f"rt/lowcmd; the refusal shape must be module-local."
    )
    assert "FSM" not in _REFUSAL_TEXT, f"_REFUSAL_TEXT names the motion FSM: {_REFUSAL_TEXT!r}."


def test_g1_list_dds_max_buffer_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp and the refusal."""
    result = _call(g1_list_dds_max_buffer_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["max_buffer_min"] == _MAX_BUFFER_MIN
    assert env["max_buffer_max"] == _MAX_BUFFER_MAX
    assert env["max_buffer_neon_default"] == _MAX_BUFFER_NEON_DEFAULT
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["text"] == _REFUSAL_TEXT


def test_g1_max_buffer_admits_admits_the_neon_default() -> None:
    """The verb admits the neon-tuned default without a refusal."""
    result = _call(g1_max_buffer_admits, max_buffer=_MAX_BUFFER_NEON_DEFAULT)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_buffer_admits_admits_the_lower_bound() -> None:
    """The lower-bound value ``1`` is inside the envelope (inclusive floor)."""
    result = _call(g1_max_buffer_admits, max_buffer=_MAX_BUFFER_MIN)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_buffer_admits_admits_the_upper_bound() -> None:
    """The upper-bound value ``10000`` is inside the envelope (inclusive ceiling)."""
    result = _call(g1_max_buffer_admits, max_buffer=_MAX_BUFFER_MAX)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_buffer_admits_refuses_zero() -> None:
    """The verb refuses ``max_buffer=0`` (empty-read subscription).

    A caller passing ``0`` reserves a deque that decodes every
    message and drops it before :func:`g1_dds_read` can observe it;
    the refusal surfaces the ``max_buffer_min`` clamp and the
    module-local text.
    """
    result = _call(g1_max_buffer_admits, max_buffer=0)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "max_buffer"
    assert refusal["value"] == 0
    assert refusal["bound_key"] == "max_buffer_min"
    assert refusal["bound"] == _MAX_BUFFER_MIN
    assert refusal["text"] == _REFUSAL_TEXT


def test_g1_max_buffer_admits_refuses_negative() -> None:
    """The verb refuses a negative value with the module-local text.

    :func:`collections.deque` raises ``ValueError`` on a negative
    ``maxlen`` at construction time on the driver side; the refusal
    at the envelope layer surfaces the shape mistake before the
    subscribe path is entered.
    """
    result = _call(g1_max_buffer_admits, max_buffer=-1)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["bound_key"] == "max_buffer_min"


def test_g1_max_buffer_admits_refuses_above_the_ceiling() -> None:
    """The verb refuses values above the practical RSS ceiling."""
    result = _call(g1_max_buffer_admits, max_buffer=_MAX_BUFFER_MAX + 1)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["bound_key"] == "max_buffer_max"
    assert result["refusals"][0]["value"] == _MAX_BUFFER_MAX + 1


def test_g1_max_buffer_admits_refuses_boolean_at_type_boundary() -> None:
    """The verb refuses ``bool`` values before an ``int`` lookup silently admits.

    Python's ``bool`` is a subclass of ``int``; ``True`` would
    otherwise silently look up ``1`` (a legitimate one-message
    window) and hide the type mistake.
    """
    for value in (True, False):
        result = _call(g1_max_buffer_admits, max_buffer=value)
        assert result["status"] == "success"
        assert result["admits"] is False, f"boolean {value!r} was silently admitted"
        assert result["refusals"][0]["comparison"] == "non-int"


def test_g1_max_buffer_admits_refuses_float() -> None:
    """The verb refuses ``float`` values before ``deque`` raises ``TypeError``."""
    result = _call(g1_max_buffer_admits, max_buffer=20.0)  # type: ignore[arg-type]
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["refusals"][0]["comparison"] == "non-int"


def test_g1_max_buffer_admits_refuses_decimal() -> None:
    """The verb refuses ``Decimal`` values with the same shape as ``float``."""
    result = _call(g1_max_buffer_admits, max_buffer=Decimal("20"))  # type: ignore[arg-type]
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["refusals"][0]["comparison"] == "non-int"


@pytest.mark.parametrize("value", [_MAX_BUFFER_MIN, 5, 20, 100, 1000, _MAX_BUFFER_MAX])
def test_g1_max_buffer_admits_admits_every_value_across_the_envelope(value: int) -> None:
    """Every value inside the envelope admits."""
    result = _call(g1_max_buffer_admits, max_buffer=value)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_buffer_admits_carries_the_envelope_on_a_refusal() -> None:
    """A refused payload still names the envelope for the caller."""
    result = _call(g1_max_buffer_admits, max_buffer=-999)
    assert result["status"] == "success"
    assert result["admits"] is False
    env = result["envelope"]
    assert env["max_buffer_min"] == _MAX_BUFFER_MIN
    assert env["max_buffer_max"] == _MAX_BUFFER_MAX
    assert env["max_buffer_neon_default"] == _MAX_BUFFER_NEON_DEFAULT
