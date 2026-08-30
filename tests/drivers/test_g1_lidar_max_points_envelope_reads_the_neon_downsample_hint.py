"""The lidar-max-points envelope lookup tools name what the neon snapshot verb admits.

The neon bundle's lidar snapshot verb
(``cagataycali/neon-the-g1/tools/g1_lidar.py::g1_lidar_snapshot``) takes
a ``max_points`` argument that names the stride-based downsample target
for one Livox MID-360 :class:`PointCloud2_` frame.  The parser
(``_cloud_to_numpy`` in the same neon module) computes the stride as
``max(1, total_points // max_points)`` and slices the raw bytes at that
step; a value of ``0`` would raise ``ZeroDivisionError`` and a negative
value would flip the sign of the stride and yield an empty slice, so
the envelope's floor is ``1``.  The SLAM feeder path
(``cagataycali/neon-the-g1/tools/g1_slam.py``) calls the parser with
``max_points=50000`` -- the pipeline-observed ceiling above which the
stride collapses to ``1`` on every Livox MID-360 frame (which fires
~24000 points per 100 ms sweep) and the ICP registration cost stops
scaling.  The
:mod:`strands_robots.tools.g1.g1_lidar_max_points_envelope` module
snapshots that observed range into module-level constants and exposes
two agent-facing verbs -
:func:`g1_list_lidar_max_points_envelope` (name the whole envelope)
and :func:`g1_max_points_admits` (decide one query) - so a caller can
decide the refusal decidably before a future lidar-frame write path is
attempted.  The tests here fix that contract without pulling the SDK
or ``numpy``: the module is loadable on a host without
``unitree_sdk2py`` and without a numpy submodule import at load
time (the same SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
observed range surfaces here as a shape change rather than as a
diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The Livox MID-360 firmware's own answer at wire time.  The envelope
  is the neon bundle's observed range, not the firmware's compile-time
  per-sweep point count (which is an implementation detail of the
  Livox capture pipeline and not part of the ``rt/utlidar/cloud``
  Python surface).  A driver-side wrapper for the lidar snapshot that
  lands later will re-check the envelope at wire time and its refusal
  string will surface the same module-local :data:`_REFUSAL_TEXT` the
  admits-verb quotes today.
* The live lidar state.  Whether the ``_ensure_subs`` singleton is
  currently subscribed to ``rt/utlidar/cloud``, whether the latest
  cache carries a fresh frame, whether the Livox firmware is
  publishing at all: those are live driver-instance reads and belong
  on the existing ``g1_lidar_state`` verb; the envelope surfaces only
  the numeric bound decision.

One property this file explicitly refuses to pin: the ``7404``
motion-FSM refusal code from
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`.  That code is
the driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refusal on ``rt/lowcmd`` writes and its decoded text reads
``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a locomotion FSM
remedy.  The lidar frame parser runs on the caller's Python thread
in-process and never touches ``rt/lowcmd``; the lidar-frame pipeline
ships no distinct rc for a bounds-violated downsample argument, and
the refusal text this module surfaces is module-local so a planner
reading a downsample refusal sees a remedy on the same surface the
read belongs on, not a re-borrowed motion FSM code.  Cells below pin
only the module-local text; a re-borrowing of ``7404`` would fail
``test_the_refusal_text_names_the_downsample_envelope_not_the_motion_fsm``.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_lidar_max_points_envelope import (
    _MAX_POINTS_MAX,
    _MAX_POINTS_MIN,
    _MAX_POINTS_NEON_DEFAULT,
    _MAX_POINTS_SLAM_INTERNAL,
    _REFUSAL_TEXT,
    g1_list_lidar_max_points_envelope,
    g1_max_points_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358); a
    module that pulled a submodule at import time would break every
    headless CI runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_max_points_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_lidar_max_points_envelope imports "
        f"pulled SDK submodules: {leaked}. The rule for this package "
        f"is that the SDK loads only inside function bodies (refs "
        f"strands-labs/robots#358)."
    )


def test_the_import_pulls_no_numpy_module() -> None:
    """The tool module is loadable without a ``numpy`` submodule at load.

    The neon bundle's ``g1_lidar._cloud_to_numpy`` imports numpy at
    module load to run the parser's stride math; the envelope port
    must not close that dependency on this module so a headless CI
    runner can decide the numeric refusal without numpy present.
    Pinned here so a future edit that reaches into ``numpy`` at
    import time (for a dtype-derived clamp, say) fails this cell
    first, not as a dependency surprise on a mesh peer that never
    installs numpy.
    """
    # numpy may already be in sys.modules from an earlier test import
    # (importing pytest can pull it transitively on some environments),
    # so the cell checks the *delta* rather than the absolute state:
    # a fresh submodule newly imported by this module's load path is
    # a rule violation; an already-loaded ``numpy`` from an unrelated
    # test is not.
    before = set(sys.modules)
    # force a reimport of only the envelope module to observe the
    # delta of what it pulls at its own load time.
    sys.modules.pop("strands_robots.tools.g1.g1_lidar_max_points_envelope", None)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_max_points_envelope")
    after = set(sys.modules)
    added = after - before
    numpy_added = {name for name in added if name == "numpy" or name.startswith("numpy.")}
    assert numpy_added == set(), (
        f"strands_robots.tools.g1.g1_lidar_max_points_envelope imports "
        f"newly pulled numpy submodules: {numpy_added}. The envelope "
        f"port is numeric-only; the parser's stride math belongs "
        f"inside the driver-side wrapper that lands later, refs "
        f"strands-labs/robots#358."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """The envelope bounds are integers with min <= max.

    An inverted min/max pair (min > max) would reject every integer
    value; a non-integer bound would let a caller passing ``4000.5``
    slip through the type-refusal path.  Pins the invariant so a
    widen or narrow of the observed range that inverts a pair
    surfaces here rather than as a silently unreachable envelope in
    production.
    """
    assert isinstance(_MAX_POINTS_MIN, int) and not isinstance(_MAX_POINTS_MIN, bool), (
        f"_MAX_POINTS_MIN is not a plain int: {_MAX_POINTS_MIN!r}"
    )
    assert isinstance(_MAX_POINTS_MAX, int) and not isinstance(_MAX_POINTS_MAX, bool), (
        f"_MAX_POINTS_MAX is not a plain int: {_MAX_POINTS_MAX!r}"
    )
    assert _MAX_POINTS_MIN <= _MAX_POINTS_MAX, (
        f"max-points bounds inverted: min={_MAX_POINTS_MIN} > "
        f"max={_MAX_POINTS_MAX}. g1_max_points_admits would refuse "
        f"every value."
    )


def test_the_neon_default_sits_inside_the_envelope() -> None:
    """The neon-tuned default is inside the observed clamp pair.

    The neon bundle's ``g1_lidar_snapshot`` names ``max_points=4000``
    as the "downsample target (stride-based). Default 4000." on its
    verb docstring - the agent-facing token-budgeted value.
    Surfacing the default at the envelope layer without also keeping
    it inside the clamp pair would hand a caller a "neon-observed"
    value the admits verb would then refuse - contradiction.  Pinned
    so a narrow of the observed range that accidentally drops the
    neon default fails this cell first, not as an admit/refuse
    contradiction the caller has to reason about at runtime.
    """
    assert isinstance(_MAX_POINTS_NEON_DEFAULT, int) and not isinstance(_MAX_POINTS_NEON_DEFAULT, bool), (
        f"_MAX_POINTS_NEON_DEFAULT is not a plain int: {_MAX_POINTS_NEON_DEFAULT!r}"
    )
    assert _MAX_POINTS_MIN <= _MAX_POINTS_NEON_DEFAULT <= _MAX_POINTS_MAX, (
        f"neon default {_MAX_POINTS_NEON_DEFAULT} is outside "
        f"[{_MAX_POINTS_MIN}, {_MAX_POINTS_MAX}]. A caller pinning "
        f"the neon-observed value would then be refused by "
        f"g1_max_points_admits."
    )


def test_the_slam_internal_value_sits_inside_the_envelope() -> None:
    """The SLAM-feeder internal downsample value is inside the clamp pair.

    The neon bundle's ``g1_slam.py`` calls the parser with
    ``max_points=50000`` on every frame the ICP feeder consumes; that
    value is the pipeline-observed ceiling above which the stride
    collapses to ``1``.  Surfacing it on the envelope without keeping
    it inside the clamp pair would hand a caller a "neon-observed"
    SLAM value the admits verb would then refuse.  Pinned so a
    narrow of the observed range that accidentally drops the SLAM
    ceiling fails this cell first.
    """
    assert isinstance(_MAX_POINTS_SLAM_INTERNAL, int) and not isinstance(_MAX_POINTS_SLAM_INTERNAL, bool), (
        f"_MAX_POINTS_SLAM_INTERNAL is not a plain int: {_MAX_POINTS_SLAM_INTERNAL!r}"
    )
    assert _MAX_POINTS_MIN <= _MAX_POINTS_SLAM_INTERNAL <= _MAX_POINTS_MAX, (
        f"SLAM internal {_MAX_POINTS_SLAM_INTERNAL} is outside "
        f"[{_MAX_POINTS_MIN}, {_MAX_POINTS_MAX}]. A caller pinning "
        f"the SLAM-feeder value would then be refused by "
        f"g1_max_points_admits."
    )


def test_the_envelope_matches_the_neon_observed_range() -> None:
    """The bounds match the neon-bundle-observed ``[1, 50000]`` range.

    The neon bundle's parser refuses ``max_points=0`` implicitly (via
    ``ZeroDivisionError``) so the floor of ``1`` is the smallest
    admitted stride divisor; the neon SLAM feeder names ``50000``
    as the pipeline-observed ceiling above which the stride
    collapses.  The agent-facing snapshot verb defaults to ``4000``
    and the SLAM feeder passes ``50000`` on every frame.  Pinning
    the numbers here surfaces a drift in either direction: a widen
    to ``[1, 100000]`` (a change in the Livox firmware per-sweep
    ceiling) or a narrow to ``[10, 10000]`` (a caller-side field-note
    correction) would fail this cell first.
    """
    assert _MAX_POINTS_MIN == 1
    assert _MAX_POINTS_MAX == 50000
    assert _MAX_POINTS_NEON_DEFAULT == 4000
    assert _MAX_POINTS_SLAM_INTERNAL == 50000


def test_the_refusal_text_names_the_downsample_envelope_not_the_motion_fsm() -> None:
    """The refusal text is module-local, not a re-borrowed motion FSM code.

    The G1 driver's :meth:`_check_motion_gates` refuses locomotion
    writes with rc=``7404`` whose text reads ``"Invalid FSM id -
    need FSM in {500, 501, 801}"``.  The lidar frame parser runs on
    the caller's Python thread in-process and never touches
    ``rt/lowcmd``; the lidar-frame pipeline ships no distinct rc for
    a bounds-violated downsample argument.  The refusal shape this
    module surfaces is module-local text that names the downsample
    envelope (not the motion FSM) so an agent planner reading a
    downsample refusal sees a remedy on the same surface the read
    belongs on.  Pinned here so a re-borrowing of ``7404`` (or any
    other motion-FSM entry from ``ERR_CODES``) fails this cell
    first, not as a wrong-remedy surprise in production.
    """
    assert isinstance(_REFUSAL_TEXT, str) and _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT is not a non-empty string: {_REFUSAL_TEXT!r}"
    )
    assert "max_points" in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT does not name the max_points dimension: "
        f"{_REFUSAL_TEXT!r}. A caller reading the refusal must see a "
        f"remedy on the lidar-parser surface."
    )
    fsm_text = ERR_CODES[7404]
    assert _REFUSAL_TEXT != fsm_text, (
        f"_REFUSAL_TEXT re-borrows the motion-FSM ``7404`` text "
        f"{fsm_text!r}. The lidar frame parser runs on the caller's "
        f"Python thread in-process and never touches rt/lowcmd; the "
        f"refusal shape must be module-local so a planner does not "
        f"read a motion FSM remedy for a downsample error."
    )
    assert "FSM" not in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT names the motion FSM: {_REFUSAL_TEXT!r}. The "
        f"lidar frame parser runs in-process and never touches the "
        f"locomotion FSM; the refusal remedy belongs on the "
        f"downsample surface, not the locomotion FSM."
    )


def test_g1_list_lidar_max_points_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp and the refusal.

    ``envelope`` carries every clamp constant (including the
    neon-observed default and the SLAM-internal value) and
    ``refusals`` names the module-local :data:`_REFUSAL_TEXT` a
    future driver-side lidar-snapshot wrapper would surface on a
    bounds violation.
    """
    result = _call(g1_list_lidar_max_points_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["max_points_min"] == _MAX_POINTS_MIN
    assert env["max_points_max"] == _MAX_POINTS_MAX
    assert env["max_points_neon_default"] == _MAX_POINTS_NEON_DEFAULT
    assert env["max_points_slam_internal"] == _MAX_POINTS_SLAM_INTERNAL
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["text"] == _REFUSAL_TEXT


def test_g1_max_points_admits_admits_the_neon_default() -> None:
    """The verb admits the neon-tuned default without a refusal.

    A caller passing no explicit argument lands on the neon-observed
    admitted default (``4000``); the admits verb must not refuse the
    value it names as the neon default.  Pinned so a narrow of the
    envelope that drops ``4000`` fails this cell first.
    """
    result = _call(g1_max_points_admits, max_points=_MAX_POINTS_NEON_DEFAULT)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_points_admits_admits_the_slam_ceiling() -> None:
    """The verb admits the SLAM-feeder ceiling without a refusal.

    A caller reaching for the SLAM-feeder-style read passes
    ``max_points=50000``; the admits verb must not refuse the value
    the neon SLAM feeder uses on every frame.  Pinned so a narrow of
    the envelope that drops ``50000`` fails this cell first.
    """
    result = _call(g1_max_points_admits, max_points=_MAX_POINTS_SLAM_INTERNAL)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_points_admits_admits_the_lower_bound() -> None:
    """The lower-bound value is inside the envelope (inclusive floor).

    A caller running a token-cheap probe with ``max_points=1`` is
    asking for exactly one point from the frame - a legitimate
    single-point bbox read that signals whether the lidar is
    publishing at all.  Pinned as inclusive so an off-by-one that
    flips to exclusive fails this cell first.
    """
    result = _call(g1_max_points_admits, max_points=_MAX_POINTS_MIN)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_points_admits_refuses_a_zero_stride_divisor() -> None:
    """The verb refuses ``max_points=0`` with the module-local text.

    A value of ``0`` would raise ``ZeroDivisionError`` on the
    parser's stride divide; the refusal shape names the
    ``max_points_min`` clamp violation and surfaces the
    module-local :data:`_REFUSAL_TEXT` a driver-side wrapper would
    quote.
    """
    result = _call(g1_max_points_admits, max_points=0)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "max_points"
    assert refusal["value"] == 0
    assert refusal["bound_key"] == "max_points_min"
    assert refusal["bound"] == _MAX_POINTS_MIN
    assert refusal["text"] == _REFUSAL_TEXT


def test_g1_max_points_admits_refuses_a_negative_stride_divisor() -> None:
    """The verb refuses a negative divisor with the module-local text.

    A negative divisor would flip the sign of the stride and yield
    an empty slice on the parser; the refusal shape names the
    ``max_points_min`` clamp violation.
    """
    result = _call(g1_max_points_admits, max_points=-1)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["bound_key"] == "max_points_min"
    assert result["refusals"][0]["value"] == -1


def test_g1_max_points_admits_refuses_above_the_slam_ceiling() -> None:
    """The verb refuses values above the SLAM-observed ceiling.

    A caller passing ``max_points > 50000`` collapses the stride to
    ``1`` on every Livox MID-360 frame anyway (which fires ~24000
    points per 100 ms sweep) and pays the token cost for a value
    that has no effect; the refusal surfaces the ceiling so the
    planner narrows the value before the parser runs.
    """
    result = _call(g1_max_points_admits, max_points=_MAX_POINTS_MAX + 1)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["bound_key"] == "max_points_max"
    assert result["refusals"][0]["value"] == _MAX_POINTS_MAX + 1


def test_g1_max_points_admits_refuses_a_boolean_at_the_type_boundary() -> None:
    """The verb refuses ``bool`` values before an ``int`` lookup silently admits.

    Python's ``bool`` is a subclass of ``int`` (``True == 1``,
    ``False == 0``); a caller passing ``True`` would otherwise
    silently look up ``1`` (a legitimate one-point probe) and hide
    the type mistake.  The refusal at the boundary surfaces the
    mistake instead.
    """
    for boolean_value in (True, False):
        result = _call(g1_max_points_admits, max_points=boolean_value)
        assert result["status"] == "success", f"boolean {boolean_value!r} did not reach a refusal path"
        assert result["admits"] is False, f"boolean {boolean_value!r} was silently admitted"
        assert len(result["refusals"]) == 1
        assert result["refusals"][0]["comparison"] == "non-int"


def test_g1_max_points_admits_refuses_a_float_argument() -> None:
    """The verb refuses ``float`` values before ``//`` silently truncates.

    A caller passing ``max_points=4000.0`` would otherwise be
    truncated by the parser's ``//`` operator silently.  Pinned so
    a caller passing the neon default as a float sees an actionable
    refusal rather than a silent behaviour change.
    """
    result = _call(g1_max_points_admits, max_points=4000.0)  # type: ignore[arg-type]
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["comparison"] == "non-int"


def test_g1_max_points_admits_refuses_a_decimal_argument() -> None:
    """The verb refuses ``Decimal`` values with the same shape as ``float``.

    A caller reaching for a ``Decimal`` (say, on a policy that
    keeps every numeric argument as ``Decimal`` for audit reasons)
    hits the same refusal shape as a ``float``; the parser's ``//``
    operator does not admit ``Decimal`` either.
    """
    result = _call(g1_max_points_admits, max_points=Decimal("4000"))  # type: ignore[arg-type]
    assert result["status"] == "success"
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["comparison"] == "non-int"


@pytest.mark.parametrize("value", [_MAX_POINTS_MIN, 100, 4000, 20000, _MAX_POINTS_MAX])
def test_g1_max_points_admits_admits_every_value_across_the_envelope(value: int) -> None:
    """Every value inside the envelope admits.

    A parametrised sweep across the observed range so a narrow that
    would silently exclude an interior value fails this cell first.
    """
    result = _call(g1_max_points_admits, max_points=value)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_max_points_admits_carries_the_envelope_on_a_refusal() -> None:
    """A refused payload still names the envelope for the caller.

    A caller reading the refusal shape must see the same envelope
    the admits verb returns on an admitted value; that way a
    planner adjusting the argument to satisfy the refusal reads the
    same clamp pair the initial call rejected against.
    """
    result = _call(g1_max_points_admits, max_points=-999)
    assert result["status"] == "success"
    assert result["admits"] is False
    env = result["envelope"]
    assert env["max_points_min"] == _MAX_POINTS_MIN
    assert env["max_points_max"] == _MAX_POINTS_MAX
    assert env["max_points_neon_default"] == _MAX_POINTS_NEON_DEFAULT
    assert env["max_points_slam_internal"] == _MAX_POINTS_SLAM_INTERNAL
