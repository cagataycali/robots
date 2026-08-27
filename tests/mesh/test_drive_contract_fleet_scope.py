"""Which drive guarantees are fleet-wide, and which only some platforms keep.

Four mesh classes expose the same ``drive(linear, angular, duration, count)``
call over three transports and two message layouts, and an operator or agent
that learns the contract from one drives the others with it. Three of the
guarantees really are kept by every class they are measured on - the numeric
domains every value is checked against, the single-shot latch, and the trailing
zero Twist - while the velocity clamp and the ``max_duration`` ceiling are kept
by ``RosbridgeRobot`` and ``AckermannRosRobot`` only.

The trailing zero is the guarantee this file was written for, because it is the
one that decides whether a *timed* drive can leave a live velocity behind. It
was ``RosbridgeRobot``'s alone: ``RtpsRobot("rover", "/cmd_vel").drive(
linear=1.0, duration=5.0)`` published fifty Twists at 1.0 m/s and then stopped
publishing, leaving the last one latched in the robot's controller, while the
prose read as though it self-stopped. Consolidating the drive contract onto the
shared mobile base made the prose true instead of correcting it downwards, and
the assertions below moved with the measurement - which is the direction this
file is meant to push.

Every check here therefore *measures* each guarantee on all three bridges and
grades the prose against the measurement, rather than pinning either half to a
hardcoded expectation. A bridge that later gains the trailing stop makes the
scope assertion fail, which is the intended signal: the guarantee has become
fleet-wide and the two places that scope it have to say so. That is not
hypothetical - it is what happened to the trailing zero.

The classes each guarantee is measured over are *derived*, not listed, because a
scope claim can only be graded against the whole fleet: a class that owns
``drive`` and lives in a public module of ``strands_robots.mesh`` is a shipped
platform bridge, and a fifth one fails the inventory below until it is given a
case. Two guarantees are read off the content of a ``Twist`` and so are measured
on the three ``Twist`` bridges only - an Ackermann car halts with a zero
``ServoCtrlMsg``, the same exclusion ``test_bridge_stop_tool_parity.py`` states
for the same reason. The other three are read only from whether a call was
forwarded at all and whether two requests forward the same thing, which is true
of any message layout, so they are measured on every drive-owning class.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import strands_robots.mesh as mesh_pkg
import strands_robots.mesh.ackermann_robot as ackermann_mod
import strands_robots.mesh.ros_bridge as ros_bridge_mod
import strands_robots.mesh.rosbridge_robot as rosbridge_mod
import strands_robots.mesh.rtps_robot as rtps_mod


class _Recorder:
    """Records the kwargs of each forwarded transport call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


#: (label, module, forwarded transport symbol, robot factory). One publish rate
#: for all three so a duration hold derives the same message count everywhere.
_BRIDGES: list[tuple[str, Any, str, Callable[[], Any]]] = [
    (
        "RosBridgedRobot",
        ros_bridge_mod,
        "use_ros",
        lambda: ros_bridge_mod.RosBridgedRobot("rover", "/cmd_vel", "/odom", publish_rate=10.0),
    ),
    ("RtpsRobot", rtps_mod, "use_rtps", lambda: rtps_mod.RtpsRobot("rover", "/cmd_vel", publish_rate=10.0)),
    (
        "RosbridgeRobot",
        rosbridge_mod,
        "use_rosbridge",
        lambda: rosbridge_mod.RosbridgeRobot("rover", "/cmd_vel", "/odom", publish_rate=10.0),
    ),
]

#: Every drive-owning class, for the guarantees that do not depend on the
#: message layout. An Ackermann car takes the same ``drive`` call over a servo
#: pair instead of a ``Twist``, so it is inside those three surveys and outside
#: the two that read a ``Twist`` field.
_DRIVE_OWNERS: list[tuple[str, Any, str, Callable[[], Any]]] = [
    *_BRIDGES,
    (
        "AckermannRosRobot",
        ackermann_mod,
        "use_ros",
        lambda: ackermann_mod.AckermannRosRobot("car", "/servo", publish_rate=10.0),
    ),
]

#: The guarantees whose probe reads only whether a call was forwarded and
#: whether two requests forward the same thing. Those two questions can be put
#: to any transport, so these are measured over ``_DRIVE_OWNERS``; the rest read
#: a ``Twist`` field and are measured over ``_BRIDGES``.
_LAYOUT_INDEPENDENT = frozenset({"validated inputs", "velocity clamp", "duration ceiling"})

#: A velocity no surveyed platform's ceiling reaches, so a request for it is
#: over every declared limit. Pinned against the real ceilings by
#: ``test_no_surveyed_ceiling_reaches_the_probe_velocity``.
_ABOVE_ANY_CEILING = 99.0

_ZERO_TWIST = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def _drive_owning_mesh_classes() -> dict[str, type]:
    """The shipped mesh classes whose public API includes ``drive``.

    Read from the package rather than listed so a new platform bridge cannot
    sit outside the scope survey: the prose graded below claims things about
    *every* mobile base, and a class the survey never learned exists makes
    every one of those claims pass unmeasured.

    A shipped platform bridge lives in a public module - a leading underscore
    marks a shared base or helper (``_mobile_base``, ``_numeric_options``),
    which declares the contract but is not a platform anyone drives - and the
    lookup is by attribute, so a bridge that inherits ``drive`` from such a
    base is still found.
    """
    package = Path(inspect.getfile(mesh_pkg)).parent
    owners: dict[str, type] = {}
    for path in sorted(package.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"{mesh_pkg.__name__}.{path.stem}")
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isclass(obj) or obj.__module__ != module.__name__:
                continue
            if callable(getattr(obj, "drive", None)):
                owners[name] = obj
    return owners


def _drive(
    monkeypatch: pytest.MonkeyPatch, bridge: tuple[str, Any, str, Callable[[], Any]], **kwargs: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drive one bridge with its transport replaced by a recorder."""
    _, module, symbol, factory = bridge
    recorder = _Recorder()
    monkeypatch.setattr(module, symbol, recorder)
    return factory().drive(**kwargs), recorder.calls


# The measurable form of each documented guarantee. Each probe returns True when
# the bridge exhibits it, so the prose can be graded against three real bridges
# instead of against a list someone kept by hand.


def _validates_before_publishing(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=float("nan"))
    return result["status"] == "error" and calls == []


def _latches_a_single_shot(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=1.0, count=1)
    return result["status"] == "success" and len(calls) == 1 and calls[0]["fields"] != _ZERO_TWIST


def _clamps_velocity(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    """A ceiling makes two different over-ceiling requests indistinguishable.

    Read as the equality of the two forwarded calls rather than off
    ``fields["linear"]["x"]``, so the same probe answers for a servo pair as
    for a ``Twist``: whatever a platform puts on the wire, a clamped request
    puts the *ceiling* there both times. That the equality reports saturation
    and not a payload that ignores ``linear`` is pinned by
    ``test_the_clamp_probe_can_tell_two_velocities_apart``.
    """
    _, at = _drive(monkeypatch, bridge, linear=_ABOVE_ANY_CEILING, count=1)
    _, above = _drive(monkeypatch, bridge, linear=_ABOVE_ANY_CEILING * 2, count=1)
    return bool(at) and bool(above) and at[0] == above[0]


def _refuses_a_hold_past_a_ceiling(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    result, calls = _drive(monkeypatch, bridge, linear=1.0, duration=3600.0)
    return result["status"] == "error" and calls == []


def _appends_a_trailing_zero(monkeypatch: pytest.MonkeyPatch, bridge: Any) -> bool:
    _, calls = _drive(monkeypatch, bridge, linear=1.0, duration=5.0)
    return bool(calls) and calls[-1]["fields"] == _ZERO_TWIST


#: guarantee -> (probe, docstring phrase, docs bullet label).
_GUARANTEES: dict[str, tuple[Callable[..., bool], str, str]] = {
    "validated inputs": (_validates_before_publishing, "validated", "Finite-input guards"),
    "single-shot latch": (_latches_a_single_shot, "latches", "Single-shot latch"),
    "velocity clamp": (_clamps_velocity, "clamped", "Velocity clamps"),
    "duration ceiling": (_refuses_a_hold_past_a_ceiling, "max_duration", "Loud duration rejection"),
    "trailing zero Twist": (_appends_a_trailing_zero, "zero Twist", "Timed-command trailing zero"),
}

_FLEET_MARKER = "Fleet-standard across all three mobile-base bridges:"
_SCOPED_MARKER = "Not carried by every mobile base:"


def _surveyed(guarantee: str) -> list[tuple[str, Any, str, Callable[[], Any]]]:
    """The classes ``guarantee`` is measured over - see ``_LAYOUT_INDEPENDENT``."""
    return _DRIVE_OWNERS if guarantee in _LAYOUT_INDEPENDENT else _BRIDGES


def _measure(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The classes that exhibit each guarantee, measured through ``drive``."""
    return {
        name: [c[0] for c in _surveyed(name) if probe(monkeypatch, c)] for name, (probe, _, _) in _GUARANTEES.items()
    }


def _holds_everywhere(guarantee: str, holders: list[str]) -> bool:
    """Whether ``guarantee`` is kept by every class it is measured over."""
    return set(holders) == {c[0] for c in _surveyed(guarantee)}


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _scoped_paragraph(marker: str) -> str:
    """The ``drive`` docstring paragraph introduced by ``marker``."""
    doc = inspect.getdoc(rosbridge_mod.RosbridgeRobot.drive) or ""
    matches = [p for p in _paragraphs(doc) if p.startswith(marker)]
    assert len(matches) == 1, (
        f"RosbridgeRobot.drive should carry exactly one paragraph opening {marker!r}, found {len(matches)}. "
        "Some of its guarantees hold on every mobile base and some on only a few, so the docstring has to "
        "label which is which - an unlabelled list reads as fleet-wide."
    )
    return matches[0]


def test_the_three_bridges_are_the_ones_under_comparison() -> None:
    """Premise: a shrunken bridge list would make every scope check vacuous."""
    assert [b[0] for b in _BRIDGES] == ["RosBridgedRobot", "RtpsRobot", "RosbridgeRobot"]


def test_every_drive_owning_mesh_class_is_surveyed() -> None:
    """No shipped platform may sit outside the survey the prose is graded on.

    The failure this exists for is silence, not a wrong answer: a class the
    survey never learned about cannot contradict a "this bridge only" claim, so
    every scope check passes while the claim it grades is false. Give the new
    class a case here and the claims are measured against it.
    """
    assert {c[0] for c in _DRIVE_OWNERS} == set(_drive_owning_mesh_classes())


def test_a_layout_independent_guarantee_is_measured_on_more_than_the_twist_bridges() -> None:
    """Premise: the wider survey has to be wider, or it grades nothing new."""
    assert _LAYOUT_INDEPENDENT, "some guarantee must be measurable on any message layout"
    assert {c[0] for c in _BRIDGES} < {c[0] for c in _DRIVE_OWNERS}
    for guarantee in _LAYOUT_INDEPENDENT:
        assert guarantee in _GUARANTEES, f"{guarantee!r} is not one of the graded guarantees"


def test_no_surveyed_ceiling_reaches_the_probe_velocity() -> None:
    """Premise: ``_ABOVE_ANY_CEILING`` has to be above the declared ceilings.

    A probe velocity a platform's own limit exceeds is not an over-ceiling
    request at all, and the clamp probe would report that platform as
    unclamped.
    """
    for label, _, _, factory in _DRIVE_OWNERS:
        limits = {n: v for n, v in vars(factory()).items() if n.startswith("max_") and isinstance(v, float)}
        for name, value in sorted(limits.items()):
            assert value < _ABOVE_ANY_CEILING, (
                f"{label}.{name} is {value}, at or above the {_ABOVE_ANY_CEILING} probe velocity"
            )


def test_the_clamp_probe_can_tell_two_velocities_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise: the clamp probe's equality must report saturation.

    It compares two forwarded calls, so a platform that ignored ``linear``
    entirely would read as clamped. An in-range request has to reach the wire
    as something different from an over-ceiling one on every surveyed class.
    """
    for case in _DRIVE_OWNERS:
        _, low = _drive(monkeypatch, case, linear=0.25, count=1)
        _, high = _drive(monkeypatch, case, linear=_ABOVE_ANY_CEILING, count=1)
        assert low and high, f"{case[0]} forwarded nothing for an in-range velocity"
        assert low[0] != high[0], (
            f"{case[0]} forwards the same call for 0.25 and {_ABOVE_ANY_CEILING} m/s, so the clamp probe "
            "cannot tell a ceiling from a payload that never carries the requested velocity"
        )


def test_each_guarantee_is_exhibited_by_at_least_one_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise: a probe that measures nothing would let any prose pass."""
    for name, bridges in _measure(monkeypatch).items():
        assert bridges, f"the probe for {name!r} found no bridge that exhibits it - the probe is broken"


def test_the_drive_guarantees_split_into_fleet_wide_shared_and_bridge_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured split the two prose surfaces have to describe.

    Fails if a class gains or loses one of these, which is the point: the
    guarantee's scope changed and the prose that scopes it is now stale. The
    middle bucket is the one a two-way split cannot express - a limit two
    platforms declare is neither fleet-wide nor one bridge's own, and calling
    it either misinforms a reader of the other three.
    """
    measured = _measure(monkeypatch)

    fleet_wide = sorted(name for name, holders in measured.items() if _holds_everywhere(name, holders))
    several = sorted(
        name for name, holders in measured.items() if not _holds_everywhere(name, holders) and len(holders) > 1
    )
    bridge_only = sorted(name for name, holders in measured.items() if holders == ["RosbridgeRobot"])

    assert fleet_wide == ["single-shot latch", "trailing zero Twist", "validated inputs"], measured
    assert several == ["duration ceiling", "velocity clamp"], measured
    # Empty since the shared mobile base took over the drive contract: the
    # trailing zero was the last entry here and every Twist bridge inherits it
    # now. This is the signal the module docstring predicts, so the tier is kept
    # rather than deleted - a sixth platform that trails a zero its siblings do
    # not lands here and the prose has to say so. The three-way split stays
    # graded because the middle tier is occupied; it is the *middle* tier a
    # two-way split cannot express, and that is the one still populated.
    assert bridge_only == [], measured
    assert sorted(fleet_wide + several + bridge_only) == sorted(_GUARANTEES), (
        f"every guarantee must be fleet-wide, shared by some, or this bridge's alone, got {measured}"
    )


def test_the_clamp_and_the_ceiling_are_carried_by_the_two_bridges_that_know_a_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measurement the two prose surfaces got wrong.

    ``RosbridgeRobot`` and ``AckermannRosRobot`` each wrap a platform whose
    limits are known, so each declares them; the ROS 2 and RTPS bridges publish
    to whatever ``cmd_vel`` topic they are pointed at and declare none. A reader
    told the clamp and the ceiling are one bridge's own plans a hold on a
    DeepRacer that its own ``max_duration`` refuses.
    """
    measured = _measure(monkeypatch)
    for guarantee in ("velocity clamp", "duration ceiling"):
        assert measured[guarantee] == ["RosbridgeRobot", "AckermannRosRobot"], measured


def test_a_timed_drive_self_stops_on_every_twist_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence the trailing-zero scope is about, read off the wire.

    ``RosBridgedRobot`` and ``RtpsRobot`` used to leave 1.0 m/s latched in the
    robot's controller here while only ``RosbridgeRobot`` trailed a zero. Both
    inherit the trailing zero from the shared mobile base now, so the last thing
    on the wire is a stop on all three - which is why the guarantee moved into
    the fleet-wide tier above. Measured rather than asserted from the class
    layout, so a transport that stops trailing the zero fails here first.
    """
    final_velocity: dict[str, float] = {}
    for bridge in _BRIDGES:
        _, calls = _drive(monkeypatch, bridge, linear=1.0, duration=5.0)
        assert [c["count"] for c in calls][0] == 50, "round(5.0 * 10.0) messages"
        final_velocity[bridge[0]] = calls[-1]["fields"]["linear"]["x"]

    assert final_velocity == {"RosBridgedRobot": 0.0, "RtpsRobot": 0.0, "RosbridgeRobot": 0.0}


#: Any wording that asserts this bridge's contract is the fleet's. Matched
#: rather than pinned so the check grades whatever phrasing is in use.
_SHARING_CLAIM = re.compile(r"shared with|[Ff]leet-standard")


def test_no_paragraph_claiming_a_shared_contract_states_a_bridge_specific_guarantee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect this file exists for, graded against whatever prose is there.

    A guarantee carried by one bridge, written into a sentence group that says
    the contract is shared with the other two, is read as theirs as well. The
    trailing zero is the costly one: it is the guarantee that a timed drive
    cannot leave a live velocity behind, and only this bridge has it.
    """
    measured = _measure(monkeypatch)
    scoped = {_GUARANTEES[name][1]: name for name, holders in measured.items() if not _holds_everywhere(name, holders)}
    assert scoped, f"premise: some guarantee must not be fleet-wide, got {measured}"

    doc = inspect.getdoc(rosbridge_mod.RosbridgeRobot.drive) or ""
    claiming = [p for p in _paragraphs(doc) if _SHARING_CLAIM.search(p)]
    assert claiming, "premise: the docstring should say which guarantees the three bridges share"

    for paragraph in claiming:
        overreach = sorted(f"{name!r} ({phrase!r})" for phrase, name in scoped.items() if phrase in paragraph)
        assert not overreach, (
            f"this paragraph of RosbridgeRobot.drive claims a shared contract and then states "
            f"{', '.join(overreach)}, which not every mobile base carries: {paragraph}"
        )


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docstring_states_each_guarantee_in_the_scope_it_holds_in(
    monkeypatch: pytest.MonkeyPatch, guarantee: str
) -> None:
    """A guarantee three bridges keep and one they do not cannot share a scope."""
    measured = _measure(monkeypatch)[guarantee]
    is_fleet_wide = _holds_everywhere(guarantee, measured)
    phrase = _GUARANTEES[guarantee][1]
    fleet, scoped = _scoped_paragraph(_FLEET_MARKER), _scoped_paragraph(_SCOPED_MARKER)
    holder, other = (fleet, scoped) if is_fleet_wide else (scoped, fleet)
    scope = "fleet-wide" if is_fleet_wide else f"carried only by {', '.join(measured)}"

    assert phrase in holder, (
        f"{guarantee!r} is {scope}, so {phrase!r} belongs in the "
        f"{'fleet-standard' if is_fleet_wide else 'scoped'} paragraph of RosbridgeRobot.drive"
    )
    assert phrase not in other, f"{guarantee!r} is {scope}, so {phrase!r} must not appear in the other scope"


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docstring_names_every_other_class_that_carries_a_scoped_guarantee(
    monkeypatch: pytest.MonkeyPatch, guarantee: str
) -> None:
    """A limit another platform also declares has to be credited to it.

    Reading ``RosbridgeRobot.drive`` is how a caller learns where each limit
    stands, and a guarantee listed as this bridge's while a sibling class keeps
    it too sends that sibling's operator looking for a ceiling they already
    have - or, worse, planning around one they will hit.
    """
    holders = _measure(monkeypatch)[guarantee]
    if _holds_everywhere(guarantee, holders):
        pytest.skip(f"{guarantee!r} is fleet-wide, so there is no co-holder to credit")
    scoped = _scoped_paragraph(_SCOPED_MARKER)
    for name in holders:
        if name == "RosbridgeRobot":
            continue
        assert name in scoped, (
            f"{guarantee!r} is carried by {', '.join(holders)}, so the scoped paragraph of "
            f"RosbridgeRobot.drive has to name {name} - as written it reads as this bridge's own"
        )


def test_the_docstring_names_what_every_other_drive_owner_does_instead() -> None:
    """A reader told a guarantee is local still needs to know the alternative.

    Derived from the survey rather than listed, so a fifth platform bridge has
    to be placed in this paragraph too instead of being quietly left out of a
    comparison that claims to cover the fleet.
    """
    specific = _scoped_paragraph(_SCOPED_MARKER)
    for label, _, _, _ in _DRIVE_OWNERS:
        if label == "RosbridgeRobot":
            continue
        assert label in specific, f"the scoped paragraph should say where {label} stands"
    # What the other two do instead is "declare no ceiling", not "leave it
    # latched": they inherit the trailing zero now, so the only guarantees still
    # scoped are the clamp and the ceiling, and an unset limit there is the thing
    # a reader has to be told is not a zero one.
    assert "unclamped" in specific, "it should say the other two publish the requested burst unclamped"


# The docs page carries the same claim, so it is graded the same way ----------


def _drive_contract_bullets() -> dict[str, str]:
    """``label -> bullet text`` for the safety-semantics list on the docs page."""
    page = Path(inspect.getfile(rosbridge_mod)).parents[2] / "docs" / "rosbridge-integration.md"
    body = page.read_text(encoding="utf-8")
    section = re.search(r"\n### Drive contract\n(.*?)(?=\n### )", body, re.DOTALL)
    assert section is not None, (
        "docs/rosbridge-integration.md should carry a '### Drive contract' section. It documented this "
        "bridge's clamp, ceiling and trailing zero under a 'Fleet drive contract' heading, which reads as "
        "though the ROS 2 and RTPS bridges carry them too."
    )
    bullets = re.findall(r"\n- \*\*(.+?)\*\*(.*?)(?=\n- \*\*|\n\n|\Z)", section.group(1), re.DOTALL)
    return {label: " ".join((label + tail).split()) for label, tail in bullets}


#: A bullet whose guarantee is not fleet-wide carries a parenthetical scope
#: note right after its bold label; the list's own heading says "fleet-wide
#: unless marked", so an unmarked bullet claims every mobile base.
_SOLE_OWNER_CLAIM = "this bridge only"


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docs_page_marks_each_guarantee_that_is_not_fleet_wide(
    monkeypatch: pytest.MonkeyPatch, guarantee: str
) -> None:
    """The page's safety list is read as the fleet's unless it says otherwise."""
    measured = _measure(monkeypatch)[guarantee]
    label = _GUARANTEES[guarantee][2]
    bullets = _drive_contract_bullets()

    assert label in bullets, f"the drive-contract list should still document {label!r}, got {sorted(bullets)}"
    bullet = bullets[label]
    marked = bullet.startswith(f"{label} (")
    if _holds_everywhere(guarantee, measured):
        assert not marked, f"{guarantee!r} holds on every mobile base surveyed, so {label!r} must not be scoped"
    else:
        assert marked, (
            f"{guarantee!r} is carried only by {', '.join(measured)}, so the {label!r} bullet has to say so - "
            "unmarked, it reads as a guarantee of every mobile-base bridge"
        )


@pytest.mark.parametrize("guarantee", sorted(_GUARANTEES))
def test_the_docs_page_calls_a_guarantee_this_bridges_own_only_when_it_is(
    monkeypatch: pytest.MonkeyPatch, guarantee: str
) -> None:
    """The defect: a limit two platforms declare, marked as one bridge's own.

    ``(this bridge only)`` is a stronger claim than "not fleet-wide", and it is
    the claim a reader acts on: told the duration ceiling is this bridge's,
    nobody expects ``AckermannRosRobot(...).drive(linear=0.5, duration=30.0)``
    to be refused by a ceiling of its own. The bullet has to name every other
    class that carries it instead.
    """
    holders = _measure(monkeypatch)[guarantee]
    if _holds_everywhere(guarantee, holders):
        pytest.skip(f"{guarantee!r} is fleet-wide, so its bullet is unmarked and credits nobody")
    others = [name for name in holders if name != "RosbridgeRobot"]
    if not others:
        return
    bullet = _drive_contract_bullets()[_GUARANTEES[guarantee][2]]
    assert _SOLE_OWNER_CLAIM not in bullet, (
        f"{guarantee!r} is carried by {', '.join(holders)}, so the "
        f"{_GUARANTEES[guarantee][2]!r} bullet must not say {_SOLE_OWNER_CLAIM!r}"
    )
    for name in others:
        assert name in bullet, (
            f"{guarantee!r} is carried by {', '.join(holders)}, so the "
            f"{_GUARANTEES[guarantee][2]!r} bullet has to name {name}"
        )
