"""The acceptance criterion for harness#361 is a positive outcome, not a body.

harness#361 has closed early thirteen times because every box on it is
satisfied by a body existing or by a mocked test passing. A checklist of
bodies closes early forever. The gap the peer reviewer named directly on
that issue is one line long:

    ``send_action`` returns ``status="success"`` on a connected driver with a
    decoded ``LowState_`` and a healthy pack.

Nothing in the tree grades that today. :mod:`test_g1_control_loop` reaches
``send_action`` reachability through a ``MagicMock`` driver whose ``_fsm_id``
is injected and whose ``_check_motion_gates`` is a ``MagicMock``: both halves
of the gate are replaced, so no cell can observe the reachability question.
:mod:`test_g1_battery_floor_is_gated_behind_the_unwired_fsm` grades the *un*
reachability (refusal must name FSM, not battery). That pins the current
state honestly and it does not answer the peer question, which is when the
positive outcome is reachable.

This file adds the positive-outcome cell as an :func:`pytest.mark.xfail` with
``strict=True``. Three consequences worth stating so the marker is the right
one:

1. Today it xfails, and the recorded reason is exactly the shipped refusal
   text: ``FSM id unknown - motion-switcher source has not been wired
   (harness#361 PR-C); see #2765 for the wire-side decision``. So the file
   is not adding a red cell to CI; it is adding a documented deferral that
   CI grades.

2. The day a motion-switcher decoder gives ``_fsm_id`` a producer, this
   cell passes -- and because ``strict=True`` an xfail that passes is a
   failure. The change that wires the FSM will fire this cell, name it in
   its failure message ("XPASSED, expected refusal, got success"), and the
   author of that change deletes the marker in the same commit. That is the
   mechanical checkpoint the issue has been missing.

3. It is not a duplicate of the un-reachability pin. That test grades the
   refusal *text* on a driver whose publisher is deliberately absent
   (``_pubs is None``, from an unconnected driver); this test drives a
   ``_pubs`` populated with a recording publisher and expects ``success``,
   so a refusal like "publisher not initialised" would fail here without
   firing the un-reachability pin. The two contracts read different halves
   of the same gate and both need to hold on the wired-FSM day.

Which wiring shapes turn the marker over.  A strict xfail is a checkpoint only
for the shapes that can reach XPASS, so the reachable set was measured by
planting a ``_fsm_id`` producer in each place one could plausibly land and
running this file plus the un-reachability pin:

===============================================  ==============  =============
planted producer                                 marker turns    other pin
===============================================  ==============  =============
in ``_on_lowstate``, beside ``_mode_machine``    yes             fires
in ``__init__``, as a non-``None`` default       yes             fires
in ``_check_motion_gates``, fetched lazily       yes             fires
in ``send_action``, fetched before gating        yes             fires
in ``connect_eagerly``, from ``CheckMode()``     no              fires
===============================================  ==============  =============

Two things follow.  First, the fixture populates ``_mode_machine`` and
``_battery`` by calling their decoders rather than by assigning the attributes:
a fixture that assigns bypasses the callbacks, and a producer added beside
either of them then never runs, leaving the marker XFAILed and silent.  That is
measured -- with a producer planted in ``_on_lowstate``, an assigning fixture
fires nothing in this file at all.

Second, ``connect_eagerly`` is the one shape out of reach, because it opens a
DDS participant and a unit fixture cannot run it.  A producer there is caught by
``test_fsm_id_has_exactly_one_assignment_in_the_driver_module`` in the
un-reachability pin, which reads the module's assignments rather than driving
it.  The two files are complementary in that direction too, and no cell here
duplicates that count.

Refutation: this whole file would be moot if the current refusal were about
anything other than ``_fsm_id``. Both parts of the message are asserted
below (the "FSM id unknown" phrase *and* the "motion-switcher" phrase), so
a change that shifted the reason to a different attribute -- say, a battery
guard that fired first -- would trip a different assertion and the reader
would know at once that the criterion this file grades has moved rather
than remain xfailed for the wrong reason.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver

# ``_HEALTHY_MODE_MACHINE`` is what the ``rt/lowstate`` decoder produces on a
# real G1 (uint8 layout id, in ``[0, 255]``).  Any populated value gets past
# the ``mode_machine is None`` refusal and to the ``_fsm_id`` check.  Literal,
# not derived, so a rename in the module cannot silently make this file grade
# a different gate.
_HEALTHY_MODE_MACHINE = 9

# A healthy pack, well above any configured floor.  This is the ``soc`` a real
# ``rt/lf/bmsstate`` frame carries; ``_on_bms`` is what turns it into the ``pct``
# the gate reads, so the fixture hands the decoder its input rather than
# fabricating the decoder's output.
_HEALTHY_PACK_PCT = 92.0

# The battery floor is well below the pack, so the floor guard cannot be the
# reason if the FSM guard's refusal ever moves away from ``_fsm_id``.
_HEALTHY_FLOOR_PCT = 15.0


class _RecordingPublisher:
    """A minimally-real ``_pubs`` stand-in.

    ``send_action`` reads ``_pubs`` and calls ``.publish(topic, LowCmd_,
    cmd)``.  A ``MagicMock()`` here would return a ``MagicMock`` for
    ``publish``, whose truthiness (``!= None``) would make the driver report
    a refusal.  Returning ``None`` is the "success" contract publishers use,
    so this class matches production shape.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def publish(self, topic: str, msg_type: Any, cmd: Any) -> str | None:
        self.calls.append((topic, msg_type, cmd))
        return None


def _lowstate_frame() -> SimpleNamespace:
    """A duck-typed stand-in for the ``LowState_`` the driver decodes.

    :meth:`G1Driver._on_lowstate` reads its fields with ``getattr``, so this
    needs no ``unitree_sdk2py`` and the file stays importable and runnable on
    the CI box, where the SDK is not installed.  The field set is what the
    decoder reads -- ``mode_machine`` plus an ``imu_state`` whose four vectors
    it copies -- so a decoder that began reading a further field would find it
    absent here rather than silently defaulted.
    """
    return SimpleNamespace(
        mode_machine=_HEALTHY_MODE_MACHINE,
        imu_state=SimpleNamespace(
            rpy=[0.0, 0.0, 0.0],
            gyroscope=[0.0, 0.0, 0.0],
            accelerometer=[0.0, 0.0, 9.81],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
    )


def _bms_frame() -> SimpleNamespace:
    """A duck-typed stand-in for the ``rt/lf/bmsstate`` frame.

    :meth:`G1Driver._on_bms` reads ``soc`` and turns it into the ``pct`` the
    battery floor compares against, so the pack percentage is supplied here in
    the shape the wire carries it.
    """
    return SimpleNamespace(soc=_HEALTHY_PACK_PCT, charge=0, current=0.0, cycle=0)


def _healthy_driver() -> G1Driver:
    """Return a driver whose every field is what a real, healthy G1 produces.

    ``_fsm_id`` is deliberately not set here.  This is the point of the test:
    a driver on real hardware with a real ``LowState_`` decoded and a real
    battery pack still has ``_fsm_id is None`` today, because nothing in the
    driver writes it.  The refusal that produces is what the acceptance
    criterion has to remove.
    """
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
    )
    # ``_connected`` and ``_pubs`` are what a completed ``connect_eagerly``
    # leaves behind.  That method opens a DDS participant, so it is out of a
    # unit fixture's reach and these two are set directly; the consequence is
    # stated in the module docstring (a producer added *inside*
    # ``connect_eagerly`` is not reachable from this file).
    driver._connected = True
    # ``_pubs`` is typed ``DDSPublisher | None`` on the driver; the stand-in
    # matches the .publish() shape send_action reads, which is the only
    # surface this test needs.  Cast is narrower than a blanket ignore.
    driver._pubs = _RecordingPublisher()  # type: ignore[assignment]

    # ``_mode_machine`` and ``_battery`` are driven *through their decoders*
    # rather than assigned.  Those callbacks are where their producers live, so
    # a motion-switcher decoder that gives ``_fsm_id`` a producer beside either
    # of them runs here -- which is what makes the xfail below an XPASS trigger
    # for that shape rather than a marker nothing turns over.
    driver._on_lowstate(_lowstate_frame())
    driver._on_bms(_bms_frame())
    return driver


@pytest.mark.xfail(
    strict=True,
    reason=(
        "harness#361 acceptance criterion: send_action returns success on a "
        "connected driver with a decoded LowState_ and a healthy pack.  Today "
        "the driver refuses with 'FSM id unknown - motion-switcher source "
        "has not been wired (harness#361 PR-C); see #2765 for the wire-side "
        "decision'.  When a motion-switcher decoder gives _fsm_id a producer, "
        "this cell passes and (strict=True) fires an XPASS failure so the "
        "author of that change deletes the marker in the same commit."
    ),
)
def test_send_action_returns_success_on_a_healthy_driver_that_has_a_decoded_lowstate() -> None:
    """The one line the harness#361 checklist has been missing.

    Every field is populated the way a real, healthy G1 populates it:

    * ``_connected=True`` from a completed ``connect_eagerly``.
    * ``_mode_machine`` from a real ``rt/lowstate`` decode (uint8 layout id).
    * ``_battery`` from a real ``rt/lf/bmsstate`` decode, well above the
      configured floor.
    * ``_pubs`` from a real ``connect_eagerly`` (a publisher that returns
      ``None`` on ``.publish``, exactly what production carries).

    The one field left at its ``None`` initialiser is ``_fsm_id``, because
    on today's ``main`` there is no writer for it -- the same measurement
    :mod:`test_g1_battery_floor_is_gated_behind_the_unwired_fsm` pins from
    the un-reachability side.  Once wired, this cell moves from XFAIL to
    XPASS and (strict=True) that is a failure the wiring commit clears by
    removing the marker.
    """
    driver = _healthy_driver()

    # No ``_fsm_id is None`` assertion here, deliberately.  An xfail cell that
    # first asserts the un-wired state can never XPASS: on the wired-FSM day
    # that assertion is the thing that fails, so the cell stays XFAIL and the
    # strict marker never turns over.  The un-wired state is claimed by
    # :func:`test_the_fsm_id_is_still_unwritten_on_a_healthy_driver` instead,
    # which is a passing cell and so fails loudly on that day.

    # A minimal action the joint index knows about.  ``left_shoulder_pitch``
    # is in ``_G1_JOINT_INDEX`` at slot 15 (arm range); the exact joint does
    # not matter, only that the action is well-formed enough that a
    # reachable path reaches the publisher.
    result = driver.send_action({"left_shoulder_pitch": 0.0})

    # This is the criterion.  Both halves are stated: the envelope's status,
    # and the publisher recorded exactly one call (so the "success" is not
    # a return that skipped the wire).
    assert result["status"] == "success"
    assert isinstance(driver._pubs, _RecordingPublisher)
    assert len(driver._pubs.calls) == 1
    topic, _msg_type, _cmd = driver._pubs.calls[0]
    assert topic == "rt/lowcmd"


def test_the_fsm_id_is_still_unwritten_on_a_healthy_driver() -> None:
    """``_fsm_id`` is the one field a healthy driver still leaves unset.

    The xfail above is documented against that fact and deliberately does not
    assert it: an xfail cell whose first assertion is "the state is still
    un-wired" cannot XPASS, because on the wired-FSM day that assertion is what
    fails, leaving the cell XFAIL and the strict marker un-turned.  The claim
    therefore lives here, where it is a *passing* cell -- so the day a producer
    appears, this fails by name while the xfail above XPASSes.  Both halves turn
    over in the same commit, which is the point.

    ``_healthy_driver`` reaches this state the way hardware does, through
    :meth:`G1Driver._on_lowstate` and :meth:`G1Driver._on_bms`, so a producer
    added beside either decoder's write is observed here rather than bypassed.
    """
    driver = _healthy_driver()
    assert driver._fsm_id is None, (
        "A producer for _fsm_id has appeared.  That is the wired-FSM day: the "
        "xfail in this file should now be XPASSing (strict=True turns that "
        "into a failure), and both this cell and the marker are deleted by the "
        "commit that wired it."
    )
    # The fields that *do* have producers arrived through them, which is what
    # makes the sentence above a measurement rather than an assumption.
    assert driver._mode_machine == _HEALTHY_MODE_MACHINE
    assert (driver._battery or {}).get("pct") == _HEALTHY_PACK_PCT


def test_the_current_refusal_still_names_the_fsm_and_the_motion_switcher() -> None:
    """The xfail above is documented against a specific reason; hold it.

    :func:`test_send_action_returns_success_on_a_healthy_driver_that_has_a_decoded_lowstate`
    is xfailed with a reason that quotes the shipped refusal text.  If the
    refusal text changes, the xfail reason is misleading and the cell above
    might xfail for the wrong cause.  This cell reads the refusal on the
    same driver shape and asserts the two phrases the xfail reason names.

    On the wired-FSM day this cell also flips -- the refusal is gone -- and
    it is deleted alongside the marker.  Grading it here means both halves
    of the deferral (the reachability, and the reason attached to it) turn
    over in the same commit.
    """
    driver = _healthy_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # Both phrases the xfail reason cites.
    assert "FSM id unknown" in text
    assert "motion-switcher" in text


def test_the_publisher_is_populated_and_the_driver_is_otherwise_healthy() -> None:
    """The un-reachability pin uses ``_pubs is None``; hold the boundary.

    :mod:`test_g1_battery_floor_is_gated_behind_the_unwired_fsm` sets up a
    driver with no publisher, because it grades the refusal *text* rather
    than the wire outcome.  This file grades the wire outcome, so the
    publisher has to be populated -- otherwise the acceptance criterion
    would be unreachable for a second reason (publisher not initialised)
    and the xfail would flip for the wrong cause on the wired-FSM day.

    This cell asserts the boundary: on the shape ``_healthy_driver``
    produces, the *only* refusal reason is the FSM, and the publisher is
    real.  A future refactor that made the driver skip publisher setup on
    ``_fsm_id is None`` would trip this cell, and the fix is to preserve
    the current shape rather than to silence the assertion.
    """
    driver = _healthy_driver()
    assert driver._pubs is not None
    # No other guard fires: connected, mode_machine present, battery healthy,
    # so ``_check_motion_gates`` returns exactly the ``_fsm_id`` refusal.
    # The private call is deliberate here: we are grading which guard is the
    # blocker, not what a caller sees, so reading through ``send_action``
    # (which composes gate + publisher) would conflate two questions.
    refusal = driver._check_motion_gates("arm")
    assert refusal is not None
    text = refusal["content"][0]["text"]
    assert "FSM id unknown" in text
    # None of the other guards' text appears.
    assert "not connected" not in text
    assert "mode_machine unknown" not in text
    assert "battery" not in text
