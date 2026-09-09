"""``hz_actual`` must measure its numerator over the window its denominator covers.

Both sides of the mesh input stream report an achieved rate beside a cumulative
frame total::

    InputPublisher.stats -> {"frames": N,          "hz_actual": ..., "hz_target": hz}
    InputReceiver.stats  -> {"frames_received": N, "hz_actual": ...}

and ``teleop_mixin.get_teleop_status`` prints both lines from those two keys, so
one reading is compared against one ``hz_target``.

``frames`` is cumulative for the life of the object: both ``stats`` docstrings
say so, and the receiver's sampled safety audit keys its cadence on that count.
``_start_mono`` is not - every ``start()`` re-stamps it, which is exactly what
makes a *restarted* session's rate measurable, and restart is a supported flow:
the publisher's ``start()`` clears the stop event its ``stop()`` set, and the
receiver's re-declares the subscription that ``Mesh.stop`` drops ("a caller that
rejoins the mesh re-declares its own subscriptions").

Taking the rate as ``frames / elapsed`` therefore divided a lifetime numerator by
one session's denominator, and reported a rate the stream never ran at - above
the ``hz`` the publisher paces itself against, which no reading of "achieved
rate" can produce. The local teleop session in ``teleop_mixin`` already resets
its counter and its stamp together; these two did not.

The contract is pinned on the reported numbers, so a future refactor cannot
satisfy it by renaming a field: a restarted session reports the rate it ran at,
a first session is unchanged, and every reported TOTAL stays cumulative.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import input as mesh_input

#: Both sides report ``hz_actual``, so both are held to it.
SIDES = ("publisher", "receiver")

TARGET_HZ = 30.0
#: A first session long enough that its frames dominate a short second one.
FIRST_SESSION_S = 6.0
SECOND_SESSION_S = 2.0


class _MonotonicClock:
    """A monotonic clock the test advances by a known amount.

    Only ``monotonic`` is used by the code under test; ``time`` is present
    because the receiver's freshness check reads it on the frame path, which
    these cells do not drive.
    """

    def __init__(self) -> None:
        self._elapsed = 0.0

    def advance(self, seconds: float) -> None:
        self._elapsed += seconds

    def monotonic(self) -> float:
        return 1000.0 + self._elapsed

    def time(self) -> float:
        return 1_700_000_000.0 + self._elapsed


class _StubTransport:
    """Mesh stand-in that declares and drops subscriptions like the real one."""

    peer_id = "test-peer"

    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    def subscribe(self, topic: str, **_kwargs: Any) -> str:
        self.subscribed.append(topic)
        return f"sub-{len(self.subscribed)}"

    def unsubscribe(self, name: str) -> None:
        self.unsubscribed.append(name)


def _build(kind: str, monkeypatch: pytest.MonkeyPatch, clock: _MonotonicClock) -> Any:
    """One started-and-stoppable input side, with the loop body left out.

    The frame counters are driven directly (``_frame_count``), the way the
    sibling clock-domain suite drives them: what is under test is the
    arithmetic ``stats`` reports over a window, not the loop that fills it.
    """
    monkeypatch.setattr(mesh_input, "time", clock)
    transport = _StubTransport()
    side: Any
    if kind == "publisher":
        side = mesh_input.InputPublisher(
            mesh=transport,  # type: ignore[arg-type]
            teleoperator=object(),
            device_name="leader",
            hz=TARGET_HZ,
        )
        # The publish loop is not under test; without this its thread would
        # poll a teleoperator that has no get_action().
        monkeypatch.setattr(side, "_publish_loop", lambda: None)
    else:
        side = mesh_input.InputReceiver(
            mesh=transport,  # type: ignore[arg-type]
            robot=object(),
            source_peer_id="leader-peer",
            device_name="leader",
        )
    return side


def _rate(side: Any) -> float:
    return float(side.stats["hz_actual"])


def _total(side: Any) -> int:
    stats = side.stats
    return int(stats["frames"] if "frames" in stats else stats["frames_received"])


def _run_session(side: Any, clock: _MonotonicClock, seconds: float, hz: float) -> None:
    """Run one session of ``seconds`` at ``hz``, then stop."""
    side.start()
    clock.advance(seconds)
    side._frame_count += round(seconds * hz)
    side.stop()


@pytest.fixture
def clock() -> _MonotonicClock:
    return _MonotonicClock()


@pytest.mark.parametrize("kind", SIDES)
def test_a_restarted_session_reports_the_rate_it_ran_at(
    kind: str, clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frames of a previous session must not be charged to this one's clock."""
    side = _build(kind, monkeypatch, clock)
    _run_session(side, clock, FIRST_SESSION_S, TARGET_HZ)

    side.start()
    clock.advance(SECOND_SESSION_S)
    side._frame_count += round(SECOND_SESSION_S * TARGET_HZ)
    try:
        assert _rate(side) == pytest.approx(TARGET_HZ), (
            f"a {kind} restarted after {round(FIRST_SESSION_S * TARGET_HZ)} frames reported "
            f"{_rate(side):.2f}Hz for a stream running at {TARGET_HZ}Hz - the first session's "
            "frames were divided by the second session's elapsed time"
        )
    finally:
        side.stop()


@pytest.mark.parametrize("kind", SIDES)
def test_the_reported_rate_never_exceeds_the_rate_the_stream_ran_at(
    kind: str, clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate above the paced one is unreachable, so reporting one is a defect.

    Stated as the invariant rather than a number: an achieved rate is bounded by
    the rate the frames were produced at, whichever session is being measured.
    """
    side = _build(kind, monkeypatch, clock)
    for _ in range(3):
        _run_session(side, clock, FIRST_SESSION_S, TARGET_HZ)
        # Idle between sessions: real time passes with no frames on the wire.
        clock.advance(11.0)

    side.start()
    try:
        for _ in range(5):
            clock.advance(1.0)
            side._frame_count += round(TARGET_HZ)
            assert _rate(side) <= TARGET_HZ + 1e-6, (
                f"a {kind} reported {_rate(side):.2f}Hz, above the {TARGET_HZ}Hz the stream ran at"
            )
    finally:
        side.stop()


def test_a_frame_the_loop_published_after_stop_is_not_charged_to_the_next_session(
    clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window opens where the session does, which is why ``start()`` stamps it.

    ``InputPublisher.stop`` documents that a teleoperator whose ``get_action()``
    blocks past the join budget "leaves the loop free to publish one more frame
    after this call returns", so the counter can still move between a stop and
    the next start. Stamping the window at ``stop()`` instead would be
    indistinguishable on every other path and would charge those frames to the
    session that had not begun when they went out.
    """
    side = _build("publisher", monkeypatch, clock)
    _run_session(side, clock, FIRST_SESSION_S, TARGET_HZ)
    # The documented case: the loop outlived the join and put one more frame out.
    side._frame_count += 1

    side.start()
    clock.advance(1.0)
    side._frame_count += round(TARGET_HZ)
    try:
        assert _rate(side) == pytest.approx(TARGET_HZ), (
            f"a frame published after stop() returned was charged to the next session: "
            f"{_rate(side):.2f}Hz for a stream running at {TARGET_HZ}Hz"
        )
    finally:
        side.stop()


@pytest.mark.parametrize("kind", SIDES)
def test_a_first_session_is_unchanged(kind: str, clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: with no restart the two operands already agreed."""
    side = _build(kind, monkeypatch, clock)
    side.start()
    clock.advance(SECOND_SESSION_S)
    side._frame_count += round(SECOND_SESSION_S * TARGET_HZ)
    try:
        assert _rate(side) == pytest.approx(TARGET_HZ)
    finally:
        side.stop()


@pytest.mark.parametrize("kind", SIDES)
def test_the_reported_frame_total_stays_cumulative_across_a_restart(
    kind: str, clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the totals are lifetime counts, so the fix must not reset one.

    Zeroing the counter would satisfy the rate cells above and silently turn a
    documented cumulative total - the one the receiver's sampled safety audit
    keys its cadence on - into a per-session one.
    """
    side = _build(kind, monkeypatch, clock)
    _run_session(side, clock, FIRST_SESSION_S, TARGET_HZ)
    first = _total(side)
    assert first == round(FIRST_SESSION_S * TARGET_HZ), "premise: the first session was counted"

    _run_session(side, clock, SECOND_SESSION_S, TARGET_HZ)
    assert _total(side) == first + round(SECOND_SESSION_S * TARGET_HZ), (
        f"a {kind}'s reported total dropped the first session's frames"
    )


@pytest.mark.parametrize("kind", SIDES)
def test_restarting_a_stopped_side_is_a_supported_flow(
    kind: str, clock: _MonotonicClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premise: the cells above measure a flow the classes implement on purpose.

    The publisher's ``start()`` clears the stop event its ``stop()`` set, and the
    receiver's re-declares a subscription - neither is reachable unless a stopped
    side is meant to be started again.
    """
    side = _build(kind, monkeypatch, clock)
    side.start()
    side.stop()
    assert side.stats["running"] is False
    side.start()
    try:
        assert side.stats["running"] is True, f"a stopped {kind} could not be restarted"
        if kind == "publisher":
            assert not side._stop_event.is_set(), "the restart left the stop event set"
        else:
            assert len(side.mesh.subscribed) == 2, "the restart did not re-declare the subscription"
    finally:
        side.stop()
