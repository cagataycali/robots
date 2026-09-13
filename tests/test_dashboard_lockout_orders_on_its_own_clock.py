"""A lockout verdict must be ordered on the clock that stamped the observations.

``Lockout.since`` is whatever the sending peer put in the safety envelope's
``t`` - its own clock. ``first_seen`` and ``proof_at`` are stamped by the
dashboard. Ordering the second pair against the first mixes two clocks, and the
mesh admits an e-stop up to ``STRANDS_MESH_RESUME_FRESHNESS_S`` (60 s by
default) behind the receiver's clock before any cross-peer skew, so the mismatch
is the normal case rather than an edge one. Both misreadings run the unsafe way:
a peer that was present downgrades to "may never have received it", and a
command accepted before the stop was even known reads as proof the peer is
clear - the healthy-looking badge on a locked arm this module exists to prevent.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.safety_state import Lockout, apply_event, resolve_peer

#: An e-stop the coordinator stamped at 100.0 that reached this dashboard at
#: 200.0 - the shape of tests/test_dashboard_safety_state.py's incident fixture.
REPORTED = 100.0
ARRIVED = 200.0


@pytest.fixture
def skewed_estop() -> Lockout:
    return apply_event(
        Lockout(),
        kind="estop",
        data={"source": "evac-coordinator", "t": REPORTED},
        now=ARRIVED,
    )


def test_the_sender_clock_and_this_dashboards_clock_are_both_recorded(skewed_estop: Lockout) -> None:
    # Keeping only one of the two is what forced the comparisons onto the wrong one.
    assert skewed_estop.since == REPORTED, "the reported instant is what an operator reads"
    assert skewed_estop.arrived == ARRIVED, "the local instant is what observations sort against"
    fields = skewed_estop.as_fields()
    assert fields["since"] == REPORTED and fields["arrived"] == ARRIVED, (
        "a log that hides the skew makes a wrong verdict undiagnosable"
    )


@pytest.mark.parametrize("first_seen", [REPORTED + 1, ARRIVED - 1, ARRIVED])
def test_a_peer_seen_before_the_stop_arrived_stays_locked(skewed_estop: Lockout, first_seen: float) -> None:
    # It was on the mesh when the e-stop was published, so it received it.
    # Reading the sender's 100.0 made every such peer look freshly spawned.
    verdict = resolve_peer(skewed_estop, first_seen=first_seen)
    assert verdict.state == "locked"
    assert "appeared after" not in verdict.reason


def test_a_peer_that_really_appeared_after_the_stop_is_still_softened(skewed_estop: Lockout) -> None:
    # The softening is right when the peer is genuinely newer than what we know.
    verdict = resolve_peer(skewed_estop, first_seen=ARRIVED + 1)
    assert verdict.state == "unknown"
    assert "appeared after the fleet e-stop" in verdict.reason


@pytest.mark.parametrize("proof_at", [REPORTED + 1, ARRIVED - 1, ARRIVED])
def test_a_command_accepted_before_the_stop_was_known_is_not_proof(skewed_estop: Lockout, proof_at: float) -> None:
    # A command accepted while the dashboard still believed the fleet free says
    # nothing about a lockout it had not yet heard of. Counting it painted a
    # locked peer green.
    verdict = resolve_peer(skewed_estop, first_seen=10.0, proof_at=proof_at)
    assert verdict.state == "locked", "proof older than the stop must not clear it"


def test_a_command_accepted_after_the_stop_was_known_still_proves_clear(skewed_estop: Lockout) -> None:
    # The proof path must keep working, or a resumed fleet can never go green.
    assert resolve_peer(skewed_estop, first_seen=10.0, proof_at=ARRIVED + 1).state == "clear"


def test_a_resume_records_when_this_dashboard_heard_it_too(skewed_estop: Lockout) -> None:
    resumed = apply_event(skewed_estop, kind="resume", data={"t": 300.0}, now=400.0)
    assert resumed.since == 300.0 and resumed.arrived == 400.0
    # Proof between the two clocks must not be read as post-resume evidence.
    assert resolve_peer(resumed, first_seen=10.0, proof_at=350.0).state == "unknown"
    assert resolve_peer(resumed, first_seen=10.0, proof_at=450.0).state == "clear"


def test_a_lockout_with_no_local_stamp_falls_back_to_the_only_instant_it_has() -> None:
    # Assembled by hand rather than folded from an event: there is no better clock.
    hand_built = Lockout(state="locked", since=100.0, reason="locked")
    assert resolve_peer(hand_built, first_seen=150.0).state == "unknown"
    assert resolve_peer(hand_built, first_seen=50.0).state == "locked"
