"""Q118 - the SDK's slew refusal has two branches and the dashboard parsed one.

The test calls ``security.input_frame_slew_violation`` to PRODUCE the messages instead of quoting
strings I typed: a reworded SDK refusal must fail here, loudly, rather than go quiet in a browser.
Found by probing this parser with the producer's real output (the Q116/Q117 method), after the same
method found a leaked token and a lost cache header earlier today.
"""

from __future__ import annotations

import math

from strands_robots.dashboard.teleop_health import diagnose_receiver, envelope_refusal
from strands_robots.mesh.security import input_frame_slew_violation

PREV = {"shoulder_pan.pos": (10.0, 0.0)}


def _normal() -> str:
    """A speed that exceeds the bound: 30 units in 0.24s against 8 units/s."""
    msg = input_frame_slew_violation({"shoulder_pan.pos": 40.0}, PREV, 0.24, 0.0, max_slew=8.0)
    assert msg, "the SDK no longer refuses this frame - the fixture, not the parser, is wrong"
    return msg


def _no_elapsed_time() -> str:
    """The extreme case: two frames sharing a timestamp, so no speed can be computed."""
    msg = input_frame_slew_violation(
        {"shoulder_pan.pos": 40.0}, {"shoulder_pan.pos": (10.0, 5.0)}, 5.0, 0.0, max_slew=8.0
    )
    assert msg, "the SDK no longer refuses a zero-interval frame - fixture is wrong"
    return msg


def test_the_two_branches_really_are_worded_differently() -> None:
    """The premise of the bug, pinned: one has a number after the colon, the other a word."""
    assert "out of range: moved " in _no_elapsed_time()
    assert "out of range: moved " not in _normal()


def test_a_speed_violation_still_parses() -> None:
    got = envelope_refusal([_normal()])
    assert got == {"kind": "slew", "joint": "shoulder_pan.pos", "value": 125.0, "bound": 8.0}


def test_a_zero_interval_violation_is_no_longer_invisible() -> None:
    """This returned None before Q118 - the most extreme violation was the unexplained one."""
    got = envelope_refusal([_no_elapsed_time()])
    assert got is not None
    assert got["kind"] == "slew"
    assert got["joint"] == "shoulder_pan.pos"
    assert got["instant"] is True
    assert math.isinf(got["value"])  # no interval: any bound is exceeded
    assert got["bound"] == 8.0
    assert got["delta"] == 30.0  # what the frame actually asked for


def test_the_newest_line_still_wins_across_the_two_kinds() -> None:
    tail = [_no_elapsed_time(), _normal()]
    assert envelope_refusal(tail)["value"] == 125.0  # type: ignore[index]
    assert envelope_refusal(list(reversed(tail)))["instant"] is True  # type: ignore[index]


def test_the_sentence_does_not_blame_the_envelope_for_a_clock_problem() -> None:
    """`inf units/s` is arithmetic, not an explanation - and widening the bound cannot help."""
    out = diagnose_receiver(
        {"running": True, "frames_received": 0, "rejected": 0, "slew_rejected": 4},
        log_tail=[_no_elapsed_time()],
    )
    assert out["state"] == "refusing"
    detail = out["detail"]
    assert "SAME timestamp" in detail
    assert "30 units" in detail
    assert "inf" not in detail
    assert "radians" not in detail  # the degrees/radians story belongs to VALUE refusals
    assert "widening the bound cannot help" in detail
