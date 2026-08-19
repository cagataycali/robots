"""Teleop that says what is actually happening (U3).

Measured on real hardware 2026-08-19: a real SO-101 leader published 176 frames
to a follower that applied NONE of them, and every surface the dashboard had said
success - the receive call returned "started", the receiver reported running:true,
/api/fleet showed both peers healthy. The truth was in the follower's child log:
the mesh's per-frame envelope is 4*pi (RADIANS) and an SO-101 reports degrees.

Then the opposite failure, same session: nothing arrived at all while the leader
published 200+ frames, and the first version of this diagnosis told the operator
"the leader is not publishing" - sending them to the working end of the problem.
Both are pinned here.
"""

from __future__ import annotations

from strands_robots.dashboard.consent import classify_refusal, env_patch, revoke_patch
from strands_robots.dashboard.teleop_health import (
    diagnose_receiver,
    envelope_refusal,
    published_frames,
    teleop_health,
)

# The real line, from the real follower's log.
REAL_REFUSAL = (
    "09:16:22 [mesh] input frame rejected from so101-arm-1: input frame value for "
    "'shoulder_lift.pos' out of range: |-46.417582417582416| > 12.566370614359172"
)


def _receiver(**kw):
    base = {"source": "so101-arm-1", "device": "leader", "running": True,
            "frames_received": 0, "errors": 0, "drops": 0, "rejected": 0,
            "rate_dropped": 0, "slew_rejected": 0, "hz_actual": 0.0}
    base.update(kw)
    return base


# ------------------------------------------------------- the refusal, parsed


def test_the_real_log_line_is_parsed():
    assert envelope_refusal([REAL_REFUSAL]) == {
        "kind": "value", "joint": "shoulder_lift.pos",
        "value": 46.417582417582416, "bound": 12.566370614359172,
    }


def test_the_newest_bound_wins():
    """An operator who already widened the envelope once must see the bound they
    are hitting NOW, not the first one they ever hit."""
    older = REAL_REFUSAL
    newer = ("input frame value for 'wrist_roll.pos' out of range: |170.0| > 400.0")
    got = envelope_refusal([older, newer])
    assert got["bound"] == 400.0 and got["joint"] == "wrist_roll.pos"


def test_a_slew_refusal_is_recognised_too():
    got = envelope_refusal(["input frame slew for 'elbow_flex.pos' out of range: 91.2 > 25.1 units/s"])
    assert got["kind"] == "slew" and got["bound"] == 25.1


def test_unparseable_logs_are_not_a_crash():
    assert envelope_refusal(None) is None
    assert envelope_refusal(["nothing to see"]) is None
    assert envelope_refusal(["input frame value for 'x' out of range: |abc| > def"]) is None


# ------------------------------------------------------------- the verdicts


def test_refusing_every_frame_is_not_reported_as_running():
    v = diagnose_receiver(_receiver(rejected=176), [REAL_REFUSAL])
    assert v["state"] == "refusing"
    assert "every frame" in v["headline"]
    # The sentence must name the unit mistake, not just the number.
    assert "DEGREES" in v["detail"] and "12.566" in v["detail"]
    assert v["refusal"]["joint"] == "shoulder_lift.pos"


def test_refusing_without_a_log_still_says_the_follower_has_not_moved():
    v = diagnose_receiver(_receiver(rejected=3), None)
    assert v["state"] == "refusing"
    assert "has not moved" in v["detail"]
    assert v["refusal"] is None


def test_nothing_arriving_blames_the_leader_only_when_it_is_quiet():
    v = diagnose_receiver(_receiver(), None, source_frames=0)
    assert v["state"] == "silent"
    assert "not publishing" in v["detail"]


def test_nothing_arriving_while_the_leader_publishes_is_a_ROUTE_problem():
    """The bug this test exists for: blaming the leader while it publishes 209
    frames sends the operator to the wrong end of the problem."""
    v = diagnose_receiver(_receiver(), None, source_frames=209)
    assert v["state"] == "unrouted"
    assert "nothing reaches this follower" in v["headline"]
    assert "209" in v["detail"]
    assert "not publishing" not in v["detail"]


def test_no_evidence_about_the_leader_is_not_evidence_of_zero():
    v = diagnose_receiver(_receiver(), None, source_frames=None)
    assert v["state"] == "silent"
    assert "its publisher reports 0 frames" not in (v["detail"] or "")


def test_following_reports_the_rate_and_any_losses():
    v = diagnose_receiver(_receiver(frames_received=500, hz_actual=8.4, rate_dropped=12))
    assert v["state"] == "following"
    assert "8.4Hz" in v["headline"] and "12 dropped" in v["detail"]


def test_a_stopped_receiver_says_so():
    assert diagnose_receiver(_receiver(running=False))["state"] == "stopped"


# --------------------------------------------------------- the whole payload


def _envelope(receivers=None, publishers=None):
    """A status as it crosses the mesh: prose plus a json block."""
    return {"status": "success", "content": [
        {"text": "Teleop status: ..."},
        {"json": {"receivers": receivers or {}, "publishers": publishers or {}}},
    ]}


def test_the_worst_receiver_is_the_one_worth_showing():
    status = _envelope(receivers={
        "a/leader": _receiver(frames_received=100, hz_actual=8.0),
        "b/leader": _receiver(rejected=10),
    })
    h = teleop_health(status, [REAL_REFUSAL])
    assert h["worst"]["peer_key"] == "b/leader"
    assert h["worst"]["state"] == "refusing"


def test_a_publisher_far_under_its_target_is_explained_not_flagged():
    """Measured: 20Hz requested, 6.2Hz achieved - because the state probe and the
    camera publisher share that servo bus. Reading it as a fault invites someone
    to 'fix' it by asking for more."""
    h = teleop_health(_envelope(publishers={
        "leader": {"running": True, "frames": 209, "hz_actual": 6.2, "hz_target": 20.0},
    }))
    p = h["publishers"]["leader"]
    assert p["state"] == "publishing"
    assert "shared" in p["detail"]


def test_a_publisher_at_its_target_needs_no_excuse():
    h = teleop_health(_envelope(publishers={
        "leader": {"running": True, "frames": 100, "hz_actual": 19.0, "hz_target": 20.0},
    }))
    assert h["publishers"]["leader"]["detail"] is None


def test_published_frames_distinguishes_zero_from_unknown():
    status = _envelope(publishers={"leader": {"running": True, "frames": 0}})
    assert published_frames(status, "leader") == 0
    assert published_frames(status, "gamepad") is None
    assert published_frames(None, "leader") is None
    assert published_frames({"nonsense": 1}, "leader") is None


def test_an_unrecognisable_status_decorates_nothing():
    h = teleop_health("not a status")
    assert h == {"receivers": {}, "publishers": {}, "worst": None}


# ----------------------------------------- the refusal is CONTINUABLE, safely


def test_the_real_refusal_becomes_a_consent_request():
    req = classify_refusal(REAL_REFUSAL)
    assert req is not None and req.kind == "teleop_degree_units"
    # The dialog must state what widening a safety bound costs.
    assert "RADIANS" in req.risk and "DEGREES" in req.risk
    assert "longer reach" in req.risk
    # And it must not promise "unlimited".
    assert "still refused" in req.risk


def test_the_grant_widens_both_halves_of_the_envelope():
    req = classify_refusal(REAL_REFUSAL)
    patch = env_patch(req, {})
    assert patch == {"STRANDS_MESH_INPUT_VALUE_ABS": "400", "STRANDS_MESH_INPUT_SLEW_ABS": "800"}
    # Distance and speed are one unit decision; granting only one leaves teleop
    # broken the moment the arm actually moves.
    assert len(req.grants) == 2


def test_approving_twice_changes_nothing():
    req = classify_refusal(REAL_REFUSAL)
    assert env_patch(req, {"STRANDS_MESH_INPUT_VALUE_ABS": "400",
                           "STRANDS_MESH_INPUT_SLEW_ABS": "800"}) == {}


def test_revoking_clears_rather_than_freezing_todays_default():
    """Writing 12.566... back would silently override a future SDK default."""
    req = classify_refusal(REAL_REFUSAL)
    assert revoke_patch(req, {"STRANDS_MESH_INPUT_VALUE_ABS": "400",
                              "STRANDS_MESH_INPUT_SLEW_ABS": "800"}) == {
        "STRANDS_MESH_INPUT_VALUE_ABS": "", "STRANDS_MESH_INPUT_SLEW_ABS": ""}
    assert revoke_patch(req, {}) == {}


def test_a_slew_refusal_asks_the_same_question():
    req = classify_refusal("input frame slew for 'elbow_flex.pos' out of range: 91.2 > 25.1 units/s")
    assert req is not None and req.kind == "teleop_degree_units"


def test_an_ordinary_error_is_not_turned_into_a_consent_prompt():
    assert classify_refusal("connection refused") is None
    assert classify_refusal("input frame value for 'x' is weird") is None
