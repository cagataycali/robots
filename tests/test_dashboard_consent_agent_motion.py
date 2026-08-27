"""Q80 follow-up: the agent-motion grant is a first-class consent, not a hidden env var.

Q80 put the gate in agent_motion.py; this pins the other half of the promise the dashboard makes about
every guard -- that a refusal is recognisable, grantable in one place, VISIBLE afterwards, and revocable.
granted_state exists because a grant with no surface is a grant nobody can take back (see its docstring).
"""

from __future__ import annotations

from strands_robots.dashboard.agent_motion import MOTION_ENV, agent_motion_allowed
from strands_robots.dashboard.consent import (
    KINDS,
    attach_consent,
    build_request,
    classify_refusal,
    env_patch,
    granted_state,
    revoke_patch,
)

ARM = {"presence": {"hw": "so_follower"}}


def test_the_real_refusal_text_is_recognised_end_to_end():
    """Not a hand-written string: the exact text agent_motion.py produces."""
    refusal = agent_motion_allowed("task", peer=ARM, target="so101-arm-1", env={})["reason"]
    req = classify_refusal(refusal)
    assert req is not None, "the gate's own refusal must be continuable"
    assert req.kind == "agent_physical_motion"
    assert req.subject == "so101-arm-1"
    assert req.env_var == MOTION_ENV
    d = req.as_dict()
    # The dialog has to say what is being widened and how far it reaches.
    assert "real robots" in d["title"]
    assert "ANY real robot" in d["risk"]
    assert "everyone stop" in d["risk"], "the reassurance belongs in the dialog too"


def test_the_grant_is_machine_wide_and_says_so():
    """Scope is not per-peer: the gate reads one env var, so promising 'just this arm' would lie."""
    a = build_request("agent_physical_motion", "so101-arm-1", "")
    b = build_request("agent_physical_motion", "so101-arm-2", "")
    assert a.scope == b.scope == "agent_physical_motion"


def test_a_hostile_subject_is_dropped_not_echoed():
    req = build_request("agent_physical_motion", "<img src=x onerror=alert(1)>", "")
    assert req.subject is None
    assert "a real robot" in req.title or "a real robot" in req.risk


def test_grant_then_revoke_round_trips_and_is_idempotent():
    req = build_request("agent_physical_motion", "so101-arm-1", "")
    assert env_patch(req, {}) == {MOTION_ENV: "1"}
    # Already granted: an empty patch is the signal that approving changes nothing.
    for held in ("1", "true", "YES", "on"):
        assert env_patch(req, {MOTION_ENV: held}) == {}
    # Revocation CLEARS rather than writing "0", so a stale 1 in a shell profile cannot win a restart.
    assert revoke_patch(req, {MOTION_ENV: "1"}) == {MOTION_ENV: ""}
    assert revoke_patch(req, {}) == {}, "nothing to take back"


def test_the_permissions_screen_can_see_it():
    assert "agent_physical_motion" in KINDS
    assert granted_state({})["agent_physical_motion"] is False
    assert granted_state({MOTION_ENV: "1"})["agent_physical_motion"] is True
    # A hand-set value is IN FORCE and must be shown, not normalised away.
    assert granted_state({MOTION_ENV: "on"})["agent_physical_motion"] is True
    assert granted_state({MOTION_ENV: "0"})["agent_physical_motion"] is False


def test_the_verdict_and_the_grant_agree_on_what_counts_as_granted():
    """Two modules read the same variable; a mismatch would mean the screen says 'granted' while the
    gate still refuses (or worse, the reverse)."""
    for value in ("1", "true", "TRUE", "yes", "on", "", "0", "no", "off", "maybe", " "):
        gate = agent_motion_allowed("task", peer=ARM, env={MOTION_ENV: value})["allowed"]
        screen = granted_state({MOTION_ENV: value})["agent_physical_motion"]
        assert gate is screen, value


def test_an_error_payload_carries_the_consent_offer():
    refusal = agent_motion_allowed("task", peer=ARM, target="so101-arm-2", env={})["reason"]
    payload = attach_consent({"error": refusal}, refusal)
    assert payload["needs_consent"]["kind"] == "agent_physical_motion"
    assert payload["needs_consent"]["subject"] == "so101-arm-2"


def test_the_other_kinds_still_classify():
    """A new branch in classify_refusal must not shadow the existing three."""
    assert classify_refusal("set STRANDS_TRUST_REMOTE_CODE=1").kind == "trust_remote_code"
    assert (
        classify_refusal("pretrained_name_or_path='org/x' not in allowlist. Set STRANDS_MESH_HF_REPO_ALLOW").kind
        == "hf_repo_allow"
    )
    assert classify_refusal("input frame value for wrist_roll is out of range").kind == "teleop_degree_units"
    assert classify_refusal("just a sentence") is None
