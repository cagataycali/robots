"""Consent surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_consent_agent_motion.py, test_dashboard_consent_policy_host.py, test_dashboard_consent_state.py, test_dashboard_consent.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import consent
from strands_robots.dashboard.agent_motion import MOTION_ENV, agent_motion_allowed
from strands_robots.dashboard.consent import (
    KINDS,
    ConsentRequest,
    attach_consent,
    build_request,
    classify_refusal,
    env_patch,
    granted_state,
    revoke_patch,
)
from strands_robots.mesh import security as sec

# ============================================================================
# from tests/test_dashboard_consent_agent_motion.py
# Q80 follow-up: the agent-motion grant is a first-class consent, not a hidden env var.
# ============================================================================

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


# ============================================================================
# from tests/test_dashboard_consent_policy_host.py
# Q119 part two - the policy-host refusals, closing the allowlist family at 5 of 5.
# ============================================================================

ENV = "STRANDS_MESH_POLICY_HOST_ALLOW"
BASE = {"action": "start", "instruction": "pick up the cube", "policy_provider": "mock"}


@pytest.fixture(autouse=True)
def _clean_allowlist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV, raising=False)
    sec._clear_security_caches_for_tests()
    yield
    sec._clear_security_caches_for_tests()


def _refusal(**extra: object) -> tuple[dict, str]:
    cmd = {**BASE, **extra}
    try:
        sec.validate_command(cmd)
    except sec.ValidationError as exc:
        return cmd, str(exc)
    pytest.fail(f"the SDK no longer refuses {extra} - the fixture is out of date, not the parser")


@pytest.mark.parametrize(
    "field,value,expected_entry",
    [
        ("policy_host", "gpu.lan", "gpu.lan"),
        # Literal match: the port MUST stay, or the approval changes nothing.
        ("policy_host", "gpu.lan:8000", "gpu.lan:8000"),
        # server_address strips scheme/path/port before checking, so the entry is the bare host.
        ("server_address", "http://gpu.lan:8000", "gpu.lan"),
        ("server_address", "tcp://192.168.1.151:9000", "192.168.1.151"),
        ("server_address", "http://[fe80::1]:9000", "fe80::1"),
    ],
)
def test_the_grant_actually_unblocks_the_command(
    field: str, value: str, expected_entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd, message = _refusal(**{field: value})
    assert ENV in message
    req = classify_refusal(message)
    assert req is not None, f"{field}={value} is a dead end again"
    assert req.kind == "policy_host_allow"
    assert req.subject == expected_entry
    patch = env_patch(req, {})
    assert patch == {ENV: expected_entry}

    monkeypatch.setenv(ENV, patch[ENV])
    sec._clear_security_caches_for_tests()
    sec.validate_command(cmd)  # raises if the approval did not actually let it through


def test_the_grant_is_additive_and_revocation_is_narrow() -> None:
    req = classify_refusal(_refusal(policy_host="gpu.lan")[1])
    assert env_patch(req, {ENV: "10.0.0.0/24"}) == {ENV: "10.0.0.0/24,gpu.lan"}  # type: ignore[arg-type]
    assert env_patch(req, {ENV: "gpu.lan"}) == {}  # type: ignore[arg-type]  # already granted: nothing would change
    assert revoke_patch(req, {ENV: "10.0.0.0/24,gpu.lan"}) == {ENV: "10.0.0.0/24"}  # type: ignore[arg-type]
    # A hand-written CIDR may still cover the host; do not narrow a range we did not write.
    assert revoke_patch(req, {ENV: "10.0.0.0/24"}) == {}  # type: ignore[arg-type]


def test_the_risk_names_what_the_host_is_trusted_with() -> None:
    req = classify_refusal(_refusal(server_address="http://gpu.lan:8000")[1])
    assert "gpu.lan" in req.risk  # type: ignore[union-attr]
    assert "camera frames" in req.risk and "drive real hardware" in req.risk  # type: ignore[union-attr]
    assert "matched literally" in req.risk  # type: ignore[union-attr]  # DNS trust is part of the decision


def test_a_hostile_or_unreadable_subject_grants_nothing() -> None:
    for junk in ("gpu.lan; rm -rf /", "gpu lan", "$(whoami)", "", None):
        req = build_request("policy_host_allow", junk, "refused")
        assert req is not None and req.subject is None, junk
        assert env_patch(req, {}) == {}
        assert "that address" in req.title


def test_the_grant_is_visible_and_defaults_are_not_listed_as_grants() -> None:
    state = granted_state({ENV: "gpu.lan, 10.0.0.0/24"})
    assert state["policy_host_allow"] == ["gpu.lan", "10.0.0.0/24"]
    assert "policy_host_allow" in KINDS and "policy_host_allow" in state["kinds"]
    # Loopback is the SDK's own default and nobody approved it - it must not appear as a grant.
    assert granted_state({})["policy_host_allow"] == []


def test_the_whole_allowlist_family_is_now_continuable() -> None:
    """5 of 5: the point of Q119. Each field's refusal must produce SOME consent request."""
    fields = [
        {"pretrained_name_or_path": "HashtagRobotics/smolvla-tic-tac-toe-games-1-5-80k"},
        {"policy_provider": "lerobot_weird"},
        {"policy_type": "smolvla_x"},
        {"policy_host": "gpu.lan"},
        {"server_address": "http://gpu.lan:8000"},
    ]
    for extra in fields:
        cmd = {**BASE, **extra}
        try:
            sec.validate_command(cmd)
        except sec.ValidationError as exc:
            assert classify_refusal(str(exc)) is not None, f"{extra} has no way through"


# ============================================================================
# from tests/test_dashboard_consent_state.py
# ============================================================================


def test_every_kind_is_reported():
    """Was pinned to the three kinds of the day; a fourth (agent_physical_motion, Q80) then made the
    list wrong rather than making the omission visible. Generic now: whatever KINDS says must have a
    key in the payload, so a kind added later cannot reach the permissions screen unreportable."""
    s = consent.granted_state({})
    assert set(s["kinds"]) == set(consent.KINDS)
    for kind in consent.KINDS:
        assert kind in s, f"{kind} is grantable but the permissions screen cannot see it"
    assert s["trust_remote_code"] is False and s["hf_repo_allow"] == []
    # the bug: this key did not exist, so the permissions screen could not show or revoke it
    assert s["teleop_degree_units"]["granted"] is False
    assert s["agent_physical_motion"] is False


def test_degree_preset_is_recognised_as_such():
    granted = consent.env_patch(consent.build_request("teleop_degree_units", "so101-arm-2"), {})
    s = consent.granted_state(granted)
    t = s["teleop_degree_units"]
    assert t["granted"] is True and t["is_degree_preset"] is True
    assert t["value_abs"] == granted["STRANDS_MESH_INPUT_VALUE_ABS"]


def test_a_hand_tuned_bound_is_granted_but_not_called_the_preset():
    s = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "9999", "STRANDS_MESH_INPUT_SLEW_ABS": "9999"})
    t = s["teleop_degree_units"]
    assert t["granted"] is True and t["is_degree_preset"] is False and t["value_abs"] == "9999"


def test_half_a_pair_is_still_in_force():
    # reach widened, speed bound untouched: the widened half already applies, so hiding it lies
    t = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "180"})["teleop_degree_units"]
    assert t["granted"] is True and t["slew_abs"] is None and t["is_degree_preset"] is False


def test_blank_and_whitespace_are_not_grants():
    t = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "  ", "STRANDS_MESH_INPUT_SLEW_ABS": ""})[
        "teleop_degree_units"
    ]
    assert t["granted"] is False
    assert consent.granted_state({"STRANDS_TRUST_REMOTE_CODE": " "})["trust_remote_code"] is False
    assert consent.granted_state({"STRANDS_MESH_HF_REPO_ALLOW": " a , ,b "})["hf_repo_allow"] == ["a", "b"]


# ============================================================================
# from tests/test_dashboard_consent.py
# U18: a safety refusal must be answerable, and the answer must be minimal.
# ============================================================================

TRUST_REFUSAL = (
    # The glyph is written as an escape on purpose: this string has to stay byte-exact
    # (it is what the product actually prints), and the repo forbids the literal in source.
    "\u2717 trust: Policy provider 'lerobot_local' loads HuggingFace models with "
    "trust_remote_code=True, which allows arbitrary code execution from the model "
    "repository. Only load models from organisations you trust.\n\n"
    "To acknowledge this risk and proceed, set the environment variable:\n"
    "    export STRANDS_TRUST_REMOTE_CODE=1\n"
)

HF_REFUSAL = (
    "pretrained_name_or_path='HashtagRobotics/smolvla-tic-tac-toe-games-1-5-80k' not in "
    "allowlist. Set STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix."
)


def test_trust_refusal_names_the_provider_and_the_var():
    req = classify_refusal(TRUST_REFUSAL)
    assert req is not None
    assert req.kind == "trust_remote_code"
    assert req.scope == "trust_remote_code"
    assert req.subject == "lerobot_local"
    assert req.env_var == "STRANDS_TRUST_REMOTE_CODE"
    assert "arbitrary" in req.risk or "executes" in req.risk
    assert "lerobot_local" in req.title


def test_hf_refusal_extracts_the_repo():
    req = classify_refusal(HF_REFUSAL)
    assert req is not None
    assert req.kind == "hf_repo_allow"
    assert req.subject == "HashtagRobotics/smolvla-tic-tac-toe-games-1-5-80k"
    assert req.scope.endswith(req.subject)
    assert req.env_var == "STRANDS_MESH_HF_REPO_ALLOW"


@pytest.mark.parametrize("text", ["", None, 42, "connection refused", "ValueError: bad port"])
def test_unrelated_errors_are_not_consent(text):
    assert classify_refusal(text) is None


def test_trust_patch_is_one_variable():
    req = classify_refusal(TRUST_REFUSAL)
    assert env_patch(req, {}) == {"STRANDS_TRUST_REMOTE_CODE": "1"}


def test_trust_patch_is_empty_when_already_granted():
    req = classify_refusal(TRUST_REFUSAL)
    assert env_patch(req, {"STRANDS_TRUST_REMOTE_CODE": "true"}) == {}


def test_hf_patch_appends_and_keeps_existing_entries():
    req = classify_refusal(HF_REFUSAL)
    patch = env_patch(req, {"STRANDS_MESH_HF_REPO_ALLOW": "nvidia,lerobot"})
    assert patch == {"STRANDS_MESH_HF_REPO_ALLOW": "nvidia,lerobot,HashtagRobotics/smolvla-tic-tac-toe-games-1-5-80k"}


def test_hf_patch_grants_the_repo_not_the_org():
    req = classify_refusal(HF_REFUSAL)
    value = env_patch(req, {})["STRANDS_MESH_HF_REPO_ALLOW"]
    assert value == "HashtagRobotics/smolvla-tic-tac-toe-games-1-5-80k"
    assert value != "HashtagRobotics"


def test_hf_patch_empty_when_org_already_allowed():
    req = classify_refusal(HF_REFUSAL)
    assert env_patch(req, {"STRANDS_MESH_HF_REPO_ALLOW": "HashtagRobotics"}) == {}


def test_hostile_repo_name_asks_but_grants_nothing():
    req = classify_refusal(
        "pretrained_name_or_path='org/repo;rm -rf /' not in allowlist. Set "
        "STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix."
    )
    assert req is not None
    assert req.subject is None
    assert req.scope == "hf_repo_allow"
    assert env_patch(req, {}) == {}


def test_unknown_kind_patches_nothing():
    req = ConsentRequest(kind="teleop", scope="teleop", title="t", risk="r", env_var="X")
    assert env_patch(req, {}) == {}


def test_message_is_carried_but_bounded():
    req = classify_refusal(TRUST_REFUSAL + "x" * 5000)
    assert len(req.message) <= 2000
    assert "STRANDS_TRUST_REMOTE_CODE" in req.message


def test_attach_consent_adds_the_key_only_when_continuable():
    body = attach_consent({"state": "failed", "error": HF_REFUSAL}, "boom", HF_REFUSAL)
    assert body["state"] == "failed"
    assert body["needs_consent"]["kind"] == "hf_repo_allow"

    plain = attach_consent({"error": "port in use"}, "port in use")
    assert "needs_consent" not in plain


def test_as_dict_is_json_shaped():
    payload = classify_refusal(TRUST_REFUSAL).as_dict()
    assert isinstance(payload["grants"], list)
    assert set(payload) == {
        "kind",
        "scope",
        "title",
        "risk",
        "env_var",
        "subject",
        "message",
        "grants",
        # Q120: the client used to decide approvability from a hardcoded kind list, which went
        # stale the moment consent.py grew two more allowlist kinds (an ENABLED Approve button for
        # a host the server could not read). The server owns env_patch, so it ships the answer.
        "grantable",
    }
    assert payload["grantable"] is True
    # And the case the field exists for: an allowlist kind whose subject could not be read safely
    # names its refusal honestly, but says plainly that there is nothing to grant.
    unreadable = build_request("policy_host_allow", "gpu lan; rm -rf /", "refused").as_dict()
    assert unreadable["grantable"] is False
    assert unreadable["subject"] is None
