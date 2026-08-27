"""U18: a safety refusal must be answerable, and the answer must be minimal."""

from __future__ import annotations

import pytest

from strands_robots.dashboard.consent import (
    ConsentRequest,
    attach_consent,
    build_request,
    classify_refusal,
    env_patch,
)

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
