"""A consent request is built from a refusal's code, never from its prose.

Every refusal here is produced the way the dashboard will meet it: raised by
``validate_command`` / ``validate_input_frame`` / ``_check_trust_remote_code``
and caught, or arriving as the wire mapping ``{"code", "subject", "message"}``,
or as the verdict dict ``agent_motion_allowed`` returns. The environment
variable each card offers is asserted against ``refusal_codes.REFUSAL_GRANTS``,
so this file cannot pass while the two drift apart.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from strands_robots import refusal_codes
from strands_robots.dashboard import agent_motion
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
from strands_robots.mesh.security import ValidationError, validate_command, validate_input_frame
from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

GRANTS = refusal_codes.REFUSAL_GRANTS


@pytest.fixture(autouse=True)
def _no_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse by default: an inherited grant would silence the refusal under test."""
    for name in (*GRANTS.values(), "STRANDS_MESH_INPUT_SLEW_ABS", agent_motion.MOTION_ENV):
        monkeypatch.delenv(name, raising=False)


def _command(**overrides: Any) -> dict[str, Any]:
    cmd: dict[str, Any] = {
        "action": "execute",
        "instruction": "pick up the cube",
        "sender": "operator",
        "policy_provider": "mock",
    }
    cmd.update(overrides)
    return cmd


def _refused(fn: Any) -> ValidationError | UntrustedRemoteCodeError:
    try:
        result = fn()
    except (ValidationError, UntrustedRemoteCodeError) as refusal:
        return refusal
    raise AssertionError(f"expected a refusal, got {result!r}")


def _classified(fn: Any) -> ConsentRequest:
    request = classify_refusal(_refused(fn))
    assert request is not None
    return request


# --- each coded refusal reaches the right card ----------------------------------------------


def test_an_untrusted_provider_asks_for_trust_remote_code() -> None:
    req = _classified(lambda: _check_trust_remote_code("lerobot_local"))
    assert req.kind == "trust_remote_code"
    assert req.subject == "lerobot_local"
    assert req.env_var == GRANTS[refusal_codes.TRUST_REMOTE_CODE_REQUIRED]
    assert "lerobot_local" in req.title
    assert env_patch(req, {}) == {req.env_var: "1"}
    assert env_patch(req, {req.env_var: "1"}) == {}
    assert revoke_patch(req, {req.env_var: "1"}) == {req.env_var: ""}


def test_a_repo_outside_the_allowlist_asks_for_exactly_that_repo() -> None:
    req = _classified(lambda: validate_command(_command(pretrained_name_or_path="sketchy-org/model")))
    assert req.kind == "hf_repo_allow"
    assert req.subject == "sketchy-org/model"
    assert req.env_var == GRANTS[refusal_codes.HF_REPO_NOT_ALLOWED]
    assert req.scope == "hf_repo_allow:sketchy-org/model"
    assert env_patch(req, {}) == {req.env_var: "sketchy-org/model"}
    assert env_patch(req, {req.env_var: "nvidia"}) == {req.env_var: "nvidia,sketchy-org/model"}
    assert env_patch(req, {req.env_var: "sketchy-org"}) == {}  # the org already covers it
    assert revoke_patch(req, {req.env_var: "nvidia,sketchy-org/model"}) == {req.env_var: "nvidia"}
    assert revoke_patch(req, {req.env_var: "sketchy-org"}) == {}  # never narrows what it did not write


@pytest.mark.parametrize("field", ["policy_type", "policy_provider"])
def test_type_and_provider_share_one_card_and_one_variable(field: str) -> None:
    req = _classified(lambda: validate_command(_command(**{field: "mystery_net"})))
    assert req.kind == "policy_type_allow"
    assert req.subject == "mystery_net"
    assert req.env_var == GRANTS[refusal_codes.POLICY_TYPE_NOT_ALLOWED]
    assert env_patch(req, {}) == {req.env_var: "mystery_net"}
    assert env_patch(req, {req.env_var: "mock,mystery_net"}) == {}
    assert revoke_patch(req, {req.env_var: "mock,mystery_net"}) == {req.env_var: "mock"}


def test_a_host_outside_the_allowlist_asks_for_that_host() -> None:
    req = _classified(lambda: validate_command(_command(policy_host="10.9.9.9")))
    assert req.kind == "policy_host_allow"
    assert req.subject == "10.9.9.9"
    assert req.env_var == GRANTS[refusal_codes.POLICY_HOST_NOT_ALLOWED]
    assert env_patch(req, {}) == {req.env_var: "10.9.9.9"}
    assert revoke_patch(req, {req.env_var: "10.9.9.9,10.0.0.7"}) == {req.env_var: "10.0.0.7"}


def test_a_server_address_is_reduced_to_the_host_entry_the_allowlist_takes() -> None:
    """The subject is the whole address; the grant is the host inside it."""
    req = _classified(lambda: validate_command(_command(server_address="tcp://10.9.9.9:5555")))
    assert req.kind == "policy_host_allow"
    assert req.subject == "10.9.9.9"
    assert env_patch(req, {}) == {GRANTS[refusal_codes.POLICY_HOST_NOT_ALLOWED]: "10.9.9.9"}


def test_a_teleop_frame_past_the_envelope_asks_for_the_degrees_preset() -> None:
    req = _classified(lambda: validate_input_frame({"shoulder.pos": 900.0}))
    assert req.kind == "teleop_degree_units"
    assert req.subject == "shoulder.pos"
    assert req.env_var == GRANTS[refusal_codes.TELEOP_VALUE_OUT_OF_RANGE]
    assert "shoulder.pos" in req.risk
    patch = env_patch(req, {})
    assert patch == {req.env_var: "400", "STRANDS_MESH_INPUT_SLEW_ABS": "800"}
    assert env_patch(req, patch) == {}
    assert revoke_patch(req, patch) == {req.env_var: "", "STRANDS_MESH_INPUT_SLEW_ABS": ""}


def test_every_code_in_the_contract_has_a_card() -> None:
    for code in refusal_codes.REFUSAL_CODES:
        req = classify_refusal(ValidationError("", code=code, subject="x"))
        assert req is not None, code
        assert req.kind in KINDS
        assert req.env_var == GRANTS[code]


# --- recognition is by code, not by wording ---------------------------------------------------


def test_the_same_code_with_different_prose_gives_the_same_request() -> None:
    a = classify_refusal(
        ValidationError("policy_host='10.9.9.9' not in allowlist.", code="POLICY_HOST_NOT_ALLOWED", subject="10.9.9.9")
    )
    b = classify_refusal(ValidationError("nope", code="POLICY_HOST_NOT_ALLOWED", subject="10.9.9.9"))
    assert a is not None and b is not None
    assert dataclasses.replace(a, message="") == dataclasses.replace(b, message="")
    assert a.message == "policy_host='10.9.9.9' not in allowlist."


def test_prose_that_names_a_grant_but_carries_no_code_is_not_continuable() -> None:
    text = "pretrained_name_or_path='evil/repo' not in allowlist. Set STRANDS_MESH_HF_REPO_ALLOW to add it."
    assert classify_refusal(text) is None
    assert classify_refusal(ValidationError(text)) is None
    assert classify_refusal({"message": text}) is None


def test_a_rejection_the_operator_cannot_grant_has_no_card() -> None:
    refusal = _refused(lambda: validate_command(_command(instruction="x" * 99999)))
    assert refusal.code is None
    assert classify_refusal(refusal) is None


def test_an_unknown_code_falls_through_to_the_message() -> None:
    assert classify_refusal(ValidationError("later", code="SOMETHING_NEW", subject="x")) is None
    assert classify_refusal({"code": "SOMETHING_NEW", "subject": "x", "message": "later"}) is None
    assert classify_refusal({"code": 7, "subject": "x"}) is None


@pytest.mark.parametrize("source", [None, "", 42, ValueError("boom"), {}, {"status": "error"}])
def test_things_that_are_not_refusals_are_not_consent(source: object) -> None:
    assert classify_refusal(source) is None


# --- the wire shape --------------------------------------------------------------------------


def test_the_wire_mapping_classifies_like_the_exception() -> None:
    exc = _refused(lambda: validate_command(_command(pretrained_name_or_path="sketchy-org/model")))
    wire = {"code": exc.code, "subject": exc.subject, "message": str(exc)}
    assert classify_refusal(wire) == classify_refusal(exc)


def test_the_message_is_carried_for_display_and_bounded() -> None:
    req = classify_refusal({"code": "HF_REPO_NOT_ALLOWED", "subject": "a/b", "message": "x" * 5000})
    assert req is not None
    assert len(req.message) == 2000
    assert classify_refusal({"code": "HF_REPO_NOT_ALLOWED", "subject": "a/b", "message": 3}) is not None


# --- the dashboard's own refusal -------------------------------------------------------------


def test_the_agent_motion_verdict_is_accepted_as_the_dict_it_is() -> None:
    verdict = agent_motion.agent_motion_allowed("task", peer=None, target="arm-left", env={})
    assert verdict["allowed"] is False
    req = classify_refusal(verdict, subject="arm-left")
    assert req is not None
    assert req.kind == "agent_physical_motion"
    assert req.subject == "arm-left"
    assert req.env_var == agent_motion.MOTION_ENV
    assert req.message == verdict["reason"]
    assert env_patch(req, {}) == {agent_motion.MOTION_ENV: "1"}
    assert revoke_patch(req, {agent_motion.MOTION_ENV: "on"}) == {agent_motion.MOTION_ENV: ""}


def test_a_verdict_that_allows_or_is_granted_is_not_a_request() -> None:
    assert classify_refusal(agent_motion.agent_motion_allowed("status", peer=None, env={})) is None
    granted = agent_motion.agent_motion_allowed("task", peer=None, env={agent_motion.MOTION_ENV: "1"})
    assert granted["allowed"] is True
    assert classify_refusal(granted) is None


def test_the_grant_and_the_gate_agree_on_what_counts_as_granted() -> None:
    req = build_request("agent_physical_motion", "arm-left")
    assert req is not None
    env = dict(env_patch(req, {}))
    assert agent_motion.agent_motion_allowed("task", peer=None, env=env)["allowed"] is True
    env.update(revoke_patch(req, env))
    assert agent_motion.agent_motion_allowed("task", peer=None, env=env)["allowed"] is False


# --- a hostile subject is shown as nothing and grants nothing --------------------------------


@pytest.mark.parametrize(
    ("code", "subject"),
    [
        ("HF_REPO_NOT_ALLOWED", "evil,nvidia"),
        ("POLICY_TYPE_NOT_ALLOWED", "mock,evil"),
        ("POLICY_HOST_NOT_ALLOWED", "10.9.9.9,0.0.0.0/0"),
        ("TRUST_REMOTE_CODE_REQUIRED", "x y"),
    ],
)
def test_a_subject_that_is_not_one_entry_asks_but_grants_nothing(code: str, subject: str) -> None:
    req = classify_refusal(ValidationError("", code=code, subject=subject))
    assert req is not None
    assert req.subject is None
    assert subject not in req.title
    assert env_patch(req, {}) == {} or req.kind == "trust_remote_code"
    assert not req.grantable or req.kind == "trust_remote_code"


# --- the grants the machine holds, and the payload that offers them --------------------------


def test_granted_state_reads_every_variable_from_the_contract() -> None:
    env = {
        GRANTS["TRUST_REMOTE_CODE_REQUIRED"]: "1",
        GRANTS["HF_REPO_NOT_ALLOWED"]: "nvidia, my-org/model",
        GRANTS["POLICY_TYPE_NOT_ALLOWED"]: "mock",
        GRANTS["POLICY_HOST_NOT_ALLOWED"]: "10.9.9.9",
        GRANTS["TELEOP_VALUE_OUT_OF_RANGE"]: "400",
        "STRANDS_MESH_INPUT_SLEW_ABS": "800",
        agent_motion.MOTION_ENV: "on",
    }
    state = granted_state(env)
    assert state["kinds"] == list(KINDS)
    assert state["trust_remote_code"] is True
    assert state["hf_repo_allow"] == ["nvidia", "my-org/model"]
    assert state["policy_type_allow"] == ["mock"]
    assert state["policy_host_allow"] == ["10.9.9.9"]
    assert state["agent_physical_motion"] is True
    assert state["teleop_degree_units"] == {
        "granted": True,
        "value_abs": "400",
        "slew_abs": "800",
        "is_degree_preset": True,
    }


def test_a_hand_tuned_teleop_bound_is_granted_but_not_called_the_preset() -> None:
    env = {GRANTS["TELEOP_VALUE_OUT_OF_RANGE"]: "1000"}
    teleop = granted_state(env)["teleop_degree_units"]
    assert teleop == {"granted": True, "value_abs": "1000", "slew_abs": None, "is_degree_preset": False}
    assert granted_state({})["teleop_degree_units"]["granted"] is False


def test_attach_consent_offers_the_first_continuable_source_only() -> None:
    plain = ValidationError("schema failure")
    coded = _refused(lambda: validate_command(_command(policy_host="10.9.9.9")))
    payload = attach_consent({"status": "error"}, plain, coded, "some text")
    assert payload["needs_consent"]["kind"] == "policy_host_allow"
    assert payload["needs_consent"]["env_var"] == GRANTS["POLICY_HOST_NOT_ALLOWED"]
    assert payload["needs_consent"]["grantable"] is True
    assert "needs_consent" not in attach_consent({"status": "error"}, plain, "text", None)


def test_as_dict_is_json_shaped() -> None:
    req = build_request("hf_repo_allow", "my-org/model", "why")
    assert req is not None
    d = req.as_dict()
    assert set(d) == {"kind", "scope", "title", "risk", "env_var", "subject", "message", "grants", "grantable"}
    assert d["grants"] == ["load my-org/model"]
    assert d["env_var"] == GRANTS["HF_REPO_NOT_ALLOWED"]


def test_build_request_refuses_a_kind_it_does_not_know() -> None:
    assert build_request("sudo") is None
    assert env_patch(ConsentRequest("sudo", "sudo", "t", "r", "X"), {}) == {}
    assert revoke_patch(ConsentRequest("sudo", "sudo", "t", "r", "X"), {"X": "1"}) == {}
