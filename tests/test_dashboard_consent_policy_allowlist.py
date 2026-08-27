"""Q119 - the SDK refuses five things by allowlist; only one of them could be approved.

Measured by calling ``security.validate_command`` for each refusal rather than by quoting strings:
``pretrained_name_or_path`` was classified, while ``policy_provider``, ``policy_type`` and
``policy_host`` returned None from classify_refusal - a dead end on the screen, which is the exact
complaint consent.py exists to answer. This file covers the provider/type pair (one env var, as the
SDK's own sentence says); policy_host is a separate variable and a separate step.

The refusal text comes from the SDK on every run, so a reworded refusal fails HERE instead of
quietly removing the only way through in a browser.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.consent import (
    KINDS,
    build_request,
    classify_refusal,
    env_patch,
    granted_state,
    revoke_patch,
)
from strands_robots.mesh.security import ValidationError, validate_command

ENV = "STRANDS_MESH_POLICY_TYPE_ALLOW"
BASE = {"action": "start", "instruction": "pick up the cube"}


def _sdk_refusal(**extra: object) -> str:
    """The SDK's real message for a disallowed policy field."""
    try:
        validate_command({**BASE, **extra})
    except ValidationError as exc:
        return str(exc)
    pytest.fail(f"the SDK no longer refuses {extra} - this fixture, not the parser, is out of date")


@pytest.mark.parametrize("field,value", [("policy_provider", "lerobot_weird"), ("policy_type", "smolvla_x")])
def test_both_fields_that_share_the_variable_are_continuable(field: str, value: str) -> None:
    message = _sdk_refusal(**{field: value})
    assert ENV in message, "the SDK stopped naming the variable this classification keys off"
    req = classify_refusal(message)
    assert req is not None, f"{field} refusal is a dead end again"
    assert req.kind == "policy_type_allow"
    assert req.subject == value
    assert req.env_var == ENV
    assert value in req.risk  # the operator is told WHAT they are approving
    assert env_patch(req, {}) == {ENV: value}


def test_the_grant_is_narrow_and_additive() -> None:
    req = classify_refusal(_sdk_refusal(policy_type="smolvla_x"))
    assert env_patch(req, {ENV: "mock,lerobot"}) == {ENV: "mock,lerobot,smolvla_x"}
    # already granted: an empty patch tells the caller approving would change nothing
    assert env_patch(req, {ENV: "mock,smolvla_x"}) == {}


def test_revoking_leaves_the_rest_of_the_allowlist_alone() -> None:
    req = classify_refusal(_sdk_refusal(policy_type="smolvla_x"))
    assert revoke_patch(req, {ENV: "mock,smolvla_x,lerobot"}) == {ENV: "mock,lerobot"}
    assert revoke_patch(req, {ENV: "mock"}) == {}  # nothing to take back


def test_the_grant_is_visible_on_the_permissions_screen() -> None:
    """A grant with no surface is a grant nobody can take back - granted_state's own lesson."""
    state = granted_state({ENV: "mock, smolvla_x"})
    assert state["policy_type_allow"] == ["mock", "smolvla_x"]
    assert "policy_type_allow" in state["kinds"]
    assert "policy_type_allow" in KINDS


def test_a_hostile_subject_is_dropped_not_echoed() -> None:
    """`subject` is what the approval endpoint rebuilds from, so it cannot carry junk."""
    req = build_request("policy_type_allow", "lerobot;rm -rf /", "refused")
    assert req is not None and req.subject is None
    assert env_patch(req, {}) == {}  # ask, but grant nothing automatically
    assert "the requested policy" in req.title


def test_an_unrelated_refusal_is_still_not_continuable() -> None:
    assert classify_refusal("Port is in use!") is None
    assert classify_refusal("") is None
