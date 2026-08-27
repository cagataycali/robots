"""Q119 part two - the policy-host refusals, closing the allowlist family at 5 of 5.

The test that matters here is a ROUND TRIP: take the SDK's refusal, classify it, apply the patch
consent.py computes, and assert the SDK now ACCEPTS the same command. A grant that parses but does
not unblock is worse than a refusal, because the operator believes they are through - and the
symmetric first draft of this code did exactly that for `policy_host='gpu.lan:8000'` (policy_host is
matched LITERALLY while server_address strips the port). Measurement caught it, so it is pinned here.
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
from strands_robots.mesh import security as sec

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
