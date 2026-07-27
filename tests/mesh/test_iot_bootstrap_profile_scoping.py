"""bootstrap_account / teardown_account must honour ``profile`` end to end.

Two verified defects:

* ``profile`` was passed to ``boto3.Session`` only for the STS identity check;
  every resource-creating client (iot/iam/lambda/dynamodb/logs) fell back to the
  default credential chain. Resources could land in a different account than the
  one the adjacent ``account_id_expected`` guard verified.
* The ``dry_run`` preview listed resources that do not match what the create
  path builds (wrong table / log-group / rule / role names, plus a phantom Thing
  Type and Policy that bootstrap never creates).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.iot import bootstrap as boot_mod
from strands_robots.mesh.iot.bootstrap import (
    ESTOP_LAMBDA_NAME,
    ESTOP_LAMBDA_ROLE,
    IOT_ACTION_ROLE,
    LOG_GROUP_NAME,
    PROVISIONING_HOOK_LAMBDA_NAME,
    PROVISIONING_HOOK_ROLE,
    PROVISIONING_ROLE,
    PROVISIONING_TEMPLATE,
    RULE_ESTOP_FANOUT,
    RULE_SAFETY_TO_DYNAMODB,
    SAFETY_TABLE_NAME,
)

PROFILE = "fleet-ops-prod"


class _FakeSession:
    """Records the profile every client is created under."""

    def __init__(self, profile_name: str | None = None) -> None:
        self.profile_name = profile_name
        self.created: list[str] = []

    def client(self, svc: str, **kwargs: Any) -> Any:
        self.created.append(svc)
        c = MagicMock(name=f"client:{svc}")
        if svc == "sts":
            c.get_caller_identity.return_value = {"Account": "111122223333"}
            c.meta.region_name = "us-west-2"
        return c


class _FakeBoto3:
    """Stand-in for the boto3 module. ``.client`` here is the DEFAULT chain -
    it must never be used once a profile is requested."""

    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []
        self.default_client_calls: list[str] = []

    def Session(self, profile_name: str | None = None) -> _FakeSession:  # noqa: N802 - mirrors boto3 API
        s = _FakeSession(profile_name=profile_name)
        self.sessions.append(s)
        return s

    def client(self, svc: str, **kwargs: Any) -> Any:  # pragma: no cover - asserted absent
        self.default_client_calls.append(svc)
        return MagicMock()


def _stub_all_ensures(monkeypatch) -> None:
    for name, ret in [
        ("_ensure_log_group", "arn:logs"),
        ("_ensure_safety_table", "arn:ddb"),
        ("_ensure_lambda_role", "arn:lam-role"),
        ("_ensure_estop_lambda", "arn:estop"),
        ("_ensure_safety_to_dynamodb_rule", "arn:rule-safety"),
        ("_ensure_estop_rule", "arn:rule-estop"),
        ("_grant_iot_invoke_lambda", None),
        ("_ensure_provisioning_hook_role", "arn:hook-role"),
        ("_ensure_provisioning_hook_lambda", "arn:hook"),
        ("_ensure_provisioning_template", "arn:template"),
        ("_grant_iot_invoke_provisioning_hook", None),
    ]:
        monkeypatch.setattr(boot_mod, name, lambda *a, _r=ret, **k: _r)


class TestProfileThreadedToEveryClient:
    def test_all_resource_clients_use_profile_session(self, monkeypatch):
        fake = _FakeBoto3()
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)
        _stub_all_ensures(monkeypatch)

        boot_mod.bootstrap_account(confirm=True, dry_run=False, profile=PROFILE)

        # Exactly one session, created under the requested profile.
        assert len(fake.sessions) == 1
        session = fake.sessions[0]
        assert session.profile_name == PROFILE
        # STS + every mutating client came from that profile session.
        assert set(session.created) == {"sts", "iot", "iam", "lambda", "dynamodb", "logs"}
        # The default credential chain (boto3.client) was NEVER touched.
        assert fake.default_client_calls == []

    def test_teardown_threads_profile_to_every_client(self, monkeypatch):
        fake = _FakeBoto3()
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)

        boot_mod.teardown_account(region="us-west-2", profile=PROFILE)

        assert len(fake.sessions) == 1
        session = fake.sessions[0]
        assert session.profile_name == PROFILE
        assert set(session.created) == {"iot", "iam", "lambda", "dynamodb", "logs"}
        assert fake.default_client_calls == []


class TestDryRunPreviewMatchesCreatedResources:
    def test_preview_lists_real_names_only(self, monkeypatch, capsys):
        sts = MagicMock()
        sts.get_caller_identity.return_value = {"Account": "111122223333"}
        sts.meta.region_name = "us-west-2"
        boto3_mock = MagicMock()
        boto3_mock.client.return_value = sts
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        boot_mod.bootstrap_account(region="us-west-2")  # dry_run defaults True
        err = capsys.readouterr().err

        # Every real resource name the create path uses appears verbatim.
        for name in (
            LOG_GROUP_NAME,
            SAFETY_TABLE_NAME,
            ESTOP_LAMBDA_ROLE,
            ESTOP_LAMBDA_NAME,
            IOT_ACTION_ROLE,
            RULE_SAFETY_TO_DYNAMODB,
            RULE_ESTOP_FANOUT,
            PROVISIONING_HOOK_ROLE,
            PROVISIONING_HOOK_LAMBDA_NAME,
            PROVISIONING_ROLE,
            PROVISIONING_TEMPLATE,
        ):
            assert name in err, f"preview missing real resource {name!r}"

        # Phantom / wrong names from the old preview must be gone.
        for phantom in (
            "DynamoDB Table: strands-mesh-fleet",  # old wrong table name (vs -safety-events)
            "/strands/mesh",  # wrong log group
            "strands_mesh_audit",  # wrong / nonexistent rule
            "IoT Thing Type",  # never created by bootstrap
            "strands-mesh-robot-policy",  # created by provision_robot, not here
        ):
            assert phantom not in err, f"preview still advertises phantom {phantom!r}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
