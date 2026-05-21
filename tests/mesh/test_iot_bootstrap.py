"""Unit tests for the account-wide bootstrap.

Real AWS calls are mocked. The end-to-end "really creates resources" test
is in /tmp/test_bootstrap.py and was run manually against us-west-2 during
implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import bootstrap as boot_mod
from strands_robots.mesh.iot.bootstrap import (
    ESTOP_LAMBDA_NAME,
    LOG_GROUP_NAME,
    PROVISIONING_TEMPLATE,
    RULE_ESTOP_FANOUT,
    RULE_SAFETY_TO_DYNAMODB,
    SAFETY_TABLE_NAME,
    BootstrappedAccount,
    _build_lambda_zip,
)


class TestLambdaZipBuild:
    def test_zip_contains_handler(self):
        zb = _build_lambda_zip()
        # First two bytes are the zip magic (PK)
        assert zb[:2] == b"PK"
        import zipfile
        from io import BytesIO

        with zipfile.ZipFile(BytesIO(zb)) as zf:
            names = zf.namelist()
            assert "lambda_function.py" in names
            src = zf.read("lambda_function.py").decode()
            assert "def lambda_handler" in src
            assert "iot_data.publish" in src
            assert "strands-mesh-role" in src


class TestResourceNames:
    """Lock in the canonical names — changing them is a breaking change."""

    def test_safety_table_name(self):
        assert SAFETY_TABLE_NAME == "strands-mesh-safety-events"

    def test_estop_lambda_name(self):
        assert ESTOP_LAMBDA_NAME == "strands-mesh-estop-fanout"

    def test_rule_names(self):
        assert RULE_SAFETY_TO_DYNAMODB == "strands_safety_to_dynamodb"
        assert RULE_ESTOP_FANOUT == "strands_estop_fanout"

    def test_provisioning_template_name(self):
        assert PROVISIONING_TEMPLATE == "strands-mesh-fleet-provisioning"

    def test_log_group_under_aws_iot(self):
        # Must be under /aws/iot/ for IoT Logs Role to deliver logs natively.
        assert LOG_GROUP_NAME.startswith("/aws/iot/")


class TestBootstrappedAccountDataclass:
    def test_default_lists_empty(self):
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        assert a.created == []
        assert a.skipped == []


class TestBootstrapRequireBoto3:
    def test_helpful_import_error(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                boot_mod._require_boto3()


class TestEnsureSafetyTable:
    def test_skips_when_present(self):
        ddb = MagicMock()
        ddb.describe_table.return_value = {"Table": {"TableArn": "arn:test"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        result = boot_mod._ensure_safety_table(ddb, a)
        assert result == "arn:test"
        ddb.create_table.assert_not_called()
        assert "dynamodb:strands-mesh-safety-events" in a.skipped

    def test_creates_when_missing(self):
        class _NotFound(Exception):
            pass

        ddb = MagicMock()
        ddb.exceptions = MagicMock()
        ddb.exceptions.ResourceNotFoundException = _NotFound
        ddb.describe_table.side_effect = _NotFound("not found")
        ddb.create_table.return_value = {"TableDescription": {"TableArn": "arn:new"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")

        result = boot_mod._ensure_safety_table(ddb, a)

        assert result == "arn:new"
        ddb.create_table.assert_called_once()
        kwargs = ddb.create_table.call_args.kwargs
        # Must request KMS encryption — safety audit is a security-sensitive table.
        assert kwargs["SSESpecification"]["Enabled"] is True
        assert kwargs["SSESpecification"]["SSEType"] == "KMS"
        # Pay-per-request (so test accounts don't get autoscaling surprises).
        assert kwargs["BillingMode"] == "PAY_PER_REQUEST"
        # Tags must include strands-mesh=managed for IaC discoverability.
        tag_keys = {t["Key"] for t in kwargs["Tags"]}
        assert "strands-mesh" in tag_keys
        assert "dynamodb:strands-mesh-safety-events" in a.created


class TestEnsureRuleIdempotence:
    def test_skips_when_get_returns_existing(self):
        iot = MagicMock()
        iot.get_topic_rule.return_value = {"ruleArn": "arn:rule"}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_iot_action_role",
            return_value="arn:iam:role",
        ):
            result = boot_mod._ensure_safety_to_dynamodb_rule(iot, "arn:t", a)
        assert result == "arn:rule"
        iot.create_topic_rule.assert_not_called()

    def test_skips_when_get_raises_unauthorised(self):
        """AWS IoT returns UnauthorizedException when a rule doesn't exist."""

        class _NotFound(Exception):
            pass

        class _Unauthorized(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.exceptions.UnauthorizedException = _Unauthorized
        iot.get_topic_rule.side_effect = _Unauthorized("denied")

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_iot_action_role",
            return_value="arn:iam:role",
        ):
            boot_mod._ensure_safety_to_dynamodb_rule(iot, "arn:t", a)

        # Rule was created (UnauthorizedException treated as not-found).
        iot.create_topic_rule.assert_called_once()


class TestEnsureLogGroup:
    def test_creates_when_missing(self):
        from strands_robots.mesh.iot.bootstrap import LOG_GROUP_NAME, BootstrappedAccount, _ensure_log_group

        logs = MagicMock()
        # First describe returns no matching group
        logs.describe_log_groups.side_effect = [
            {"logGroups": []},  # initial check
            {
                "logGroups": [{"arn": "arn:aws:logs:us-west-2:123:log-group:test", "logGroupName": LOG_GROUP_NAME}]
            },  # post-create check
        ]
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_log_group(logs, a)
        assert arn.startswith("arn:aws:logs")
        logs.create_log_group.assert_called_once()
        logs.put_retention_policy.assert_called_once()
        assert f"logs:{LOG_GROUP_NAME}" in a.created

    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import LOG_GROUP_NAME, BootstrappedAccount, _ensure_log_group

        logs = MagicMock()
        logs.describe_log_groups.return_value = {"logGroups": [{"arn": "arn:logs", "logGroupName": LOG_GROUP_NAME}]}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_log_group(logs, a)
        assert arn == "arn:logs"
        logs.create_log_group.assert_not_called()
        assert f"logs:{LOG_GROUP_NAME}" in a.skipped


class TestEnsureLambdaRole:
    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import ESTOP_LAMBDA_ROLE, BootstrappedAccount, _ensure_lambda_role

        iam = MagicMock()
        iam.get_role.return_value = {"Role": {"Arn": "arn:iam:role"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_lambda_role(iam, a)
        assert arn == "arn:iam:role"
        iam.create_role.assert_not_called()
        assert f"iam:{ESTOP_LAMBDA_ROLE}" in a.skipped


class TestEnsureEstopLambda:
    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _ensure_estop_lambda

        lam = MagicMock()
        lam.get_function.return_value = {"Configuration": {"FunctionArn": "arn:lambda"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_estop_lambda(lam, "arn:role", a)
        assert arn == "arn:lambda"
        lam.create_function.assert_not_called()


class TestGrantIotInvokeLambda:
    def test_grants_permission_when_missing(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _grant_iot_invoke_lambda

        lam = MagicMock()
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        _grant_iot_invoke_lambda(lam, "arn:lambda", a)
        lam.add_permission.assert_called_once()
        kwargs = lam.add_permission.call_args.kwargs
        assert kwargs["Principal"] == "iot.amazonaws.com"
        assert kwargs["Action"] == "lambda:InvokeFunction"

    def test_swallows_existing_permission(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _grant_iot_invoke_lambda

        class _Conflict(Exception):
            pass

        lam = MagicMock()
        lam.exceptions = MagicMock()
        lam.exceptions.ResourceConflictException = _Conflict
        lam.add_permission.side_effect = _Conflict("already granted")
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        # Must not raise
        _grant_iot_invoke_lambda(lam, "arn:lambda", a)


class TestBootstrappedAccountEnv:
    """Sanity checks — names locked in."""

    def test_lambda_zip_size_reasonable(self):
        from strands_robots.mesh.iot.bootstrap import _build_lambda_zip

        zb = _build_lambda_zip()
        # Lambda source is ~2 KB; zipped should be well under 10 KB.
        assert 500 < len(zb) < 10_000
