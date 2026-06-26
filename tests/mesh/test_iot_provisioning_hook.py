"""Tests for the Fleet Provisioning PreProvisioningHook.

Without a PreProvisioningHook, any holder of the shared claim cert can
register an arbitrary Thing. These tests pin the hook's deny-by-default
behaviour and that the template wires it.
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

from strands_robots.mesh.iot import bootstrap as b


def test_hook_source_is_valid_python():
    ast.parse(b._PROVISIONING_HOOK_SOURCE)


def test_hook_zip_builds():
    assert len(b._build_provisioning_hook_zip()) > 0


def test_template_wires_pre_provisioning_hook():
    src = inspect.getsource(b._ensure_provisioning_template)
    assert "preProvisioningHook" in src
    assert "hook_lambda_arn" in src


def test_template_create_includes_hook_when_arn_supplied():
    """create_provisioning_template must receive preProvisioningHook."""
    iot = MagicMock()
    iot.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    iot.exceptions.InvalidRequestException = type("IRE", (Exception,), {})
    # describe -> not found so it proceeds to create
    iot.describe_provisioning_template.side_effect = iot.exceptions.ResourceNotFoundException()
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    # stub the role helper to avoid IAM calls
    import strands_robots.mesh.iot.bootstrap as mod

    orig = mod._ensure_provisioning_role
    mod._ensure_provisioning_role = lambda *a, **k: "arn:aws:iam::123456789012:role/x"
    try:
        b._ensure_provisioning_template(
            iot, acct, hook_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:hook"
        )
    finally:
        mod._ensure_provisioning_role = orig

    kwargs = iot.create_provisioning_template.call_args.kwargs
    assert "preProvisioningHook" in kwargs
    assert kwargs["preProvisioningHook"]["targetArn"].endswith(":function:hook")


def test_template_omits_hook_when_no_arn():
    iot = MagicMock()
    iot.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    iot.exceptions.InvalidRequestException = type("IRE", (Exception,), {})
    iot.describe_provisioning_template.side_effect = iot.exceptions.ResourceNotFoundException()
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    import strands_robots.mesh.iot.bootstrap as mod

    orig = mod._ensure_provisioning_role
    mod._ensure_provisioning_role = lambda *a, **k: "arn:aws:iam::123456789012:role/x"
    try:
        b._ensure_provisioning_template(iot, acct)  # no hook arn
    finally:
        mod._ensure_provisioning_role = orig

    kwargs = iot.create_provisioning_template.call_args.kwargs
    assert "preProvisioningHook" not in kwargs


# --- Behavioural tests of the hook handler itself ------------------------


def _run_handler(event, *, thing_exists=False, serial_allowed=True):
    """Exec the hook source with a controllable fake boto3 and invoke it."""
    fake_boto3 = MagicMock()

    iot_client = MagicMock()
    ssm_client = MagicMock()

    class _RNF(Exception):
        pass

    class _PNF(Exception):
        pass

    iot_client.exceptions.ResourceNotFoundException = _RNF
    ssm_client.exceptions.ParameterNotFound = _PNF

    if thing_exists:
        iot_client.describe_thing.return_value = {"thingName": "x"}
    else:
        iot_client.describe_thing.side_effect = _RNF()

    if not serial_allowed:
        ssm_client.get_parameter.side_effect = _PNF()

    def _client(name, *a, **k):
        return {"iot": iot_client, "ssm": ssm_client}[name]

    fake_boto3.client.side_effect = _client

    # The hook source does `import boto3` at module level, which shadows
    # any exec-global we inject. Patch sys.modules so the import resolves
    # to our fake instead of the real SDK (which would hit AWS).
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        g: dict = {}
        exec(compile(b._PROVISIONING_HOOK_SOURCE, "<hook>", "exec"), g)
        return g["lambda_handler"](event, MagicMock())


def test_hook_allows_valid_allowlisted_serial():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-001", "ThingName": "g1-robot-001"}},
        thing_exists=False,
        serial_allowed=True,
    )
    assert res == {"allowProvisioning": True}


def test_hook_denies_bad_serial():
    res = _run_handler(
        {"parameters": {"SerialNumber": "../../etc", "ThingName": "x"}},
    )
    assert res == {"allowProvisioning": False}


def test_hook_denies_missing_serial():
    res = _run_handler({"parameters": {"ThingName": "x"}})
    assert res == {"allowProvisioning": False}


def test_hook_denies_existing_thing():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-001", "ThingName": "g1-robot-001"}},
        thing_exists=True,
    )
    assert res == {"allowProvisioning": False}


def test_hook_denies_serial_not_in_allowlist():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-999", "ThingName": "g1-robot-999"}},
        thing_exists=False,
        serial_allowed=False,
    )
    assert res == {"allowProvisioning": False}


def test_hook_role_grants_describe_thing_and_ssm_getparameter():
    """The hook role must permit the two reads the hook makes (F-19/B-13).

    Regression for the review blocker: the hook was originally created with
    the E-stop Lambda role, which grants neither iot:DescribeThing nor
    ssm:GetParameter. Those calls would then AccessDenied, get swallowed by
    the deny-on-error envelope, and refuse *every* registration.
    """
    iam = MagicMock()

    class _NoSuchEntity(Exception):
        pass

    iam.exceptions.NoSuchEntityException = _NoSuchEntity
    iam.get_role.side_effect = _NoSuchEntity()
    iam.create_role.return_value = {
        "Role": {"Arn": "arn:aws:iam::123456789012:role/strands-mesh-provisioning-hook-role"}
    }
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    import strands_robots.mesh.iot.bootstrap as mod

    orig_sleep = mod.time.sleep
    mod.time.sleep = lambda *_a, **_k: None
    try:
        arn = b._ensure_provisioning_hook_role(iam, acct)
    finally:
        mod.time.sleep = orig_sleep

    assert arn.endswith(":role/strands-mesh-provisioning-hook-role")

    # The inline policy must grant both actions the hook needs.
    inline = iam.put_role_policy.call_args.kwargs
    import json as _json

    doc = _json.loads(inline["PolicyDocument"])
    actions = {a for stmt in doc["Statement"] for a in stmt["Action"]}
    assert "iot:DescribeThing" in actions
    assert "ssm:GetParameter" in actions
    # SSM read must be scoped to the allowlist namespace, not "*".
    ssm_stmt = next(s for s in doc["Statement"] if "ssm:GetParameter" in s["Action"])
    assert all("provisioning/allow/" in r for r in ssm_stmt["Resource"])
    assert all(r != "*" for r in ssm_stmt["Resource"])


def test_bootstrap_uses_dedicated_hook_role():
    """bootstrap_account must wire the hook to its own role, not the E-stop role."""
    src = inspect.getsource(b.bootstrap_account)
    assert "_ensure_provisioning_hook_role" in src


def test_hook_lambda_stamps_version_description():
    """Create path stamps the version tag so drift can be detected later."""
    lam = MagicMock()

    class _RNF(Exception):
        pass

    lam.exceptions.ResourceNotFoundException = _RNF
    lam.exceptions.InvalidParameterValueException = type("IPV", (Exception,), {})
    lam.get_function.side_effect = _RNF()
    lam.create_function.return_value = {
        "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    }
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)

    desc = lam.create_function.call_args.kwargs["Description"]
    assert f"[v{b._PROVISIONING_HOOK_VERSION}]" in desc


# --- Idempotent reuse + force-update + role-propagation retry ------------
#
# _ensure_provisioning_hook_lambda must (a) reuse an existing function and
# only replace its code under force_update, (b) warn on a stale version tag
# so drift is visible, and (c) survive the IAM role-propagation race that
# makes create_function transiently reject a freshly-created role.


def _hook_lambda_client():
    """A MagicMock Lambda client wired with the exception types the code uses."""
    lam = MagicMock()
    lam.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    lam.exceptions.InvalidParameterValueException = type("IPV", (Exception,), {})
    return lam


def _current_version_function(arn):
    return {
        "Configuration": {
            "FunctionArn": arn,
            "Description": f"strands-mesh hook [v{b._PROVISIONING_HOOK_VERSION}]",
        }
    }


def test_hook_lambda_reuses_existing_without_force_update():
    """Existing, current-version function is skipped (not recreated/updated)."""
    arn = "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    lam = _hook_lambda_client()
    lam.get_function.return_value = _current_version_function(arn)
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    result = b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)

    assert result == arn
    assert f"lambda:{b.PROVISIONING_HOOK_LAMBDA_NAME}" in acct.skipped
    lam.create_function.assert_not_called()
    lam.update_function_code.assert_not_called()


def test_hook_lambda_warns_on_stale_version(caplog):
    """A version-tag mismatch on an existing function logs a drift warning."""
    arn = "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    lam = _hook_lambda_client()
    lam.get_function.return_value = {"Configuration": {"FunctionArn": arn, "Description": "strands-mesh hook [v0]"}}
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    with caplog.at_level("WARNING"):
        result = b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)

    assert result == arn
    assert any("stale version" in r.message for r in caplog.records)


def test_hook_lambda_force_update_replaces_code_and_config():
    """force_update rewrites code + configuration and records an update."""
    arn = "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    lam = _hook_lambda_client()
    lam.get_function.return_value = _current_version_function(arn)
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    result = b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct, force_update=True)

    assert result == arn
    lam.update_function_code.assert_called_once()
    lam.update_function_configuration.assert_called_once()
    assert f"lambda:{b.PROVISIONING_HOOK_LAMBDA_NAME} (updated)" in acct.created
    assert f"lambda:{b.PROVISIONING_HOOK_LAMBDA_NAME}" not in acct.skipped


def test_hook_lambda_retries_until_role_is_assumable(monkeypatch):
    """create_function is retried with backoff while the new role propagates."""
    arn = "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    lam = _hook_lambda_client()
    lam.get_function.side_effect = lam.exceptions.ResourceNotFoundException()
    role_err = lam.exceptions.InvalidParameterValueException(
        "The role defined for the function cannot be assumed by Lambda."
    )
    lam.create_function.side_effect = [role_err, role_err, {"FunctionArn": arn}]
    sleeps: list = []
    monkeypatch.setattr(b.time, "sleep", lambda s: sleeps.append(s))
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    result = b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)

    assert result == arn
    assert lam.create_function.call_count == 3
    assert len(sleeps) == 2  # slept once per transient failure, not after success
    assert f"lambda:{b.PROVISIONING_HOOK_LAMBDA_NAME}" in acct.created


def test_hook_lambda_reraises_non_role_parameter_error(monkeypatch):
    """A non-role InvalidParameterValue is a real error and must not be retried."""
    import pytest

    lam = _hook_lambda_client()
    lam.get_function.side_effect = lam.exceptions.ResourceNotFoundException()
    lam.create_function.side_effect = lam.exceptions.InvalidParameterValueException(
        "Runtime python3.12 is not supported"
    )
    monkeypatch.setattr(b.time, "sleep", lambda _s: None)
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    with pytest.raises(lam.exceptions.InvalidParameterValueException):
        b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)
    lam.create_function.assert_called_once()


def test_hook_lambda_raises_after_retry_exhaustion(monkeypatch):
    """If the role never becomes assumable, a clear RuntimeError is raised."""
    lam = _hook_lambda_client()
    lam.get_function.side_effect = lam.exceptions.ResourceNotFoundException()
    lam.create_function.side_effect = lam.exceptions.InvalidParameterValueException(
        "The role defined for the function cannot be assumed by Lambda."
    )
    monkeypatch.setattr(b.time, "sleep", lambda _s: None)
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    import pytest

    with pytest.raises(RuntimeError, match="Provisioning-hook Lambda create failed after retries"):
        b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)
    assert lam.create_function.call_count == 6


# --- IoT invoke-permission grant (idempotent) ---------------------------


def test_grant_invoke_permission_added_once():
    """The IoT service principal is granted invoke and the grant is recorded."""
    lam = _hook_lambda_client()
    lam.exceptions.ResourceConflictException = type("RCE", (Exception,), {})
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    b._grant_iot_invoke_provisioning_hook(lam, "arn:aws:lambda:us-east-1:123456789012:function:hook", acct)

    kwargs = lam.add_permission.call_args.kwargs
    assert kwargs["Principal"] == "iot.amazonaws.com"
    assert kwargs["Action"] == "lambda:InvokeFunction"
    assert "lambda-permission:provisioning-hook-invoke" in acct.created


def test_grant_invoke_permission_is_idempotent():
    """A pre-existing statement (ResourceConflict) is skipped, not an error."""
    lam = _hook_lambda_client()
    lam.exceptions.ResourceConflictException = type("RCE", (Exception,), {})
    lam.add_permission.side_effect = lam.exceptions.ResourceConflictException()
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    b._grant_iot_invoke_provisioning_hook(lam, "arn:aws:lambda:us-east-1:123456789012:function:hook", acct)

    assert "lambda-permission:provisioning-hook-invoke" in acct.skipped
    assert "lambda-permission:provisioning-hook-invoke" not in acct.created
