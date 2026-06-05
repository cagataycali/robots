"""Tests for the Fleet Provisioning PreProvisioningHook (pentest F-19 / B-13).

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
    iot.exceptions.ResourceNotFoundException = type(
        "RNF", (Exception,), {}
    )
    iot.exceptions.InvalidRequestException = type("IRE", (Exception,), {})
    # describe -> not found so it proceeds to create
    iot.describe_provisioning_template.side_effect = (
        iot.exceptions.ResourceNotFoundException()
    )
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
    iot.describe_provisioning_template.side_effect = (
        iot.exceptions.ResourceNotFoundException()
    )
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
