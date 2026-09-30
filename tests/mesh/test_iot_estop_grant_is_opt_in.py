"""Regression tests: a robot certificate may not originate a fleet stop unless designated.

``provision_robot`` attached the ``strands-robot`` policy by default, whose
``AllowSafetyEstop`` statement lets the certificate publish on
``strands/safety/estop`` and ``strands/safety/resume``. Obeying a stop needs
only subscribe and receive, which both policy variants carry, so every robot
in a fleet held an authority only a safety operator should have: one extracted
robot certificate could halt the whole fleet, arm an MQTT Will that halts it
when the connection drops, or clear a lockout a human engaged. The Fleet
Provisioning template hardcoded the same grant-bearing policy for every
zero-touch device with no way to opt out.

The default is now the posture the module always called the common case: a
robot obeys stops and cannot originate one. The estop grant is an explicit
opt-in on the Python path (``allow_estop_publish=True``), the CLI
(``--estop-publish``) and the Fleet Provisioning template, which now attaches
``strands-robot-no-estop``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import provision
from strands_robots.mesh.iot.provision import (
    ROBOT_NO_ESTOP_POLICY_NAME,
    ROBOT_POLICY_NAME,
    provision_robot,
)

_SAFETY_TOPICS = {
    "arn:aws:iot:*:*:topic/strands/safety/estop",
    "arn:aws:iot:*:*:topic/strands/safety/resume",
}


def _publish_resources(doc: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for st in doc["Statement"]:
        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        if st.get("Effect") == "Allow" and any(a in ("iot:Publish", "iot:RetainPublish") for a in actions):
            res = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            out.update(res)
    return out


@pytest.fixture
def iot() -> MagicMock:
    c = MagicMock()
    c.meta.region_name = "us-west-2"
    c.exceptions = MagicMock()
    c.exceptions.ResourceNotFoundException = type("NotFound", (Exception,), {})
    c.describe_thing.side_effect = c.exceptions.ResourceNotFoundException()
    c.get_policy.side_effect = c.exceptions.ResourceNotFoundException()
    c.create_thing.return_value = {"thingArn": "arn:aws:iot:us-west-2:123456789012:thing/r"}
    c.create_policy.return_value = {"policyArn": "arn:aws:iot:us-west-2:123456789012:policy/p"}
    c.create_certificate_from_csr.return_value = {
        "certificateArn": "arn:aws:iot:us-west-2:123456789012:cert/abc123def456",
        "certificateId": "abc123def456",
        "certificatePem": "-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n",
    }
    c.describe_endpoint.return_value = {"endpointAddress": "fake-ats.iot.us-west-2.amazonaws.com"}
    c.list_thing_principals.return_value = {"principals": []}
    c.list_attached_policies.return_value = {"policies": []}
    return c


def _provision(iot: MagicMock, tmp_path: Any, **kw: Any) -> Any:
    with (
        patch("strands_robots.mesh.iot.provision._require_boto3", lambda: MagicMock(client=lambda *a, **k: iot)),
        patch("strands_robots.mesh.iot.provision._ensure_ca", lambda ca_path: None),
    ):
        return provision_robot("robot-01", cert_dir=tmp_path, **kw)


class TestTheDefaultIsObeyOnly:
    def test_an_unqualified_provision_robot_attaches_the_no_estop_policy(self, iot: MagicMock, tmp_path: Any) -> None:
        result = _provision(iot, tmp_path)

        assert result.policy_name == ROBOT_NO_ESTOP_POLICY_NAME
        assert iot.attach_policy.call_args.kwargs["policyName"] == ROBOT_NO_ESTOP_POLICY_NAME
        created = iot.create_policy.call_args.kwargs
        doc = json.loads(created["policyDocument"])
        assert not (_publish_resources(doc) & _SAFETY_TOPICS), "default robot cert may publish a fleet stop"

    def test_a_designated_safety_authority_opts_in(self, iot: MagicMock, tmp_path: Any) -> None:
        result = _provision(iot, tmp_path, allow_estop_publish=True)

        assert result.policy_name == ROBOT_POLICY_NAME
        doc = json.loads(iot.create_policy.call_args.kwargs["policyDocument"])
        assert _SAFETY_TOPICS <= _publish_resources(doc)

    def test_the_default_robot_still_receives_both_safety_topics(self) -> None:
        doc = provision._robot_policy_doc(allow_estop_publish=False)
        receive = {
            r
            for st in doc["Statement"]
            if "iot:Receive" in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])
            for r in (st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]])
        }
        assert any("safety/estop" in r for r in receive), receive
        assert any("safety/resume" in r for r in receive), receive


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    from strands_robots.mesh.iot import cli

    seen: dict[str, Any] = {}

    def _fake(name: str, **kw: Any) -> Any:
        seen.update(kw)
        result = MagicMock()
        result.thing_name = name
        result.cert_id = "abc123def456"
        result.subject_cn = name
        result.policy_name = "p"
        result.stale_certificates = []
        result.export_lines.return_value = []
        return result

    monkeypatch.setattr(provision, "provision_robot", _fake)
    assert cli.main(argv) == 0
    return seen


class TestTheCliFlagIsAnOptIn:
    def test_provision_robot_without_a_flag_is_obey_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _run_cli(monkeypatch, ["provision-robot", "robot-01"])["allow_estop_publish"] is False

    def test_estop_publish_flag_opts_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _run_cli(monkeypatch, ["provision-robot", "robot-01", "--estop-publish"])["allow_estop_publish"] is True


class TestFleetProvisioningTemplate:
    def _create_template(self) -> MagicMock:
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _ensure_provisioning_template

        class _NotFound(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.describe_provisioning_template.side_effect = _NotFound()
        iot.get_policy.side_effect = _NotFound()
        iot.create_policy.return_value = {"policyArn": "arn:aws:iot:us-west-2:1:policy/p"}
        iot.create_provisioning_template.return_value = {"templateArn": "arn:iot:template"}
        with patch("strands_robots.mesh.iot.bootstrap._ensure_provisioning_role", return_value="arn:iam:role"):
            _ensure_provisioning_template(iot, MagicMock(), BootstrappedAccount(region="us-west-2", account_id="1"))
        return iot

    def test_zero_touch_devices_get_the_no_estop_policy(self) -> None:
        iot = self._create_template()
        body = json.loads(iot.create_provisioning_template.call_args.kwargs["templateBody"])

        assert body["Resources"]["policy"]["Properties"]["PolicyName"] == ROBOT_NO_ESTOP_POLICY_NAME

    def test_bootstrap_creates_the_policy_the_template_names(self) -> None:
        """A template naming a policy nobody created fails every registration; the robot path created it before."""
        iot = self._create_template()

        assert iot.create_policy.call_args.kwargs["policyName"] == ROBOT_NO_ESTOP_POLICY_NAME
        doc = json.loads(iot.create_policy.call_args.kwargs["policyDocument"])
        assert not (_publish_resources(doc) & _SAFETY_TOPICS)
