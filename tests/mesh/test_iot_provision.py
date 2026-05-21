"""Unit tests for the AWS IoT provisioning module.

These tests use mocked boto3 clients — no real AWS calls are made.
A full end-to-end provision + teardown happens in
``tests_integ/test_iot_transport.py`` and in the manually-validated
``/tmp/test_magic.py`` smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot.provision import (
    _OPERATOR_POLICY_DOC,
    _ROBOT_POLICY_DOC,
    OPERATOR_POLICY_NAME,
    ROBOT_POLICY_NAME,
    ProvisionedThing,
    _ensure_policy,
    _ensure_thing,
    provision_operator,
    provision_robot,
)

# Test fixtures


@pytest.fixture
def tmp_cert_dir(tmp_path):
    """Isolated cert dir so we don't write to ~/.strands_robots."""
    d = tmp_path / "iot"
    d.mkdir()
    return d


@pytest.fixture
def fake_iot_client():
    """A boto3 IoT client mock with sensible defaults for happy-path provisioning."""
    iot = MagicMock()
    iot.meta.region_name = "us-west-2"

    # ResourceNotFoundException — exposed at iot.exceptions
    class _NotFound(Exception):
        pass

    iot.exceptions = MagicMock()
    iot.exceptions.ResourceNotFoundException = _NotFound

    # describe_thing: not found by default → CreateThing path
    iot.describe_thing.side_effect = _NotFound("not found")
    iot.create_thing.return_value = {
        "thingName": "test-thing",
        "thingArn": "arn:aws:iot:us-west-2:123456789012:thing/test-thing",
        "thingId": "abc-123",
    }

    # get_policy: not found → CreatePolicy path
    iot.get_policy.side_effect = _NotFound("not found")
    iot.create_policy.return_value = {
        "policyName": "strands-robot",
        "policyArn": "arn:aws:iot:us-west-2:123456789012:policy/strands-robot",
        "policyDocument": "{}",
        "policyVersionId": "1",
    }

    # create_keys_and_certificate
    iot.create_keys_and_certificate.return_value = {
        "certificateArn": "arn:aws:iot:us-west-2:123456789012:cert/abc123def456",
        "certificateId": "abc123def456",
        "certificatePem": "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n",
        "keyPair": {
            "PrivateKey": "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----\n",
            "PublicKey": "-----BEGIN PUBLIC KEY-----\nFAKE\n-----END PUBLIC KEY-----\n",
        },
    }

    iot.describe_endpoint.return_value = {
        "endpointAddress": "fake-ats.iot.us-west-2.amazonaws.com",
    }

    return iot


# Policy documents — schema sanity checks


class TestPolicyDocuments:
    """Both policy docs must be valid IoT policy JSON with the substitution
    variable in the right places."""

    def test_robot_policy_has_substitution(self):
        doc_str = json.dumps(_ROBOT_POLICY_DOC)
        assert "${iot:Connection.Thing.ThingName}" in doc_str
        # Verify the right Sids
        sids = [s["Sid"] for s in _ROBOT_POLICY_DOC["Statement"]]
        assert "AllowConnect" in sids
        assert "AllowOwnTopics" in sids
        assert "AllowResponseToAnyOperator" in sids
        assert "AllowSafetyEstop" in sids

    def test_robot_policy_has_retain_publish(self):
        """iot:RetainPublish must be granted alongside iot:Publish for retained
        topics. Discovered as a footgun in the spike (PUBACK 135 otherwise)."""
        # Find the AllowOwnTopics statement
        own = next(s for s in _ROBOT_POLICY_DOC["Statement"] if s["Sid"] == "AllowOwnTopics")
        assert "iot:RetainPublish" in own["Action"]
        assert "iot:Publish" in own["Action"]

    def test_operator_policy_has_substitution(self):
        doc_str = json.dumps(_OPERATOR_POLICY_DOC)
        assert "${iot:Connection.Thing.ThingName}" in doc_str

    def test_operator_can_publish_to_fleet(self):
        pub = next(s for s in _OPERATOR_POLICY_DOC["Statement"] if s["Sid"] == "OperatorPublishToFleet")
        # Must allow operator to publish to ANY robot's /cmd
        assert any("strands/*/cmd" in r for r in pub["Resource"])
        assert any("strands/broadcast" in r for r in pub["Resource"])

    def test_operator_response_topics_use_substitution(self):
        """OperatorReceiveResponses must use the variable so each operator
        can only see its OWN responses."""
        resp = next(s for s in _OPERATOR_POLICY_DOC["Statement"] if s["Sid"] == "OperatorReceiveResponses")
        for r in resp["Resource"]:
            assert "${iot:Connection.Thing.ThingName}" in r


class TestEnsureThing:
    """``_ensure_thing`` is idempotent."""

    def test_creates_when_missing(self, fake_iot_client):
        """First call creates the thing."""
        arn = _ensure_thing(fake_iot_client, "new-thing", None)
        fake_iot_client.create_thing.assert_called_once()
        assert "thing/test-thing" in arn

    def test_skips_when_present(self, fake_iot_client):
        """Existing thing is reused — no CreateThing call."""
        fake_iot_client.describe_thing.side_effect = None  # found
        fake_iot_client.describe_thing.return_value = {
            "thingArn": "arn:aws:iot:us-west-2:123456789012:thing/existing",
        }
        arn = _ensure_thing(fake_iot_client, "existing", None)
        fake_iot_client.create_thing.assert_not_called()
        assert "thing/existing" in arn

    def test_attributes_passed_through(self, fake_iot_client):
        _ensure_thing(fake_iot_client, "t", {"robot_type": "so100"})
        call_kwargs = fake_iot_client.create_thing.call_args.kwargs
        assert call_kwargs["attributePayload"]["attributes"]["robot_type"] == "so100"


class TestEnsurePolicy:
    """``_ensure_policy`` is idempotent."""

    def test_creates_when_missing(self, fake_iot_client):
        arn = _ensure_policy(fake_iot_client, "strands-robot", _ROBOT_POLICY_DOC)
        fake_iot_client.create_policy.assert_called_once()
        assert "policy/strands-robot" in arn

    def test_skips_when_present(self, fake_iot_client):
        fake_iot_client.get_policy.side_effect = None
        fake_iot_client.get_policy.return_value = {
            "policyArn": "arn:aws:iot:us-west-2:123456789012:policy/strands-robot",
            "defaultVersionId": "3",
        }
        arn = _ensure_policy(fake_iot_client, "strands-robot", _ROBOT_POLICY_DOC)
        fake_iot_client.create_policy.assert_not_called()
        assert "policy/strands-robot" in arn


class TestProvisionRobot:
    """End-to-end provisioning with all AWS calls mocked."""

    def test_writes_certs_with_correct_permissions(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """provision_robot must write the cert + key with mode 0o600."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        # Skip the urlopen call for the CA — write a fake one.
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        result = provision_robot(
            "test-robot-01",
            cert_dir=tmp_cert_dir,
        )

        assert isinstance(result, ProvisionedThing)
        assert result.thing_name == "test-robot-01"
        assert result.policy_name == ROBOT_POLICY_NAME
        assert result.endpoint == "fake-ats.iot.us-west-2.amazonaws.com"
        assert result.cert_path.exists()
        assert result.key_path.exists()
        # Mode 0o600 — owner R/W only
        assert oct(result.cert_path.stat().st_mode)[-3:] == "600"
        assert oct(result.key_path.stat().st_mode)[-3:] == "600"

    def test_attaches_policy_to_cert_and_thing(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        provision_robot("r", cert_dir=tmp_cert_dir)

        # attach_policy was called with the robot policy
        assert fake_iot_client.attach_policy.called
        attach_kwargs = fake_iot_client.attach_policy.call_args.kwargs
        assert attach_kwargs["policyName"] == ROBOT_POLICY_NAME

        # attach_thing_principal was called
        assert fake_iot_client.attach_thing_principal.called

    def test_env_vars_helper(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        result = provision_robot("e2", cert_dir=tmp_cert_dir)
        env = result.env_vars()
        assert env["STRANDS_IOT_THING_NAME"] == "e2"
        assert env["STRANDS_MESH_BACKEND"] == "iot"
        assert env["STRANDS_IOT_ENDPOINT"] == "fake-ats.iot.us-west-2.amazonaws.com"
        # export_lines is the eval-friendly form
        lines = result.export_lines()
        assert any("STRANDS_MESH_BACKEND=iot" in line for line in lines)

    def test_injects_strands_mesh_role_attribute(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """provision_robot auto-injects strands-mesh-role=robot attribute for ACL."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        provision_robot("my-robot", cert_dir=tmp_cert_dir)

        # create_thing should have been called with the role attribute
        call_kwargs = fake_iot_client.create_thing.call_args.kwargs
        assert call_kwargs["attributePayload"]["attributes"]["strands-mesh-role"] == "robot"

    def test_preserves_user_attributes(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """User-supplied attributes are preserved alongside the injected role."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        provision_robot("my-robot", cert_dir=tmp_cert_dir, attributes={"hw": "so100"})

        call_kwargs = fake_iot_client.create_thing.call_args.kwargs
        attrs = call_kwargs["attributePayload"]["attributes"]
        assert attrs["strands-mesh-role"] == "robot"
        assert attrs["hw"] == "so100"

    def test_user_can_override_role_attribute(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """If user explicitly sets strands-mesh-role, their value is kept."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        provision_robot("my-robot", cert_dir=tmp_cert_dir, attributes={"strands-mesh-role": "custom"})

        call_kwargs = fake_iot_client.create_thing.call_args.kwargs
        attrs = call_kwargs["attributePayload"]["attributes"]
        assert attrs["strands-mesh-role"] == "custom"


class TestProvisionOperator:
    """Operator provisioning uses the operator policy, not the robot policy."""

    def test_uses_operator_policy(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        result = provision_operator("ops-1", cert_dir=tmp_cert_dir)

        assert result.policy_name == OPERATOR_POLICY_NAME
        attach_kwargs = fake_iot_client.attach_policy.call_args.kwargs
        assert attach_kwargs["policyName"] == OPERATOR_POLICY_NAME


class TestRequireBoto3:
    """Helpful error when boto3 is missing."""

    def test_raises_clear_import_error(self):
        from strands_robots.mesh.iot import provision as prov_mod

        # Force the import to fail
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                prov_mod._require_boto3()


class TestProvisionedThingDataclass:
    """Smoke tests on the dataclass."""

    def test_env_vars_keys(self):
        p = ProvisionedThing(
            thing_name="t",
            thing_arn="arn",
            cert_arn="arn",
            cert_id="id",
            cert_path=Path("/tmp/t.cert.pem"),
            key_path=Path("/tmp/t.private.key"),
            ca_path=Path("/tmp/AmazonRootCA1.pem"),
            endpoint="x.iot",
            policy_name="strands-robot",
            region="us-west-2",
        )
        env = p.env_vars()
        assert set(env.keys()) == {
            "STRANDS_IOT_THING_NAME",
            "STRANDS_IOT_ENDPOINT",
            "STRANDS_IOT_CERT_DIR",
            "STRANDS_MESH_BACKEND",
        }


class TestCleanupStaleCerts:
    """Re-running provision_robot must not accumulate certs.

    Regression coverage for the security-relevant bug found in cycle 9 of
    the deep-test sweep: AWS IoT CreateKeysAndCertificate always returns a
    new cert (private keys aren't recoverable post-issuance), so without
    explicit cleanup a Thing accumulates ACTIVE certs across re-runs —
    each one a potential impersonation credential.
    """

    def test_cleanup_runs_before_new_cert_issuance(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """provision_robot must call _cleanup_stale_certs *before* creating
        the new cert."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        # Pretend the Thing already has an old cert attached.
        old_cert_arn = "arn:aws:iot:us-west-2:123:cert/old-cert-id-aaaaa"
        fake_iot_client.list_thing_principals.return_value = {"principals": [old_cert_arn]}
        fake_iot_client.list_attached_policies.return_value = {"policies": [{"policyName": "strands-robot"}]}

        provision_robot("test-thing", cert_dir=tmp_cert_dir)

        # The old cert must have been detached + deleted.
        fake_iot_client.detach_thing_principal.assert_called_once_with(thingName="test-thing", principal=old_cert_arn)
        fake_iot_client.detach_policy.assert_called_with(policyName="strands-robot", target=old_cert_arn)
        fake_iot_client.update_certificate.assert_called_once()
        fake_iot_client.delete_certificate.assert_called_once_with(certificateId="old-cert-id-aaaaa", forceDelete=True)
        # Then the new cert is created.
        fake_iot_client.create_keys_and_certificate.assert_called_once()
        # And attached.
        fake_iot_client.attach_thing_principal.assert_called()

    def test_cleanup_swallows_cert_delete_failures(self, fake_iot_client, tmp_cert_dir, monkeypatch):
        """If the old cert can't be deleted (e.g. revoked elsewhere), the
        new cert MUST still be issued. Cleanup is best-effort."""
        monkeypatch.setattr(
            "strands_robots.mesh.iot.provision._require_boto3",
            lambda: MagicMock(client=lambda *a, **kw: fake_iot_client),
        )
        (tmp_cert_dir / "AmazonRootCA1.pem").write_text("fake-ca")

        fake_iot_client.list_thing_principals.return_value = {
            "principals": ["arn:aws:iot:us-west-2:123:cert/cant-delete"]
        }
        fake_iot_client.list_attached_policies.return_value = {"policies": []}
        fake_iot_client.delete_certificate.side_effect = RuntimeError("cannot delete")

        # Must NOT raise — proceeds to create the new cert.
        result = provision_robot("test-thing", cert_dir=tmp_cert_dir)
        assert result.thing_name == "test-thing"
        fake_iot_client.create_keys_and_certificate.assert_called_once()

    def test_cleanup_handles_missing_thing(self):
        """When list_thing_principals raises NotFound, _cleanup_stale_certs
        returns 0 cleanly (no detach/delete attempted)."""
        from strands_robots.mesh.iot.provision import _cleanup_stale_certs

        iot = MagicMock()

        class _NotFound(Exception):
            pass

        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.list_thing_principals.side_effect = _NotFound("missing")

        n = _cleanup_stale_certs(iot, "missing-thing")
        assert n == 0
        iot.detach_thing_principal.assert_not_called()
        iot.delete_certificate.assert_not_called()
