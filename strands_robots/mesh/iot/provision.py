"""One-command AWS IoT provisioning for strands-robots.

This module is the magical out-of-box experience: a developer with AWS
credentials runs::

    from strands_robots.mesh.iot import provision_robot
    provision_robot("so100-arm-01")

…and the function:

1. Creates an AWS IoT Thing named ``so100-arm-01``.
2. Generates an X.509 keypair + cert (AWS-issued, ``CreateKeysAndCertificate``).
3. Creates the canonical strands-robot IoT Policy if it doesn't exist (idempotent).
4. Attaches policy → cert → Thing.
5. Writes ``cert.pem`` / ``private.key`` / ``AmazonRootCA1.pem`` to
   ``~/.strands_robots/iot/`` with mode 0o600.
6. Discovers the IoT data endpoint and writes it to ``~/.strands_robots/iot/endpoint``.

After provisioning, the next ``Robot("so100", peer_id="so100-arm-01")`` call
with ``STRANDS_MESH_BACKEND=iot`` joins the AWS IoT mesh transparently.

All operations are idempotent: re-running ``provision_robot("so100-arm-01")``
re-uses the existing Thing and policy if they're there. A new cert is created
each time (you can't list private keys after the fact, so re-running
generates fresh credentials and keeps the file naming stable).

Operator provisioning
---------------------
:func:`provision_operator` is the analogue for fleet operators (Bedrock
agents, ops consoles). The two policies differ — robots can publish to
their own topic prefix and respond to any operator; operators can publish
``cmd`` / ``broadcast`` and observe the whole fleet.

CLI
---
The same logic is exposed as a CLI entry point (registered in
``pyproject.toml``)::

    strands-robots iot provision so100-arm-01
    strands-robots iot provision-operator bedrock-agent-01
    strands-robots iot teardown so100-arm-01    # cleanup
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_AMAZON_ROOT_CA1_URL = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"

DEFAULT_CERT_DIR = Path.home() / ".strands_robots" / "iot"
ROBOT_POLICY_NAME = "strands-robot"
OPERATOR_POLICY_NAME = "strands-operator"


# Provisioning result


@dataclass
class ProvisionedThing:
    """The artefacts of a single :func:`provision_robot` /
    :func:`provision_operator` call.

    Attributes:
        thing_name: The AWS IoT Thing name (== Mesh peer_id).
        thing_arn: ARN of the Thing.
        cert_arn: ARN of the active certificate.
        cert_id: AWS IoT certificate id (last segment of the ARN).
        cert_path: Local path to the cert PEM (mode 0o600).
        key_path: Local path to the private key (mode 0o600).
        ca_path: Local path to the Amazon Root CA1.
        endpoint: The IoT Data ATS endpoint to connect to.
        policy_name: The policy attached (``strands-robot`` or ``strands-operator``).
        region: The AWS region these resources live in.
    """

    thing_name: str
    thing_arn: str
    cert_arn: str
    cert_id: str
    cert_path: Path
    key_path: Path
    ca_path: Path
    endpoint: str
    policy_name: str
    region: str

    def env_vars(self) -> dict[str, str]:
        """Return env vars a process can export to use these artefacts."""
        return {
            "STRANDS_IOT_THING_NAME": self.thing_name,
            "STRANDS_IOT_ENDPOINT": self.endpoint,
            "STRANDS_IOT_CERT_DIR": str(self.cert_path.parent),
            "STRANDS_MESH_BACKEND": "iot",
        }

    def export_lines(self) -> list[str]:
        """Shell-export lines suitable for ``eval $(...)``."""
        return [f"export {k}={v}" for k, v in self.env_vars().items()]


# Policy documents — verified working in the spike

_ROBOT_POLICY_DOC: dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowConnect",
            "Effect": "Allow",
            "Action": "iot:Connect",
            "Resource": "arn:aws:iot:*:*:client/${iot:Connection.Thing.ThingName}",
        },
        {
            "Sid": "AllowOwnTopics",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/*",
            ],
        },
        {
            "Sid": "AllowResponseToAnyOperator",
            "Effect": "Allow",
            "Action": "iot:Publish",
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*/response/*",
            ],
        },
        {
            "Sid": "AllowSafetyEstop",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/safety/estop",
            ],
        },
        {
            "Sid": "AllowOwnSubscriptions",
            "Effect": "Allow",
            "Action": "iot:Subscribe",
            "Resource": [
                "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/*",
                "arn:aws:iot:*:*:topicfilter/strands/broadcast",
                "arn:aws:iot:*:*:topicfilter/strands/safety/estop",
                "arn:aws:iot:*:*:topicfilter/strands/+/presence",
            ],
        },
        {
            "Sid": "AllowReceiveOthers",
            "Effect": "Allow",
            "Action": "iot:Receive",
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*",
            ],
        },
        {
            "Sid": "AllowShadow",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:Subscribe", "iot:Receive"],
            "Resource": [
                "arn:aws:iot:*:*:topic/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
                "arn:aws:iot:*:*:topicfilter/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
            ],
        },
    ],
}


_OPERATOR_POLICY_DOC: dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "OperatorConnect",
            "Effect": "Allow",
            "Action": "iot:Connect",
            "Resource": "arn:aws:iot:*:*:client/${iot:Connection.Thing.ThingName}",
        },
        {
            "Sid": "OperatorPublishToFleet",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*/cmd",
                "arn:aws:iot:*:*:topic/strands/broadcast",
                "arn:aws:iot:*:*:topic/strands/safety/estop",
            ],
        },
        {
            "Sid": "OperatorReceiveResponses",
            "Effect": "Allow",
            "Action": ["iot:Subscribe", "iot:Receive"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/response/*",
                "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/response/*",
            ],
        },
        {
            "Sid": "OperatorObserveFleet",
            "Effect": "Allow",
            "Action": ["iot:Subscribe", "iot:Receive"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*",
                "arn:aws:iot:*:*:topicfilter/strands/+/presence",
                "arn:aws:iot:*:*:topicfilter/strands/+/state",
                "arn:aws:iot:*:*:topicfilter/strands/+/health",
                "arn:aws:iot:*:*:topicfilter/strands/+/safety/event",
                "arn:aws:iot:*:*:topicfilter/strands/safety/estop",
            ],
        },
        {
            "Sid": "OperatorShadow",
            "Effect": "Allow",
            "Action": ["iot:GetThingShadow", "iot:UpdateThingShadow"],
            "Resource": ["arn:aws:iot:*:*:thing/*"],
            "Condition": {
                "StringEquals": {
                    "iot:Connection.Thing.Attributes.strands-mesh-role": "robot",
                },
            },
        },
    ],
}


# Public API


def provision_robot(
    thing_name: str,
    *,
    region: str | None = None,
    cert_dir: Path | str | None = None,
    attributes: dict[str, str] | None = None,
) -> ProvisionedThing:
    """Provision a robot Thing and write its credentials to disk.

    Args:
        thing_name: The Thing name. MUST equal the intended Mesh peer_id —
            the IoT Policy uses ``${iot:Connection.Thing.ThingName}`` for
            topic ACL substitution. Should be DNS-safe (alphanumeric + ``-_``).
        region: AWS region. Defaults to the default boto3 session region.
        cert_dir: Where to write certs. Defaults to ``~/.strands_robots/iot``.
        attributes: Optional thing-attribute dict (≤3 keys, ≤800 chars total).

    Returns:
        :class:`ProvisionedThing` describing the artefacts.

    Raises:
        ImportError: If ``boto3`` is not installed.
        botocore.exceptions.ClientError: For AWS-side failures (auth, throttling).

    Idempotence:
        - Thing creation: ``CreateThing`` is idempotent if the attributes match.
        - Policy creation: skipped if the policy name already exists.
        - Cert creation: a new cert is always issued (private keys aren't
          recoverable). Old certs from prior runs remain on the Thing —
          call :func:`teardown_thing` to clean them up.
    """
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)
    region = iot.meta.region_name

    # Inject strands-mesh-role attribute for ACL — the OperatorShadow policy
    # uses an attribute condition to scope shadow access to robot Things only.
    attributes = dict(attributes) if attributes else {}
    attributes.setdefault("strands-mesh-role", "robot")

    cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    cert_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cert_dir, 0o700)
    except OSError:
        pass

    # 1. Thing
    thing_arn = _ensure_thing(iot, thing_name, attributes)

    # 2. Policy
    policy_arn = _ensure_policy(iot, ROBOT_POLICY_NAME, _ROBOT_POLICY_DOC)
    logger.info("[provision] %s: using policy %s", thing_name, policy_arn)

    # 3. Cert + key
    # Clean up stale certs from prior provision_robot runs on the same Thing.
    # Each call to AWS IoT CreateKeysAndCertificate yields a brand-new cert
    # (private keys cannot be recovered after issuance), so without cleanup
    # the Thing would accumulate certs across re-runs — every leftover is
    # an active credential that could impersonate the robot.
    _cleanup_stale_certs(iot, thing_name)

    cert_path = cert_dir / f"{thing_name}.cert.pem"
    key_path = cert_dir / f"{thing_name}.private.key"
    cert_arn, cert_id = _create_cert(iot, cert_path, key_path)

    # 4. Attach policy → cert → thing
    iot.attach_policy(policyName=ROBOT_POLICY_NAME, target=cert_arn)
    iot.attach_thing_principal(thingName=thing_name, principal=cert_arn)
    logger.info("[provision] %s: cert %s attached", thing_name, cert_id)

    # 5. CA + endpoint
    ca_path = cert_dir / "AmazonRootCA1.pem"
    _ensure_ca(ca_path)
    endpoint = _discover_endpoint(iot)
    (cert_dir / "endpoint").write_text(endpoint)

    return ProvisionedThing(
        thing_name=thing_name,
        thing_arn=thing_arn,
        cert_arn=cert_arn,
        cert_id=cert_id,
        cert_path=cert_path,
        key_path=key_path,
        ca_path=ca_path,
        endpoint=endpoint,
        policy_name=ROBOT_POLICY_NAME,
        region=region,
    )


def provision_operator(
    thing_name: str,
    *,
    region: str | None = None,
    cert_dir: Path | str | None = None,
    attributes: dict[str, str] | None = None,
) -> ProvisionedThing:
    """Provision an operator Thing (Bedrock agent / fleet ops console).

    Same as :func:`provision_robot` but with the operator policy
    (``strands-operator``) which can publish ``cmd`` / ``broadcast`` and
    observe the whole fleet.
    """
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)
    region = iot.meta.region_name

    # Inject strands-mesh-role attribute — operators get role=operator so the
    # OperatorShadow attribute condition (role=robot) excludes their shadows.
    attributes = dict(attributes) if attributes else {}
    attributes.setdefault("strands-mesh-role", "operator")

    cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    cert_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cert_dir, 0o700)
    except OSError:
        pass

    thing_arn = _ensure_thing(iot, thing_name, attributes)
    policy_arn = _ensure_policy(iot, OPERATOR_POLICY_NAME, _OPERATOR_POLICY_DOC)
    logger.info("[provision] %s: using policy %s", thing_name, policy_arn)

    # Clean up stale certs from prior provision_operator runs.
    _cleanup_stale_certs(iot, thing_name)

    cert_path = cert_dir / f"{thing_name}.cert.pem"
    key_path = cert_dir / f"{thing_name}.private.key"
    cert_arn, cert_id = _create_cert(iot, cert_path, key_path)

    iot.attach_policy(policyName=OPERATOR_POLICY_NAME, target=cert_arn)
    iot.attach_thing_principal(thingName=thing_name, principal=cert_arn)

    ca_path = cert_dir / "AmazonRootCA1.pem"
    _ensure_ca(ca_path)
    endpoint = _discover_endpoint(iot)
    (cert_dir / "endpoint").write_text(endpoint)

    return ProvisionedThing(
        thing_name=thing_name,
        thing_arn=thing_arn,
        cert_arn=cert_arn,
        cert_id=cert_id,
        cert_path=cert_path,
        key_path=key_path,
        ca_path=ca_path,
        endpoint=endpoint,
        policy_name=OPERATOR_POLICY_NAME,
        region=region,
    )


def teardown_thing(thing_name: str, *, region: str | None = None) -> None:
    """Detach + delete every cert attached to *thing_name*, then delete the Thing.

    Cleans up the cert files in ``DEFAULT_CERT_DIR`` if they're named after
    this Thing. Does NOT delete the policies — those are shared across all
    robots and removing them would break siblings.

    Idempotent: missing Thing or no certs is a silent success.
    """
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)

    try:
        principals = iot.list_thing_principals(thingName=thing_name).get("principals", [])
    except iot.exceptions.ResourceNotFoundException:
        logger.info("[teardown] thing %s not found, skipping", thing_name)
        principals = []

    for cert_arn in principals:
        cert_id = cert_arn.rsplit("/", 1)[-1]
        try:
            iot.detach_thing_principal(thingName=thing_name, principal=cert_arn)
        except Exception as exc:
            logger.debug("[teardown] detach %s from %s: %s", cert_id, thing_name, exc)
        # Detach all attached policies first
        try:
            for pol in iot.list_attached_policies(target=cert_arn).get("policies", []):
                iot.detach_policy(policyName=pol["policyName"], target=cert_arn)
        except Exception as exc:
            logger.debug("[teardown] detach policies from %s: %s", cert_id, exc)
        try:
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id, forceDelete=True)
        except Exception as exc:
            logger.warning("[teardown] could not delete cert %s: %s", cert_id, exc)

    # Delete the Thing
    try:
        iot.delete_thing(thingName=thing_name)
        logger.info("[teardown] deleted thing %s", thing_name)
    except iot.exceptions.ResourceNotFoundException:
        pass

    # Remove local cert files
    for suffix in (".cert.pem", ".private.key", ".public.key"):
        p = DEFAULT_CERT_DIR / f"{thing_name}{suffix}"
        if p.exists():
            try:
                p.unlink()
            except OSError as exc:
                logger.debug("[teardown] could not unlink %s: %s", p, exc)


# Internals


def _require_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS IoT provisioning. Install with: pip install 'strands-robots[mesh-iot]'"
        ) from exc
    return boto3


def _ensure_thing(iot: Any, thing_name: str, attributes: dict[str, str] | None) -> str:
    """Create the Thing if absent, otherwise return its ARN unchanged."""
    try:
        existing = iot.describe_thing(thingName=thing_name)
        logger.info("[provision] thing %s already exists", thing_name)
        return existing["thingArn"]
    except iot.exceptions.ResourceNotFoundException:
        pass

    payload: dict[str, Any] = {"thingName": thing_name}
    if attributes:
        payload["attributePayload"] = {"attributes": attributes}
    resp = iot.create_thing(**payload)
    logger.info("[provision] thing %s created", thing_name)
    return resp["thingArn"]


def _ensure_policy(iot: Any, name: str, document: dict[str, Any]) -> str:
    """Create the policy if absent. Idempotent — does not update an existing
    policy; users who want to update should bump the policy version manually."""
    try:
        existing = iot.get_policy(policyName=name)
        logger.info("[provision] policy %s already exists (v%s)", name, existing.get("defaultVersionId", "?"))
        return existing["policyArn"]
    except iot.exceptions.ResourceNotFoundException:
        pass

    resp = iot.create_policy(
        policyName=name,
        policyDocument=json.dumps(document),
    )
    logger.info("[provision] policy %s created", name)
    return resp["policyArn"]


def _cleanup_stale_certs(iot: Any, thing_name: str) -> int:
    """Detach + delete any certificates already attached to *thing_name*.

    Re-running :func:`provision_robot` on the same Thing has historically
    caused certs to accumulate (each run issues a fresh cert because
    AWS doesn't expose previously-generated private keys). That left
    Things with 5–10 ACTIVE certs after a few dev iterations, which is
    a footgun: every old cert is a credential that *could* be used to
    impersonate the robot.

    This helper detaches every existing principal, removes its policy
    attachments, marks the cert INACTIVE, and force-deletes it. Failures
    are logged at DEBUG and swallowed so a partial cleanup never blocks
    the new cert issuance — the new cert is what users actually want.

    Returns the number of certs cleaned up (for logging in the caller).
    """
    cleaned = 0
    try:
        existing = iot.list_thing_principals(thingName=thing_name).get("principals", [])
    except iot.exceptions.ResourceNotFoundException:
        return 0

    for cert_arn in existing:
        cert_id = cert_arn.rsplit("/", 1)[-1]
        try:
            iot.detach_thing_principal(thingName=thing_name, principal=cert_arn)
        except Exception as exc:
            logger.debug("[provision] detach %s from %s: %s", cert_id, thing_name, exc)
        try:
            for pol in iot.list_attached_policies(target=cert_arn).get("policies", []):
                iot.detach_policy(policyName=pol["policyName"], target=cert_arn)
        except Exception as exc:
            logger.debug("[provision] detach policies from %s: %s", cert_id, exc)
        try:
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id, forceDelete=True)
            cleaned += 1
        except Exception as exc:
            logger.warning("[provision] could not delete stale cert %s: %s", cert_id, exc)
    if cleaned:
        logger.info(
            "[provision] cleaned up %d stale cert(s) on %s before issuing new one",
            cleaned,
            thing_name,
        )
    return cleaned


def _create_cert(iot: Any, cert_path: Path, key_path: Path) -> tuple[str, str]:
    """Issue a fresh cert+key and write them to disk with mode 0o600."""
    resp = iot.create_keys_and_certificate(setAsActive=True)
    cert_arn = resp["certificateArn"]
    cert_id = resp["certificateId"]

    cert_path.write_text(resp["certificatePem"])
    key_path.write_text(resp["keyPair"]["PrivateKey"])
    try:
        os.chmod(cert_path, 0o600)
        os.chmod(key_path, 0o600)
    except OSError as exc:
        logger.warning("[provision] could not chmod certs: %s", exc)
    return cert_arn, cert_id


def _ensure_ca(ca_path: Path) -> None:
    """Download the Amazon Root CA1 to *ca_path* if not already present."""
    if ca_path.exists() and ca_path.stat().st_size > 0:
        return
    logger.info("[provision] downloading Amazon Root CA1 → %s", ca_path)
    with urllib.request.urlopen(_AMAZON_ROOT_CA1_URL) as resp:
        ca_path.write_bytes(resp.read())
    try:
        os.chmod(ca_path, 0o644)
    except OSError:
        pass


def _discover_endpoint(iot: Any) -> str:
    """Return the iot:Data-ATS endpoint for this region+account."""
    return iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
