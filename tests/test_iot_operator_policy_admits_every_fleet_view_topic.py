"""The strands-operator IoT policy admits every topic the dashboard fleet view subscribes to.

The dashboard joins the mesh as an operator. Over Zenoh the ACL is the LAN's;
over AWS IoT Core the ``strands-operator`` policy decides which topics the
broker delivers to that identity. The bridge used to subscribe to eleven key
expressions while the policy granted six, so a robot reached over IoT showed
a name, joints and a health row and never a camera, a stream, a pose or a
lidar summary: the broker answered the extra subscriptions with "not
authorized" and nothing in the dashboard said so.

These cells pin the two lists to each other. A topic added to the fleet view
must be granted here in the same change, and the grant stays read-only: the
operator observes the fleet, it never receives another operator's commands or
responses, and teleop input never crosses the WAN.
"""

from __future__ import annotations

import ast
import fnmatch
import pathlib

import pytest

from strands_robots.dashboard import mesh_bridge
from strands_robots.mesh.iot import provision
from strands_robots.mesh.iot.provision import _OPERATOR_OBSERVE_POLICY_DOC, _OPERATOR_POLICY_DOC, _ROBOT_POLICY_DOC
from strands_robots.mesh.transport.iot_transport import _zenoh_to_mqtt_filter

_BRIDGE_SOURCE = pathlib.Path(mesh_bridge.__file__)


#: Every policy an operator certificate carries; the grants are graded as their union.
OPERATOR_DOCS = (_OPERATOR_POLICY_DOC, _OPERATOR_OBSERVE_POLICY_DOC)


def _resources(docs: dict | tuple[dict, ...], action: str) -> list[str]:
    out: list[str] = []
    for doc in (docs,) if isinstance(docs, dict) else docs:
        out.extend(_statement_resources(doc, action))
    return out


def _statement_resources(doc: dict, action: str) -> list[str]:
    out: list[str] = []
    for statement in doc["Statement"]:
        if statement.get("Effect") != "Allow":
            continue
        actions = statement["Action"]
        if isinstance(actions, str):
            actions = [actions]
        if action not in actions:
            continue
        resources = statement["Resource"]
        if isinstance(resources, str):
            resources = [resources]
        out.extend(resources)
    return out


def _arn_matches(resource_arn: str, concrete: str) -> bool:
    """An IoT policy resource ARN against a concrete topic (or filter) ARN.

    ``*`` in a resource ARN matches any run of characters including ``/``;
    a policy variable never appears in the operator's observe grants.
    """
    return fnmatch.fnmatchcase(concrete, resource_arn)


def _concrete_topic(mqtt_filter: str) -> str:
    """One topic a robot really publishes that the MQTT filter would deliver."""
    parts = []
    for segment in mqtt_filter.split("/"):
        if segment == "+":
            parts.append("robot-a")
        elif segment == "#":
            parts.append("front/ref")
        else:
            parts.append(segment)
    return "/".join(parts)


@pytest.mark.parametrize("key_expr", mesh_bridge.FLEET_SUBSCRIPTIONS)
def test_every_fleet_view_subscription_is_granted_to_the_operator(key_expr: str) -> None:
    mqtt_filter = _zenoh_to_mqtt_filter(key_expr)
    filter_arn = f"arn:aws:iot:*:*:topicfilter/{mqtt_filter}"
    subscribe = _resources(OPERATOR_DOCS, "iot:Subscribe")
    assert any(_arn_matches(r, filter_arn) for r in subscribe), (
        f"the fleet view subscribes to {key_expr!r} but no iot:Subscribe grant in the "
        f"strands-operator policy covers topicfilter {mqtt_filter!r}"
    )
    topic_arn = f"arn:aws:iot:*:*:topic/{_concrete_topic(mqtt_filter)}"
    receive = _resources(OPERATOR_DOCS, "iot:Receive")
    assert any(_arn_matches(r, topic_arn) for r in receive), (
        f"the operator may subscribe to {mqtt_filter!r} but no iot:Receive grant delivers "
        f"{_concrete_topic(mqtt_filter)!r}; the broker would accept the SUBSCRIBE and drop every message"
    )


def test_the_bridge_subscribes_from_the_roster_and_nowhere_else() -> None:
    """Every ``declare_subscriber`` key in the bridge comes from ``FLEET_SUBSCRIPTIONS``.

    A literal ``sub("strands/*/thing", ...)`` next to the roster would be a
    subscription the operator policy is never graded against.
    """
    tree = ast.parse(_BRIDGE_SOURCE.read_text(encoding="utf-8"))
    literal_keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name not in {"sub", "declare_subscriber"}:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            literal_keys.append(node.args[0].value)
    assert literal_keys == [], f"subscriptions outside FLEET_SUBSCRIPTIONS: {literal_keys}"
    assert len(set(mesh_bridge.FLEET_SUBSCRIPTIONS)) == len(mesh_bridge.FLEET_SUBSCRIPTIONS)


def test_the_operator_still_cannot_read_commands_responses_or_teleop_input() -> None:
    receive = _resources(OPERATOR_DOCS, "iot:Receive")
    subscribe = _resources(OPERATOR_DOCS, "iot:Subscribe")
    for forbidden in (
        "arn:aws:iot:*:*:topic/strands/robot-a/cmd",
        "arn:aws:iot:*:*:topic/strands/other-operator/response/robot-a/turn-1",
        "arn:aws:iot:*:*:topic/strands/robot-a/input/teleop",
        "arn:aws:iot:*:*:topic/strands/robot-a/hand/left",
    ):
        assert not any(_arn_matches(r, forbidden) for r in receive), f"operator Receive covers {forbidden!r}"
    for resource in subscribe + receive:
        assert "/input/" not in resource and "/hand/" not in resource, resource
        assert not resource.endswith(":topic/strands/*"), resource


def test_every_shipped_policy_document_fits_the_aws_cap() -> None:
    """AWS refuses a document over 2048 characters; the robot doc alone is 1971 compact.

    The failure is at publish time, on the first fleet whose document changed, so
    it is graded here for every document the package ships.
    """
    docs = {
        provision.OPERATOR_POLICY_NAME: _OPERATOR_POLICY_DOC,
        provision.OPERATOR_OBSERVE_POLICY_NAME: _OPERATOR_OBSERVE_POLICY_DOC,
        "strands-robot": provision._robot_policy_doc(allow_estop_publish=True),
        "strands-robot-no-estop": provision._robot_policy_doc(allow_estop_publish=False),
    }
    for name, doc in docs.items():
        assert provision.policy_document_size_error(name, doc) is None, provision.policy_document_size_error(name, doc)
    assert provision.POLICY_DOCUMENT_CAP == 2048
    too_big = {"Version": "2012-10-17", "Statement": [{"Sid": "x" * 2100}]}
    text = provision.policy_document_size_error("big", too_big)
    assert text is not None and "2048" in text and "second policy" in text


def test_an_operator_certificate_carries_both_policies() -> None:
    """provision_operator attaches strands-operator AND strands-operator-observe; reprovision adds the second to an old operator."""
    import inspect

    src = inspect.getsource(provision.provision_operator)
    assert "attach_policy(policyName=OPERATOR_POLICY_NAME" in src
    assert "attach_policy(policyName=OPERATOR_OBSERVE_POLICY_NAME" in src
    src = inspect.getsource(provision.reprovision_thing)
    assert "policy_names.append(OPERATOR_OBSERVE_POLICY_NAME)" in src
    assert provision.OPERATOR_OBSERVE_POLICY_NAME == "strands-operator-observe"


def test_a_robot_may_publish_what_the_fleet_view_reads() -> None:
    """The robot side of the same wire: its own camera, stream and sensor topics."""
    publish = _resources(_ROBOT_POLICY_DOC, "iot:Publish")
    for key_expr in mesh_bridge.FLEET_SUBSCRIPTIONS:
        if key_expr.startswith("strands/safety/"):
            continue
        topic = _concrete_topic(_zenoh_to_mqtt_filter(key_expr)).replace("robot-a", "${iot:Connection.Thing.ThingName}")
        arn = f"arn:aws:iot:*:*:topic/{topic}"
        assert any(_arn_matches(r, arn) for r in publish), f"robot policy does not let a Thing publish {topic!r}"
