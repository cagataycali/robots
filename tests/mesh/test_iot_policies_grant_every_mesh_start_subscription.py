"""Every topic filter ``Mesh.start()`` subscribes to is granted by the policies the peer's certificate carries.

AWS IoT Core answers an unauthorized filter in a SUBSCRIBE with reason code 135
and then closes the whole MQTT session, not only that subscription. An operator
Thing on the shipped ``strands-operator`` policies used to be refused its own
``cmd`` topic and ``strands/broadcast``, so its mesh never came alive and the
dashboard's safety rail (an operator peer) reported "safety mesh unavailable".

The filter list is read from a real ``Mesh.start()`` against a recording
session, so a subscription added there is graded against the policies the
moment it lands. The grader is a small IoT policy matcher: ``*`` in a resource
matches any run of characters, ``${iot:Connection.Thing.ThingName}`` resolves to
the connecting Thing, and an explicit ``Deny`` wins over any ``Allow``.
"""

from __future__ import annotations

import fnmatch

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.iot import provision
from strands_robots.mesh.transport.iot_transport import _suback_refusal_text, _zenoh_to_mqtt_filter

THING = "strands-op-1"

#: The policies each provisioning path attaches to one certificate.
CERTIFICATES = {
    "operator": (provision._OPERATOR_POLICY_DOC, provision._OPERATOR_OBSERVE_POLICY_DOC),
    "robot": (provision._robot_policy_doc(allow_estop_publish=True), provision._ROBOT_CHILDREN_POLICY_DOC),
    "robot-no-estop": (provision._robot_policy_doc(allow_estop_publish=False), provision._ROBOT_CHILDREN_POLICY_DOC),
}


class _Sub:
    def undeclare(self) -> None:
        return None


def _mesh_start_filters(monkeypatch: pytest.MonkeyPatch, robot: object, peer_type: str) -> list[str]:
    """The MQTT topic filters a real ``Mesh.start()`` subscribes to for *THING*."""
    keys: list[str] = []

    class _RecordingSession:
        def declare_subscriber(self, key_expr: str, _handler: object) -> _Sub:
            keys.append(key_expr)
            return _Sub()

    monkeypatch.setattr(mesh_core, "get_session", lambda: _RecordingSession())
    monkeypatch.setattr(mesh_core, "release_session", lambda: None)
    monkeypatch.setattr(mesh_core.Mesh, "_refuse_under_permissive_default_acl", lambda self: False)
    mesh = mesh_core.Mesh(robot, peer_id=THING, peer_type=peer_type)
    for loop in ("_heartbeat_loop", "_state_loop"):
        monkeypatch.setattr(mesh, loop, lambda *a, **k: None)
    try:
        mesh.start()
    finally:
        mesh.stop()
    assert keys, "Mesh.start() declared no subscribers"
    return [_zenoh_to_mqtt_filter(k) for k in keys]


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _permits(docs: tuple[dict, ...], action: str, arn: str) -> bool:
    """IoT policy evaluation for one action on one resource: any Allow, and no Deny."""
    verdicts = set()
    for doc in docs:
        for st in doc["Statement"]:
            if action not in _as_list(st["Action"]):
                continue
            for resource in _as_list(st["Resource"]):
                if fnmatch.fnmatchcase(arn, resource.replace("${iot:Connection.Thing.ThingName}", THING)):
                    verdicts.add(st["Effect"])
    return "Allow" in verdicts and "Deny" not in verdicts


def _a_topic_the_filter_delivers(mqtt_filter: str) -> str:
    return "/".join({"+": "robot-a", "#": "robot-a/turn-1"}.get(s, s) for s in mqtt_filter.split("/"))


@pytest.mark.parametrize("posture", sorted(CERTIFICATES))
def test_every_mesh_start_filter_is_subscribable_and_receivable(monkeypatch: pytest.MonkeyPatch, posture: str) -> None:
    """Subscribe on the filter AND Receive on a topic it delivers: a missing Receive drops every message silently."""
    docs = CERTIFICATES[posture]
    # The dashboard's safety rail is a robot-less gateway peer; a robot peer has a body.
    robot = None if posture == "operator" else object()
    filters = _mesh_start_filters(monkeypatch, robot, "gateway" if robot is None else "robot")
    refused = [
        f"{action} {arn}"
        for mqtt_filter in filters
        for action, arn in (
            ("iot:Subscribe", f"arn:aws:iot:*:*:topicfilter/{mqtt_filter}"),
            ("iot:Receive", f"arn:aws:iot:*:*:topic/{_a_topic_the_filter_delivers(mqtt_filter)}"),
        )
        if not _permits(docs, action, arn)
    ]
    assert refused == [], f"the {posture} certificate would get SUBACK 135 and lose its session on: {refused}"


def test_the_grader_refuses_what_the_broker_refuses() -> None:
    """Planted negatives: another Thing's command topic, and a Deny over an Allow."""
    operator = CERTIFICATES["operator"]
    assert not _permits(operator, "iot:Receive", "arn:aws:iot:*:*:topic/strands/robot-a/cmd")
    assert not _permits(operator, "iot:Subscribe", "arn:aws:iot:*:*:topicfilter/strands/+/cmd")
    deny = {
        "Statement": [
            {"Effect": "Allow", "Action": "iot:Publish", "Resource": "arn:aws:iot:*:*:topic/strands/*"},
            {"Effect": "Deny", "Action": "iot:Publish", "Resource": "arn:aws:iot:*:*:topic/strands/broadcast"},
        ]
    }
    assert not _permits((deny,), "iot:Publish", "arn:aws:iot:*:*:topic/strands/broadcast")
    assert _permits((deny,), "iot:Publish", "arn:aws:iot:*:*:topic/strands/x")


def test_every_owned_policy_is_republished_and_fits_the_cap() -> None:
    """``reprovision_thing`` republishes only owned documents; the observe policy carries the new grant."""
    owned = provision._OWNED_POLICY_DOCUMENTS
    assert owned[provision.OPERATOR_OBSERVE_POLICY_NAME]() is provision._OPERATOR_OBSERVE_POLICY_DOC
    for name, build in owned.items():
        assert provision.policy_document_size_error(name, build()) is None


@pytest.mark.parametrize(("code", "names_the_policy"), [(135, True), (151, False)])
def test_a_refused_subscribe_says_whether_the_policy_is_the_cause(code: int, names_the_policy: bool) -> None:
    """135 is the policy refusal that drops the session; it names the grant and the command that republishes it."""
    text = _suback_refusal_text(code)
    assert f"SUBACK reason code {code}" in text
    assert ("iot:Subscribe" in text and "strands-robots iot reprovision" in text) is names_the_policy
