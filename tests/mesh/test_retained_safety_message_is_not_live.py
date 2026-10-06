"""Regression tests: a retained MQTT message is never mistaken for a live fleet stop.

Four decisions composed into a durable fleet lockout on the AWS IoT transport:
the robot and operator policies granted ``iot:RetainPublish`` on the safety
topics, the transport published ``safety/estop`` and ``safety/resume`` with
``retain=True``, subscriptions used the MQTT 5 default retain handling (the
broker replays the stored message the moment a robot subscribes), and the
handler saw only topic and payload, so a message stored an hour ago looked
like an operator pressing the button now. One re-publish per freshness window
kept every robot that booted, reconnected or was newly provisioned locked out.

Now: no ``RetainPublish`` on the safety topics in either policy, the two
safety commands are published as events (``retain=False``), the safety
subscriptions ask the broker not to send retained messages at subscribe time,
the transport carries the packet's retain flag on the sample, and the shared
safety envelope decoder refuses a retained delivery with an audit record.
The same refusal holds for a ``Mesh.subscribe`` reader, the dashboard and a
wildcard subscription, an unreadable packet flag reads as retained, and
``clear_retained_safety_messages`` deletes what an older policy let a peer store.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.iot import provision
from strands_robots.mesh.transport import iot_transport
from strands_robots.mesh.transport.iot_transport import _MqttSample, _qos_and_retain_for

_SAFETY = ("arn:aws:iot:*:*:topic/strands/safety/estop", "arn:aws:iot:*:*:topic/strands/safety/resume")


def _statements_publishing(doc: dict[str, Any], resource: str) -> list[dict[str, Any]]:
    out = []
    for st in doc["Statement"]:
        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        resources = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
        if resource in resources and any(a.startswith("iot:") and "Publish" in a for a in actions):
            out.append(st)
    return out


class TestNoRetainPublishOnSafetyTopics:
    @pytest.mark.parametrize("resource", _SAFETY)
    def test_the_safety_authority_robot_policy_publishes_without_retain(self, resource: str) -> None:
        doc = provision._robot_policy_doc(allow_estop_publish=True)
        statements = _statements_publishing(doc, resource)
        assert statements, "the safety-authority robot policy must still be able to publish a stop"
        for st in statements:
            assert "iot:RetainPublish" not in st["Action"], st

    @pytest.mark.parametrize("resource", _SAFETY)
    def test_the_operator_policy_publishes_without_retain(self, resource: str) -> None:
        statements = _statements_publishing(provision._OPERATOR_POLICY_DOC, resource)
        assert statements, "the operator must still be able to publish a stop"
        for st in statements:
            assert "iot:RetainPublish" not in st["Action"], st

    def test_the_operator_still_retains_its_own_presence(self) -> None:
        """Retain earns its keep on state topics; only the safety commands lose it."""
        st = next(s for s in provision._OPERATOR_POLICY_DOC["Statement"] if s["Sid"] == "OperatorAnnounceSelf")
        assert "iot:RetainPublish" in st["Action"]


class TestSafetyCommandsAreEventsNotState:
    @pytest.mark.parametrize("topic", ["strands/safety/estop", "strands/safety/resume"])
    def test_published_with_retain_false(self, topic: str) -> None:
        qos, retain = _qos_and_retain_for(topic)
        assert qos == 1
        assert retain is False

    def test_presence_and_the_per_robot_safety_event_keep_retain(self) -> None:
        assert _qos_and_retain_for("strands/arm-1/presence") == (1, True)
        assert _qos_and_retain_for("strands/arm-1/safety/event") == (1, True)


class TestSubscriptionsDoNotReplayRetainedSafetyMessages:
    def _transport_with_fake_mqtt5(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
        subscriptions: list[Any] = []

        class _Subscription:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw
                subscriptions.append(self)

        class _SubscribePacket:
            def __init__(self, subscriptions: list[Any]) -> None:
                self.subscriptions = subscriptions

        fake_mqtt5 = SimpleNamespace(
            Subscription=_Subscription,
            SubscribePacket=_SubscribePacket,
            QoS=SimpleNamespace(AT_LEAST_ONCE="qos1"),
            RetainHandlingType=SimpleNamespace(SEND_ON_SUBSCRIBE="send", DONT_SEND="dont_send"),
        )
        monkeypatch.setitem(sys.modules, "awscrt", SimpleNamespace(mqtt5=fake_mqtt5))
        monkeypatch.setitem(sys.modules, "awscrt.mqtt5", fake_mqtt5)

        transport = iot_transport.IotMqttTransport.__new__(iot_transport.IotMqttTransport)
        transport._client = MagicMock()
        transport._client.subscribe.return_value.result.return_value = None
        transport._connected = threading.Event()
        transport._connected.set()
        transport._lock = threading.Lock()
        transport._handlers = {}
        return transport, subscriptions

    def test_safety_topics_ask_the_broker_not_to_send_retained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport, subs = self._transport_with_fake_mqtt5(monkeypatch)

        transport.declare_subscriber("strands/safety/estop", lambda s: None)
        transport.declare_subscriber("strands/safety/resume", lambda s: None)

        assert [s.kw["retain_handling_type"] for s in subs] == ["dont_send", "dont_send"]

    def test_presence_subscriptions_still_receive_retained_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport, subs = self._transport_with_fake_mqtt5(monkeypatch)

        transport.declare_subscriber("strands/*/presence", lambda s: None)

        assert "retain_handling_type" not in subs[0].kw or subs[0].kw["retain_handling_type"] == "send"


class TestARetainedDeliveryIsRefusedByTheHandler:
    def test_the_sample_carries_the_packet_retain_flag(self) -> None:
        transport = iot_transport.IotMqttTransport.__new__(iot_transport.IotMqttTransport)
        transport._lock = threading.Lock()
        seen: list[Any] = []
        transport._handlers = {"strands/safety/estop": [seen.append]}
        transport._thing_name = "arm-1"
        transport._unmatched_inbound = 0
        packet = SimpleNamespace(
            topic="strands/safety/estop", payload=b"{}", retain=True, response_topic=None, correlation_data=None
        )

        transport._on_publish_received(SimpleNamespace(publish_packet=packet))

        assert seen[0].retain is True
        assert _MqttSample("strands/safety/estop", b"{}").retain is False

    def test_a_retained_estop_does_not_engage_the_lockout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mesh = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="arm-1")
        audits: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(mesh, "_audit_local", lambda e, p: audits.append((e, p)))
        envelope = {"peer_id": "op-1", "t": time.time(), "responses_received": 1, "peers_not_stopped": []}
        sample = _MqttSample("strands/safety/estop", json.dumps(envelope).encode())
        sample.retain = True

        mesh._on_safety_estop(sample)

        assert not mesh._estop_lockout.is_set()
        assert [e for e, _ in audits] == ["safety_retained_delivery_rejected"]
        assert audits[0][1]["kind"] == "estop"

    def test_the_same_envelope_delivered_live_engages_it(self) -> None:
        mesh = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="arm-1")
        envelope = {"peer_id": "op-1", "t": time.time(), "responses_received": 1, "peers_not_stopped": []}

        mesh._on_safety_estop(_MqttSample("strands/safety/estop", json.dumps(envelope).encode()))

        assert mesh._estop_lockout.is_set()

    def test_a_zenoh_sample_has_no_retain_flag_and_reads_as_live(self) -> None:
        mesh = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="arm-1")
        envelope = {"peer_id": "op-1", "t": time.time(), "responses_received": 1, "peers_not_stopped": []}
        sample = SimpleNamespace(
            payload=SimpleNamespace(to_bytes=lambda: json.dumps(envelope).encode()), source_info=None
        )

        mesh._on_safety_estop(sample)

        assert mesh._estop_lockout.is_set()

    def test_a_magicmock_sample_reads_as_live(self) -> None:
        """The unit fixtures' MagicMock attributes are truthy; only a real True is a retained delivery."""
        mesh = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="arm-1")
        envelope = {"peer_id": "op-1", "t": time.time(), "responses_received": 1, "peers_not_stopped": []}
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(envelope).encode()

        mesh._on_safety_estop(sample)

        assert mesh._estop_lockout.is_set()


class TestEverySubscriberRefusesARetainedSafetyCommand:
    """The robot's refusal holds for every first-party reader of the safety topics."""

    @pytest.mark.parametrize(
        ("topic_filter", "dont_send"),
        [
            ("strands/safety/estop", True),
            ("strands/safety/#", True),
            ("strands/safety/+", True),
            ("strands/+/estop", True),
            ("strands/#", True),
            ("#", True),
            ("strands/+/presence", False),
            ("strands/safety/event", False),
            ("strands/arm-1/safety/#", False),
        ],
    )
    def test_a_filter_that_delivers_a_safety_command_asks_for_no_retained(
        self, topic_filter: str, dont_send: bool
    ) -> None:
        assert iot_transport._is_safety_command_filter(topic_filter) is dont_send

    @pytest.mark.parametrize(("flag", "retained"), [(True, True), (False, False), (None, True), ("0", True)])
    def test_a_packet_flag_that_cannot_be_read_is_treated_as_retained(self, flag: Any, retained: bool) -> None:
        transport = iot_transport.IotMqttTransport.__new__(iot_transport.IotMqttTransport)
        transport._lock = threading.Lock()
        seen: list[Any] = []
        transport._handlers = {"strands/safety/estop": [seen.append]}
        transport._thing_name = "arm-1"
        transport._unmatched_inbound = 0
        attrs: dict[str, Any] = {"topic": "strands/safety/estop", "payload": b"{}"}
        if flag is not None:
            attrs["retain"] = flag
        transport._on_publish_received(SimpleNamespace(publish_packet=SimpleNamespace(**attrs)))
        assert seen[0].retain is retained

    def test_a_mesh_subscription_drops_a_retained_stop_and_keeps_a_live_one(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        handlers: list[Any] = []

        def declare_subscriber(key: str, handler: Any) -> MagicMock:
            handlers.append(handler)
            return MagicMock()

        session = SimpleNamespace(declare_subscriber=declare_subscriber)
        monkeypatch.setattr("strands_robots.mesh.core.current_session", lambda: session)
        mesh = Mesh(SimpleNamespace(tool_name_str="agent"), peer_id="agent-1")
        mesh._running = True
        assert mesh.subscribe("strands/safety/**", name="safety") == "safety"
        body = json.dumps({"peer_id": "op-1", "t": time.time()}).encode()
        retained = _MqttSample("strands/safety/estop", body, retain=True)

        with caplog.at_level("WARNING"):
            handlers[0](retained)
            handlers[0](retained)
        handlers[0](_MqttSample("strands/safety/estop", body))

        assert [key for key, _ in mesh.inbox["safety"]] == ["strands/safety/estop"]
        assert sum("dropped a retained message" in r.getMessage() for r in caplog.records) == 1

    def test_the_dashboard_does_not_fold_a_retained_stop_into_the_fleet_lockout(self) -> None:
        from strands_robots.dashboard.mesh_bridge import MeshBridge

        bridge = MeshBridge(peer_id="dash")
        body = json.dumps({"source": "operator-laptop", "t": time.time()}).encode()

        bridge._on_safety(_MqttSample("strands/safety/estop", body, retain=True))
        assert bridge._lockout.state != "locked"
        refused = [e for e in bridge.activity_log() if e["action"] == "estop_refused"]
        assert refused and refused[0]["detail"]["why"] == "retained"

        bridge._on_safety(_MqttSample("strands/safety/estop", body))
        assert bridge._lockout.state == "locked"


class _RetainedStore:
    """The ``iot`` and ``iot-data`` clients ``clear_retained_safety_messages`` reaches, holding retained topics."""

    def __init__(self, topics: list[str]) -> None:
        self.topics = topics
        self.published: list[dict[str, Any]] = []
        self.meta = SimpleNamespace(region_name="us-west-2")

    def describe_endpoint(self, endpointType: str) -> dict[str, str]:  # noqa: N803 - boto3 keyword
        return {"endpointAddress": "abc-ats.iot.us-west-2.amazonaws.com"}

    def list_retained_messages(self, maxResults: int, nextToken: str | None = None) -> dict[str, Any]:  # noqa: N803
        start = int(nextToken or 0)
        page: dict[str, Any] = {"retainedTopics": [{"topic": t} for t in self.topics[start : start + 2]]}
        if start + 2 < len(self.topics):
            page["nextToken"] = str(start + 2)
        return page

    def publish(self, **kw: Any) -> dict[str, Any]:
        self.published.append(kw)
        return {}


class TestClearingRetainedSafetyMessages:
    TOPICS = ["strands/arm-1/presence", "strands/safety/estop", "strands/arm-1/safety/event", "strands/safety/resume"]

    @pytest.fixture
    def store(self, monkeypatch: pytest.MonkeyPatch) -> _RetainedStore:
        store = _RetainedStore(list(self.TOPICS))
        monkeypatch.setattr(provision, "_require_boto3", lambda: SimpleNamespace(client=lambda *a, **k: store))
        return store

    def test_a_dry_run_names_the_fleet_safety_topics_and_publishes_nothing(self, store: _RetainedStore) -> None:
        report = provision.clear_retained_safety_messages()
        assert report.topics == ("strands/safety/estop", "strands/safety/resume")
        assert report.applied is False and store.published == []

    def test_apply_clears_each_with_a_zero_byte_retained_publish(self, store: _RetainedStore) -> None:
        provision.clear_retained_safety_messages(apply=True)
        assert store.published == [
            {"topic": t, "qos": 1, "retain": True, "payload": b""}
            for t in ("strands/safety/estop", "strands/safety/resume")
        ]

    def test_the_cli_verb_is_a_dry_run_unless_told_to_apply(
        self, store: _RetainedStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from strands_robots.mesh.iot.cli import main

        assert main(["clear-retained-safety"]) == 0
        assert "strands/safety/estop" in capsys.readouterr().out and store.published == []
        assert main(["clear-retained-safety", "--apply"]) == 0
        assert len(store.published) == 2
