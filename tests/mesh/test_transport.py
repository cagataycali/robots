"""Unit tests for the MeshTransport abstraction layer.

These tests do NOT touch real AWS or Zenoh — they verify protocol shape,
wildcard translation, QoS lookup, topic-filter matching, and ZenohTransport
delegation. The real AWS-backed integration test lives in
``tests_integ/test_iot_transport.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.transport import (
    IotMqttTransport,
    MeshTransport,
    Sample,
    ZenohTransport,
)
from strands_robots.mesh.transport.iot_transport import (
    _mqtt_topic_matches,
    _MqttSample,
    _qos_and_retain_for,
    _should_drop,
    _zenoh_to_mqtt_filter,
)

# Protocol shape


class TestProtocolShape:
    """Both transports satisfy the runtime-checkable Protocol."""

    def test_zenoh_satisfies_protocol(self):
        """ZenohTransport should satisfy the MeshTransport protocol."""
        t = ZenohTransport()
        assert isinstance(t, MeshTransport)

    def test_iot_satisfies_protocol(self):
        """IotMqttTransport should satisfy the MeshTransport protocol."""
        t = IotMqttTransport(thing_name="test", endpoint="x.iot.us-west-2.amazonaws.com")
        assert isinstance(t, MeshTransport)

    def test_mqtt_sample_satisfies_sample_protocol(self):
        """_MqttSample exposes .key_expr and .payload.to_bytes()."""
        s = _MqttSample("strands/foo/state", b'{"k":1}')
        assert isinstance(s, Sample)
        assert s.key_expr == "strands/foo/state"
        assert s.payload.to_bytes() == b'{"k":1}'


# Wildcard translation (Zenoh -> MQTT)


class TestZenohToMqttFilter:
    """Zenoh key-expression syntax → MQTT topic-filter syntax."""

    @pytest.mark.parametrize(
        "zenoh,mqtt",
        [
            # Concrete patterns Mesh actually uses
            ("strands/*/presence", "strands/+/presence"),
            ("strands/{peer}/cmd", "strands/{peer}/cmd"),
            ("strands/{peer}/response/**", "strands/{peer}/response/#"),
            ("strands/broadcast", "strands/broadcast"),
            # Edge cases
            ("strands/*/*/state", "strands/+/+/state"),
            ("**", "#"),
            ("*", "+"),
            ("a/b/c", "a/b/c"),
        ],
    )
    def test_translation(self, zenoh, mqtt):
        assert _zenoh_to_mqtt_filter(zenoh) == mqtt


# Per-topic QoS lookup


class TestTopicPolicy:
    """QoS / retain / drop defaults for our topic scheme."""

    def test_presence_is_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/presence")
        assert qos == 1
        assert retain is True

    def test_state_is_qos0_no_retain(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/state")
        assert qos == 0
        assert retain is False

    def test_cmd_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/cmd")
        assert qos == 1
        assert retain is False

    def test_response_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/response/abc123def")
        assert qos == 1

    def test_broadcast_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/broadcast")
        assert qos == 1

    def test_health_is_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/health")
        assert retain is True

    def test_safety_event_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/safety/event")
        assert qos == 1
        assert retain is True

    def test_safety_estop_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/safety/estop")
        assert qos == 1
        assert retain is True

    def test_lidar_summary(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/lidar/summary")
        assert qos == 0
        assert retain is False

    def test_lidar_state_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/lidar/state")
        assert retain is True

    def test_unknown_topic_default_qos0_no_retain(self):
        qos, retain = _qos_and_retain_for("strands/peer/somethingnew")
        assert qos == 0
        assert retain is False

    def test_camera_returns_drop(self):
        qos, retain = _qos_and_retain_for("strands/peer/camera/wrist")
        assert qos == -1  # explicit DROP


# Drop list (LAN-only topics)


class TestShouldDrop:
    @pytest.mark.parametrize(
        "topic,expected",
        [
            ("strands/peer/camera/wrist", True),
            ("strands/peer/camera/front", True),
            ("strands/peer/input/leader", True),
            ("strands/peer/input/gamepad", True),
            ("strands/peer/hand/right/state", True),
            ("strands/peer/presence", False),
            ("strands/peer/state", False),
            ("strands/peer/cmd", False),
            ("strands/peer/response/abc", False),
        ],
    )
    def test_should_drop(self, topic, expected):
        assert _should_drop(topic) is expected


# MQTT topic-filter matching


class TestMqttMatcher:
    """Standard MQTT v5 wildcard semantics."""

    @pytest.mark.parametrize(
        "filter_,topic,expected",
        [
            # Exact
            ("strands/peer1/cmd", "strands/peer1/cmd", True),
            ("strands/peer1/cmd", "strands/peer2/cmd", False),
            # + matches one segment
            ("strands/+/presence", "strands/peer1/presence", True),
            ("strands/+/presence", "strands/peer2/presence", True),
            ("strands/+/presence", "strands/peer1/state", False),
            ("strands/+/presence", "strands/a/b/presence", False),
            # # matches tail (zero or more)
            ("strands/peer/response/#", "strands/peer/response/abc", True),
            ("strands/peer/response/#", "strands/peer/response/a/b/c", True),
            ("strands/peer/response/#", "strands/peer/response", True),
            ("strands/peer/response/#", "strands/peer/cmd", False),
            # Different lengths
            ("strands/peer/cmd", "strands/peer/cmd/extra", False),
            ("strands/peer/cmd/extra", "strands/peer/cmd", False),
        ],
    )
    def test_match(self, filter_, topic, expected):
        assert _mqtt_topic_matches(filter_, topic) is expected


# IotMqttTransport — no live broker


class TestIotMqttTransportConfig:
    """Transport config validation without touching a live broker."""

    def test_missing_thing_name_returns_false(self, monkeypatch):
        monkeypatch.delenv("STRANDS_IOT_THING_NAME", raising=False)
        monkeypatch.delenv("STRANDS_IOT_ENDPOINT", raising=False)
        t = IotMqttTransport()
        assert t.connect() is False

    def test_missing_endpoint_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_IOT_THING_NAME", "test-thing")
        monkeypatch.delenv("STRANDS_IOT_ENDPOINT", raising=False)
        monkeypatch.setenv("STRANDS_IOT_CERT_DIR", str(tmp_path))
        t = IotMqttTransport()
        assert t.connect() is False

    def test_missing_cert_files_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_IOT_THING_NAME", "test-thing")
        monkeypatch.setenv("STRANDS_IOT_ENDPOINT", "x.iot.us-west-2.amazonaws.com")
        monkeypatch.setenv("STRANDS_IOT_CERT_DIR", str(tmp_path))
        # No cert files in tmp_path
        t = IotMqttTransport()
        assert t.connect() is False

    def test_thing_name_property(self):
        t = IotMqttTransport(
            thing_name="so100-spike-01",
            endpoint="x.iot.us-west-2.amazonaws.com",
        )
        assert t.thing_name == "so100-spike-01"

    def test_put_no_op_when_disconnected(self):
        """put() must NOT raise even when the client is None."""
        t = IotMqttTransport(thing_name="test", endpoint="x")
        # Should not raise
        t.put("strands/test/state", {"k": 1})

    def test_close_idempotent(self):
        """close() is safe to call before connect() and twice in a row."""
        t = IotMqttTransport(thing_name="test", endpoint="x")
        t.close()  # before connect
        t.close()  # double-close

    def test_is_alive_false_when_not_connected(self):
        t = IotMqttTransport(thing_name="test", endpoint="x")
        assert t.is_alive() is False


# ZenohTransport — delegating to mesh.session


class TestZenohTransportDelegation:
    """ZenohTransport is a thin wrapper over mesh.session."""

    def test_satisfies_protocol(self):
        t = ZenohTransport()
        assert isinstance(t, MeshTransport)

    def test_close_before_connect_is_safe(self):
        t = ZenohTransport()
        t.close()  # No-op, no exception

    def test_put_no_op_when_disconnected(self):
        """put() must NOT raise when no session exists."""
        # Make sure no session is open from a prior test
        from strands_robots.mesh import session as sess_mod

        with sess_mod._SESSION_LOCK:
            if sess_mod._SESSION is not None:
                try:
                    sess_mod._SESSION.close()
                except Exception:
                    pass
                sess_mod._SESSION = None
                sess_mod._SESSION_REFS = 0
        t = ZenohTransport()
        # Should be safe — delegates to session.put which is a no-op.
        t.put("strands/test/state", {"k": 1})

    def test_is_alive_false_when_no_session(self):
        from strands_robots.mesh import session as sess_mod

        with sess_mod._SESSION_LOCK:
            if sess_mod._SESSION is not None:
                try:
                    sess_mod._SESSION.close()
                except Exception:
                    pass
                sess_mod._SESSION = None
                sess_mod._SESSION_REFS = 0
        t = ZenohTransport()
        assert t.is_alive() is False

    def test_connect_pre_seeds_session_then_close_releases(self):
        """When session is pre-seeded, connect() takes a ref; close() releases it."""
        from strands_robots.mesh import session as sess_mod

        # Pre-seed: simulate "session already open with 0 refs" — this is
        # the state right after a get_session→release_session cycle would
        # leave it if it didn't auto-close. We construct it manually here
        # because we don't want this test to require zenoh as a hard dep.
        mock_session = MagicMock()
        with sess_mod._SESSION_LOCK:
            # Save state
            saved_session = sess_mod._SESSION
            saved_refs = sess_mod._SESSION_REFS
            sess_mod._SESSION = mock_session
            sess_mod._SESSION_REFS = 1  # already-open singleton
        try:
            t = ZenohTransport()
            ok = t.connect()
            assert ok is True
            assert t.is_alive() is True
            assert sess_mod._SESSION is mock_session
            assert sess_mod._SESSION_REFS == 2  # transport added one ref

            # Second connect on same instance: no-op (we already hold the ref)
            assert t.connect() is True
            assert sess_mod._SESSION_REFS == 2

            t.close()
            assert sess_mod._SESSION_REFS == 1  # transport released its one ref
            assert sess_mod._SESSION is mock_session  # still open

            # Double close: idempotent
            t.close()
            assert sess_mod._SESSION_REFS == 1
        finally:
            with sess_mod._SESSION_LOCK:
                sess_mod._SESSION = saved_session
                sess_mod._SESSION_REFS = saved_refs
