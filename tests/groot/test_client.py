"""Tests for strands_robots.policies.groot.client — ZMQ serialization and client.

Covers: MsgSerializer roundtrips, Gr00tInferenceClient construction, api_token
handling, and error paths.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from strands_robots.policies.groot.client import Gr00tInferenceClient, MsgSerializer
from strands_robots.policies.groot.data_config import ModalityConfig

# ---------------------------------------------------------------------------
# MsgSerializer
# ---------------------------------------------------------------------------


class TestMsgSerializer:
    def test_roundtrip_plain_dict(self):
        data = {"endpoint": "ping", "data": {"foo": 42}}
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(data)) == data

    def test_roundtrip_numpy_array(self):
        array = np.random.rand(3, 4).astype(np.float32)
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"obs": array}))
        np.testing.assert_array_almost_equal(result["obs"], array)

    def test_roundtrip_numpy_uint8(self):
        array = np.random.randint(0, 255, (2, 3, 3), dtype=np.uint8)
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"img": array}))
        np.testing.assert_array_equal(result["img"], array)

    def test_roundtrip_modality_config(self):
        config = ModalityConfig(delta_indices=[0, 1], modality_keys=["state.arm"])
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"config": config}))
        decoded = result["config"]
        assert isinstance(decoded, ModalityConfig)
        assert decoded.delta_indices == [0, 1]
        assert decoded.modality_keys == ["state.arm"]

    def test_roundtrip_nested_arrays(self):
        data = {
            "video": np.zeros((1, 1, 64, 64, 3), dtype=np.uint8),
            "state": np.ones((1, 1, 6), dtype=np.float32),
        }
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(data))
        assert result["video"].shape == (1, 1, 64, 64, 3)
        assert result["state"].dtype == np.float32

    def test_encode_non_custom_returns_as_is(self):
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(data))
        assert result["key"] == "value"
        assert result["num"] == 42


# ---------------------------------------------------------------------------
# Gr00tInferenceClient — construction & api_token
# ---------------------------------------------------------------------------


class TestGr00tInferenceClient:
    def test_construction_defaults(self):
        client = Gr00tInferenceClient(host="localhost", port=5555)
        assert client.host == "localhost"
        assert client.port == 5555
        assert client.timeout_ms == 15000
        assert client.api_token is None

    def test_construction_with_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=5555, api_token="secret")
        assert client.api_token == "secret"

    def test_api_token_warning_on_remote_host(self, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.groot.client"):
            Gr00tInferenceClient(host="10.0.0.1", port=5555, api_token="tok")
        assert any("plaintext" in record.message for record in caplog.records)

    def test_no_warning_for_localhost_token(self, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.groot.client"):
            Gr00tInferenceClient(host="localhost", port=5555, api_token="tok")
        assert not any("plaintext" in record.message for record in caplog.records)

    def test_custom_timeout(self):
        client = Gr00tInferenceClient(host="localhost", port=5555, timeout_ms=5000)
        assert client.timeout_ms == 5000

    def test_call_endpoint_includes_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=9999, api_token="mytoken")
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert len(sent_data) == 1
        assert sent_data[0]["api_token"] == "mytoken"

    def test_call_endpoint_without_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert "api_token" not in sent_data[0]

    def test_call_endpoint_raises_on_server_error(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"error": "bad input"}))
        with pytest.raises(RuntimeError, match="Server error: bad input"):
            client.call_endpoint("get_action", {"obs": {}})

    def test_ping_returns_false_on_failure(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.send = MagicMock(side_effect=Exception("timeout"))
        assert client.ping() is False

    def test_ping_does_not_reconnect(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        original_socket = client.socket
        client.socket.send = MagicMock(side_effect=Exception("timeout"))
        client.ping()
        assert client.socket is original_socket

    def test_get_action_calls_endpoint(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        action = {"single_arm": np.zeros((1, 16, 5), dtype=np.float32)}
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes(action))
        result = client.get_action({"some": "obs"})
        assert "single_arm" in result

    def test_call_endpoint_data_none_omits_data_key(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert "data" not in sent_data[0]

    def test_call_endpoint_data_present_includes_data_key(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("get_action", {"obs": "test"})
        assert "data" in sent_data[0]
        assert sent_data[0]["data"] == {"obs": "test"}


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


class TestZmqDeps:
    def test_require_optional_loads_zmq(self):
        from strands_robots.utils import require_optional

        zmq = require_optional("zmq", pip_install="pyzmq")
        assert hasattr(zmq, "Context")

    def test_require_optional_loads_msgpack(self):
        from strands_robots.utils import require_optional

        msgpack = require_optional("msgpack")
        assert hasattr(msgpack, "packb")
