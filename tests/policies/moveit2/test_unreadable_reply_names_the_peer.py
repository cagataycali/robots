# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MoveIt2 sidecar reply this client cannot read names the peer, not the codec.

The reference sidecar grades the request direction at two doors and refuses both
in one class - bytes that are not msgpack, and a value that decodes but is not a
map - because "either way the peer did not send a request"
(MODULE strands_robots.policies.moveit2.server.zmq_node). These pin the same two
doors in the reply direction.

Only the second door is a silent one, and it is the reason this file exists:
``"error" in reply`` is a membership test, so a list and a string answer it
``False`` without raising and the non-map was returned as the declared
``dict[str, Any]``, to fail an ``AttributeError`` later inside the policy; a
string that happens to contain ``"error"`` took the server-error branch and
raised ``TypeError: string indices must be integers`` instead.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

msgpack = pytest.importorskip(
    "msgpack",
    reason="msgpack not installed - pip install 'strands-robots[moveit2]'",
)
pytest.importorskip(
    "zmq",
    reason="zmq not installed - pip install 'strands-robots[moveit2]'",
)

# E402: importorskip must execute before these imports to skip cleanly.
from strands_robots.policies.moveit2 import (  # noqa: E402
    MoveIt2InferenceClient,
    MoveIt2Policy,
    MsgSerializer,
)

_HOST = "127.0.0.1"
_PORT = 19998
_URI = f"tcp://{_HOST}:{_PORT}"

# Frames that are not one msgpack object. Every one of these reached the caller
# as a ``msgpack`` exception naming the codec and nothing else.
UNDECODABLE = [
    pytest.param(b"<html><body>502 Bad Gateway</body></html>", id="http-error-page"),
    pytest.param(b"", id="empty-frame"),
    pytest.param(b'{"trajectory": [], "success": true}', id="json-text"),
    pytest.param(msgpack.packb({"status": "ok"}, use_bin_type=True)[:1], id="truncated-map"),
    pytest.param(msgpack.packb({"status": "ok"}, use_bin_type=True) + b"tail", id="trailing-bytes"),
]

# Values that decode perfectly and are not a request/reply map. The sidecar
# refuses exactly this class in the request direction.
NOT_A_MAP = [
    pytest.param(42, "int", id="int"),
    pytest.param("ok", "str", id="str"),
    pytest.param([1, 2], "list", id="list"),
    pytest.param(None, "NoneType", id="nil"),
    pytest.param(True, "bool", id="bool"),
]


def _client_answering(frame: bytes) -> MoveIt2InferenceClient:
    """Return a client whose next recv yields *frame*, with no server involved."""
    client = MoveIt2InferenceClient(host=_HOST, port=_PORT)
    client.socket.send = MagicMock()  # type: ignore[method-assign]
    client.socket.recv = MagicMock(return_value=frame)  # type: ignore[method-assign]
    return client


def _packed(value: object) -> bytes:
    return bytes(msgpack.packb(value, use_bin_type=True))


class TestAnUnreadableReplyNamesThePeer:
    @pytest.mark.parametrize("frame", UNDECODABLE)
    def test_bytes_that_are_not_msgpack_name_the_peer_and_the_request(self, frame: bytes) -> None:
        """The report carries the peer, the endpoint, the codec fault and the frame.

        The endpoint is what a caller with a ping, a reset and a plan behind it
        needs to know; the frame preview is what lets an operator recognise an
        HTTP error page or JSON as the wrong wire format.
        """
        client = _client_answering(frame)
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("plan", {})
        report = str(excinfo.value)
        assert "MoveIt2 sidecar" in report
        assert _URI in report
        assert "'plan'" in report
        assert "not msgpack" in report
        assert repr(frame)[:24] in report
        assert "zmq_node" in report
        # The codec failure is preserved as the cause, not discarded, and its own
        # words are quoted: "received extra data" and "incomplete input" are what
        # tell a trailing-byte frame from a truncated one, and dropping them
        # would make every frame in this table report identically.
        cause = excinfo.value.__cause__
        assert isinstance(cause, TypeError | ValueError)
        assert type(cause).__name__ in report
        assert str(cause) in report

    @pytest.mark.parametrize(("value", "type_name"), NOT_A_MAP)
    def test_a_value_that_decodes_but_is_not_a_map_is_refused(self, value: object, type_name: str) -> None:
        """A decodable non-map is refused in the same class as undecodable bytes.

        Pre-fix this was the silent door: the value was returned as the declared
        ``dict``, or raised a ``TypeError`` from the membership test, depending
        only on whether the type happened to be iterable.
        """
        client = _client_answering(_packed(value))
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("plan", {})
        report = str(excinfo.value)
        assert f"expected a msgpack map, got {type_name}" in report
        assert _URI in report
        assert "'plan'" in report

    def test_both_doors_report_in_one_class_and_name_the_same_exchange(self) -> None:
        """Undecodable bytes and a decodable non-map differ only in the problem.

        This is the sidecar's own rule for the request direction - either way the
        peer did not send a reply - so the two reports must agree on everything
        except the sentence naming what is wrong.
        """
        with pytest.raises(ConnectionError) as undecodable:
            _client_answering(b"<html>502</html>").call_endpoint("reset")
        with pytest.raises(ConnectionError) as non_map:
            _client_answering(_packed([1, 2])).call_endpoint("reset")
        prefix = f"MoveIt2 sidecar at {_URI} answered 'reset' with an unreadable reply: "
        assert str(undecodable.value).startswith(prefix)
        assert str(non_map.value).startswith(prefix)
        assert "zmq_node" in str(undecodable.value)
        assert "zmq_node" in str(non_map.value)

    def test_the_endpoint_is_named_not_assumed(self) -> None:
        """The report names the request that came back unreadable, not a fixed one."""
        reports = {}
        for endpoint in ("ping", "plan", "reset"):
            with pytest.raises(ConnectionError) as excinfo:
                _client_answering(_packed(42)).call_endpoint(endpoint)
            reports[endpoint] = str(excinfo.value)
        assert reports["ping"].replace("'ping'", "'plan'") == reports["plan"]
        assert reports["reset"].replace("'reset'", "'plan'") == reports["plan"]

    def test_a_string_reply_containing_error_is_refused_not_indexed(self) -> None:
        """A string is refused as unreadable rather than read as a server error.

        ``"error" in reply`` is a substring test on a string, so a reply of
        ``"internal error"`` entered the server-error branch and then raised
        ``TypeError: string indices must be integers`` from ``reply['error']`` -
        a third report for one wire fault, naming neither the peer nor the
        endpoint.
        """
        client = _client_answering(_packed("internal error"))
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("plan", {})
        report = str(excinfo.value)
        assert "expected a msgpack map, got str" in report
        assert "string indices" not in report
        assert "Server error" not in report


class TestTheGradedReplyLeavesTheRestOfTheContractAlone:
    def test_a_map_reply_still_round_trips(self) -> None:
        """The healthy path is unchanged: a map is returned as it decoded."""
        client = _client_answering(_packed({"status": "ok", "success": True}))
        assert client.call_endpoint("ping") == {"status": "ok", "success": True}

    def test_a_server_error_map_is_still_a_runtime_error(self) -> None:
        """A readable reply carrying ``error`` keeps its own report and type."""
        client = _client_answering(_packed({"error": "no plan found"}))
        with pytest.raises(RuntimeError, match="Server error: no plan found") as excinfo:
            client.call_endpoint("plan", {})
        assert not isinstance(excinfo.value, ConnectionError)

    def test_ping_still_reports_unreachable_instead_of_raising(self) -> None:
        """``ping`` absorbs the refusal - any failure means "not reachable"."""
        assert _client_answering(b"<html>502</html>").ping() is False
        assert _client_answering(_packed(42)).ping() is False

    def test_predict_names_the_endpoint_rather_than_an_attribute_error(self) -> None:
        """Through the policy, a non-map reply names the plan request it answered.

        Pre-fix ``get_actions`` failed with ``AttributeError: 'str' object has no
        attribute 'get'`` from ``response.get("status", ...)``, which names the
        Python type of a decoded frame and nothing about the sidecar.
        """
        policy = MoveIt2Policy(host=_HOST, port=_PORT, planning_group="arm")
        policy._client.socket.send = MagicMock()  # type: ignore[method-assign]
        policy._client.socket.recv = MagicMock(return_value=_packed("ok"))  # type: ignore[method-assign]
        with pytest.raises(ConnectionError) as excinfo:
            asyncio.run(
                policy.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": 0.5},
                )
            )
        report = str(excinfo.value)
        assert "'plan'" in report
        assert _URI in report


class TestTheSerializerNoLongerPromisesAMap:
    """``from_bytes`` is a codec: it decodes whatever the bytes encode."""

    @pytest.mark.parametrize(("value", "type_name"), NOT_A_MAP)
    def test_from_bytes_returns_the_value_the_bytes_encode(self, value: object, type_name: str) -> None:
        decoded = MsgSerializer.from_bytes(_packed(value))
        assert decoded == value
        assert type(decoded).__name__ == type_name

    def test_a_request_the_client_packed_still_decodes_to_its_map(self) -> None:
        """The round-trip the tests and the sidecar rely on is untouched."""
        request = {"endpoint": "plan", "data": {"planning_group": "arm"}}
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(request)) == request
