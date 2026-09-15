# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A GR00T policy-server reply this client cannot read names the peer, not the codec.

The reference client (``gr00t.policy.server_client.PolicyClient``) refuses one
wrong-peer frame by name - a bare ``b"ERROR"`` gets "Make sure we are running the
correct policy server" - and reads the ``error`` field only off a reply that is a
map. These pin the general form of both: bytes that are not msgpack, and a value
that decodes but is neither a map nor the ``(action, info)`` list the reference
server sends, are refused in one class naming the peer and the request.

Only the second door is a silent one, and it is the reason this file exists:
``"error" in response`` is a membership test, so a string answers it ``False``
without raising and was returned as the declared ``dict``, to fail an
``AttributeError: 'str' object has no attribute 'items'`` later inside the
policy; an ``int`` raised ``TypeError`` from the test itself; and a string that
happens to contain ``"error"`` took the server-error branch and raised
``TypeError: string indices must be integers`` instead.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

msgpack = pytest.importorskip(
    "msgpack",
    reason="msgpack not installed - pip install 'strands-robots[groot-service]'",
)
pytest.importorskip(
    "zmq",
    reason="zmq not installed - pip install 'strands-robots[groot-service]'",
)

# E402: importorskip must execute before these imports to skip cleanly.
from strands_robots.policies.groot.client import (  # noqa: E402
    Gr00tInferenceClient,
    MsgSerializer,
)
from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402

_HOST = "127.0.0.1"
_PORT = 19997
_URI = f"tcp://{_HOST}:{_PORT}"

# Frames that are not one msgpack object. Every one of these reached the caller
# as a ``msgpack`` exception naming the codec and nothing else.
UNDECODABLE = [
    pytest.param(b"<html><body>502 Bad Gateway</body></html>", id="http-error-page"),
    pytest.param(b"", id="empty-frame"),
    pytest.param(b'{"action.single_arm": [[0.0]]}', id="json-text"),
    pytest.param(b"ERROR", id="legacy-error-sentinel"),
    pytest.param(msgpack.packb({"status": "ok"})[:1], id="truncated-map"),
    pytest.param(msgpack.packb({"status": "ok"}) + b"tail", id="trailing-bytes"),
]

# Values that decode perfectly and are neither a map nor a list. The reference
# server never sends one: every handler returns a dict, and ``get_action``
# returns a tuple, which msgpack carries as a list.
NOT_A_MAP_OR_LIST = [
    pytest.param(42, "int", id="int"),
    pytest.param("ok", "str", id="str"),
    pytest.param(None, "NoneType", id="nil"),
    pytest.param(True, "bool", id="bool"),
    pytest.param(1.5, "float", id="float"),
]

# Lists that are not ``(action, info)``: wrong length, or a first element that
# is not the action map. Pre-fix a 2-list returned its first element whatever
# it was, and any other list was returned whole as the action dict.
NOT_AN_ACTION_ENVELOPE = [
    pytest.param([1, 2], "['int', 'int']", id="two-scalars"),
    pytest.param([{"action.x": [[0.0]]}], "['dict']", id="one-element"),
    pytest.param([{"action.x": [[0.0]]}, {}, {}], "['dict', 'dict', 'dict']", id="three-elements"),
    pytest.param([], "[]", id="empty-list"),
]


def _client_answering(frame: bytes) -> Gr00tInferenceClient:
    """Return a client whose next recv yields *frame*, with no server involved."""
    client = Gr00tInferenceClient(host=_HOST, port=_PORT)
    client.socket.send = MagicMock()  # type: ignore[method-assign]
    client.socket.recv = MagicMock(return_value=frame)  # type: ignore[method-assign]
    return client


def _packed(value: object) -> bytes:
    return bytes(MsgSerializer.to_bytes(value))  # type: ignore[arg-type]


class TestAnUnreadableReplyNamesThePeer:
    @pytest.mark.parametrize("frame", UNDECODABLE)
    def test_bytes_that_are_not_msgpack_name_the_peer_and_the_request(self, frame: bytes) -> None:
        """The report carries the peer, the endpoint, the codec fault and the frame.

        The endpoint is what a caller with a ping, a reset and a get_action
        behind it needs to know; the frame preview is what lets an operator
        recognise an HTTP error page or JSON as the wrong wire format.
        """
        client = _client_answering(frame)
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("get_action", {})
        report = str(excinfo.value)
        assert "GR00T policy server" in report
        assert _URI in report
        assert "'get_action'" in report
        assert "not msgpack" in report
        assert repr(frame)[:24] in report
        assert "server_client.PolicyServer" in report
        # The codec failure is preserved as the cause, not discarded, and its own
        # words are quoted: "received extra data" and "incomplete input" are what
        # tell a trailing-byte frame from a truncated one, and dropping them
        # would make every frame in this table report identically.
        cause = excinfo.value.__cause__
        assert isinstance(cause, TypeError | ValueError)
        assert type(cause).__name__ in report
        assert str(cause) in report

    @pytest.mark.parametrize(("value", "type_name"), NOT_A_MAP_OR_LIST)
    def test_a_value_that_decodes_but_is_neither_a_map_nor_a_list_is_refused(
        self, value: object, type_name: str
    ) -> None:
        """A decodable scalar is refused in the same class as undecodable bytes.

        Pre-fix this was the silent door: the value was returned as the declared
        ``dict``, or raised a ``TypeError`` from the membership test, depending
        only on whether the type happened to be iterable.
        """
        client = _client_answering(_packed(value))
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("reset")
        report = str(excinfo.value)
        assert f"expected a msgpack map or list, got {type_name}" in report
        assert _URI in report
        assert "'reset'" in report

    def test_both_doors_report_in_one_class_and_name_the_same_exchange(self) -> None:
        """Undecodable bytes and a decodable scalar differ only in the problem.

        Either way the peer did not send a reply, so the two reports must agree
        on everything except the sentence naming what is wrong.
        """
        with pytest.raises(ConnectionError) as undecodable:
            _client_answering(b"<html>502</html>").call_endpoint("reset")
        with pytest.raises(ConnectionError) as scalar:
            _client_answering(_packed(42)).call_endpoint("reset")
        prefix = f"GR00T policy server at {_URI} answered 'reset' with an unreadable reply: "
        assert str(undecodable.value).startswith(prefix)
        assert str(scalar.value).startswith(prefix)
        assert "server_client.PolicyServer" in str(undecodable.value)
        assert "server_client.PolicyServer" in str(scalar.value)

    def test_the_endpoint_is_named_not_assumed(self) -> None:
        """The report names the request that came back unreadable, not a fixed one."""
        reports = {}
        for endpoint in ("ping", "get_action", "reset"):
            with pytest.raises(ConnectionError) as excinfo:
                _client_answering(_packed(42)).call_endpoint(endpoint)
            reports[endpoint] = str(excinfo.value)
        assert reports["ping"].replace("'ping'", "'get_action'") == reports["get_action"]
        assert reports["reset"].replace("'reset'", "'get_action'") == reports["get_action"]

    def test_a_string_reply_containing_error_is_refused_not_indexed(self) -> None:
        """A string is refused as unreadable rather than read as a server error.

        ``"error" in response`` is a substring test on a string, so a reply of
        ``"internal error"`` entered the server-error branch and then raised
        ``TypeError: string indices must be integers`` from ``response['error']``
        - a third report for one wire fault, naming neither the peer nor the
        endpoint.
        """
        client = _client_answering(_packed("internal error"))
        with pytest.raises(ConnectionError) as excinfo:
            client.call_endpoint("get_action", {})
        report = str(excinfo.value)
        assert "expected a msgpack map or list, got str" in report
        assert "string indices" not in report
        assert "Server error" not in report


class TestAnActionReplyThatIsNotTheEnvelopeNamesThePeer:
    """``get_action`` admits ``(action, info)`` and a bare action map, and nothing else."""

    @pytest.mark.parametrize(("value", "element_types"), NOT_AN_ACTION_ENVELOPE)
    def test_a_list_that_is_not_action_info_is_refused(self, value: list[object], element_types: str) -> None:
        """A list of the wrong shape is refused naming the peer and the shape.

        Pre-fix ``[1, 2]`` returned ``1`` as the action dict and every other
        length returned the list itself, each to fail on ``.items()`` one frame
        later with a report naming a Python type and not the server.
        """
        client = _client_answering(_packed(value))
        with pytest.raises(ConnectionError) as excinfo:
            client.get_action({"state": np.zeros(1)})
        report = str(excinfo.value)
        assert "GR00T policy server" in report
        assert _URI in report
        assert "'get_action'" in report
        assert f"got a list of {len(value)} whose elements are {element_types}" in report
        assert "server_client.PolicyServer" in report

    def test_the_report_names_element_types_rather_than_rendering_the_elements(self) -> None:
        """An array in a malformed envelope is named by type, not printed."""
        client = _client_answering(_packed([np.zeros((64, 64, 3)), {}, {}]))
        with pytest.raises(ConnectionError) as excinfo:
            client.get_action({"state": np.zeros(1)})
        report = str(excinfo.value)
        assert "['ndarray', 'dict', 'dict']" in report
        assert "0., 0., 0." not in report


class TestTheGradedReplyLeavesTheRestOfTheContractAlone:
    def test_a_map_reply_still_round_trips(self) -> None:
        """The healthy path is unchanged: a map is returned as it decoded."""
        client = _client_answering(_packed({"status": "ok", "message": "Server is running"}))
        assert client.call_endpoint("ping") == {"status": "ok", "message": "Server is running"}

    def test_a_server_error_map_is_still_a_runtime_error(self) -> None:
        """A readable reply carrying ``error`` keeps its own report and type."""
        client = _client_answering(_packed({"error": "Unauthorized: Invalid API token"}))
        with pytest.raises(RuntimeError, match="Server error: Unauthorized") as excinfo:
            client.call_endpoint("get_action", {})
        assert not isinstance(excinfo.value, ConnectionError)

    def test_the_action_info_list_still_unpacks_to_the_action(self) -> None:
        """The reference server's ``(action, info)`` shape is unchanged."""
        action = {"action.single_arm": np.ones((1, 16, 5))}
        client = _client_answering(_packed((action, {})))
        got = client.get_action({"state": np.zeros(5)})
        assert set(got) == {"action.single_arm"}
        np.testing.assert_array_equal(got["action.single_arm"], action["action.single_arm"])

    def test_a_bare_action_map_still_returns_as_is(self) -> None:
        """The legacy bare-dict shape is unchanged."""
        action = {"action.single_arm": np.ones((1, 16, 5))}
        got = _client_answering(_packed(action)).get_action({"state": np.zeros(5)})
        assert set(got) == {"action.single_arm"}

    def test_a_list_reply_to_a_non_action_endpoint_passes_the_error_test_without_raising(self) -> None:
        """A list reply is not indexed for ``error``; only a map carries that field.

        This is the reference client's own guard (``isinstance(response, dict)
        and "error" in response``), and it is what lets ``(action, info)`` through
        ``call_endpoint`` unrefused.
        """
        assert _client_answering(_packed([{"x": 1}, {}])).call_endpoint("get_action", {}) == [{"x": 1}, {}]

    def test_ping_still_reports_unreachable_instead_of_raising(self) -> None:
        """``ping`` absorbs the refusal - any failure means "not reachable"."""
        assert _client_answering(b"<html>502</html>").ping() is False
        assert _client_answering(_packed(42)).ping() is False

    def test_get_actions_names_the_request_rather_than_an_attribute_error(self) -> None:
        """Through the policy, a string reply names the get_action request it answered.

        Pre-fix ``get_actions`` failed with ``AttributeError: 'str' object has no
        attribute 'items'`` from ``_unpack_service_actions``, which names the
        Python type of a decoded frame and nothing about the server.
        """
        policy = Gr00tPolicy(data_config="so100", host=_HOST, port=_PORT)
        assert policy._client is not None
        policy._client.socket.send = MagicMock()  # type: ignore[method-assign]
        policy._client.socket.recv = MagicMock(return_value=_packed("ok"))  # type: ignore[method-assign]
        with pytest.raises(ConnectionError) as excinfo:
            asyncio.run(policy.get_actions({"webcam": np.zeros((64, 64, 3), dtype=np.uint8)}, "t"))
        report = str(excinfo.value)
        assert "'get_action'" in report
        assert _URI in report

    def test_reset_still_continues_and_logs_the_peer(self, caplog: pytest.LogCaptureFixture) -> None:
        """``Gr00tPolicy.reset`` is best-effort: it logs the refusal and continues.

        What changes is the sentence it logs - the peer and the request, where it
        used to log ``unpack(b) received extra data.`` from the codec.
        """
        policy = Gr00tPolicy(data_config="so100", host=_HOST, port=_PORT)
        assert policy._client is not None
        policy._client.socket.send = MagicMock()  # type: ignore[method-assign]
        policy._client.socket.recv = MagicMock(return_value=b"<html>502</html>")  # type: ignore[method-assign]
        with caplog.at_level("INFO", logger="strands_robots.policies.groot.policy"):
            policy.reset(seed=7)
        assert _URI in caplog.text
        assert "'reset'" in caplog.text


class TestTheSerializerNoLongerPromisesAMap:
    """``from_bytes`` is a codec: it decodes whatever the bytes encode."""

    @pytest.mark.parametrize(("value", "type_name"), NOT_A_MAP_OR_LIST)
    def test_from_bytes_returns_the_value_the_bytes_encode(self, value: object, type_name: str) -> None:
        decoded = MsgSerializer.from_bytes(_packed(value))
        assert decoded == value
        assert type(decoded).__name__ == type_name

    def test_a_request_the_client_packed_still_decodes_to_its_map(self) -> None:
        """The round-trip the tests and the server rely on is untouched."""
        request = {"endpoint": "get_action", "data": {"observation": {"state": [0.0]}, "options": None}}
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(request)) == request
