# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An array in a GR00T policy-server reply is decoded, not returned as its envelope.

The two directions of this ZMQ wire are not symmetric.
``gr00t.policy.server_client.MsgSerializer`` decodes both envelopes - the
``__ndarray_class__`` / ``as_npy`` pair this client packs and its own - so a
request is read by the reference server either way. A reply, though, is always
packed by ``msgpack_numpy.encode``: a map carrying ``nd`` / ``type`` / ``kind``
/ ``shape`` / ``data``.

Decoding only our own envelope left every array in a reply as that raw map. A
well-formed ``get_action`` chunk of ``(1, 16, 5)`` float32 therefore reached
MODULE strands_robots.policies.groot.policy as a dict, where ``np.asarray``
made it a 0-D object array and ``_unpack_service_actions`` refused it with
``scalar (0-D) action value(s) ... The action chunk is malformed`` - a report
blaming the model for a chunk this client never decoded, on every request to
every reference server.

The envelope is written out here rather than built with ``msgpack_numpy``
(which is the reference server's dependency, not ours) so a change in that wire
form shows up as a loud failure in this file.
"""

from __future__ import annotations

from typing import Any
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
from strands_robots.policies.groot.client import Gr00tInferenceClient  # noqa: E402
from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402

_HOST = "127.0.0.1"
_PORT = 19996


def _envelope(array: np.ndarray) -> dict[bytes, Any]:
    """The array as ``msgpack_numpy`` 0.4.8's ``encode`` wires a non-object dtype."""
    return {
        b"nd": True,
        b"type": array.dtype.str,
        b"kind": b"",
        b"shape": list(array.shape),
        b"data": array.tobytes(),
    }


def _client_answering(reply: object) -> Gr00tInferenceClient:
    """A client whose socket answers one packed *reply*."""
    client = Gr00tInferenceClient(host=_HOST, port=_PORT)
    socket = MagicMock()
    socket.recv.return_value = msgpack.packb(reply, default=lambda o: o)
    client.socket = socket
    return client


class TestAReplyArrayDecodesFromTheReferenceEnvelope:
    """The reference server's own encoding is read back as the array it holds."""

    @pytest.mark.parametrize(
        "array",
        [
            pytest.param(np.arange(80, dtype=np.float32).reshape(1, 16, 5), id="action-chunk-f32"),
            pytest.param(np.arange(16, dtype=np.float32).reshape(1, 16, 1), id="gripper-column-f32"),
            pytest.param(np.arange(6, dtype=np.float64).reshape(2, 3), id="f64"),
            pytest.param(np.arange(4, dtype=np.int64).reshape(4, 1), id="i64"),
            pytest.param(np.zeros((1, 1, 0), dtype=np.float32), id="empty-trailing-axis"),
        ],
    )
    def test_the_array_arrives_with_its_shape_dtype_and_values(self, array: np.ndarray) -> None:
        client = _client_answering([{"single_arm": _envelope(array)}, {}])

        got = client.get_action({"video": {}, "state": {}, "language": {}})["single_arm"]

        assert isinstance(got, np.ndarray)
        assert got.shape == array.shape
        assert got.dtype == array.dtype
        np.testing.assert_array_equal(got, array)
        # Writable, like the ``as_npy`` path, so a caller can normalise in place.
        assert got.flags.writeable

    def test_a_numpy_scalar_arrives_as_a_scalar(self) -> None:
        """``encode`` wires a ``np.generic`` with ``nd`` false and no shape."""
        client = _client_answering([{"step": {b"nd": False, b"type": "<i8", b"data": np.int64(7).tobytes()}}, {}])

        assert client.get_action({})["step"] == 7

    def test_the_chunk_the_server_sends_unpacks_into_one_dict_per_step(self) -> None:
        """The frame that read as ``scalar (0-D) action value(s) ... malformed``."""
        chunk = {
            "single_arm": _envelope(np.arange(80, dtype=np.float32).reshape(1, 16, 5)),
            "gripper": _envelope(np.arange(16, dtype=np.float32).reshape(1, 16, 1)),
        }
        client = _client_answering([chunk, {}])
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy._action_mapping = None

        steps = policy._unpack_service_actions(client.get_action({}))

        assert len(steps) == 16
        assert steps[0]["single_arm"] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert steps[15]["gripper"] == [15.0]


class TestAnEnvelopeThatCannotHoldAnArrayNamesThePeer:
    """A pickle-bearing or malformed envelope is refused where it is decoded."""

    @pytest.mark.parametrize(
        ("envelope", "expected"),
        [
            pytest.param(
                {b"nd": True, b"type": "|O8", b"kind": b"O", b"shape": [1], b"data": b"\x80\x04"},
                "pickle-bearing",
                id="object-kind",
            ),
            pytest.param(
                {b"nd": True, b"type": "|O8", b"kind": b"", b"shape": [1], b"data": b"\x80\x04"},
                "only pickle can read",
                id="object-dtype-without-the-kind-marker",
            ),
            pytest.param(
                {b"nd": True, b"type": "<f4", b"kind": b"", b"shape": [1], b"data": "not-bytes"},
                "do not describe an array",
                id="data-is-not-bytes",
            ),
            pytest.param(
                {b"nd": True, b"type": 4, b"kind": b"", b"shape": [1], b"data": b"\x00\x00\x00\x00"},
                "do not describe an array",
                id="type-is-not-a-dtype-string",
            ),
            pytest.param(
                {b"nd": True, b"type": "<f4", b"kind": b"", b"shape": "16", b"data": b"\x00\x00\x00\x00"},
                "is not a sequence of lengths",
                id="shape-is-not-a-sequence",
            ),
        ],
    )
    def test_the_report_names_the_peer_the_endpoint_and_the_reason(
        self, envelope: dict[bytes, Any], expected: str
    ) -> None:
        client = _client_answering([{"single_arm": envelope}, {}])

        with pytest.raises(ConnectionError) as caught:
            client.get_action({})

        text = str(caught.value)
        assert expected in text
        assert f"tcp://{_HOST}:{_PORT}" in text
        assert "'get_action'" in text
