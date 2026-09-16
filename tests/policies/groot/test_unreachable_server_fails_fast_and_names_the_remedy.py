"""A GR00T request that times out says where it dialled, what expired and what to do.

Pre-fix ``call_endpoint`` let ``zmq.Again`` escape raw, so the policy runner
printed ``Policy failed: Resource temporarily unavailable`` - no host, no port,
no remedy - and ``Gr00tPolicy`` dropped ``timeout_ms`` into ``**kwargs``, so a
caller lowering the budget still waited the 15 s default on a dead port.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("zmq", reason="pyzmq is optional (groot extra)")
pytest.importorskip("msgpack", reason="msgpack is optional (groot extra)")

from strands_robots.policies.groot.client import (  # noqa: E402
    Gr00tInferenceClient,
    unreachable_server_error,
)
from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402

DEAD_PORT = 9  # discard; nothing listens there


def test_a_dead_port_is_reported_within_the_budget_with_uri_and_remedy():
    client = Gr00tInferenceClient(host="127.0.0.1", port=DEAD_PORT, timeout_ms=300)
    t0 = time.monotonic()
    with pytest.raises(ConnectionError) as info:
        client.call_endpoint("ping")
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, elapsed
    text = str(info.value)
    assert text.startswith("GR00T policy server at tcp://127.0.0.1:9 did not answer 'ping' within timeout_ms=300.")
    assert "Nothing is listening on that port" in text
    assert "gr00t_inference(action='start', port=9)" in text
    assert "python -m gr00t.eval.run_gr00t_server --port 9" in text
    assert type(info.value.__cause__).__name__ == "Again"


def test_the_socket_is_usable_again_after_a_timeout():
    # A REQ socket that timed out is stuck in its lockstep; the client must
    # reconnect so the caller's retry is a real attempt, not an EFSM error.
    client = Gr00tInferenceClient(host="127.0.0.1", port=DEAD_PORT, timeout_ms=200)
    for _ in range(2):
        with pytest.raises(ConnectionError) as info:
            client.call_endpoint("ping")
        assert "did not answer 'ping'" in str(info.value)
    assert client.ping() is False


def test_a_listening_but_silent_port_is_told_apart_from_an_absent_one():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    accepted: list[socket.socket] = []

    def accept_forever():
        try:
            while True:
                accepted.append(srv.accept()[0])
        except OSError:
            pass

    threading.Thread(target=accept_forever, daemon=True).start()
    try:
        client = Gr00tInferenceClient(host="127.0.0.1", port=port, timeout_ms=300)
        with pytest.raises(ConnectionError) as info:
            client.call_endpoint("get_action", {"observation": {}, "options": None})
        text = str(info.value)
        assert f"tcp://127.0.0.1:{port} did not answer 'get_action' within timeout_ms=300." in text
        assert "The port is listening" in text
        assert "raise timeout_ms" in text
        assert "Nothing is listening" not in text
    finally:
        srv.close()
        for s in accepted:
            s.close()


def test_the_undecided_probe_names_both_possibilities():
    text = unreachable_server_error(uri="tcp://gpu-box:5555", endpoint="ping", timeout_ms=1500, listener=None)
    assert text.startswith("GR00T policy server at tcp://gpu-box:5555 did not answer 'ping' within timeout_ms=1500.")
    assert "Could not tell whether anything is listening" in text
    assert "raise timeout_ms" in text


def test_gr00t_policy_forwards_timeout_ms_to_its_client():
    policy = Gr00tPolicy(data_config="so101", host="127.0.0.1", port=DEAD_PORT, timeout_ms=250)
    assert policy._client is not None
    assert policy._client.timeout_ms == 250
    t0 = time.monotonic()
    with pytest.raises(ConnectionError) as info:
        policy._client.call_endpoint("ping")
    assert time.monotonic() - t0 < 3.0
    assert "timeout_ms=250" in str(info.value)


def test_gr00t_policy_default_budget_is_unchanged():
    policy = Gr00tPolicy(data_config="so101", host="127.0.0.1", port=DEAD_PORT)
    assert policy._client is not None
    assert policy._client.timeout_ms == 15000
