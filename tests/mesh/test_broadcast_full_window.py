"""Regression: broadcast() must collect acks for the FULL window, not return
~0.3s after the first response.

``broadcast`` waited on an Event that ``_on_response`` sets on the FIRST reply,
then slept only 0.3s more -- so ``emergency_stop``'s ``responses_received``
systematically under-counted the fleet: an operator could not distinguish
"1 of N robots acknowledged the stop" from "all N acknowledged". The window
must be honoured in full so every peer that replies within it is counted.
"""

import json
import threading
import time
import types

from strands_robots.mesh import core


def _sample(payload: dict) -> object:
    s = types.SimpleNamespace()
    s.payload = types.SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    return s


def test_broadcast_collects_responses_across_the_full_window(monkeypatch):
    m = core.Mesh(robot=object(), peer_id="op")
    m._running = True
    monkeypatch.setattr(core._security, "validate_command", lambda cmd: cmd)

    def fake_publish(key, msg):
        turn = msg["turn_id"]

        def responder():
            # ack #1 arrives immediately (old code returned ~0.3s after this)
            time.sleep(0.05)
            m._on_response(_sample({"turn_id": turn, "responder_id": "r1", "result": 1}))
            # acks #2 and #3 arrive AFTER the old 0.3s early-return window
            time.sleep(0.5)
            m._on_response(_sample({"turn_id": turn, "responder_id": "r2", "result": 2}))
            m._on_response(_sample({"turn_id": turn, "responder_id": "r3", "result": 3}))

        threading.Thread(target=responder, daemon=True).start()

    monkeypatch.setattr(m, "publish", fake_publish)

    start = time.monotonic()
    resps = m.broadcast({"action": "stop"}, timeout=1.0)
    elapsed = time.monotonic() - start

    # All three peers that replied within the window are counted (old code
    # returned after ack #1 only).
    assert len(resps) == 3
    # The full window was honoured rather than returning right after ack #1.
    assert elapsed >= 0.9


def test_broadcast_returns_empty_when_not_running():
    m = core.Mesh(robot=object(), peer_id="op")
    m._running = False
    assert m.broadcast({"action": "stop"}, timeout=0.1) == []
