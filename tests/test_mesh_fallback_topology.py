"""A non-hub session must be able to RECEIVE what a sibling publishes.

The bug this pins: every child of the dashboard connects to the one process
that won the ``STRANDS_MESH_PORT`` listener, and a Zenoh 1.x *peer* will not
accept traffic relayed by that hub (nor by a router - ``routing/peer/mode``
was removed in 1.10). Child-to-dashboard streams hid it, because a child opens
that link itself; teleop was the first child-to-child topic and its receiver
sat at zero frames while the leader published hundreds.

The config assertions are cheap. The one that matters is
``test_late_subscriber_receives_a_running_publisher_through_the_hub``: three
real sessions, hub as the only configured endpoint, subscriber started LAST -
the shape that delivered 0 of 62 frames before this fix.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time

import pytest

from strands_robots.mesh import session as mesh_session

zenoh = pytest.importorskip("zenoh")

KEY = "strands/fallback-probe/input/leader"


def _cfg_dict(**env: str) -> dict:
    cfg = zenoh.Config()
    mesh_session._apply_fallback_topology(cfg, "tcp/127.0.0.1:7447", "tcp")
    return json.loads(str(cfg))


def test_default_fallback_is_client_mode(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_FALLBACK_MODE", raising=False)
    cfg = zenoh.Config()
    assert mesh_session._apply_fallback_topology(cfg, "tcp/127.0.0.1:7447", "tcp") == "client"
    d = json.loads(str(cfg))
    assert d["mode"] == "client"
    # No listener is OVERRIDDEN: zenoh's default listen/endpoints is a per-mode
    # map ({"peer": ["tcp/[::]:0"], "router": [...]}) with no "client" entry, so
    # a client listens on nothing. Asserting "empty" would have been wrong -
    # measured, not assumed.
    assert isinstance(d["listen"]["endpoints"], dict), d["listen"]["endpoints"]
    assert "client" not in d["listen"]["endpoints"]
    assert d["connect"]["endpoints"] == ["tcp/127.0.0.1:7447"]
    # The property the old peer-mode fallback was chosen for, kept.
    assert d["connect"]["exit_on_failure"] is False
    assert d["connect"]["retry"]["period_init_ms"] == mesh_session.FALLBACK_RETRY["period_init_ms"]


def test_peer_mode_is_still_available_and_keeps_its_ephemeral_listener(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_FALLBACK_MODE", "PEER")  # case-insensitive
    cfg = zenoh.Config()
    assert mesh_session._apply_fallback_topology(cfg, "tls/127.0.0.1:7447", "tls") == "peer"
    d = json.loads(str(cfg))
    assert d["listen"]["endpoints"] == ["tls/127.0.0.1:0"]
    assert d["connect"]["endpoints"] == ["tls/127.0.0.1:7447"]
    assert d.get("mode") in (None, "peer")


def test_a_typo_warns_and_keeps_the_mesh_working(monkeypatch, caplog):
    monkeypatch.setenv("STRANDS_MESH_FALLBACK_MODE", "cleint")
    with caplog.at_level("WARNING"):
        assert mesh_session._fallback_mode() == "client"
    assert "STRANDS_MESH_FALLBACK_MODE" in caplog.text


# --------------------------------------------------------------------------
# The behavioural gate: does a frame actually arrive?
# --------------------------------------------------------------------------


def _hub(ep: str, secs: float) -> None:
    cfg = zenoh.Config()
    cfg.insert_json5("mode", json.dumps("peer"))
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("listen/endpoints", json.dumps([ep]))
    s = zenoh.open(cfg)
    time.sleep(secs)
    s.close()


def _child_cfg(ep: str) -> "zenoh.Config":
    cfg = zenoh.Config()
    cfg.insert_json5("scouting/multicast/enabled", "false")
    mesh_session._apply_fallback_topology(cfg, ep, "tcp")
    return cfg


def _publisher(ep: str, secs: float, out) -> None:
    s = zenoh.open(_child_cfg(ep))
    time.sleep(1.5)
    n = 0
    end = time.time() + secs
    while time.time() < end:
        s.put(KEY, json.dumps({"seq": n}))
        n += 1
        time.sleep(0.05)
    out.put(("published", n))
    s.close()


def _late_subscriber(ep: str, secs: float, out) -> None:
    s = zenoh.open(_child_cfg(ep))
    hits = []
    handle = s.declare_subscriber(KEY, lambda _s: hits.append(1))  # noqa: F841 - keep alive
    time.sleep(secs)
    out.put(("received", len(hits)))
    s.close()


@pytest.mark.slow
def test_late_subscriber_receives_a_running_publisher_through_the_hub(monkeypatch):
    """0 of 62 before the fix; must be > 0 now.

    Two children, the hub as their only configured endpoint, and the subscriber
    starting after the publisher is already streaming - the teleop shape.
    """
    monkeypatch.delenv("STRANDS_MESH_FALLBACK_MODE", raising=False)
    monkeypatch.delenv("ZENOH_CONNECT", raising=False)
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)
    ep = "tcp/127.0.0.1:7549"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    hub = ctx.Process(target=_hub, args=(ep, 16.0))
    hub.start()
    time.sleep(2.0)
    pub = ctx.Process(target=_publisher, args=(ep, 9.0, q))
    pub.start()
    time.sleep(4.0)  # publisher is already running when the subscriber joins
    sub = ctx.Process(target=_late_subscriber, args=(ep, 6.0, q))
    sub.start()
    try:
        pub.join(30)
        sub.join(30)
    finally:
        for proc in (pub, sub, hub):
            if proc.is_alive():
                proc.terminate()
            proc.join(5)
    results = {}
    while not q.empty():
        key, value = q.get()
        results[key] = value
    assert results.get("published", 0) > 10, results
    assert results.get("received", 0) > 0, (
        f"a late subscriber received {results.get('received')} of {results.get('published')} frames "
        "through the hub - child-to-child delivery is broken again (check the session mode)"
    )
