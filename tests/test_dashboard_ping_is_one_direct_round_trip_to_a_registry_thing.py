"""``POST /api/robots/{thing}/ping``: one direct-message round trip to a Thing in the IoT registry.

A grey registry card asks "is dm-so101-02 down, or did it never boot?". The
answer is one ``status`` read sent point to point over AWS IoT Core: a 200 means
the Thing is connected and answered, a 404 in about 80 ms is the broker's
offline verdict, a 403 says this operator identity may not address it. The
route never moves anything (``status`` is a read verb) and it is still closed
by default: a session is required, the target must be a Thing the registry
view lists, and on a backend without an addressed send (plain Zenoh) the route
answers ``unavailable`` without touching the wire.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.dashboard import routes_mesh
from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh.iot import registry


class _Bridge(MeshBridge):
    """A bridge whose send is scripted; records what it was asked to send."""

    def __init__(self, backend: str, result: dict[str, Any]) -> None:
        super().__init__(peer_id="dash")
        self._running = True
        self._endpoints = {"backend": backend}
        self.sent: list[tuple[str, dict[str, Any], float]] = []
        self._scripted = result

    def send_cmd(
        self, target: str, cmd: dict[str, Any], timeout: float = 30.0, *, source: str = "api"
    ) -> dict[str, Any]:
        self.sent.append((target, cmd, timeout))
        return dict(self._scripted)


@pytest.fixture
def things(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list_things(*args: Any, **kwargs: Any) -> registry.RegistryView:
        return registry.RegistryView(
            status="ok", things=tuple(registry.RegistryThing(thing_name=n) for n in ("dm-so101-01", "dm-so101-02"))
        )

    monkeypatch.setattr(registry, "list_things", fake_list_things)
    monkeypatch.setattr(
        routes_mesh, "_IOT_REGISTRY_CACHE", type(routes_mesh._IOT_REGISTRY_CACHE)(routes_mesh.IOT_REGISTRY_TTL_S)
    )


def _ping(bridge: _Bridge, thing: str) -> dict[str, Any]:
    return asyncio.run(routes_mesh.ping_thing_verdict(bridge, thing))


ANSWERED = {
    "status": "ok",
    "state": {},
    "delivery": {"via": "direct", "confirmed": True, "latency_ms": 91.0, "reason": ""},
}
OFFLINE = {
    "status": "error",
    "error": "peer offline (iot 404)",
    "delivery": {"via": "direct", "confirmed": False, "latency_ms": 80.0, "reason": "offline"},
}
FORBIDDEN = {
    "status": "timeout",
    "delivery": {"via": "publish", "confirmed": False, "latency_ms": 0.0, "reason": "forbidden"},
}
SILENT = {"status": "timeout", "delivery": {"via": "direct", "confirmed": True, "latency_ms": 95.0, "reason": ""}}


def test_a_connected_thing_answers(things: None) -> None:
    bridge = _Bridge("bridge", ANSWERED)
    out = _ping(bridge, "dm-so101-01")
    assert out["verdict"] == "answered"
    assert out["latency_ms"] == 91.0
    assert bridge.sent == [("dm-so101-01", {"action": "status"}, routes_mesh.PING_TIMEOUT_S)]
    assert routes_mesh.PING_TIMEOUT_S == 3.0


def test_the_broker_offline_verdict_is_reported_as_offline(things: None) -> None:
    out = _ping(_Bridge("iot", OFFLINE), "dm-so101-02")
    assert out["verdict"] == "offline"
    assert out["latency_ms"] == 80.0


def test_a_forbidden_identity_is_named_not_hidden(things: None) -> None:
    out = _ping(_Bridge("bridge", FORBIDDEN), "dm-so101-02")
    assert out["verdict"] == "forbidden"
    assert "operator" in out["reason"]


def test_delivered_but_silent_is_its_own_verdict(things: None) -> None:
    out = _ping(_Bridge("bridge", SILENT), "dm-so101-01")
    assert out["verdict"] == "silent"


def test_no_addressed_send_on_plain_zenoh(things: None) -> None:
    bridge = _Bridge("zenoh", ANSWERED)
    out = _ping(bridge, "dm-so101-01")
    assert out["verdict"] == "unavailable"
    assert bridge.sent == [], "nothing goes to the wire on a backend that cannot address a Thing"


def test_a_name_outside_the_registry_is_refused_before_any_send(things: None) -> None:
    bridge = _Bridge("bridge", ANSWERED)
    out = _ping(bridge, "so101-lan")
    assert out["verdict"] == "refused"
    assert "registry" in out["reason"]
    assert bridge.sent == []


def test_the_route_requires_a_session() -> None:
    import inspect

    from strands_robots.dashboard import access

    route: Any = next(r for r in routes_mesh.router.routes if getattr(r, "path", "") == "/api/robots/{thing}/ping")
    assert set(route.methods) == {"POST"}
    params = inspect.signature(route.endpoint).parameters
    depends = [p.default for p in params.values() if getattr(p.default, "dependency", None) is not None]
    assert any(d.dependency is access.require_session for d in depends)


def test_the_verdict_words_are_closed() -> None:
    assert routes_mesh.PING_VERDICTS == (
        "answered",
        "offline",
        "forbidden",
        "silent",
        "unavailable",
        "refused",
        "error",
    )
