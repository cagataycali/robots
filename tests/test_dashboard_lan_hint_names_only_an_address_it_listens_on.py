"""The LAN hint must not offer an address the server is not bound to.

``python -m strands_robots dashboard`` binds ``127.0.0.1`` by default, and
``cli.bind_verdict`` refuses a LAN bind until a passkey or a static token guards
the API. The hint banner nevertheless read the machine's private addresses and
offered ``http://192.168.0.195:8090`` to a viewer on the same network -- an
address nothing listened on, so "open the local address" failed to load.
Measured 2026-10-01 on the default bind.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import lan_hint

_OWN = ["127.0.0.1", "192.168.0.195", "fe80::1%en0"]


@pytest.mark.parametrize("bind_host", ["127.0.0.1", "localhost", "::1"])
def test_a_loopback_bind_offers_no_lan_url_and_says_why(bind_host: str) -> None:
    body = lan_hint.hint("127.0.0.1", _OWN, 8090, bind_host=bind_host)
    assert body["same_network"] is True
    assert body["lan_urls"] == [], "an address nothing listens on must not be offered"
    assert bind_host in body["why"]
    assert "--host 0.0.0.0" in body["why"]


@pytest.mark.parametrize(
    ("bind_host", "urls"),
    [
        ("0.0.0.0", ["http://192.168.0.195:8090", "http://10.0.0.5:8090"]),
        ("::", ["http://192.168.0.195:8090", "http://10.0.0.5:8090"]),
        (None, ["http://192.168.0.195:8090", "http://10.0.0.5:8090"]),
        ("10.0.0.5", ["http://10.0.0.5:8090"]),
    ],
)
def test_a_reachable_bind_offers_only_the_urls_it_listens_on(bind_host: str | None, urls: list[str]) -> None:
    body = lan_hint.hint("127.0.0.1", [*_OWN, "10.0.0.5"], 8090, bind_host=bind_host)
    assert body["lan_urls"] == urls


def test_the_bind_question_is_decided_by_the_address_not_a_string_match() -> None:
    assert lan_hint.bound_to_loopback("127.0.0.1")
    assert lan_hint.bound_to_loopback("127.1.2.3")
    assert lan_hint.bound_to_loopback("LOCALHOST ")
    assert not lan_hint.bound_to_loopback("0.0.0.0")
    assert not lan_hint.bound_to_loopback("not-an-address")
    assert not lan_hint.bound_to_loopback(None)


def test_the_route_answers_for_the_bind_the_cli_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cli.main`` sets ``app.state.host``; ``GET /api/network/hint`` must honour it."""
    pytest.importorskip("fastapi")
    psutil = pytest.importorskip("psutil")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app
    from tests._dashboard_bootstrap import configure_bootstrap

    nic = {"en0": [SimpleNamespace(address=a) for a in _OWN]}
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: nic)
    app = create_app()
    app.state.port = 8090
    headers = configure_bootstrap(monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1:8090", headers=headers, client=("127.0.0.1", 50000)) as client:
        app.state.host = "127.0.0.1"
        loopback = client.get("/api/network/hint").json()
        app.state.host = "0.0.0.0"
        wildcard = client.get("/api/network/hint").json()
    assert loopback["lan_urls"] == [] and "127.0.0.1" in loopback["why"]
    assert wildcard["lan_urls"] == ["http://192.168.0.195:8090"]
