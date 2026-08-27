"""Q52: a viewer in the same house should not stream video through Cloudflare."""

from strands_robots.dashboard.lan_hint import hint, lan_urls, same_network

# The real addresses from the incident: the Mac's own global v6 and the client that
# hammered it for 21 hours. Same /64 - the whole 20.7 GB never needed to leave.
OWN = ["2600:4041:4256:7e00:9d32:feac:b2a9:4086", "192.168.1.164", "127.0.0.1", "fe80::1%en1"]
CLIENT_SAME_HOUSE = "2600:4041:4256:7e00:a13b:93dc:7ae4:c1a4"


def test_the_measured_incident_is_detected():
    assert same_network(CLIENT_SAME_HOUSE, OWN) is True


def test_a_different_network_is_detected():
    assert same_network("2605:59ca:801b:5a80::1234", OWN) is False


def test_ipv4_says_it_does_not_know_rather_than_guessing():
    """A shared public v4 is not evidence of a shared LAN, and NAT hides the rest.

    A wrong 'you are local' sends the operator to an unreachable URL and makes the
    dashboard look broken - worse than silence.
    """
    assert same_network("100.35.227.18", OWN) is None
    assert same_network("192.168.1.55", OWN) is None
    assert same_network(None, OWN) is None
    assert same_network("not-an-ip", OWN) is None


def test_loopback_is_trivially_local_and_no_global_address_means_unknown():
    assert same_network("127.0.0.1", []) is True
    assert same_network(CLIENT_SAME_HOUSE, ["192.168.1.164", "fe80::1"]) is None


def test_only_private_v4_urls_are_offered():
    """Handing out our GLOBAL address would route the stream back out through the ISP."""
    assert lan_urls(OWN, 8090) == ["http://192.168.1.164:8090"]
    assert lan_urls(["2600:4041:4256:7e00:9d32:feac:b2a9:4086", "127.0.0.1"], 8090) == []


def test_unknown_renders_nothing_and_local_without_an_address_admits_it():
    assert hint("100.35.227.18", OWN, 8090)["same_network"] is None
    blind = hint(CLIENT_SAME_HOUSE, ["2600:4041:4256:7e00:9d32:feac:b2a9:4086"], 8090)
    assert blind["same_network"] is True and blind["lan_urls"] == []
    assert "could not find a private address" in blind["why"]
    good = hint(CLIENT_SAME_HOUSE, OWN, 8090)
    assert good["lan_urls"] == ["http://192.168.1.164:8090"] and "Cloudflare" in good["why"]


# --- the endpoint: whose address is it, really? -----------------------------------
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch, tmp_path):
    """This machine has an enrolled passkey + live settings token; point auth and
    settings at empty temp stores so the guard stays in open posture (repo gotcha)."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings
    from strands_robots.dashboard.server import create_app

    monkeypatch.setenv("STRANDS_MESH", "false")
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}
    dsettings.override("security", "auth_token", "test-token")
    return TestClient(create_app())


def test_a_forwarded_request_must_authenticate(client):
    """Not a bug - a PIN. The hint names private LAN addresses, so it must not become an
    unauthenticated topology oracle. The guard's rule (fb5f2a0a) is that a forwarded
    request is never loopback, and this endpoint inherits it rather than opting out.
    """
    assert client.get("/api/network/hint", headers={"CF-Connecting-IP": "2605:59ca:801b:5a80::9"}).status_code == 401


def test_the_forwarded_address_wins_over_the_socket_peer(client):
    """THE BUG THIS TEST EXISTS FOR: every socket this process sees is 127.0.0.1
    (cloudflared), so trusting request.client would call every remote viewer 'local'."""
    r = client.get(
        "/api/network/hint",
        headers={"CF-Connecting-IP": "2605:59ca:801b:5a80::9", "Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["client_ip"] == "2605:59ca:801b:5a80::9"
    assert body["same_network"] is False, "a real remote viewer must not be told it is local"
    assert body["lan_urls"] == []


def test_a_direct_visitor_falls_back_to_the_peer_address(client):
    body = client.get("/api/network/hint", headers={"Authorization": "Bearer test-token"}).json()
    assert body["client_ip"] in ("testclient", "127.0.0.1")
    # 'testclient' is not an address, so the honest answer is 'unknown', not 'local'.
    assert body["same_network"] in (True, None)


def test_x_forwarded_for_takes_only_the_first_hop(client):
    body = client.get(
        "/api/network/hint",
        headers={
            "X-Forwarded-For": "2605:59ca:801b:5a80::9, 172.16.0.1, 10.0.0.2",
            "Authorization": "Bearer test-token",
        },
    ).json()
    assert body["client_ip"] == "2605:59ca:801b:5a80::9"
