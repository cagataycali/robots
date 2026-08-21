"""Q88: refused handshakes are COUNTED, and /api/health says so.

The incident this exists for: a phone's camera tiles reopened refused websockets for 19.3 hours.
Every refusal was correct, and completely invisible — /api/health said ``ok`` with cheerful
coalescer stats, and the only record was a 34 MB log. These tests pin both halves: the pure
tally's judgement, and the fact that the real middleware feeds it while still refusing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import settings
from strands_robots.dashboard.refusals import STORM_THRESHOLD, WINDOW_S, RefusalTally
from strands_robots.dashboard.server import create_app

NOW = 1_000_000.0


def test_nothing_refused_says_nothing() -> None:
    # A health payload that always carries a refusals section is a section nobody reads.
    assert RefusalTally().summary(NOW) is None


def test_a_single_refusal_is_not_called_a_storm() -> None:
    t = RefusalTally()
    t.record(client="10.0.0.5", path="/api/fleet", now=NOW)
    s = t.summary(NOW)
    assert s is not None and s["storm"] is False
    assert s["total"] == 1 and s["recent"] == 1
    # Someone typing a token wrong must not be described as a broken client.
    assert "signing in" in str(s["text"])


def test_a_loop_is_named_with_the_client_the_path_and_the_one_fix() -> None:
    t = RefusalTally()
    for i in range(STORM_THRESHOLD * 3):
        t.record(client="192.168.1.44", path="/ws/camera/so101-leader/main", now=NOW + i * 2)
    s = t.summary(NOW + 60)
    assert s is not None and s["storm"] is True
    assert s["worst"] == {
        "client": "192.168.1.44",
        "path": "/ws/camera/so101-leader/main",
        "kind": "credential",
        "count": STORM_THRESHOLD * 3,
    }
    text = str(s["text"])
    assert "192.168.1.44" in text and "/ws/camera/so101-leader/main" in text
    assert "sign in again" in text
    # The whole point of Q88: the reader must not go looking at the robot.
    assert "Nothing is wrong with the robots" in text
    # And it must not accuse: a stale tab is overwhelmingly more likely than an attack.
    assert "attack" not in text.lower()


def test_a_cross_origin_loop_gets_the_CORS_fix_not_the_signin_fix() -> None:
    t = RefusalTally()
    for i in range(STORM_THRESHOLD + 1):
        t.record(client="10.1.2.3", path="/ws/mesh", now=NOW + i, kind="origin")
    text = str(t.summary(NOW + 30)["text"])  # type: ignore[index]
    assert "cors_origins" in text
    assert "sign in again" not in text


def test_a_storm_that_stopped_reports_that_it_stopped() -> None:
    # "Did my fix work?" is a question the status endpoint should answer.
    t = RefusalTally()
    for i in range(50):
        t.record(client="10.0.0.5", path="/ws/mesh", now=NOW + i)
    s = t.summary(NOW + WINDOW_S * 2)
    assert s is not None and s["recent"] == 0 and "storm" not in s
    assert "none in the last" in str(s["text"]) and s["total"] == 50


def test_the_tally_is_bounded_and_never_undercounts() -> None:
    # A storm is exactly when an unbounded structure becomes the second bug — but the total
    # must still be the truth.
    t = RefusalTally()
    for i in range(500):
        t.record(client=f"10.0.0.{i}", path="/ws/mesh", now=NOW + i)
    assert t.total == 500
    s = t.summary(NOW + 500)
    assert s is not None and int(s["total"]) == 500
    assert int(s["untracked"]) > 0
    assert len(t._recent) <= 64


@pytest.fixture()
def sealed_app():
    settings.override("security", "auth_token", "the-real-token")
    try:
        yield create_app()
    finally:
        settings.clear_overrides()


def test_the_real_middleware_counts_while_still_refusing(sealed_app) -> None:
    client = TestClient(sealed_app)
    # A wrong-token REST call and a refused websocket handshake: both are refusals, and the
    # refusal itself must be unchanged by the bookkeeping.
    for _ in range(STORM_THRESHOLD + 2):
        assert client.get("/api/fleet", headers={"Authorization": "Bearer stale"}).status_code == 401
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/mesh?token=stale"):
            pass

    # The identities are for the operator; the token rides along (the withholding rule itself
    # is pinned by test_an_unauthenticated_reader_gets_counts_but_no_identities).
    health = client.get("/api/health", headers={"Authorization": "Bearer the-real-token"}).json()
    assert health["status"] == "ok"  # health stays public and still answers
    block = health["refused_handshakes"]
    assert block["total"] >= STORM_THRESHOLD + 3
    assert block["storm"] is True
    assert "/api/fleet" in str(block["worst"]["path"])
    assert "sign in again" in str(block["text"])

    # And an accepted request adds nothing: only refusals are counted.
    before = block["total"]
    assert client.get("/api/fleet", headers={"Authorization": "Bearer the-real-token"}).status_code == 200
    after = client.get(
        "/api/health", headers={"Authorization": "Bearer the-real-token"}
    ).json()["refused_handshakes"]["total"]
    assert after == before


def test_an_unauthenticated_reader_gets_counts_but_no_identities(sealed_app) -> None:
    """/api/health is PUBLIC by design — so this block must not become reconnaissance.

    Counted, then reviewed one iteration later: the sentence named a LAN address and the exact
    screens being refused, and it was readable by the one caller who could NOT authenticate.
    The counts are still public (a "something is hammering me" number gives nothing away).
    """
    client = TestClient(sealed_app)
    for _ in range(STORM_THRESHOLD + 2):
        client.get("/api/fleet", headers={"Authorization": "Bearer stale"})

    # A remote caller with no credential: counts yes, who/where no.
    public = client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.9"}).json()
    block = public["refused_handshakes"]
    assert block["total"] >= STORM_THRESHOLD + 2
    assert block["storm"] is True                      # the fact survives
    assert "worst" not in block                        # the address does not
    body = str(block)
    assert "/api/fleet" not in body and "192.168." not in body and "testclient" not in body
    assert "Sign in to see" in str(block["text"])

    # The same server, the same moment, to the operator who CAN act on it.
    trusted = client.get(
        "/api/health", headers={"Authorization": "Bearer the-real-token"}
    ).json()["refused_handshakes"]
    assert "/api/fleet" in str(trusted["worst"]["path"])
    assert "sign in again" in str(trusted["text"])


def test_health_has_no_refusal_section_on_a_clean_app(sealed_app) -> None:
    assert "refused_handshakes" not in TestClient(sealed_app).get("/api/health").json()
