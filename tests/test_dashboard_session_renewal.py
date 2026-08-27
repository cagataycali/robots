"""U21: a remote session renews itself, or the home-lab dashboard logs you out daily.

Evidence this exists (BUGS.md Q109): the JWT lives 24h and had NO renewal route, so
the phone that signed in at 07:04Z on Aug 19 was refused at 07:04Z on Aug 20 — and
its websocket kept knocking for 44 hours, 18,968 refusals.

The clock is passed in, never read, so every boundary here is provable.
"""

from strands_robots.dashboard.auth import renewal_verdict

TTL = 86400  # the shipped default
MAX = 30 * 86400  # absolute cap
NOW = 1_787_000_000


def _claims(**over):
    """A token issued NOW-ish, valid for TTL."""
    base = {"sub": "cagatay", "iat": NOW, "iat0": NOW, "exp": NOW + TTL}
    base.update(over)
    return base


def test_a_fresh_token_is_left_alone():
    v = renewal_verdict(_claims(), NOW + 60, ttl=TTL, max_age=MAX)
    assert v["renew"] is False
    assert "fresh" in v["reason"]


def test_past_the_half_life_it_renews():
    # THE POINT: an active session must never die mid-use.
    v = renewal_verdict(_claims(), NOW + TTL // 2 + 1, ttl=TTL, max_age=MAX)
    assert v["renew"] is True
    assert v["exp"] > NOW + TTL, "a renewal that does not extend is not a renewal"


def test_the_half_life_boundary_is_exact():
    just_before = renewal_verdict(_claims(), NOW + TTL // 2 - 1, ttl=TTL, max_age=MAX)
    at_it = renewal_verdict(_claims(), NOW + TTL // 2, ttl=TTL, max_age=MAX)
    assert just_before["renew"] is False and at_it["renew"] is True


def test_an_expired_token_is_never_renewed():
    # "Almost valid" is how a session becomes unrevokable: revocation works by
    # waiting one TTL out.
    v = renewal_verdict(_claims(), NOW + TTL + 1, ttl=TTL, max_age=MAX)
    assert v["renew"] is False
    assert "expired" in v["reason"] and v["exp"] is None


def test_renewal_cannot_outlive_the_absolute_cap():
    # A session renewed by a background poller would otherwise be immortal and the
    # authenticator would never be asked again.
    # One hour of cap left, a token expiring in 10s: it renews, but only to the cap.
    old = _claims(iat0=NOW - MAX + 3600, exp=NOW + 10)
    v = renewal_verdict(old, NOW, ttl=TTL, max_age=MAX)
    assert v["renew"] is True
    assert v["exp"] == NOW + 3600, "clamped to iat0 + max_age, not now + ttl"
    assert v["exp"] < NOW + TTL

    dead = _claims(iat0=NOW - MAX - 1, exp=NOW + 10)
    v2 = renewal_verdict(dead, NOW, ttl=TTL, max_age=MAX)
    assert v2["renew"] is False
    assert "maximum age" in v2["reason"] and "passkey" in v2["reason"]


def test_a_renewal_that_would_shorten_the_session_is_refused():
    # At the cap the clamp can produce an exp EARLIER than what the client holds; a
    # downgrade the client cannot refuse is worse than no renewal.
    at_cap = _claims(iat0=NOW - MAX + 5, exp=NOW + 5)
    v = renewal_verdict(at_cap, NOW + 1, ttl=TTL, max_age=MAX)
    assert v["renew"] is False and "not extend" in v["reason"]


def test_a_token_predating_the_iat0_claim_keeps_its_real_age():
    # cagatay's phone holds one of these. Treating it as brand new would restart its
    # 30-day cap on a session that is already old.
    legacy = {"sub": "cagatay", "exp": NOW + 10}  # no iat, no iat0
    v = renewal_verdict(legacy, NOW, ttl=TTL, max_age=MAX)
    assert v["iat0"] == NOW + 10 - TTL, "age inferred from exp - ttl, not from now"
    assert v["renew"] is True


def test_no_evidence_no_renewal():
    for bad in (None, {}, {"exp": "soon"}, {"exp": None}, "not-claims"):
        v = renewal_verdict(bad, NOW, ttl=TTL, max_age=MAX)  # type: ignore[arg-type]
        assert v["renew"] is False, bad
        assert v["reason"], "a refusal must say why - a silent one is the bug being fixed"


def test_every_reason_is_written_for_a_person():
    seen = {
        renewal_verdict(c, t, ttl=TTL, max_age=MAX)["reason"]
        for c, t in [
            (_claims(), NOW + 60),
            (_claims(), NOW + TTL // 2 + 1),
            (_claims(), NOW + TTL + 1),
            (_claims(iat0=NOW - MAX - 1, exp=NOW + 10), NOW),
            (None, NOW),
        ]
    }
    assert len(seen) == 5, "each refusal must be distinguishable, or the UI cannot explain it"
    for r in seen:
        assert r == r.strip() and "{" not in r and "Traceback" not in r


# --- the plumbing: renew_if_due, and the header the browser must be able to read ----
import asyncio  # noqa: E402

from strands_robots.dashboard import auth  # noqa: E402
from strands_robots.dashboard import server as srv  # noqa: E402


def test_renew_if_due_preserves_who_you_are_and_when_you_signed_in(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_JWT_SECRET", "test-secret-for-renewal")
    token = auth.issue_token("cred1", name="phone")
    first = auth.verify_token(token)
    assert first["iat0"] == first["iat"], "a first sign-in is its own origin"

    assert auth.renew_if_due(token) is None, "a token issued a second ago is fresh"

    fresh = auth.renew_if_due(token, now=first["iat"] + 50_000)  # past the 24h half-life
    assert fresh and fresh != token
    after = auth.verify_token(fresh)
    assert after["sub"] == "cred1" and after["name"] == "phone", "renewal is not a new identity"
    assert after["iat0"] == first["iat0"], "the ORIGINAL sign-in rides along, or the cap resets"
    assert after["exp"] > first["exp"]


def test_renew_if_due_says_nothing_for_a_token_it_cannot_trust(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_JWT_SECRET", "test-secret-for-renewal")
    assert auth.renew_if_due("") is None
    assert auth.renew_if_due("not.a.jwt") is None
    # An expired session is a LOGIN problem; renewing it would make expiry meaningless.
    old = auth.issue_token("cred1")
    assert auth.renew_if_due(old, now=auth.verify_token(old)["exp"] + 1) is None


def _run(scope, headers):
    """Drive the auth middleware over one request and collect what it sent."""
    sent: list = []

    async def app(scope, receive, send):  # the protected app behind the guard
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    guard = srv.TokenAuthMiddleware(app)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(guard(scope, receive, send))
    finally:
        loop.close()
    return sent


def _header(sent, name: bytes):
    for m in sent:
        if m["type"] == "http.response.start":
            for k, v in m["headers"]:
                if k.lower() == name:
                    return v
    return None


def test_a_renewed_token_rides_home_on_the_response(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    monkeypatch.setenv("STRANDS_DASH_JWT_SECRET", "test-secret-for-renewal")
    monkeypatch.setattr(srv.settings, "get", lambda *a, **k: None)
    token = auth.issue_token("cred1", name="phone")
    aged = auth.issue_token("cred1", name="phone", exp=int(__import__("time").time()) + 100)

    scope = {
        "type": "http",
        "path": "/api/fleet",
        "method": "GET",
        "client": ("192.168.1.9", 5000),
        "headers": [(b"x-dashboard-token", aged.encode())],
    }
    sent = _run(scope, None)
    got = _header(sent, b"x-session-token")
    assert got, "a session about to expire must be handed a new one, or the phone is locked out daily"
    renewed = auth.verify_token(got.decode())
    assert renewed["sub"] == "cred1"
    assert _header(sent, b"access-control-expose-headers"), "a header the browser cannot read is no renewal"

    # and a fresh session gets NO header: the client's credential is not rewritten per request
    scope["headers"] = [(b"x-dashboard-token", token.encode())]
    assert _header(_run(scope, None), b"x-session-token") is None
