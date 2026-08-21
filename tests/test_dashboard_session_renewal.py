"""U21: a remote session renews itself, or the home-lab dashboard logs you out daily.

Evidence this exists (BUGS.md Q109): the JWT lives 24h and had NO renewal route, so
the phone that signed in at 07:04Z on Aug 19 was refused at 07:04Z on Aug 20 — and
its websocket kept knocking for 44 hours, 18,968 refusals.

The clock is passed in, never read, so every boundary here is provable.
"""
from strands_robots.dashboard.auth import renewal_verdict

TTL = 86400          # the shipped default
MAX = 30 * 86400     # absolute cap
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
