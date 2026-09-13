"""A dashboard session is extended, but never past the cap its sign-in set.

The dashboard commands real hardware, so how long a session keeps that right is
the whole point of :func:`auth.renewal_verdict` and its caller
:func:`auth.renew_if_due`. Four invariants are stated in that code and were
pinned nowhere:

* The absolute cap. ``iat0`` is "carried unchanged through every renewal so the
  absolute cap in renewal_verdict() cannot be reset by re-issuing" - a session
  that renews forever is a session that never needs a passkey again.
* No downgrade. A renewal is "never SHORTER than what the client already holds:
  a renewal that shaved time off would be a downgrade the client cannot
  refuse".
* A token that does not verify is "a login problem, not a renewal one", so
  :func:`auth.renew_if_due` hands back ``None`` rather than minting from
  unverified claims.
* A handoff token "never outlives the session it came from", because it rides in
  a URL and URLs land in history, logs and screenshots.

The two modules that already name these functions only borrow them to MEASURE a
duration knob's window (``test_dashboard_auth_duration_knobs_are_not_widened``
bisects for the renewable age); neither grades the decision. So every refusal
arm, the half-life gate, the ``iat0`` fallback and the whole of
:func:`auth.renew_if_due` ran ungraded, and a renewal that reset the cap or
shortened a session would have been reported green.

The ladder cell drives the real surface across a simulated clock rather than
asserting one verdict, because the cap is enforced by CLAMPING each renewal to
``iat0 + max_age``: no single call shows the session running out of headroom.
"""

from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from fastapi import HTTPException

import strands_robots.dashboard.auth as auth

# Short, exact windows so the ladder below lands on whole boundaries. TTL is the
# token lifetime; MAX_AGE is three of them, so a session renews four times and
# then runs out of cap.
TTL = 3600
MAX_AGE = 10800

# Long enough not to trip pyjwt's InsecureKeyLengthWarning, and not the
# module's own secret, so a token signed with it must fail to verify.
FOREIGN_SECRET = "a-different-signing-key-of-a-respectable-length"

# The ladder's session signed in this long before the wall clock, so a renewal
# that re-dated the session from NOW instead of carrying ``iat0`` forward is
# visible. A session minted at the current second hides exactly that bug.
SIGNED_IN_AGO = TTL // 2


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point every store reader in this file at a per-test file.

    The cells below mint tokens, and :func:`auth.issue_token` signs with
    ``_jwt_secret()``, which reads the credential store. :func:`auth._store_path`
    resolves an unset ``STORE`` to ``~/.strands_dashboard/auth.json`` - the file
    that decides whether a dashboard on this machine is sealed - so without this
    redirect these cells would read and WRITE it. The sibling
    ``test_dashboard_auth_*`` modules that reach the store all carry the same
    redirect, for the same reason.
    """
    monkeypatch.setenv(auth._ENV + "STORE", str(tmp_path / "auth.json"))
    auth._cache = {}
    yield


@pytest.fixture
def windows(monkeypatch):
    """Set the two renewal windows through the environment the module reads."""
    monkeypatch.setenv(auth._ENV + "TOKEN_TTL", str(TTL))
    monkeypatch.setenv(auth._ENV + "SESSION_MAX_AGE", str(MAX_AGE))


NOW = 1_000_000.0

# Claims that cannot be renewed, and the refusal each must draw. A refusal here
# has to be distinguishable: "no expiry to extend" sends an operator to the
# token, "already expired" sends them to their passkey.
UNRENEWABLE: list[tuple[Any, str]] = [
    (None, "no session claims to renew"),
    ([], "no session claims to renew"),
    ("eyJhbGciOi", "no session claims to renew"),
    ({}, "session has no expiry to extend"),
    ({"exp": None}, "session has no expiry to extend"),
    ({"exp": "soon"}, "session has no expiry to extend"),
    ({"exp": NOW}, "session already expired - sign in again"),
    ({"exp": NOW - 1}, "session already expired - sign in again"),
]


@pytest.mark.parametrize(("claims", "reason"), UNRENEWABLE, ids=[r for _, r in UNRENEWABLE])
def test_a_session_that_cannot_be_renewed_says_which_way_it_cannot(claims, reason):
    """Each unrenewable session is refused, named, and offers no expiry to trust."""
    verdict = auth.renewal_verdict(claims, NOW, ttl=TTL, max_age=MAX_AGE)

    assert verdict["renew"] is False
    assert verdict["reason"] == reason
    # No expiry and no sign-in date, so a caller cannot read a window off a
    # verdict that refused to establish one.
    assert verdict["exp"] is None
    assert verdict["iat0"] is None


def test_a_session_before_its_half_life_keeps_the_expiry_it_already_holds():
    """A fresh session is not renewed, but the verdict still reports its window.

    The refusal carries ``exp``/``iat0`` where the arms above carry ``None``:
    there IS a live session here, it simply does not need a new token yet, and a
    caller telling the client when it expires must not be handed ``None``.
    """
    exp = NOW + TTL

    verdict = auth.renewal_verdict({"sub": "c", "iat0": NOW, "exp": exp}, NOW, ttl=TTL, max_age=MAX_AGE)

    assert verdict["renew"] is False
    assert verdict["reason"] == "session still fresh"
    assert verdict["exp"] == int(exp)
    assert verdict["iat0"] == int(NOW)


@pytest.mark.parametrize(
    ("claims", "label"),
    [
        ({"sub": "c", "exp": NOW + TTL / 2}, "no iat0 and no iat"),
        ({"sub": "c", "iat0": "yesterday", "exp": NOW + TTL / 2}, "an iat0 that is not a number"),
    ],
    ids=["no-iat0-and-no-iat", "unparseable-iat0"],
)
def test_a_session_with_no_usable_sign_in_date_is_dated_from_its_token(claims, label):
    """A session whose sign-in date is missing or unreadable is not brand new.

    ``iat0`` falls back to ``exp - ttl`` - the token's own issue - and NOT to
    ``now``. A token already sitting in a phone's storage that was dated from
    ``now`` would restart its cap on every renewal, which is exactly the forever
    session ``iat0`` exists to prevent.
    """
    verdict = auth.renewal_verdict(claims, NOW, ttl=TTL, max_age=MAX_AGE)

    dated_from_the_token = int(claims["exp"] - TTL)
    assert verdict["iat0"] == dated_from_the_token
    assert verdict["iat0"] < NOW, f"{label} was treated as a sign-in happening now"


# One renewal decision per half token-lifetime, from sign-in to the cap. The
# verdict at each offset, and the expiry the session holds AFTER that decision,
# both measured off the shipped surface.
LADDER: list[tuple[int, str, int]] = [
    (0, "session still fresh", TTL),
    (1800, "past half-life, extended", 5400),
    (3600, "past half-life, extended", 7200),
    (5400, "past half-life, extended", 9000),
    (7200, "past half-life, extended", MAX_AGE),
    (9000, "renewal would not extend this session", MAX_AGE),
    (10800, "session already expired - sign in again", MAX_AGE),
]


def test_a_session_renews_until_its_cap_and_never_past_it(windows):
    """Drive ``renew_if_due`` across a simulated clock, from sign-in to the cap.

    Each row decides on the token the previous row produced, so this is the
    surface an operator actually rides: ``renew_if_due`` -> ``verify_token`` ->
    ``renewal_verdict`` -> ``issue_token``. ``now`` is passed in; the tokens stay
    valid against the wall clock throughout, so nothing here waits on real time.

    The last two rows are how the cap is felt. Every renewal is CLAMPED to
    ``iat0 + max_age``, so the session does not get refused for its age - it
    lands exactly on the cap, is then refused because extending it further would
    shave time off the token held, and finally expires there.
    """
    t0 = int(time.time()) - SIGNED_IN_AGO
    cap = t0 + MAX_AGE
    token = auth.issue_token("cred1", "Ada", iat0=t0, exp=t0 + TTL)

    renewals = 0
    observed: list[tuple[int, str, int]] = []
    for offset, _, _ in LADDER:
        held = auth.verify_token(token)
        reason = auth.renewal_verdict(held, t0 + offset)["reason"]
        fresh = auth.renew_if_due(token, now=t0 + offset)
        if fresh is not None:
            claims = auth.verify_token(fresh)
            # The cap cannot be reset by re-issuing: every renewal carries the
            # ORIGINAL sign-in forward, and lands on or before the cap that set.
            assert claims["iat0"] == t0
            assert claims["exp"] <= cap
            # And never a downgrade: a renewal only ever adds time.
            assert claims["exp"] > held["exp"]
            token, renewals = fresh, renewals + 1
        observed.append((offset, reason, auth.verify_token(token)["exp"] - t0))

    assert observed == LADDER
    # A session that renewed and then stopped, rather than one that never
    # renewed (which would satisfy every assertion above vacuously).
    assert renewals == 4


def test_a_cap_shorter_than_a_token_stops_renewal_while_the_token_still_works(monkeypatch):
    """The cap is a renewal gate, not a revocation.

    ``SESSION_MAX_AGE`` below ``TOKEN_TTL`` is a legal pair, and it is the only
    way a session can be past its cap while still holding a valid token. The
    held token keeps working until it expires; it is simply never extended
    again.
    """
    monkeypatch.setenv(auth._ENV + "TOKEN_TTL", "86400")
    monkeypatch.setenv(auth._ENV + "SESSION_MAX_AGE", str(TTL))
    t0 = int(time.time())
    token = auth.issue_token("cred1", iat0=t0, exp=t0 + 86400)

    verdict = auth.renewal_verdict(auth.verify_token(token), t0 + TTL)

    assert verdict["renew"] is False
    assert "maximum age" in verdict["reason"]
    # Still a live session, not a rejected one.
    assert auth.session_is_valid(token) is True


@pytest.mark.parametrize(
    "spelling",
    ["empty", "not-a-jwt", "signed-by-someone-else", "already-expired"],
)
def test_a_token_that_does_not_verify_is_a_login_problem_not_a_renewal(spelling):
    """``renew_if_due`` mints nothing from claims it could not verify.

    Reading claims out of an unverified token and re-signing them would turn a
    forged or long-dead token into a live session, so each of these is ``None``
    rather than a fresh token.
    """
    stale = {"sub": "c", "exp": int(time.time()) - 10}
    token = {
        "empty": "",
        "not-a-jwt": "not-a-token",
        "signed-by-someone-else": jwt.encode(
            {"sub": "c", "exp": int(time.time()) + TTL}, FOREIGN_SECRET, algorithm="HS256"
        ),
        "already-expired": jwt.encode(stale, auth._jwt_secret(), algorithm="HS256"),
    }[spelling]

    assert auth.renew_if_due(token, now=time.time()) is None


# Sessions that may not be copied into a URL, and the refusal each draws.
UNHANDOFFABLE: list[tuple[Any, str]] = [
    (None, "no session claims to hand off"),
    ([], "no session claims to hand off"),
    ({}, "session has no expiry"),
    ({"exp": None}, "session has no expiry"),
    ({"exp": "soon"}, "session has no expiry"),
    ({"exp": NOW}, "session already expired - sign in again"),
]


@pytest.mark.parametrize(("claims", "reason"), UNHANDOFFABLE, ids=[f"{c}" for c, _ in UNHANDOFFABLE])
def test_a_session_that_cannot_be_handed_off_is_refused_and_named(claims, reason):
    """The verdict refuses, and :func:`auth.issue_handoff` raises 401 with its reason.

    The refusal reaches the operator: a handoff that failed silently would look
    like a handoff that worked.
    """
    assert auth.handoff_verdict(claims, NOW) == {"ok": False, "reason": reason}

    with pytest.raises(HTTPException) as raised:
        auth.issue_handoff(claims if isinstance(claims, dict) else {}, now=NOW)
    assert raised.value.status_code == 401
    assert raised.value.detail


def test_a_handoff_never_outlives_the_session_it_came_from(monkeypatch):
    """A handoff expires at its own TTL or with the session, whichever is sooner.

    The URL-borne token is the copy most likely to be screenshotted or logged,
    so a session with one minute left must not hand out five.
    """
    monkeypatch.setenv(auth._ENV + "HANDOFF_TTL", "300")
    expiring_soon = auth.handoff_verdict({"exp": NOW + 60}, NOW)
    plenty_of_time = auth.handoff_verdict({"exp": NOW + 10**6}, NOW)

    assert expiring_soon == {"ok": True, "exp": int(NOW + 60)}
    assert plenty_of_time == {"ok": True, "exp": int(NOW + 300)}


def test_a_handoff_carries_the_session_identity_so_the_cap_survives_the_copy(monkeypatch):
    """The minted handoff keeps ``sub``/``name``/``iat0`` and is marked as a handoff.

    Dropping ``iat0`` here would hand the copy a fresh cap, which is the same
    forever session by another route.
    """
    monkeypatch.setenv(auth._ENV + "HANDOFF_TTL", "300")
    t0 = int(time.time())
    session = {"sub": "cred1", "name": "Ada", "iat0": t0 - 5000, "exp": t0 + 10**6}

    minted = auth.issue_handoff(session, now=t0)
    claims = auth.verify_token(minted["token"])

    assert claims["sub"] == "cred1"
    assert claims["name"] == "Ada"
    assert claims["iat0"] == t0 - 5000
    assert claims["via"] == "handoff"
    assert minted["exp"] == t0 + 300
    assert minted["expires_in"] == 300
