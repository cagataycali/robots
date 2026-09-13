"""A renewed dashboard session stays recognisable as the sign-in it came from.

:func:`auth.issue_token` writes a ``via`` claim that "marks how the token was
minted (e.g. "handoff") for later forensics", and :func:`auth.issue_handoff` is
its only writer: the short-lived token that rides in a URL to move a session to
a phone. :func:`auth.renew_if_due` re-minted that session WITHOUT ``via``, so the
marker was erased and a session that began as a URL handoff became
indistinguishable from one signed by a passkey.

It was erased at the first use, not eventually. Renewal is due once
``now >= exp - TOKEN_TTL / 2``, and that threshold is measured against the
configured ``TOKEN_TTL`` (a day) rather than the lifetime the presented token
actually has. A handoff token lives for ``HANDOFF_TTL`` (five minutes), so it is
already 42,900 seconds past that threshold the moment it is minted - and
:func:`auth.issue_handoff` documents renewal as the expected next step for it
("carrying the session's identity (sub/name/iat0) so renewal caps survive the
copy"). The one claim that recorded where the session came from was the one claim
the copy did not survive.

``via`` is now carried exactly as ``iat0`` is, which the module already describes
as "carried unchanged through every renewal". These pins hold both halves: the
marker survives a renewal and a chain of them, and a session that never came
from a handoff still gains no marker. The renewal arithmetic that was already
right is pinned alongside it - a renewal never shortens what the client holds,
never passes the absolute cap, and is refused for a token that does not verify -
because this verb had no test at all before.
"""

from __future__ import annotations

import time

import jwt
import pytest

import strands_robots.dashboard.auth as auth


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the store reader at a per-test file.

    Every cell here mints a token, and :func:`auth.issue_token` signs with
    ``_jwt_secret()``, which reads the credential store. An unset ``STORE``
    resolves to ``~/.strands_dashboard/auth.json`` - the file that decides
    whether a dashboard on this machine is sealed - so without this redirect
    these cells would create one on a machine that has none, or rename an
    unparseable one aside and write a fresh JWT secret, invalidating every live
    session token. The sibling ``test_dashboard_auth_*`` modules that reach the
    store all carry this redirect.
    """
    monkeypatch.setenv(auth._ENV + "STORE", str(tmp_path / "auth.json"))
    auth._cache = {}
    yield


def claims_of(token: str) -> dict:
    """The claims of ``token``, read without re-checking its expiry."""
    return jwt.decode(token, auth._jwt_secret(), algorithms=["HS256"], options={"verify_exp": False})


def a_handoff_of(session_exp: float, now: float) -> str:
    """The URL token :func:`auth.issue_handoff` mints for a live session."""
    session = {"sub": "cred1", "name": "operator", "iat": now, "iat0": now, "exp": session_exp}
    return str(auth.issue_handoff(session, now=now)["token"])


# --- the marker survives the renewal that erased it ---------------------------


def test_a_renewed_handoff_session_is_still_marked_as_one():
    now = time.time()
    handoff = a_handoff_of(now + 10 * auth._token_ttl(), now)
    assert claims_of(handoff)["via"] == "handoff"

    renewed = auth.renew_if_due(handoff)

    assert renewed is not None, "a handoff token is past the renewal threshold when minted"
    assert claims_of(renewed).get("via") == "handoff"


def test_the_marker_survives_a_chain_of_renewals():
    """``via`` is carried "unchanged through every renewal", as ``iat0`` is."""
    now = time.time()
    ttl = auth._token_ttl()
    token = a_handoff_of(now + 10 * ttl, now)
    iat0 = claims_of(token)["iat0"]

    # The first hop is due at once (a handoff is minted past the threshold); each
    # later hop is asked for once the token the previous one issued is past its
    # own half-life.
    for renewal, elapsed in enumerate([0.0, 0.6, 1.2], start=1):
        token = auth.renew_if_due(token, now=now + elapsed * ttl)
        assert token is not None, f"renewal {renewal} was refused"
        assert claims_of(token).get("via") == "handoff", f"marker lost at renewal {renewal}"
        assert claims_of(token)["iat0"] == iat0, f"original sign-in lost at renewal {renewal}"


def test_a_passkey_session_gains_no_marker():
    """Only a handoff is marked: renewal must not invent provenance."""
    now = time.time()
    ttl = auth._token_ttl()
    signed_in = auth.issue_token("cred1", "operator", iat0=int(now - 0.6 * ttl), exp=int(now + 0.4 * ttl))
    assert "via" not in claims_of(signed_in)

    renewed = auth.renew_if_due(signed_in, now=now)

    assert renewed is not None
    assert "via" not in claims_of(renewed)


# --- the arithmetic this verb had no pin for ----------------------------------


def test_a_renewal_extends_without_shortening_or_passing_the_cap():
    now = time.time()
    ttl, max_age = auth._token_ttl(), auth._session_max_age()
    iat0 = int(now - 0.6 * ttl)
    held_exp = int(now + 0.4 * ttl)

    renewed = auth.renew_if_due(auth.issue_token("cred1", "operator", iat0=iat0, exp=held_exp), now=now)

    assert renewed is not None
    fresh = claims_of(renewed)
    assert fresh["exp"] > held_exp, "a renewal the client cannot refuse must not shave time off"
    assert fresh["exp"] <= iat0 + max_age, "a renewal must not outlive the absolute cap"
    assert fresh["iat0"] == iat0, "the original sign-in decides the cap, so it is carried"


# reason a session is handed no fresh token -> a token in exactly that state.
# The verb answers None for all of them; the reason is what distinguishes a
# login problem (sign in again) from a renewal that simply is not due yet.
#
# Only the *names* are known at collection. The tokens are minted inside the
# test, under ``isolated_store``: a ``parametrize`` argument is evaluated when
# pytest imports this module, which it does while collecting every suite run,
# and ``auth.issue_token`` signs with ``_jwt_secret()``, which reads - and on a
# missing or unparseable file, writes - the store at the unredirected path.
NO_RENEWAL_STATES = (
    "nothing presented",
    "not a token at all",
    "signed with another secret",
    "already expired",
    "still fresh",
    "past its maximum age",
)


def _no_renewal_cases(now: float) -> dict[str, str]:
    ttl, max_age = auth._token_ttl(), auth._session_max_age()
    return {
        "nothing presented": "",
        "not a token at all": "not.a.token",
        "signed with another secret": jwt.encode({"sub": "x", "exp": now + ttl}, "not-the-secret", algorithm="HS256"),
        "already expired": auth.issue_token("cred1", exp=int(now - 5), iat0=int(now - ttl)),
        "still fresh": auth.issue_token("cred1", "operator"),
        "past its maximum age": auth.issue_token(
            "cred1", "operator", iat0=int(now - max_age - 10), exp=int(now + 0.4 * ttl)
        ),
    }


@pytest.mark.parametrize("state", NO_RENEWAL_STATES)
def test_no_fresh_token_for_a_session_that_cannot_or_need_not_renew(state):
    now = time.time()
    cases = _no_renewal_cases(now)
    assert set(cases) == set(NO_RENEWAL_STATES), "a state was added on one side only"
    assert auth.renew_if_due(cases[state], now=now) is None
