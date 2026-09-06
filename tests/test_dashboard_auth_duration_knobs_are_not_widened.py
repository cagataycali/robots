"""The three auth duration knobs resolve through one domain, or are refused.

``TOKEN_TTL``, ``SESSION_MAX_AGE`` and ``HANDOFF_TTL`` are how an operator
NARROWS the window in which a dashboard session can command real hardware. Each
was read with a bare ``int(os.getenv(...))`` under ``except ValueError: return
<the default>``, so every spelling that is not a plain integer -- ``1h``,
``30s``, ``15m``, a trailing comment -- resolved to the shipped default, and the
default is the WIDER value in all three cases. The operator who shortened the
window kept the long one and was told nothing.

The module already states the rule this broke: :func:`auth._challenge_cap`
refuses a cap it cannot use precisely so that "an operator who narrowed a cap
and mistyped it must hear about it, not silently be handed the wide default
back". These pins hold the duration knobs to the same rule, and hold the three
things deliberately left alone: an EMPTY variable still means unset, a bare
integer is still honoured end to end, and a maximum age below the token
lifetime is still a legal pair.
"""

from __future__ import annotations

import time

import jwt
import pytest

import strands_robots.dashboard.auth as auth

# The documented contract, spelled here rather than imported, so these pins fail
# if the shipped defaults move rather than moving with them.
DOCUMENTED_DEFAULTS = {"TOKEN_TTL": 86400, "SESSION_MAX_AGE": 2592000, "HANDOFF_TTL": 300}

# Durations that mean something to a human and nothing to ``int()``. Each is a
# plausible way to ask for one hour, or to annotate the number.
UNIT_SPELLINGS = ["1h", "60m", "3600s", "1 hour", "3600 # one hour", "0x10", "1e3", "3600.0", "banana", ""]


def _load_auth(monkeypatch, **env):
    """Import a private copy of the auth module under ``env``.

    A fresh module object rather than a reload, so a refused configuration
    cannot leave ``strands_robots.dashboard.auth`` unusable for the rest of the
    session. Mirrors the helper the challenge-cap pins use.
    """
    import importlib.util

    for key, value in env.items():
        monkeypatch.setenv(auth._ENV + key, value)
    spec = importlib.util.spec_from_file_location("_auth_duration_probe", auth.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- what each knob actually buys, measured through the surface that uses it --


def _token_lifetime(module) -> int:
    """Seconds a freshly minted session token is valid for."""
    token = module.issue_token("cred1")
    claims = jwt.decode(token, module._jwt_secret(), algorithms=["HS256"], options={"verify_exp": False})
    return int(claims["exp"]) - int(claims["iat"])


def _renewable_age(module) -> int:
    """Largest age, in seconds, at which a session is still handed a fresh token."""
    now = 1_000_000.0
    ttl = 86400

    def renewable(age: int) -> bool:
        claims = {"sub": "c", "iat0": now - age, "iat": now - age, "exp": now + 1}
        return "maximum age" not in module.renewal_verdict(claims, now, ttl=ttl)["reason"]

    lo, hi = 0, 400 * 86400
    if not renewable(0):
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if renewable(mid) else (lo, mid - 1)
    return lo


def _handoff_window(module) -> int:
    """Seconds a minted handoff token, the one that rides in a URL, stays usable."""
    now = time.time()
    claims = {"sub": "c", "iat": now, "iat0": now, "exp": now + 10**7}
    return int(module.issue_handoff(claims, now=now)["expires_in"])


# knob -> (its reader, the surface that spends it). The domain is only worth
# anything at the surface: a reader that refuses while its caller still resolves
# the old way would pass a reader-only pin.
KNOBS = {
    "TOKEN_TTL": ("_token_ttl", _token_lifetime),
    "SESSION_MAX_AGE": ("_session_max_age", _renewable_age),
    "HANDOFF_TTL": ("handoff_ttl", _handoff_window),
}


def test_every_duration_knob_this_module_reads_is_pinned_here():
    """A fourth duration knob must not arrive with its own bare int() and no pin."""
    assert set(KNOBS) == set(auth._DURATION_DEFAULTS) == set(DOCUMENTED_DEFAULTS)


# --- the defect: a narrowing spelled with a unit resolved to the wide default --


@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("spelling", [s for s in UNIT_SPELLINGS if s])
def test_a_duration_that_is_not_a_whole_number_of_seconds_is_refused_by_name(knob, spelling, monkeypatch):
    reader, _surface = KNOBS[knob]
    monkeypatch.setenv(auth._ENV + knob, spelling)
    with pytest.raises(ValueError, match=knob):
        getattr(auth, reader)()


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_the_refusal_names_the_value_and_the_default_it_would_have_used(knob, monkeypatch):
    """The message has to carry both, or the operator cannot see the substitution
    they were about to get."""
    monkeypatch.setenv(auth._ENV + knob, "1h")
    reader, _surface = KNOBS[knob]
    with pytest.raises(ValueError) as exc:
        getattr(auth, reader)()
    assert "'1h'" in str(exc.value)
    assert str(DOCUMENTED_DEFAULTS[knob]) in str(exc.value)


@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("value", ["0", "-1", "-3600"])
def test_a_lifetime_at_or_below_zero_is_refused(knob, value, monkeypatch):
    """It parses, and it means every token minted under it is dead on arrival."""
    monkeypatch.setenv(auth._ENV + knob, value)
    reader, _surface = KNOBS[knob]
    with pytest.raises(ValueError, match=knob):
        getattr(auth, reader)()


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_a_misspelled_duration_stops_the_import_not_a_later_login(knob, monkeypatch):
    """Where a deployment learns about it: startup, not the operator's sign-in on
    a dashboard that is already serving."""
    with pytest.raises(ValueError, match=knob):
        _load_auth(monkeypatch, **{knob: "1h"})


# --- controls: what the domain must NOT change --------------------------------


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_a_bare_integer_is_honoured_at_the_surface_that_spends_it(knob, monkeypatch):
    """The narrowing the operator asked for reaches the token, not just the reader."""
    _reader, surface = KNOBS[knob]
    module = _load_auth(monkeypatch, **{knob: "3600"})
    assert surface(module) == pytest.approx(3600, abs=1)


@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_variable_still_means_unset(knob, raw, monkeypatch):
    """Unchanged, and cross-referenced by the challenge-cap pins next door."""
    monkeypatch.setenv(auth._ENV + knob, raw)
    reader, _surface = KNOBS[knob]
    assert getattr(auth, reader)() == DOCUMENTED_DEFAULTS[knob]


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_an_unset_variable_resolves_to_the_documented_default(knob, monkeypatch):
    monkeypatch.delenv(auth._ENV + knob, raising=False)
    reader, _surface = KNOBS[knob]
    assert getattr(auth, reader)() == DOCUMENTED_DEFAULTS[knob]


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_surrounding_whitespace_is_still_accepted(knob, monkeypatch):
    monkeypatch.setenv(auth._ENV + knob, " 3600 ")
    reader, _surface = KNOBS[knob]
    assert getattr(auth, reader)() == 3600


def test_a_maximum_age_below_the_token_lifetime_is_a_legal_pair(monkeypatch):
    """Deliberately NOT a cross-knob rule, unlike the challenge-cap pair. "Tokens
    last a day, but re-authenticate every hour" is a coherent request, and the
    cap simply wins."""
    module = _load_auth(monkeypatch, TOKEN_TTL="86400", SESSION_MAX_AGE="3600")
    assert (module._token_ttl(), module._session_max_age()) == (86400, 3600)
    assert _renewable_age(module) == pytest.approx(3600, abs=1)


def test_the_underscore_grouping_python_accepts_is_still_a_whole_number(monkeypatch):
    """``3_600`` is a bare integer to ``int()``; the domain does not narrow that."""
    monkeypatch.setenv(auth._ENV + "TOKEN_TTL", "3_600")
    assert auth._token_ttl() == 3600
