"""A request that sent no ``Host`` header is not read as a browser on this machine.

Both halves of a WebAuthn expectation are derived from the ``Host`` header --
the ``rp_id`` a credential is cryptographically bound to, and the origin the
authenticator's signed ``clientDataJSON`` is compared against. Each door stood
in a value when the header was absent, and the values they stood in were the
two most permissive readings available:

* ``_derive_rp_id`` read ``"localhost"``, and
  :func:`~strands_robots.dashboard.auth.rp_id_verdict`
  answers ``"loopback"`` for that -- a verdict that outranks both
  ``STRANDS_DASH_AUTH_RP_ID`` and the enrolled-credential set, and exists
  because a browser on this machine is the operator. So a host that was WRONG
  was refused while a host that was MISSING was trusted absolutely.
* ``_served_origin`` read ``"localhost:8090"``, an authority and a port nobody
  configured.
  :func:`~strands_robots.dashboard.auth.origin_verdict`
  then compares the caller's own ``Origin`` against that invention, which the
  caller passes by offering the invented value -- the tautology that function
  exists to refuse, reached through a default spelled in the source.

Absence is not a hostname. The pin still answers for the deployment it exists
for (a proxy that rewrites ``Host``); everything else is refused, for the same
reason ``_connection_scheme`` refuses a transport that reports no scheme: an
expectation is refused rather than guessed.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from strands_robots.dashboard import auth

ENROLLED = {"dash.example.com"}


def request(scheme: str = "https", **headers: str) -> Request:
    """A request that really arrived over ``scheme`` carrying only ``headers``."""
    return Request(
        {
            "type": "http",
            "scheme": scheme,
            "method": "POST",
            "path": "/auth/login/begin",
            "query_string": b"",
            "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()],
            "server": ("dash.example.com", 443),
            "client": ("203.0.113.9", 5555),
        }
    )


@pytest.fixture
def enrolled(monkeypatch):
    """A deployment with one enrolled credential and no pins."""
    monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_ORIGIN", raising=False)
    monkeypatch.setattr(auth, "known_rp_ids", lambda store=None: set(ENROLLED))


# --- the rp_id door -------------------------------------------------------

#: ``(label, host, pin, enrolled, expected_rp_id_or_None)``. ``None`` means the
#: ceremony is refused. A missing host must never be answered from the store's
#: own name, and never with the loopback verdict.
RP_ID = [
    ("the host it was reached at", "dash.example.com", "", ENROLLED, "dash.example.com"),
    ("a host nobody enrolled", "evil.example", "", ENROLLED, None),
    ("no host at all", None, "", ENROLLED, None),
    ("no host, nothing enrolled yet", None, "", set(), None),
    ("no host, but the operator pinned one", None, "dash.example.com", ENROLLED, "dash.example.com"),
    ("loopback really is loopback", "localhost", "", ENROLLED, "localhost"),
]


@pytest.mark.parametrize(("label", "host", "pin", "known", "expected"), RP_ID, ids=[r[0] for r in RP_ID])
def test_the_rp_id_is_decided_from_a_host_that_arrived(label, host, pin, known, expected, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_RP_ID", pin)
    monkeypatch.setattr(auth, "known_rp_ids", lambda store=None: set(known))
    req = request() if host is None else request(host=host)
    if expected is None:
        with pytest.raises(HTTPException) as e:
            auth._derive_rp_id(req)
        assert e.value.status_code == 400
    else:
        assert auth._derive_rp_id(req) == expected


def test_an_absent_host_does_not_reach_the_loopback_verdict():
    """The verdict is what makes this a defect and not a cosmetic default:
    loopback is checked before the pin and before the store, so a host read in
    from absence is trusted more than any host a caller could send."""
    assert auth.rp_id_verdict("localhost", "pinned.example")[1] == "loopback"
    rp_id, why = auth.rp_id_verdict("", "", ENROLLED)
    assert rp_id is None
    assert "no Host header" in why


def test_the_refusal_names_the_reading_that_was_missing(enrolled):
    """An operator behind a proxy that drops Host has to be able to tell this
    apart from a host that was refused for being unenrolled."""
    with pytest.raises(HTTPException) as e:
        auth._derive_rp_id(request())
    assert "no Host header" in e.value.detail["detail"]
    assert "STRANDS_DASH_AUTH_RP_ID" in e.value.detail["hint"]


# --- the origin door ------------------------------------------------------


def test_the_served_origin_is_not_invented_from_an_absent_host(enrolled):
    with pytest.raises(HTTPException) as e:
        auth._served_origin(request())
    assert e.value.status_code == 400
    assert "no Host header" in e.value.detail["detail"]
    assert "STRANDS_DASH_AUTH_ORIGIN" in e.value.detail["hint"]


def test_the_served_origin_is_the_authority_that_arrived(enrolled):
    assert auth._served_origin(request(host="dash.example.com")) == "https://dash.example.com"
    assert auth._served_origin(request("http", host="dash.example.com:8443")) == "http://dash.example.com:8443"


def test_a_configured_origin_still_answers_without_a_host(monkeypatch):
    """The pin exists for the deployments whose proxy rewrites Host, so it is
    read before the header and a missing header cannot refuse them."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_ORIGIN", "https://robots.example/")
    assert auth._served_origin(request()) == "https://robots.example"


def test_a_caller_cannot_choose_the_expectation_by_dropping_the_host(enrolled):
    """The whole exploit in one call: with no Host header the expectation used
    to be ``http://localhost:8090``, a constant spelled in the source, so a
    caller offering exactly that matched an expectation it had chosen."""
    with pytest.raises(HTTPException) as e:
        auth._derive_origin(request("http", origin="http://localhost:8090"))
    assert e.value.status_code == 400
