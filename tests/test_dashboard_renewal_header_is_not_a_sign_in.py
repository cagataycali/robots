"""A response header never replaces the operator's sign-in unless it is a renewal the page asked for.

``absorbRenewedSession`` read ``X-Session-Token`` off EVERY response the
dashboard received, success or refusal, from whichever host answered, and
wrote the value over the held bearer after one shape test (three dot
separated segments). No server route in this repository sends that header
(``POST /api/auth/renew`` renews by setting the ``strands_dash`` cookie), so
the only party that could ever exercise the path was one that is not the real
backend: a 404 with a header from a re-pointed or injected host swapped the
credential silently and persistently, either fixing the operator onto an
identity the attacker holds or turning every guarded request into a 401
(finding f018, Medium).

These cells hold the channel to the narrow shape a renewal has: only a
successful answer, only to the renewal route this page called, only when the
page holds a token it is sending to that host (``authToken()``, which the
token-to-host binding of finding f003 narrows further), and only when the
offered token decodes to the same subject with a later expiry. Everything else
leaves the stored sign-in exactly as it was.
"""

from __future__ import annotations

import base64
import json
import re
import time

from tests._dashboard_frontend import LIB, requires_node, run_frontend

ENDPOINTS = LIB / "endpoints.ts"


def _jwt(sub: str = "operator", exp_in_s: int = 3600, payload: str | None = None) -> str:
    """A JWT-shaped token whose payload the browser can decode, minted at call time (the signature is opaque to it)."""
    if payload is None:
        payload = json.dumps({"sub": sub, "exp": int(time.time()) + exp_in_s, "via": "passkey"})
    seg = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{seg}.c2ln"


def _current() -> str:
    return _jwt(exp_in_s=1800)


def _renewed() -> str:
    return _jwt(exp_in_s=7200)


def _after_answer(path: str, status: int, offered: str | None, *, setup: str = "", stored: str | None = None) -> dict:
    if stored is None:
        stored = _current()
    header = {} if offered is None else {"X-Session-Token": offered}
    return run_frontend(
        f"""
const m = await import('./endpoints.ts')
if ({stored!r}) m.setAuthToken({stored!r})
{setup}
globalThis.answer = {{ status: {status}, headers: {json.dumps(header)}, body: '{{"renewed": true}}' }}
await m.api({path!r}).catch(() => null)
out({{ token: m.authToken(), renewed_at: m.lastRenewalAt() }})
"""
    )


@requires_node
class TestARefusalOrAnUnrelatedAnswerNeverSwapsTheSignIn:
    def test_a_404_carrying_the_header_leaves_the_token_alone(self) -> None:
        """The attack: any status, an empty body, one header. The credential must not move."""
        current = _current()
        renewed = _renewed()
        got = _after_answer("/api/fleet", 404, renewed, stored=current)
        assert got["token"] == current, f"a refused answer replaced the sign-in: {got}"
        assert got["renewed_at"] == 0

    def test_a_401_carrying_the_header_leaves_the_token_alone(self) -> None:
        current = _current()
        renewed = _renewed()
        got = _after_answer("/api/fleet", 401, renewed, stored=current)
        assert got["token"] == current, got

    def test_a_successful_answer_to_any_other_route_leaves_the_token_alone(self) -> None:
        """Renewal is something the page asks for; a fleet listing is not that request."""
        current = _current()
        renewed = _renewed()
        got = _after_answer("/api/fleet", 200, renewed, stored=current)
        assert got["token"] == current, f"a header on an unrelated route replaced the sign-in: {got}"

    def test_a_page_holding_no_token_accepts_none_from_a_header(self) -> None:
        renewed = _renewed()
        got = _after_answer("/api/auth/renew", 200, renewed, stored="")
        assert got["token"] in (None, ""), got


@requires_node
class TestARenewalMustBeTheSameSessionExtended:
    def test_a_token_for_another_subject_is_refused(self) -> None:
        """Session fixation proper: the attacker's own identity offered as the operator's renewal."""
        current = _current()
        got = _after_answer("/api/auth/renew", 200, _jwt(sub="attacker", exp_in_s=7200), stored=current)
        assert got["token"] == current, got

    def test_a_token_that_does_not_extend_the_session_is_refused(self) -> None:
        current = _current()
        got = _after_answer("/api/auth/renew", 200, _jwt(exp_in_s=60), stored=current)
        assert got["token"] == current, got

    def test_a_token_whose_claims_do_not_decode_is_refused(self) -> None:
        current = _current()
        for offered in ("a.b.c", "x.eyJub3QganNvbg.y", _jwt(payload='{"exp": "soon"}')):
            got = _after_answer("/api/auth/renew", 200, offered, stored=current)
            assert got["token"] == current, (offered, got)

    def test_the_renewal_the_page_asked_for_is_accepted(self) -> None:
        """The one legitimate shape: 200 from the token's host on the renewal route, same subject, later expiry."""
        renewed = _renewed()
        got = _after_answer("/api/auth/renew", 200, renewed)
        assert got["token"] == renewed, got
        assert got["renewed_at"] > 0


class TestTheAbsorbIsAfterTheSuccessCheck:
    def test_no_call_site_absorbs_before_it_knows_the_answer_was_ok(self) -> None:
        """The header is read inside the success branch, with the route it answered."""
        source = ENDPOINTS.read_text(encoding="utf-8")
        calls = [m.start() for m in re.finditer(r"^\s*absorbRenewedSession\(res, path\)", source, re.M)]
        assert len(calls) == 2, "api() and apiBlob() each absorb once, naming the route that was called"
        for at in calls:
            before = source[:at]
            ok_check = before.rfind("if (!res.ok)")
            fetch_at = before.rfind("await fetch(")
            assert ok_check > fetch_at, (
                "absorbRenewedSession(res, path) runs before the !res.ok check of its own request"
            )
