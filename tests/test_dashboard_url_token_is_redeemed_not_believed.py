"""A ``?token=`` in the dashboard's address is redeemed against the backend, never simply believed.

``absorbUrl()`` wrote whatever ``?token=`` held straight into the stored
sign-in on page load, before any component rendered, with truthiness as the
only check, and then scrubbed the parameter from the address bar. Any junk
string signed the operator out mid-session; a token the backend would verify
(a captured hand-off link, a session the attacker holds) made the operator's
browser act under that other identity, silently, for every command and motion
approval that followed (finding f019, Medium).

The one legitimate producer is the LAN hand-off (``handoffHref``): a token the
server minted with ``via="handoff"`` and a short life, carried to a plain-http
address where WebAuthn is unavailable. These cells hold the absorbing side to
that shape and to one proof: the offered token is parked in memory, never in
storage, until ``GET /api/auth/status`` on the backend this page is already
configured for answers ``authenticated: true`` for it. It is dropped without a
probe when it is not a hand-off token, has expired, arrives beside a
``?backend=`` that moves the page, or when this browser already holds a valid
sign-in for the backend (the existing session wins).
"""

from __future__ import annotations

import base64
import json
import re
import time

from tests._dashboard_frontend import LIB, requires_node, run_frontend

ENDPOINTS = LIB / "endpoints.ts"


def _jwt(via: str = "handoff", exp: int | None = None, sub: str = "operator") -> str:
    """A JWT-shaped token the browser can decode, minted at call time (a CI run outlives a module constant)."""
    now = int(time.time())
    payload = json.dumps({"sub": sub, "exp": now + 240 if exp is None else exp, "via": via})
    seg = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{seg}.c2ln"


def _handoff() -> str:
    return _jwt()


def _session(exp_in_s: int = 8 * 3600) -> str:
    return _jwt(via="passkey", exp=int(time.time()) + exp_in_s)


STATUS_YES = '{"authenticated": true, "auth_enabled": true}'
STATUS_NO = '{"authenticated": false, "auth_enabled": true}'


def _redeem(
    page: str,
    *,
    answer_status: int = 200,
    answer_body: str = STATUS_NO,
    stored: str | None = None,
    bare_status: int = 200,
    bare_body: str = STATUS_NO,
) -> dict:
    """Redeem the page's ``?token=``; ``answer_*`` is what the backend says to the offered bearer.

    ``bare_*`` is what it says to a request carrying no bearer, where only the
    HttpOnly cookie could speak for the browser; the default is "nobody is signed
    in here", the state of a fresh page.
    """
    setup = "" if stored is None else f"localStorage.setItem('strands.token', {stored!r})\n"
    return run_frontend(
        setup
        + f"""
const m = await import('./endpoints.ts')
const before = {{ token: localStorage.getItem('strands.token'), auth: m.authToken() }}
const bearerAnswer = {{ status: {answer_status}, headers: {{}}, body: {answer_body!r} }}
const bareAnswer = {{ status: {bare_status}, headers: {{}}, body: {bare_body!r} }}
const answering = globalThis.fetch
globalThis.fetch = (url, init = {{}}) => {{
  globalThis.answer = (init.headers ?? {{}}).Authorization ? bearerAnswer : bareAnswer
  return answering(url, init)
}}
const outcome = await m.redeemUrlToken()
out({{ before, outcome, token: localStorage.getItem('strands.token'), auth: m.authToken(),
      sent: sent.map(s => ({{ url: s.url, bearer: s.headers.Authorization ?? null }})), replaced }})
""",
        page=page,
    )


@requires_node
class TestAUrlTokenIsNeverTheSignInBeforeTheBackendSaysSo:
    def test_junk_in_the_parameter_does_not_touch_the_stored_sign_in(self) -> None:
        """The denial of service: any non-empty string used to sign the operator out."""
        session = _session()
        got = _redeem("http://robot.lan:8090/?token=junk", stored=session)
        assert got["before"]["token"] == session, f"the stored sign-in was replaced on load: {got}"
        assert got["token"] == session
        assert got["outcome"] == "refused"
        assert got["sent"] == [], "junk was even tried against the backend"

    def test_a_handoff_token_is_not_in_storage_before_it_is_redeemed(self) -> None:
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}")
        assert got["before"] == {"token": None, "auth": ""}, f"the URL token was believed on load: {got}"

    def test_the_backend_is_asked_once_with_the_offered_token_and_no_is_no(self) -> None:
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_NO)
        assert got["sent"] == [
            {"url": "/api/auth/status", "bearer": None},
            {"url": "/api/auth/status", "bearer": f"Bearer {handoff}"},
        ], got
        assert got["outcome"] == "refused"
        assert got["token"] is None and got["auth"] == ""

    def test_a_yes_from_the_backend_adopts_the_token(self) -> None:
        """The LAN hand-off still works: the server that minted it says it is good."""
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES)
        assert got["outcome"] == "adopted"
        assert got["token"] == handoff and got["auth"] == handoff

    def test_an_error_or_a_shapeless_answer_is_a_no(self) -> None:
        handoff = _handoff()
        for status, body in ((500, STATUS_YES), (200, "{}"), (200, '{"authenticated": "true"}'), (200, "not json")):
            got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_status=status, answer_body=body)
            assert got["outcome"] == "refused", (status, body, got)
            assert got["token"] is None, (status, body, got)

    def test_the_parameter_is_scrubbed_whatever_the_outcome(self) -> None:
        handoff = _handoff()
        for body in (STATUS_YES, STATUS_NO):
            got = _redeem(f"http://robot.lan:8090/?token={handoff}&view=fleet", answer_body=body)
            assert got["replaced"] and all("token=" not in u for u in got["replaced"]), got["replaced"]
            assert got["replaced"][-1] == "/?view=fleet", got["replaced"]

    def test_no_parameter_is_nothing_to_redeem(self) -> None:
        session = _session()
        got = _redeem("http://robot.lan:8090/", stored=session)
        assert got["outcome"] == "none" and got["sent"] == [] and got["token"] == session


@requires_node
class TestOnlyAHandoffCanRideInTheUrl:
    def test_a_full_session_token_is_dropped_without_a_probe(self) -> None:
        """The server only ever puts ``via="handoff"`` tokens in a link; anything else did not come from it."""
        session = _session()
        got = _redeem(f"http://robot.lan:8090/?token={session}", answer_body=STATUS_YES)
        assert got["outcome"] == "refused" and got["sent"] == [] and got["token"] is None, got

    def test_an_expired_handoff_is_dropped_without_a_probe(self) -> None:
        got = _redeem(f"http://robot.lan:8090/?token={_jwt(exp=int(time.time()) - 5)}", answer_body=STATUS_YES)
        assert got["outcome"] == "refused" and got["sent"] == [], got

    def test_a_token_beside_a_backend_that_moves_the_page_is_dropped(self) -> None:
        """One link may not choose both the server and the credential."""
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?backend=https://evil.example&token={handoff}", answer_body=STATUS_YES)
        assert got["outcome"] == "refused", got
        assert got["sent"] == [], f"the token was tried somewhere: {got}"
        assert got["token"] is None

    def test_a_valid_existing_sign_in_is_kept_over_the_link(self) -> None:
        """Never silently replace a working session; the operator following their own link keeps it."""
        session = _session()
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES, stored=session)
        assert got["outcome"] == "refused" and got["sent"] == [], got
        assert got["token"] == session

    def test_an_expired_existing_sign_in_gives_way_to_a_redeemed_handoff(self) -> None:
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES, stored=_session(-60))
        assert got["outcome"] == "adopted" and got["token"] == handoff, got


@requires_node
class TestAPasskeyCookieSessionIsKeptOverTheLink:
    """The primary sign-in is the HttpOnly passkey cookie, which no script can read.

    ``authToken()`` is empty for a passkey operator, so the bearer-only guard let
    the probe proceed; the server prefers a bearer over the cookie, so a valid
    hand-off in a link answered yes, was adopted, and every ``api()`` call then
    acted under that other identity while the cookie session was shadowed
    (CWE-384 on the main auth path). The page asks the backend bare first: the
    cookie rides that same-origin fetch, and a yes means the link loses.
    """

    def test_a_cookie_signed_in_browser_refuses_the_link_without_offering_it(self) -> None:
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES, bare_body=STATUS_YES)
        assert got["outcome"] == "refused", got
        assert got["token"] is None and got["auth"] == "", f"the link shadowed the cookie session: {got}"
        assert got["sent"] == [{"url": "/api/auth/status", "bearer": None}], f"the offered token was still sent: {got}"

    def test_a_bare_answer_the_page_cannot_read_is_not_a_no(self) -> None:
        """Fail closed: when the page cannot learn whether someone is signed in, the link proves nothing."""
        handoff = _handoff()
        for status, body in ((500, STATUS_NO), (200, "{}"), (200, "not json")):
            got = _redeem(
                f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES, bare_status=status, bare_body=body
            )
            assert got["outcome"] == "refused" and got["token"] is None, (status, body, got)
            assert all(s["bearer"] is None for s in got["sent"]), (status, body, got)

    def test_a_browser_nobody_is_signed_into_still_redeems_a_good_handoff(self) -> None:
        """The rule takes nothing from the LAN hand-off: bare no, then the offered bearer's yes."""
        handoff = _handoff()
        got = _redeem(f"http://robot.lan:8090/?token={handoff}", answer_body=STATUS_YES, bare_body=STATUS_NO)
        assert got["outcome"] == "adopted" and got["token"] == handoff, got
        assert [s["bearer"] for s in got["sent"]] == [None, f"Bearer {handoff}"], got


class TestTheUrlPathNeverWritesStorage:
    def test_absorb_url_does_not_store_the_parameter(self) -> None:
        source = ENDPOINTS.read_text(encoding="utf-8")
        body = source[source.index("function absorbUrl") :]
        body = body[: body.index("\nexport function backendBase")]
        assert not re.search(r"localStorage\.setItem\(\s*TOKEN_KEY", body), (
            "absorbUrl() still writes ?token= into storage"
        )
        assert "setAuthToken(" not in body, "absorbUrl() still adopts ?token= without the backend's answer"

    def test_the_probe_is_the_public_status_route_with_the_offered_bearer(self) -> None:
        source = ENDPOINTS.read_text(encoding="utf-8")
        body = source[source.index("export async function redeemUrlToken") :]
        assert "'/api/auth/status'" in body and "authenticated === true" in body
