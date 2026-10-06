"""A ``?backend=`` in the dashboard's address re-points the page only after the operator says yes.

The SPA reads ``?backend=<origin>`` off its own URL so a robot on the LAN can be
dialled from a page served elsewhere. The question the Settings drawer asks for
a typed address (``connectionChange``) used to escalate only when a stored token
would follow the host. A passkey operator holds no token (the session is the
``HttpOnly`` cookie), so for the normal sign-in ``?backend=https://evil.example``
was persisted, scrubbed from the address bar and dialled with no question: the
login screen that followed was the other host's, and the passkey ceremony went
through it.

These cells hold the URL path to the rule as it now stands. Any move to another
host, or https to http on the same host, is ``host_changes`` and waits: until
``acceptUrlBackend()`` the page keeps talking to the backend it had, nothing is
persisted and the parameter stays visible. ``foreignBackendNotice()`` names the
backend whenever it is not the origin that served the page. A bearer the
operator types is held in page memory, bound to the host it was typed for,
dropped when it lapses or a ceremony signs in, and never written to storage.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from tests._dashboard_frontend import requires_node, run_frontend

EVIL = "https://evil.example"
PAGE = "http://robot.lan:8090/"
TOKEN = "static-access-token"


def _jwt(sub: str = "operator", exp_in_s: int = 3600) -> str:
    """A JWT-shaped token whose payload the browser decodes, minted at call time."""
    payload = json.dumps({"sub": sub, "exp": int(time.time()) + exp_in_s})
    seg = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{seg}.c2ln"


def _page(page: str, *, before: str = "", act: str = "") -> dict:
    """Load ``page``, run ``act`` after the import, send one guarded request, report what happened."""
    return run_frontend(
        before
        + f"""
const m = await import('./endpoints.ts')
{act}
await m.api('/api/fleet').catch(() => null)
const v = m.urlBackendVerdict()
out({{
  base: m.backendBase(),
  stored_base: localStorage.getItem('strands.backend'),
  storage: [...Array(localStorage.length).keys()].map(i => localStorage.key(i)),
  urls: sent.map(s => s.url),
  authorization: sent.map(s => s.headers.Authorization ?? null),
  replaced,
  verdict: v && {{ kind: v.kind, from: v.fromHost, to: v.toHost }},
  notice: m.foreignBackendNotice(),
}})
""",
        page=page,
    )


@requires_node
class TestAUrlBackendIsAQuestionNotAnInstruction:
    def test_a_cookie_only_page_does_not_dial_the_url_host_before_the_yes(self) -> None:
        """The attack link against the normal sign-in: no token held, so nothing used to ask."""
        got = _page(f"{PAGE}?backend={EVIL}")
        assert got["urls"] == ["/api/fleet"], f"the page dialled {EVIL} before anyone agreed: {got}"
        assert got["stored_base"] is None, got
        assert got["verdict"] == {"kind": "host_changes", "from": "robot.lan:8090", "to": "evil.example"}, got
        assert all("backend=" in url for url in got["replaced"]), "the evidence left the address bar unconfirmed"
        assert got["notice"] is None, "the page is still on its own origin"

    def test_the_operators_yes_moves_the_page_and_names_the_new_host(self) -> None:
        got = _page(f"{PAGE}?backend={EVIL}", act="m.acceptUrlBackend()")
        assert got["urls"] == [f"{EVIL}/api/fleet"], got
        assert got["stored_base"] == EVIL and got["verdict"] is None, got
        assert got["replaced"] and "backend=" not in got["replaced"][-1], got
        assert "evil.example" in (got["notice"] or ""), f"no banner names the backend the page now talks to: {got}"

    def test_the_operators_no_keeps_the_page_home_and_scrubs_the_link(self) -> None:
        got = _page(f"{PAGE}?backend={EVIL}", act="m.declineUrlBackend()")
        assert got["urls"] == ["/api/fleet"] and got["stored_base"] is None and got["verdict"] is None, got
        assert got["replaced"] and "backend=" not in got["replaced"][-1], got

    def test_https_to_http_on_the_same_host_is_the_same_question(self) -> None:
        before = "localStorage.setItem('strands.backend', 'https://robot.example:8090')\n"
        got = _page(f"{PAGE}?backend=http://robot.example:8090", before=before)
        assert got["urls"] == ["https://robot.example:8090/api/fleet"], got
        assert got["stored_base"] == "https://robot.example:8090", got
        assert got["verdict"]["kind"] == "host_changes", got

    def test_a_url_naming_the_backend_already_in_use_asks_nothing(self) -> None:
        got = _page(f"{PAGE}?backend=http://robot.lan:8090")
        assert got["verdict"] is None and got["urls"] == ["http://robot.lan:8090/api/fleet"], got
        assert got["notice"] is None, got

    @pytest.mark.parametrize(
        "raw", ["javascript:alert(1)", "file:///etc/passwd", "ftp://evil.example", "data:text/html,x"]
    )
    def test_a_scheme_fetch_cannot_speak_is_refused_and_the_page_stays_home(self, raw: str) -> None:
        got = _page(f"{PAGE}?backend={raw}")
        assert got["base"] == "" and got["urls"] == ["/api/fleet"] and got["stored_base"] is None, got

    def test_a_backend_already_chosen_is_named_on_every_load(self) -> None:
        got = _page(PAGE, before="localStorage.setItem('strands.backend', 'http://robot-a.lan:8090')\n")
        assert got["urls"] == ["http://robot-a.lan:8090/api/fleet"], got
        assert got["notice"] and "robot-a.lan:8090" in got["notice"] and "robot.lan:8090" in got["notice"], got


@requires_node
class TestABearerLivesInPageMemoryOnly:
    def test_a_typed_token_is_sent_to_its_host_and_never_stored(self) -> None:
        got = _page(PAGE, act=f"m.setAuthToken({TOKEN!r})")
        assert got["authorization"] == [f"Bearer {TOKEN}"], got
        assert got["storage"] == [], f"a bearer reached storage: {got}"

    def test_a_copy_an_older_build_stored_is_removed_and_never_sent(self) -> None:
        before = f"localStorage.setItem('strands.token', {TOKEN!r})\nlocalStorage.setItem('strands.token.host', 'robot.lan:8090')\n"
        got = _page(PAGE, before=before)
        assert got["authorization"] == [None], f"the stored bearer overrode the cookie: {got}"
        assert got["storage"] == [], got

    def test_the_token_does_not_follow_the_page_to_another_host(self) -> None:
        got = _page(PAGE, act=f"m.setAuthToken({TOKEN!r}); m.setBackendBase('robot-b.lan:8090')")
        assert got["urls"] == ["http://robot-b.lan:8090/api/fleet"] and got["authorization"] == [None], got

    @pytest.mark.parametrize(
        "act",
        [f"m.setAuthToken({_jwt(exp_in_s=-5)!r})", f"m.setAuthToken({TOKEN!r}); m.noteCookieSession(1_900_000_000)"],
        ids=["lapsed", "a-ceremony-signed-in"],
    )
    def test_a_lapsed_token_or_a_later_sign_in_drops_it(self, act: str) -> None:
        got = _page(PAGE, act=act)
        assert got["authorization"] == [None], got

    def test_a_bare_request_renews_nothing(self) -> None:
        """A host the token was not given for answers a renewal header: the held token is untouched."""
        got = run_frontend(
            f"""
const m = await import('./endpoints.ts')
m.setAuthToken({_jwt(exp_in_s=1800)!r})
const held = m.authToken()
m.setBackendBase('robot-b.lan:8090')
globalThis.answer = {{ status: 200, headers: {{ 'X-Session-Token': {_jwt(exp_in_s=7200)!r} }}, body: '{{}}' }}
await m.api(m.RENEWAL_PATH).catch(() => null)
m.setBackendBase('')
out({{ same: m.authToken() === held, renewed_at: m.lastRenewalAt(), sent: sent.map(s => s.headers.Authorization ?? null) }})
""",
            page=PAGE,
        )
        assert got == {"same": True, "renewed_at": 0, "sent": [None]}, got


# currentBase, currentToken, nextBase, nextToken, pageHost -> verdict kind
VERDICTS = [
    ("", "", "https://evil.example", "", "robot.lan:8090", "host_changes"),
    ("http://robot-a.lan", "", "http://robot-b.lan", "", "x", "host_changes"),
    ("https://robot.example", "", "http://robot.example", "", "x", "host_changes"),
    ("", "", "http://robot.lan:8090", "", "robot.lan:8090", "ok"),
    ("", "", "", "", "robot.lan:8090", "ok"),
    ("", "T", "https://evil.example", "T", "robot.lan:8090", "token_follows_host"),
    ("http://robot.example", "", "http://robot.example", "T", "x", "cleartext_token"),
]


@requires_node
@pytest.mark.parametrize("current,token,nxt,next_token,page,kind", VERDICTS)
def test_the_drawer_and_the_url_ask_on_a_host_change_alone(current, token, nxt, next_token, page, kind) -> None:
    """One rule for a typed address and a URL: a new host is a question whether or not a token moves."""
    args = json.dumps(
        {"currentBase": current, "currentToken": token, "nextBase": nxt, "nextToken": next_token, "pageHost": page}
    )
    got = run_frontend(
        f"""
const c = await import('./connectionChange.ts')
const v = c.connectionChange({args})
out({{ kind: v.kind, confirm: c.needsConfirm(v) }})
"""
    )
    assert got == {"kind": kind, "confirm": kind != "ok"}, got
