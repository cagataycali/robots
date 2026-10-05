"""A sign-in that rides in the dashboard's address is a one-time code the backend exchanges for a cookie.

The LAN handoff used to put a session bearer in the link (``?token=``), and the
page that opened it stored that bearer and sent it on every request, ahead of
the ``HttpOnly`` cookie. A link now carries ``?handoff=<code>``: the code is no
credential on any route, and ``redeemUrlHandoff()`` spends it once, with a POST
to ``/api/auth/handoff/redeem`` on this page's own backend, which answers with
the session as this device's cookie. The page keeps only the expiry.

The rules around the exchange are unchanged in spirit: a browser the backend
already knows keeps its session (a bare status probe first, where only the
cookie can speak), a code beside a ``?backend=`` that moves the page is dropped
unseen, and the parameter is scrubbed whatever the outcome. A ``?token=`` is
refused without a request: no link carries a bearer any more.
"""

from __future__ import annotations

import json

import pytest

from tests._dashboard_frontend import requires_node, run_frontend

STATUS_YES = '{"authenticated": true, "auth_enabled": true}'
STATUS_NO = '{"authenticated": false, "auth_enabled": true}'
CODE = "one-time-code"


def _redeem(
    page: str,
    *,
    redeem: tuple[int, str] = (200, '{"ok": true, "exp": 1900000000}'),
    bare: tuple[int, str] = (200, STATUS_NO),
) -> dict:
    """Open ``page`` and redeem what it carries; ``bare`` answers the status probe, ``redeem`` the exchange."""
    return run_frontend(
        f"""
const m = await import('./endpoints.ts')
const answers = {{
  status: {{ status: {bare[0]}, headers: {{}}, body: {bare[1]!r} }},
  redeem: {{ status: {redeem[0]}, headers: {{}}, body: {redeem[1]!r} }},
}}
const bodies = []
const answering = globalThis.fetch
globalThis.fetch = (url, init = {{}}) => {{
  bodies.push(init.body ?? null)
  globalThis.answer = String(url).endsWith('/redeem') ? answers.redeem : answers.status
  return answering(url, init)
}}
const outcome = await m.redeemUrlHandoff()
out({{ outcome, exp: m.cookieSessionExpiry(), auth: m.authToken(),
      storage: [...Array(localStorage.length).keys()].map(i => localStorage.key(i)),
      sent: sent.map((s, i) => ({{ url: s.url, method: s.method, bearer: s.headers.Authorization ?? null, body: bodies[i] }})),
      replaced }})
""",
        page=page,
    )


@requires_node
class TestAHandoffCodeIsExchangedOnce:
    def test_a_fresh_browser_redeems_the_code_for_a_cookie_and_holds_no_bearer(self) -> None:
        got = _redeem(f"http://robot.lan:8090/?handoff={CODE}")
        assert got["outcome"] == "adopted", got
        assert got["sent"] == [
            {"url": "/api/auth/status", "method": "GET", "bearer": None, "body": None},
            {
                "url": "/api/auth/handoff/redeem",
                "method": "POST",
                "bearer": None,
                "body": json.dumps({"code": CODE}, separators=(",", ":")),
            },
        ], got
        assert got["exp"] == 1_900_000_000 and got["auth"] == "" and got["storage"] == [], got

    @pytest.mark.parametrize("status", [401, 500])
    def test_a_refused_exchange_signs_nobody_in(self, status: int) -> None:
        got = _redeem(f"http://robot.lan:8090/?handoff={CODE}", redeem=(status, '{"error": "used"}'))
        assert got["outcome"] == "refused" and got["exp"] is None, got

    @pytest.mark.parametrize("bare", [(200, STATUS_YES), (500, STATUS_NO), (200, "{}"), (200, "not json")])
    def test_a_browser_the_backend_knows_or_cannot_vouch_for_keeps_its_session(self, bare: tuple[int, str]) -> None:
        got = _redeem(f"http://robot.lan:8090/?handoff={CODE}", bare=bare)
        assert got["outcome"] == "refused", got
        assert [s["url"] for s in got["sent"]] == ["/api/auth/status"], f"the code was spent anyway: {got}"

    def test_a_code_beside_a_backend_that_moves_the_page_is_dropped(self) -> None:
        got = _redeem(f"http://robot.lan:8090/?backend=https://evil.example&handoff={CODE}")
        assert got["outcome"] == "refused" and got["sent"] == [], got

    def test_the_code_is_scrubbed_whatever_the_outcome(self) -> None:
        for redeem in ((200, "{}"), (401, "{}")):
            got = _redeem(f"http://robot.lan:8090/?handoff={CODE}&view=fleet", redeem=redeem)
            assert got["replaced"][-1] == "/?view=fleet", got["replaced"]

    def test_no_parameter_is_nothing_to_redeem(self) -> None:
        got = _redeem("http://robot.lan:8090/")
        assert got["outcome"] == "none" and got["sent"] == [], got


@requires_node
@pytest.mark.parametrize("token", ["junk", "eyJhbGciOiJIUzI1NiJ9.eyJ2aWEiOiJoYW5kb2ZmIn0.c2ln"])
def test_a_bearer_in_the_url_is_refused_without_a_request_and_scrubbed(token: str) -> None:
    got = _redeem(f"http://robot.lan:8090/?token={token}")
    assert got["outcome"] == "refused" and got["sent"] == [] and got["auth"] == "", got
    assert got["replaced"] and "token=" not in got["replaced"][-1], got
