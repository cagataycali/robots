"""The credential CHANNELS the guard accepts — one test per way in.

``TokenAuthMiddleware._presented`` reads a credential from three places:
``Authorization: Bearer``, the ``X-Dashboard-Token`` header, and a ``?token=``
query parameter. The bearer and query-string channels were already pinned by
tests/test_dashboard_auth_guard.py; ``X-Dashboard-Token`` had **zero** test hits
anywhere in the suite despite being a full credential — a refactor of
``_presented`` could have dropped or loosened it silently, and the thing it
guards moves real motors.

What is pinned here, and why each one is a security statement rather than a
coverage exercise:

* the header works, for BOTH credential kinds (static token and the WebAuthn
  session JWT that is the login escape hatch) — a channel that only half works
  is worse than one that does not exist, because the operator locked out of a
  passkey would find their token rejected with the same 401;
* a token that is a PREFIX (or an extension) of the real one is refused — the
  comparison must stay whole-value and constant-time, not ``startswith``;
* when two channels disagree the request is refused rather than searched for a
  credential that happens to work: ``Authorization`` decides, and a wrong
  bearer is a wrong request even if another header carries the right secret.
  One request presents ONE credential; "try them all" turns every extra header
  into an oracle;
* ``?token=`` on plain HTTP is accepted, which is REAL EXPOSURE (access logs,
  proxy logs, browser history) — pinned as current behaviour with the reason it
  has not simply been removed, because the tightening is the owner's call.

Raw ASGI, no lifespan, no zenoh — the same harness shape as the guard tests,
because building the app opens a mesh session and a suite joined to the live
fleet is its own incident (BUGS.md Q30/Q32).
"""

from __future__ import annotations

import asyncio

import pytest

from strands_robots.dashboard import auth
from strands_robots.dashboard import server as srv
from strands_robots.dashboard.server import TokenAuthMiddleware

STATIC = "the-real-static-token-0123456789"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
    auth._cache_key = None
    auth._cache = {}
    yield


@pytest.fixture
def static_token(monkeypatch):
    """Configure the machine-level static token, nothing else."""
    monkeypatch.setattr(
        srv.settings, "get",
        lambda section, key, default=None: STATIC
        if (section, key) == ("security", "auth_token") else default,
    )
    return STATIC


class Passed(Exception):
    """The inner app was reached."""


async def _inner_app(scope, receive, send):
    raise Passed()


def _scope(path="/api/fleet", client=("203.0.113.7", 4444), headers=None, query=b""):
    # A REMOTE client throughout: loopback would pass on the open posture and
    # the test would prove nothing about the credential.
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "client": client,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": query,
    }


def run_scope(scope):
    loop = asyncio.new_event_loop()
    try:
        mw = TokenAuthMiddleware(_inner_app)
        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        try:
            loop.run_until_complete(mw(scope, receive, send))
        except Passed:
            return "passed"
        for m in sent:
            if m["type"] == "http.response.start":
                return m["status"]
            if m["type"] == "websocket.close":
                return m["code"]
        return None
    finally:
        loop.close()


class TestTheXDashboardTokenHeader:
    def test_the_static_token_is_accepted(self, static_token):
        assert run_scope(_scope(headers={"x-dashboard-token": STATIC})) == "passed"

    def test_a_session_jwt_is_accepted_too(self, monkeypatch):
        # The login escape hatch: an operator whose passkey will not open pastes
        # a session token. If it only worked on the Authorization header, the
        # lockout it exists to relieve would look identical to a wrong token.
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
        monkeypatch.setattr(srv.settings, "get", lambda *a, **k: None)
        jwt = auth.issue_token("cred1", name="phone")
        assert run_scope(_scope(headers={"x-dashboard-token": jwt})) == "passed"

    def test_surrounding_whitespace_is_tolerated(self, static_token):
        # A pasted secret often arrives with a stray space; the value is still
        # compared whole, so this loosens formatting and not the comparison.
        assert run_scope(_scope(headers={"x-dashboard-token": f"  {STATIC} "})) == "passed"

    def test_a_wrong_token_is_refused(self, static_token):
        assert run_scope(_scope(headers={"x-dashboard-token": "nope"})) == 401

    @pytest.mark.parametrize(
        "value", [STATIC[:-1], STATIC[:8], STATIC + "x", STATIC.upper(), ""],
    )
    def test_near_misses_are_refused_whole_value(self, static_token, value):
        # Not startswith, not a prefix match, not case-insensitive: a guard that
        # accepted a prefix would let an attacker confirm one character at a time.
        assert run_scope(_scope(headers={"x-dashboard-token": value})) == 401


class TestChannelsThatDisagree:
    def test_a_wrong_bearer_beats_a_correct_header_and_the_request_is_refused(
        self, static_token
    ):
        # Pinning the DECISION, not an accident: Authorization wins, and the
        # request is judged on it alone. Searching every channel for something
        # that works would make each extra header a free guess.
        got = run_scope(_scope(headers={
            "authorization": "Bearer wrong-token",
            "x-dashboard-token": STATIC,
        }))
        assert got == 401, (
            "the guard fell back to a second credential channel after the one the "
            "client named failed - one request presents one credential"
        )

    def test_a_correct_bearer_wins_over_a_wrong_header(self, static_token):
        assert run_scope(_scope(headers={
            "authorization": f"Bearer {STATIC}",
            "x-dashboard-token": "wrong-token",
        })) == "passed"

    def test_a_non_bearer_authorization_scheme_falls_through_to_the_header(
        self, static_token
    ):
        # Basic auth is not this API's scheme; _presented only claims the
        # Authorization header when it says "Bearer", so a proxy that adds Basic
        # credentials must not lock the operator out of their own dashboard.
        assert run_scope(_scope(headers={
            "authorization": "Basic dXNlcjpwYXNz",
            "x-dashboard-token": STATIC,
        })) == "passed"


class TestTheQueryStringChannelOnPlainHttp:
    def test_it_is_accepted_today(self, static_token):
        # WebSockets NEED this channel (a browser cannot set headers on a
        # handshake). Plain HTTP does not, and a token in a URL leaks into
        # access logs, proxy logs, Referer and browser history - see BUGS.md
        # Q35. Pinned as-is because narrowing it to websockets would break any
        # curl/script the operator already has, which is their call to make;
        # this test is what tells them the exposure is deliberate rather than
        # forgotten.
        assert run_scope(_scope(query=f"token={STATIC}".encode())) == "passed"

    def test_a_wrong_query_token_is_still_refused(self, static_token):
        assert run_scope(_scope(query=b"token=nope")) == 401
