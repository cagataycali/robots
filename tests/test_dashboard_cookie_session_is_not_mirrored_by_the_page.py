"""The page keeps no copy of a passkey session: the ``HttpOnly`` cookie is the session.

``AuthGate`` took the token the ceremony's answer used to carry and wrote it to
``localStorage`` through ``setAuthToken``, where every request read it back as
a bearer. That copy, not the cookie, was the credential in use, readable by any
script in the origin and sent before the cookie was ever consulted (finding
f015). The routes no longer answer with the token; these cells hold the page to
its side: a finished ceremony records WHEN the session lapses and a new
connection identity, and nothing else.
"""

from __future__ import annotations

import re

from tests._dashboard_frontend import FRONTEND_SRC, LIB, requires_node, run_frontend

AUTH_GATE = FRONTEND_SRC / "components" / "AuthGate.tsx"
PASSKEY = LIB / "passkey.ts"


@requires_node
class TestACookieSessionLeavesNothingInStorage:
    def test_recording_the_session_stores_no_token_and_changes_the_connection_key(self) -> None:
        got = run_frontend(
            """
const m = await import('./endpoints.ts')
const before = m.backendKey()
m.noteCookieSession(1_900_000_000)
await m.api('/api/fleet').catch(() => null)
out({ before, after: m.backendKey(), token: m.authToken(), stored: [...Array(localStorage.length).keys()].map(i => localStorage.key(i)),
      bearer: sent.map(s => s.headers.Authorization ?? null), exp: m.cookieSessionExpiry() })
"""
        )
        assert got["stored"] == [], f"a cookie session left something in storage: {got}"
        assert got["token"] == "" and got["bearer"] == [None], "the request must ride the cookie, not a bearer"
        assert got["before"] != got["after"], "the app would not remount on sign-in"
        assert got["exp"] == 1_900_000_000

    def test_the_expiry_verdict_reads_the_recorded_expiry(self) -> None:
        got = run_frontend(
            """
const m = await import('./endpoints.ts')
const s = await import('./sessionExpiry.ts')
const now = 1_700_000_000
m.noteCookieSession(now + 120)
const v = s.sessionVerdictAt(m.cookieSessionExpiry(), now)
out({ state: v.state, left: v.expiresInS })
"""
        )
        assert got == {"state": "expiring", "left": 120}


class TestTheCeremonyAnswerIsNeverStored:
    def test_the_gate_records_the_grant_and_does_not_store_a_token_from_it(self) -> None:
        source = AUTH_GATE.read_text(encoding="utf-8")
        run = source[source.index("async function run(") :]
        run = run[: run.index("\n  }\n") + 4]
        assert "noteCookieSession(" in run, "the gate does not record the cookie session"
        assert "setAuthToken(" not in run, "the gate still stores what the ceremony answered"

    def test_the_ceremony_helpers_do_not_hand_back_a_token(self) -> None:
        source = PASSKEY.read_text(encoding="utf-8")
        assert not re.search(r"res\.token", source), "passkey.ts still reads a token out of the ceremony's answer"
        assert "Promise<SessionGrant>" in source
