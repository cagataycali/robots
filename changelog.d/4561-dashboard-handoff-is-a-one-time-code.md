### Fixed: a dashboard handoff is a one-time code, the session cookie wins over a bearer, and a URL backend waits for the operator

`POST /api/auth/handoff` answered any caller holding the session cookie with a
bearer token in the body, the server read a bearer ahead of the cookie, and
`?backend=` re-pointed a cookie-only page without asking. A handoff now needs a
fresh passkey assertion over a `POST /api/auth/handoff/begin` challenge and
answers with a one-time code; `POST /api/auth/handoff/redeem` spends it once and
sets the session as the redeeming device's cookie, and that session is refused
as a bearer. When a request carries both, the cookie is read. The page holds a
typed bearer in memory only and deletes a copy an older build stored. Any move to
another host, or https to http, is a `host_changes` question before the page
dials it, and a persistent notice names the backend whenever it is not the page's
own origin. `connect-src` is `'self'` plus the origins listed in the new
`DASHBOARD_CONNECT_ORIGINS` environment variable. The LAN link carries
`?handoff=<code>`; a `?token=` in a URL is refused.
