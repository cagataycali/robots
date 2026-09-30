### Fixed: the dashboard's passkey session lives only in the `HttpOnly` cookie, and every response carries a Content-Security-Policy

A finished passkey ceremony set the session as the `HttpOnly` cookie and then repeated the same
token in the JSON body; the page stored that copy in `localStorage` and sent it as a bearer on
every request, so any script in the origin held a full session for the hardware routes. The
ceremony routes now answer without the token (plus `exp`, so the page can still warn before the
session lapses), the page records only the expiry and rides the cookie on same-origin requests,
and every response carries a Content-Security-Policy (scripts, styles, fonts and workers from
this origin only; no plugins, no `<base>`, no framing) and `Referrer-Policy: no-referrer`.
`connect-src` stays open to http(s) and ws(s) because the dashboard legitimately dials robots on
other hosts (finding f015).
