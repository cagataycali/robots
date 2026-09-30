### Fixed: a response header only renews the dashboard session the page asked to renew

The dashboard read `X-Session-Token` off every response, success or refusal, from whichever
host answered, and wrote the value over the stored sign-in after a shape test. No route sends
that header, so a 404 with a header from a re-pointed or injected host could swap the
operator's credential silently and persistently. A renewed session is now accepted only from a
successful answer to `/api/auth/renew`, the route the page called, while the page holds a token
it is sending to that host, and only when the offered token decodes to the same subject with a
later, unexpired expiry; both call sites absorb after their own success check (finding f018).
