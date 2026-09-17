### Quality: the passkey routes' HTTP shaping is graded, cookie flags included

`strands_robots.dashboard.routes_auth` decides no enrolment or sign-in verdict -
those are the auth module's, and are covered - but it does decide the HTTP a
browser depends on, and 25 of its 84 statements were never executed by a test.
Uncovered were the whole session-cookie path (both ceremonies and the renewal),
the `via != "passkey"` gates that stop a static token or a fresh-install loopback
caller from removing a passkey or minting a handoff, and every delegation into
the auth module. The cookie's flags are the part with no second owner: `HttpOnly`
keeps a page script out of the session, `SameSite=Strict` keeps another origin
from riding it, and `Secure` follows the connection rather than a setting,
without which `http://localhost` could never sign in.

Twenty cells now pin those, and the module is at 100%. Each attribute is read as
a token rather than matched as a substring, because `Path=/api` contains the text
`Path=/` while scoping the session away from the pages that need it.
