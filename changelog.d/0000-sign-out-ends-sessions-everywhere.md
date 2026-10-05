### Fixed: dashboard sign-out ends the session on the server, and open sockets notice

`POST /api/auth/logout` only deleted the cookie, so a copy of the token taken
before kept working and renewing until its maximum age. Each passkey now has a
session epoch stamped into its tokens (`ep`); signing out advances it, and every
earlier token of that passkey is refused on every route, renewal and handoff
included. A token without a usable epoch is refused, so sessions minted before
this release sign in once more.

`/ws/agent`, `/ws/voice`, `/ws/mesh`, `/ws/camera` and `/ws/telemetry` checked
the credential only at the handshake. They now re-check it every
`access.SOCKET_RECHECK_SECONDS` (2 s), and `/ws/agent` before every frame, and
close with `4401` once the passkey is removed or signs out, so a consent answer
from a revoked session is never delivered.

Adding a passkey now needs the bootstrap proof (`{"bootstrap": ...}`) even with a
valid session, so a copied cookie cannot enrol a passkey that outlives the
owner's. The last passkey can be removed with the same proof, which returns the
dashboard to setup.
