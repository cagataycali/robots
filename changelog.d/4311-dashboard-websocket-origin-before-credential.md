### Fixed: a dashboard WebSocket handshake is admitted by its Origin before any credential is read

`access.caller` read the `Origin` header in one branch only, the fresh-install
open posture, so a socket handshake carrying a valid passkey session cookie was
admitted with the header never read. WebSockets are exempt from CORS, the
cross-origin write middleware never sees a `websocket` scope, and
`SameSite=Strict` ignores the port, so a page on `http://localhost:3000` could
open `ws://localhost:8090/ws/agent` with the operator's cookie attached (f022,
CWE-1385). Every socket now goes through `access.admit_socket`: a foreign
`Origin` is rejected at the handshake before any cookie or token is read, a
handshake with no `Origin` is admitted only on an explicit bearer (never on the
cookie), and the credential check follows unchanged, so a same-origin page
without a session still reads 4401.
