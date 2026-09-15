### Added: `python -m strands_robots dashboard` serves the operator dashboard from the landed rails

`strands_robots/dashboard/` carried auth, consent, settings, redaction, the
lockout type and the motion gate, and nothing that listened on a port: the
server, its CLI and the pages that documented them were removed with the closed
mega-PR, and the extra that supplies FastAPI shipped with no route to serve.

`create_app()` now wires those modules to paths and nothing else. Three routes
are public - `/api/health`, `/api/auth/status` and the ceremonies the login
screen drives - and every other route takes one dependency, `access.caller`,
which admits a passkey session, the static `security.auth_token` in constant
time, or nothing at all only while no guard is configured AND the socket peer is
loopback with no forwarding header. A query-string token is never read. The
CLI binds `127.0.0.1:8090` and refuses any other address until a passkey or a
static token exists, naming the remedy.

The UI is plain files under `dashboard/static/` served by the same process, so
the wheel carries it and no build step exists. `docs/dashboard.md` is the one
page that invokes the command, which
`test_docs_module_commands_are_dispatched` now requires of a dispatched command.
