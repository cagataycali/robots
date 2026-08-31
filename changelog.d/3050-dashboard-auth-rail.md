### Added: operator dashboard auth rail, with the extra that supplies it

`strands_robots/dashboard/auth.py` is the operator authentication rail for the
web dashboard: WebAuthn passkey registration and login ceremonies, an RP-ID
verdict that separates a loopback host from a LAN one, challenge caps enforced
per process and per client IP, and a JWT session token with a renewal window.
`strands_robots/dashboard/__init__.py` carries the package's dependency gate, so
one call covers every module added to the package later, and `[dashboard]` is the
extra that supplies them.

The rail signs its session token with PyJWT, which was declared in no extra and
in no core dependency - the only copy in a developer environment arrives
transitively through an unrelated package, which is why the gap survived local
runs. `pip install 'strands-robots[dashboard]'` therefore imported the package
fine, because the gate named `fastapi`, `uvicorn` and `webauthn`, and then failed
on the one module PyJWT backs with a bare `ModuleNotFoundError: No module named
'jwt'` - exactly the reads-as-a-broken-venv failure the gate exists to prevent.
PyJWT is now in the extra and `jwt` is now gated.

`STRANDS_DASH_AUTH_ENABLED` recognizes `1`/`true`/`yes`/`on` and
`0`/`false`/`no`/`off`, and reports anything else instead of acting on it. It
previously read the variable as membership of the true-vocabulary alone behind a
non-empty check, so every other spelling - `enabled` and `y` among them -
resolved to auth-OFF *in preference to* the credential store, silently dropping
passkey auth from every guarded route on a dashboard that commands real
hardware. An unrecognized value now leaves the store as the source of truth, so
an enrolled passkey still guards the API.
