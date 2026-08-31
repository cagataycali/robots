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
