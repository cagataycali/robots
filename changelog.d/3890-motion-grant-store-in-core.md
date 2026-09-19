### Changed: the motion-grant store sits below every surface that spends one

An operator's yes to a physical motion is recorded as a one-shot grant, so the
surface that runs the motion spends it instead of asking the same question a
second time. That store lived in `strands_robots.dashboard.agent_hitl`, above
three of its four users: `pose_tool`, `serial_tool` and the `Robot` agent tool
each read it through `try: from strands_robots.dashboard import agent_hitl /
except ImportError: return False`.

With the `dashboard` extra installed, the first gated call therefore imported
fastapi, uvicorn, webauthn and PyJWT on the motion path - 232 modules, 337 ms -
to read a `set` in the same process. Without the extra, "has a human already
said yes?" was answered by a failed import rather than by the store.

The store, the identity a grant is keyed on and the one reading of the
motion-bearing field roster now live in `strands_robots._motion_grants`; the
dashboard hook imports them for the line it shows the operator and for its
deposit. No behaviour a caller sees changes - the same grant is spendable by the
same one call - and nothing below the `dashboard` layer imports that package any
more except `__main__`, which is the command that starts it.
