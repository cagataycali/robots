"""Repro: `run_policy(policy_provider='remote')` leaks a raw TypeError
out of the SimEngine error envelope when `websockets<17.1` is installed.

Context
-------
- `strands_robots` base deps declare NO `websockets` floor; `[inference]`
  extra pins `websockets>=17.1`.
- But the `remote` provider is advertised in the BASE registry
  (`list_providers()` lists `'remote'`) and reachable via
  `create_policy('remote')` / `run_policy(policy_provider='remote')`
  without the `[inference]` extra.
- `RemotePolicy._connect` (strands_robots/inference/client.py:316-348)
  calls `websockets.sync.client.connect(..., legacy=True)`. The
  `legacy=True` kwarg is a `websockets>=17.1` feature; on older
  websockets (16.0, which another dep can pull in) the unknown kwarg is
  forwarded to `socket.create_connection(..., legacy=True)` which raises
  `TypeError: create_connection() got an unexpected keyword argument
  'legacy'`.
- `RemotePolicy._connect` only catches `OSError`, `InvalidHandshake`,
  `ConnectionClosed`. `TypeError` propagates all the way out of
  `SimEngine.run_policy`, violating its documented return-envelope
  contract: `{"status": "success"|"error", "content": [...]}`.

Impact
------
End-user on a fresh `pip install -e .` with any dep tree that provides
`websockets < 17.1` sees a bare traceback when they follow the
documented `create_policy("remote", host=..., port=...)` quickstart in
`strands_robots/policies/__init__.py:24`. No `status="error"` envelope,
no "install `strands-robots[inference]`" hint, no "upgrade websockets to
17.1" hint. The crash looks like a Python-level bug - the exact UX the
error envelope exists to prevent.

Expected (any of)
-----------------
(a) The base registry gates `'remote'` behind the `[inference]` extra,
    so `create_policy('remote')` returns a structured
    "install strands-robots[inference]" error BEFORE any websockets
    import. This mirrors how other provider extras are gated.
(b) `RemotePolicy._connect` catches `TypeError` and reports it as an
    envelope error citing the websockets version mismatch.
(c) Base `pyproject.toml` floors `websockets>=17.1` whenever
    `strands_robots.inference.client` is reachable from the base
    import graph (least preferred, bloats base deps).

Repro
-----
    $ pip install -e .          # base install, no [inference] extra
    $ pip install 'websockets<17.1'   # or let another dep pull it
    $ python3 bugbash_repros/remote_provider_typeerror_repro.py

Exit code 1 (plus a raw TypeError traceback) demonstrates the defect.
On websockets>=17.1 the repro exits 0 and prints a note.
"""

import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from importlib.metadata import version

ws_ver = version("websockets")
print(f"[info] installed websockets: {ws_ver}")
if tuple(int(x) for x in ws_ver.split(".")[:2]) >= (17, 1):
    print(
        "[info] websockets >= 17.1 - defect not reproducible on this version.\n"
        "       Reproduce by: pip install 'websockets<17.1' and re-running."
    )
    sys.exit(0)

from strands_robots import Robot

sim = Robot("so101", mesh=False)

try:
    result = sim.run_policy("so101", policy_provider="remote", duration=0.1)
    print("[result] status:", result.get("status"))
    for block in result.get("content", []):
        if "text" in block:
            print("[result] text:", block["text"][:400])
    print("\n[PASS] run_policy returned the structured envelope.")
    sys.exit(0)
except TypeError as exc:
    print(
        f"\n[FAIL] run_policy leaked a raw TypeError "
        f"(envelope-contract violation):"
    )
    print(f"       {type(exc).__name__}: {exc}")
    print("\n[traceback]")
    traceback.print_exc()
    sys.exit(1)
