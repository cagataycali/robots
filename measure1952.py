"""Measure the three port surfaces of the remote-inference pair."""
import json, sys
from pathlib import Path
import strands_robots.inference.server as smod
from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies import MockPolicy

TREE = str(Path(smod.__file__).parents[2])

CASES = [("8765", 8765), ("1", 1), ("65535", 65535), ("0", 0), ("-1", -1),
         ("70000", 70000), ("2.7", 2.7), ("True", True), ("nan", float("nan")),
         ("inf", float("inf")), ("'8765'", "8765"), ("[8765]", [8765]), ("None", None)]

def _strict_int(v):
    return isinstance(v, int) and not isinstance(v, bool)

def bind_ok(v):   # a server may ask for an ephemeral port
    return _strict_int(v) and (v == 0 or 1 <= v <= 65535)

def dial_ok(v):   # a client must name a port it can address
    return _strict_int(v) and 1 <= v <= 65535

def verdict(fn):
    try:
        fn()
        return "accepted"
    except BaseException as e:      # noqa: BLE001 - an escape past the contract is the finding
        return f"refused ({type(e).__name__})"

rows = []
for label, v in CASES:
    # server constructor
    s = verdict(lambda v=v: PolicyServer(policy=MockPolicy(), port=v))
    # client constructor
    c = verdict(lambda v=v: RemotePolicy(port=v))
    # CLI (serve() stubbed so nothing binds)
    orig = PolicyServer.serve
    PolicyServer.serve = lambda self: None
    try:
        cli = verdict(lambda v=v: smod.main(["--provider", "mock", "--port", repr(v) if isinstance(v, str) else str(v)]))
    finally:
        PolicyServer.serve = orig
    rows.append({
        "label": label, "server": s, "client": c, "cli": cli,
        "bind_ok": bind_ok(v), "dial_ok": dial_ok(v),
    })

# the documented ephemeral path still works end to end
try:
    srv = PolicyServer(policy=MockPolicy(), port=0).start()
    ephemeral = srv.port
    srv.stop()
except BaseException as e:          # noqa: BLE001
    ephemeral = f"{type(e).__name__}"

out = {"tree": TREE, "rows": rows, "ephemeral_bound_port": ephemeral}
print(json.dumps(out))
