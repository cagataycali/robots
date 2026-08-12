import json, logging, pathlib, sys
from unittest.mock import MagicMock
import strands_robots.mesh.session as S
import zenoh

print("TREE:", pathlib.Path(S.__file__).parents[2])

class Cap(logging.Handler):
    def __init__(self): super().__init__(); self.recs=[]
    def emit(self, r): self.recs.append((r.levelname, r.getMessage()))

def run(fn_name, exc):
    cap = Cap(); S.logger.addHandler(cap); S.logger.setLevel(logging.DEBUG)
    sess = MagicMock(); sess.close.side_effect = exc
    with S._SESSION_LOCK:
        S._SESSION = sess; S._SESSION_REFS = 1
    raised = None
    try:
        getattr(S, fn_name)()
    except BaseException as e:  # noqa: BLE001 - classifying the outcome
        raised = f"{type(e).__name__}: {e}"
    out = {"fn": fn_name, "exc": type(exc).__name__, "raised": raised,
           "records": list(cap.recs), "session_ref_dropped": S._SESSION is None,
           "refs": S._SESSION_REFS, "close_called": sess.close.called}
    S.logger.removeHandler(cap)
    with S._SESSION_LOCK:
        S._SESSION = None; S._SESSION_REFS = 0
    return out

rows = []
for fn in ("release_session", "_atexit_cleanup"):
    for exc in (zenoh.ZError("simulated broker drop during teardown"),
                OSError("socket already closed"),
                TypeError("close() takes no arguments"),
                AttributeError("'NoneType' object has no attribute 'undeclare'")):
        rows.append(run(fn, exc))

print("\n=== CONSEQUENCE TABLE (main) ===")
for r in rows:
    print(f"  {r['fn']:<17} close raises {r['exc']:<14} -> raised={r['raised'] or 'no':<42} "
          f"records={r['records'] or '[]'} ref_dropped={r['session_ref_dropped']}")

# healthy control
cap = Cap(); S.logger.addHandler(cap); S.logger.setLevel(logging.DEBUG)
sess = MagicMock()
with S._SESSION_LOCK:
    S._SESSION = sess; S._SESSION_REFS = 1
S.release_session()
healthy = {"records": list(cap.recs), "ref": S._SESSION is None}
S.logger.removeHandler(cap)
print("\n=== HEALTHY CONTROL (release_session, close succeeds) ===")
print("  records:", healthy["records"], " session dropped:", healthy["ref"])

print("\n=== the policy the module states ===")
print("  zenoh_error_types() =", [t.__name__ for t in S.zenoh_error_types()])
print("  ZError is Exception-direct, not RuntimeError:", not issubclass(zenoh.ZError, RuntimeError))

json.dump({"rows": rows, "healthy": healthy}, open(f"/tmp/probe-pr-{sys.argv[1]}.json", "w"), indent=1)
