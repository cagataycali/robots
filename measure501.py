"""Drive every candidate verb through use_unitree headless, record what the robot ran."""
import json, os, sys, tempfile
os.environ.pop("BYPASS_TOOL_CONSENT", None)
os.environ.pop("STRANDS_UNITREE_COMMAND_ALLOW", None)
os.environ["STRANDS_MESH_AUDIT_DIR"] = tempfile.mkdtemp()
from unittest.mock import patch
import strands_robots.tools.g1.use_unitree as uu

UNKNOWN = ("Frobnicate", "Recover", "Continue", "Trigger", "Activate", "Engage", "ArmTask", "DoThing")
KNOWN_WRITES = ("SetVelocity", "Move", "ZeroTorque")
READS = ("GetFsmId", "CheckMode")
ALL = UNKNOWN + KNOWN_WRITES + READS

class Recorder:
    def __init__(self): self.calls = []
def _mk(name):
    def fn(self, **kw):
        self.calls.append(name); return 0
    fn.__name__ = name
    return fn
for v in ALL:
    setattr(Recorder, v, _mk(v))

rows = []
for verb in ALL:
    rec = Recorder()
    with patch.object(uu, "ensure_dds", lambda _i: None), patch.object(uu, "_CLIENTS", {"loco": rec}):
        res = uu.use_unitree("loco", verb, {})
    rows.append({
        "verb": verb,
        "kind": "unknown" if verb in UNKNOWN else ("read" if verb in READS else "known-write"),
        "mutative": bool(res.get("mutative")),
        "status": res["status"],
        "dispatched": rec.calls == [verb],
        "message": res.get("message", "")[:110],
    })
print(json.dumps({"tree": strands_robots_path, "rows": rows}, indent=1) if False else json.dumps(
    {"tree": __import__("strands_robots").__file__, "rows": rows}, indent=1))
