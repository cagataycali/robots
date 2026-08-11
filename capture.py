"""Measure the descriptor-shadowing effect on both descriptor kinds, in-tree."""
import json, os, pathlib, sys, types
from typing import Any

import strands_robots.tools.use_rosbridge as rb_mod
from strands_robots.tools.use_rosbridge import use_rosbridge

ROOT = pathlib.Path(rb_mod.__file__).parents[2]
FACTS: dict[str, Any] = {"tree": str(ROOT)}


def scenario(with_set: bool) -> dict[str, Any]:
    """Drive the reconnect scenario with a non-data (with_set=False) / data descriptor."""
    import tests.tools.test_use_rosbridge as T

    F = T._FakeRos
    F.instances, F.fail_next_connect = [], False
    if hasattr(F, "ready_without_connection"):
        F.ready_without_connection = False
    mod = types.ModuleType("roslibpy")
    mod.Ros, mod.Topic, mod.Service = F, T._FakeTopic, T._FakeService
    mod.Message = mod.ServiceRequest = dict
    old_mod, old_conn, old_avail = sys.modules.get("roslibpy"), rb_mod._backend._connections, rb_mod._backend._available
    sys.modules["roslibpy"] = mod
    rb_mod._backend._connections, rb_mod._backend._available = {}, None
    try:
        use_rosbridge(action="status")
        first = F.instances[0]

        body: dict[str, Any] = {}
        if with_set:
            body["__set__"] = lambda self, obj, value: None
        Flap = type("_Flapping", (), {
            "__init__": lambda self: setattr(self, "reads", 0),
            "__get__": lambda self, obj, objtype=None: (setattr(self, "reads", self.reads + 1), self.reads > 2)[1],
            **body,
        })
        flap = Flap()
        type(first).is_connected = flap
        try:
            result = use_rosbridge(action="status", timeout=1.0)
            text = "\n".join(b.get("text", "") for b in result["content"])
        finally:
            del type(first).is_connected
        return {
            "descriptor": "data (__get__ + __set__)" if with_set else "non-data (__get__ only)",
            "reads": flap.reads,
            "shadowed": flap.reads == 0,
            "assertion_passes": "connected to" in text,
            "branch": "wait loop (L217-219)" if flap.reads >= 3 else "plain cache hit (L214-215)",
        }
    finally:
        if old_mod is None:
            sys.modules.pop("roslibpy", None)
        else:
            sys.modules["roslibpy"] = old_mod
        rb_mod._backend._connections, rb_mod._backend._available = old_conn, old_avail


FACTS["shadowing"] = [scenario(False), scenario(True)]

rid = os.environ["GITHUB_RUN_ID"]
cov_b = json.load(open(f"/tmp/cov-{rid}.json"))["files"]["strands_robots/tools/use_rosbridge.py"]
cov_a = json.load(open(f"/tmp/cov-after-{rid}.json"))["files"]["strands_robots/tools/use_rosbridge.py"]
FACTS["cov"] = {
    "before": {"missing": cov_b["missing_lines"], "pct": round(cov_b["summary"]["percent_covered"], 2)},
    "after": {"missing": cov_a["missing_lines"], "pct": round(cov_a["summary"]["percent_covered"], 2)},
}
FACTS["states"] = [
    ("new connection, run() raised", 204),
    ("new connection, ready but connector down", 209),
    ("new connection, connected", 213),
    ("cached, connected", 215),
    ("cached, dropped, recovers during wait", 219),
    ("cached, dropped, never recovers", 221),
]
FACTS["mutations"] = json.load(open("/tmp/mut.json"))
pathlib.Path("/tmp/facts.json").write_text(json.dumps(FACTS, indent=1))
print(json.dumps(FACTS["shadowing"], indent=1))
print("cov:", FACTS["cov"])
