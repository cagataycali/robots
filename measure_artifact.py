"""Measure what Isaac's add_object compiles for a documented alias vs a wrong key."""
import json, sys, threading, types
from pathlib import Path
import strands_robots.simulation.isaac.simulation as isim
TREE = str(Path(isim.__file__).parents[3])
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.base import SimEngine

def stub():
    seen = []
    s = types.SimpleNamespace(
        _lock=threading.RLock(), _world_created=True, _config=IsaacConfig(),
        _objects={}, _robots={}, _cameras={}, _replicated=False, _prim_registry=[],
        _world=types.SimpleNamespace(scene=types.SimpleNamespace(add=lambda h: None)),
        _validate_mass=SimEngine._validate_mass,
    )
    def construct(**kw):
        # Reproduce the real _construct_shape_prim box fallback so the recorded
        # "compiled" value is the extent Isaac would really build.
        sz = list(kw.get("size") or [])
        box = [float(sz[i]) if len(sz) > i else 0.05 for i in range(3)]
        seen.append(box)
        return object(), box
    s._construct_shape_prim = construct
    return s, seen

WANT = [0.30, 0.30, 0.30]
CASES = [
    ("scale (documented alias)", {"scale": list(WANT)}),
    ("extents (plausible, wrong)", {"extents": list(WANT)}),
]
out = {"tree": TREE, "requested": WANT, "cases": {}}
for label, extra in CASES:
    s, seen = stub()
    try:
        r = IsaacSimulation.add_object(s, name="crate", shape="box", **extra)
        st = r["status"]
        txt = str(r["content"][0]["text"])
        j = next((b["json"] for b in r["content"] if "json" in b), {})
        out["cases"][label] = {
            "status": st, "text": txt[:220],
            "compiled": seen[0] if seen else None,
            "json_size": j.get("size"),
            "registered": "crate" in s._objects,
        }
    except BaseException as exc:
        out["cases"][label] = {"status": "raised", "text": f"{type(exc).__name__}: {exc}"[:220],
                               "compiled": seen[0] if seen else None, "registered": "crate" in s._objects}
    print(f"  {label:<28} {out['cases'][label]['status']:<8} compiled={out['cases'][label]['compiled']}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print("TREE:", TREE)
