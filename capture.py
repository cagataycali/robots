"""Measure the viewer/render dimension verdicts and the single-slot recovery."""
import json, sys, importlib.util, threading, types
from pathlib import Path
from typing import Any

import numpy as np
import strands_robots
TREE = str(Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.simulation.newton.simulation import NewtonSimEngine

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, Path.cwd() / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

vpd = _load("tests/simulation/newton/test_viewer_port_domain.py", "vpd")
acv = _load("tests/simulation/newton/test_add_camera_numeric_validation.py", "acv")

def _newton_stub():
    stub = acv._engine_stub()
    stub.default_width, stub.default_height = 640, 480
    for m in ("_resolve_camera_pose", "_resolve_camera_view", "list_cameras"):
        setattr(stub, m, types.MethodType(getattr(NewtonSimEngine, m), stub))
    return stub

NAN, INF = float("nan"), float("inf")
CASES: list[Any] = [0, -4, -1, 2.7, 640.0, True, False, "big", "640", NAN, INF, [640], 10**9]

def render_verdict(v):
    try:
        NewtonSimEngine._resolve_camera_view(_newton_stub(), "default", v, None)
    except ValueError:
        return "refused"
    except Exception as exc:
        return f"raised {type(exc).__name__}"
    return "accepted"

def viewer_verdict(v):
    stub = vpd._viewer_stub(has_display=True)
    try:
        r = NewtonSimEngine.open_viewer(stub, "gl", width=v, height=720)
    except Exception as exc:
        return f"raised {type(exc).__name__}", None
    built = stub.built[0][1] if stub.built else None
    return ("refused" if r["status"] == "error" else "accepted"), built

rows = []
for v in CASES:
    rv = render_verdict(v)
    vv, built = viewer_verdict(v)
    rows.append({"value": repr(v), "render": rv, "viewer": vv,
                 "built": (f"{built['width']}x{built['height']}" if built else None)})

# single-slot recovery ledger
stub = vpd._viewer_stub(has_display=True)
first = NewtonSimEngine.open_viewer(stub, "gl", width=0, height=0)
first_built = list(stub.built)
retry = NewtonSimEngine.open_viewer(stub, "gl", width=1280, height=720)
ledger = {
    "first_status": first["status"],
    "first_text": str(first["content"][0]["text"]),
    "first_built": (f"{first_built[0][1]['width']}x{first_built[0][1]['height']}" if first_built else None),
    "retry_status": retry["status"],
    "retry_text": str(retry["content"][0]["text"]),
    "retry_built": (f"{stub.built[-1][1]['width']}x{stub.built[-1][1]['height']}" if stub.built else None),
    "viewers_built_total": len(stub.built),
}

# grounding: a real offscreen frame at the viewer's default window size, to show
# what an honored pixel count means. Unchanged by this fix on purpose.
from strands_robots import Robot
sim = Robot("so100", mode="sim", mesh=False)
sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
r = sim.render(camera_name="look", width=1280, height=720)
png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
import io
from PIL import Image
arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
np.save(f"/tmp/ground_{Path(TREE).name}.npy", arr)
sim.cleanup()

out = {"tree": TREE, "rows": rows, "ledger": ledger, "ground_shape": list(arr.shape)}
Path(f"/tmp/art_{Path(TREE).name}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"n_divergences": sum(1 for r in rows if r["render"] != r["viewer"]),
                  "ledger": ledger, "ground_shape": list(arr.shape)}, indent=1))
