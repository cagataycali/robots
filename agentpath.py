"""Does the AGENT-TOOL path report the refusal as well as the direct call?"""
import json, pathlib
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots import Robot
sim = Robot("so101", mode="sim", mesh=False)
out = {}
# direct
d = sim.add_object(name="pen_direct", shape="cylinder", size=[0.008, 0.075], position=[0.3, 0, 0.08])
out["direct"] = {"status": d["status"], "text": " ".join(c.get("text","") for c in d.get("content",[]))[:180]}
# dispatched (what an LLM holding `sim` as a tool actually calls)
try:
    y = sim(action="add_object", name="pen_tool", shape="cylinder", size=[0.008, 0.075], position=[0.3, 0.1, 0.08])
    out["dispatched"] = {"status": y["status"], "text": " ".join(c.get("text","") for c in y.get("content",[]))[:180]}
except Exception as e:
    out["dispatched"] = {"status": f"RAISED {type(e).__name__}", "text": str(e)[:180]}
# describe(): the surface the issue names
try:
    ds = sim.describe()
    dtxt = " ".join(c.get("text","") for c in ds.get("content",[])) if isinstance(ds, dict) else str(ds)
    out["describe_mentions_objects"] = "object" in dtxt.lower()
    out["describe_len"] = len(dtxt)
    out["describe_object_lines"] = [l.strip() for l in dtxt.splitlines() if "bject" in l][:6]
except Exception as e:
    out["describe"] = f"RAISED {type(e).__name__}: {e}"
sim.cleanup()
pathlib.Path("_probe/agentpath.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)[:1800])
