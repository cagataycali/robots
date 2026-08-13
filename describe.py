import json, pathlib
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots import Robot
sim = Robot("so101", mode="sim", mesh=False)
sim.add_object(name="cube", shape="box", size=[0.05,0.05,0.05], position=[0.3,0,0.03])
sim.add_object(name="can", shape="cylinder", size=[0.06,0.0,0.12], position=[0.3,0.1,0.06])
d = sim.describe()
print("type:", type(d).__name__)
print("keys:", list(d)[:8] if isinstance(d, dict) else "-")
blocks = [list(b) for b in d.get("content", [])] if isinstance(d, dict) else []
print("content block kinds:", blocks)
for b in d.get("content", []):
    if "json" in b:
        j = b["json"]
        print("json keys:", sorted(j)[:25])
        print("  objects field:", j.get("objects"))
        print("  robots field:", j.get("robots"))
        print("  n actions:", len(j.get("available_actions", j.get("actions", []))))
    if "text" in b:
        print("text[:300]:", b["text"][:300])
lo = sim.list_objects()
print("\nlist_objects blocks:", [list(b) for b in lo.get("content", [])])
for b in lo.get("content", []):
    if "text" in b: print("  text:", b["text"][:300])
    if "json" in b: print("  json:", json.dumps(b["json"])[:400])
sim.cleanup()
