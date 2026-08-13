"""Two scenes, one variable: the cylinder size spelling. Rebuts 'renders fine'."""
import json, pathlib
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots import Robot

def build(cyl_size_fn, tag):
    sim = Robot("so101", mode="sim", mesh=False)
    ok = err = 0
    for i in range(9):
        r = sim.add_object(name=f"cube_{i}", shape="box",
                           position=[0.22 + (i % 3) * 0.07, -0.07 + (i // 3) * 0.07, 0.025],
                           size=[0.05, 0.05, 0.05],
                           color=[[0.9,.2,.2,1],[.95,.6,.1,1],[.95,.9,.2,1]][i % 3])
        ok += r["status"] == "success"; err += r["status"] != "success"
    for i in range(3):  # "pens"
        r = sim.add_object(name=f"pen_{i}", shape="cylinder", position=[0.40, -0.09 + i * 0.09, 0.06],
                           size=cyl_size_fn(0.014, 0.12), color=[0.15,0.2,0.85,1])
        ok += r["status"] == "success"; err += r["status"] != "success"
    r = sim.add_object(name="ball_green", shape="sphere", position=[0.30, 0.17, 0.03],
                       size=[0.055], color=[0.2,0.95,0.35,1])
    ok += r["status"] == "success"; err += r["status"] != "success"
    r = sim.add_object(name="coin_gold", shape="cylinder", position=[0.24, -0.20, 0.012],
                       size=cyl_size_fn(0.05, 0.024), color=[1,.84,.1,1])
    ok += r["status"] == "success"; err += r["status"] != "success"

    lo = " ".join(c.get("text","") for c in sim.list_objects().get("content",[]))
    names = sorted(l.strip().lstrip("- ").split(":")[0] for l in lo.splitlines() if l.strip().startswith("-"))
    sim.add_camera(name="look", position=[0.72, -0.30, 0.42], target=[0.30, 0.0, 0.04], fov=45)
    rr = sim.render(camera_name="look", width=760, height=560)
    assert rr["status"] == "success", rr
    png = next(c["image"]["source"]["bytes"] for c in rr["content"] if "image" in c)
    pathlib.Path(f"_probe/{tag}.png").write_bytes(png)
    import mujoco as mj
    ncyl = int(sum(1 for i in range(sim._world._model.ngeom)
                   if int(sim._world._model.geom_type[i]) == int(mj.mjtGeom.mjGEOM_CYLINDER)))
    sim.cleanup()
    return {"calls": ok + err, "ok": ok, "err": err, "listed": len(names),
            "names": names, "cylinder_geoms": ncyl, "png": f"_probe/{tag}.png"}

facts = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}
facts["reported"] = build(lambda d, h: [d / 2, h], "reported")        # reporter's [radius, height]
facts["correct"]  = build(lambda d, h: [d, 0.0, h], "correct")        # documented [diameter, unused, full height]

# the two panels must genuinely differ, and the "10/14" must reproduce exactly
from PIL import Image
a = np.asarray(Image.open("_probe/reported.png").convert("RGB")).astype(int)
b = np.asarray(Image.open("_probe/correct.png").convert("RGB")).astype(int)
facts["panel_diff_frac"] = round(float((np.abs(a - b).sum(2) > 8).mean()), 4)
assert facts["reported"]["ok"] == 10 and facts["reported"]["calls"] == 14, facts["reported"]
assert facts["reported"]["cylinder_geoms"] == 0, "reporter's scene must contain NO cylinder"
assert facts["correct"]["ok"] == 14 and facts["correct"]["cylinder_geoms"] == 4, facts["correct"]
assert facts["correct"]["listed"] == 14, facts["correct"]
# Full-frame diff is small by geometry: 4 thin cylinders are ~3% of a 760x560
# frame. The legibility gate is the derived crop below.
assert facts["panel_diff_frac"] > 0.02, facts["panel_diff_frac"]
pathlib.Path("_probe/artifact.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "names"} if isinstance(v, dict) else v
                  for k, v in facts.items()}, indent=2))

# --- crop to the derived changed-region bbox (4 thin cylinders are ~3% of frame) ---
mask = (np.abs(a - b).sum(2) > 8)
# The global bbox spans the pens AND the distant coin, so it includes a lot of
# unchanged table. Slide a fixed window and take the densest changed cluster.
side, best = 260, (-1, 0, 0)
for yy in range(0, a.shape[0] - side, 20):
    for xx in range(0, a.shape[1] - side, 20):
        n = int(mask[yy:yy + side, xx:xx + side].sum())
        if n > best[0]:
            best = (n, yy, xx)
_, y0, x0 = best
y1, x1 = y0 + side, x0 + side
ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
facts["crop"] = [int(x0), int(y0), int(x1), int(y1)]
facts["crop_diff_frac"] = round(float((np.abs(ca - cb).sum(2) > 8).mean()), 4)
Image.fromarray(ca.astype("uint8")).save("_probe/reported_crop.png")
Image.fromarray(cb.astype("uint8")).save("_probe/correct_crop.png")
assert facts["crop_diff_frac"] > 0.10, facts["crop_diff_frac"]
pathlib.Path("_probe/artifact.json").write_text(json.dumps(facts, indent=2))
print("crop", facts["crop"], "crop_diff_frac", facts["crop_diff_frac"])
