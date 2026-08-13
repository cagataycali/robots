"""Render the same commanded pose under an accepted solver and a refused one.

Both panels come from one tree; main's behaviour for the refused solver is
reproduced by neutralising the new guard (main has none), so the only variable
is the solver name. Cameras are swept and the best framing is chosen by
measurement rather than by eye.
"""
import json, pathlib
import numpy as np
import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT, flush=True)
import imageio.v3 as iio
from strands_robots.simulation import create_simulation
from strands_robots.simulation.newton import simulation as nsim
from strands_robots.simulation.newton.backend import articulated_solver_error, articulated_solvers

nsim.articulated_solver_error = lambda _s: None

ART = pathlib.Path("_art")
TARGET = {"j1": 0.9, "j2": -0.7}
JOINTS = tuple(TARGET)
W, H = 620, 520
CAMS = {
    "a": ([0.62, -0.58, 0.44], [0.20, 0.0, 0.14], 42),
    "b": ([0.48, -0.44, 0.34], [0.22, 0.0, 0.14], 48),
    "c": ([0.38, -0.36, 0.30], [0.22, 0.0, 0.14], 56),
    "d": ([0.30, -0.50, 0.26], [0.18, 0.0, 0.14], 50),
}
facts = {"tree": str(ROOT), "rows": {}, "cameras": {k: {"pos": v[0], "target": v[1], "fov": v[2]}
                                                    for k, v in CAMS.items()},
         "refusal": articulated_solver_error("xpbd"), "accepted": list(articulated_solvers())}


def save():
    (ART / "facts.json").write_text(json.dumps(facts, indent=2))


def frame(sim, cam):
    r = sim.render(camera_name=cam, width=W, height=H)
    assert r.get("status") == "success", r
    raw = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(iio.imread(raw))[:, :, :3]


for solver in ("featherstone", "xpbd"):
    row = {"frames": {}}
    facts["rows"][solver] = row
    sim = create_simulation("newton", solver=solver, mesh=False)
    try:
        sim.create_world()
        for key, (pos, tgt, fov) in CAMS.items():
            assert sim.add_camera(name=key, position=pos, target=tgt, fov=fov).get("status") == "success"
        row["add_robot"] = sim.add_robot(name="arm", urdf_path="_art/art_arm.xml").get("status")
        before = {j: float(sim.get_observation(robot_name="arm")[j]) for j in JOINTS}
        frame(sim, "a")  # warm the renderer; the first call carries a sync artifact
        row["send_action"] = sim.send_action(TARGET, robot_name="arm", n_substeps=1).get("status")
        row["step"] = sim.step(120).get("status")
        after = {j: float(sim.get_observation(robot_name="arm")[j]) for j in JOINTS}
        row["before"], row["after"] = before, after
        row["travel"] = round(max(abs(after[j] - before[j]) for j in JOINTS), 6)
        for key in CAMS:
            f = frame(sim, key)
            np.save(ART / f"{solver}_{key}.npy", f)
            sat = ((f.max(2).astype(int) - f.min(2).astype(int)) > 45).mean()
            bright = (f.mean(2) > 88).mean()
            row["frames"][key] = {"content_frac": round(float(max(sat, bright)), 4)}
    finally:
        sim.cleanup()
        save()
    print(f"{solver:14s} travel={row['travel']} "
          f"statuses={row['add_robot']}/{row['send_action']}/{row['step']}", flush=True)

print("\ncamera sweep (accepted vs refused, same commanded target):", flush=True)
best, best_frac = None, -1.0
for key in CAMS:
    acc = np.load(ART / f"featherstone_{key}.npy").astype(int)
    frz = np.load(ART / f"xpbd_{key}.npy").astype(int)
    frac = float((np.abs(acc - frz).max(2) > 8).mean())
    cf = min(facts["rows"]["featherstone"]["frames"][key]["content_frac"],
             facts["rows"]["xpbd"]["frames"][key]["content_frac"])
    facts["cameras"][key]["differing_frac"] = round(frac, 4)
    facts["cameras"][key]["min_content_frac"] = cf
    print(f"  {key}: differing={frac:.4f} min_content={cf:.4f}", flush=True)
    if frac > best_frac and cf > 0.10:
        best, best_frac = key, frac
facts["chosen_camera"] = best
save()
print(f"\nchosen camera: {best} differing={best_frac:.4f}", flush=True)
assert facts["rows"]["featherstone"]["travel"] > 0.5
assert facts["rows"]["xpbd"]["travel"] == 0.0
assert best is not None and best_frac > 0.10, (best, best_frac)
for s, r in facts["rows"].items():
    assert r["add_robot"] == r["send_action"] == r["step"] == "success", (s, r)
print("ALL GATES PASS", flush=True)
