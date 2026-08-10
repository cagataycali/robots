import pathlib, numpy as np, strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
from strands_robots.simulation.mujoco import simulation as sim_mod
from strands_robots.simulation import Simulation

CASES = {
    "3-short": [1.0, 2.0], "4-long": [1.0, 2.0, 3.0, 4.0], "empty": [],
    "str": "abc", "scalar": 3, "bool": True, "nan": [float("nan"), 0.0, 0.0],
    "inf": [float("inf"), 0.0, 0.0], "nonnum-elem": ["a", 1.0, 2.0],
    "mapping": {"x": 1}, "none-elem": [None, 1.0, 2.0], "huge-int": [10**400, 0.0, 0.0],
    "iterator": iter([1.0, 2.0, 3.0]), "0d-array": np.array(1.0),
    "ok-list": [0.4, 0.4, 0.4], "ok-numpy": np.array([0.4, 0.4, 0.4]),
    "ok-tuple": (0.4, 0.4, 0.4), "ok-npscalars": [np.float64(0.4), np.int64(1), 0.4],
    "omitted": None, "degenerate": [0.0, 0.0, 0.0],
}

def sweep(label):
    out = {}
    for k, v in CASES.items():
        s = Simulation(backend="mujoco", mesh=False)
        s.create_world()
        for param in ("position", "target"):
            kw = {"name": f"c_{k}_{param}"}
            if param == "position":
                kw["position"] = v
                kw["target"] = [0.0, 0.0, 0.0]
            else:
                kw["position"] = [1.0, 1.0, 1.0]
                kw["target"] = v
            r = s.add_camera(**({p: q for p, q in kw.items() if q is not None or p == "name"}))
            txt = ""
            for c in r.get("content", []):
                if "text" in c:
                    txt = c["text"]
            out[(k, param)] = (r.get("status"), txt[:120])
        s.cleanup()
    return out

before = sweep("with second check")
calls = []
orig = sim_mod.pose_vector_error
def spy(method, param, vec, n):
    calls.append((method, param))
    return None            # neuter: the second check can never refuse
sim_mod.pose_vector_error = spy
after = sweep("second check neutered")
sim_mod.pose_vector_error = orig

diff = [k for k in before if before[k] != after[k]]
print(f"\nsecond check invoked {len(calls)} times (so it does run)")
print(f"cases compared: {len(before)}  DIFFERING: {len(diff)}")
for k in diff:
    print("  DIFF", k, before[k], "->", after[k])
print("\nsample verdicts (identical both ways):")
for k in [("3-short","position"),("nan","target"),("nonnum-elem","position"),
          ("ok-list","position"),("omitted","position"),("degenerate","target")]:
    print(f"  {str(k):32s} {before[k][0]:7s} {before[k][1][:78]}")
