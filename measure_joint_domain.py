"""Measure IsaacSimulation.set_joint_positions against a probe set. Dumps JSON."""
import json, math, pathlib, queue, sys, threading
import numpy as np
from strands_robots.simulation.isaac.simulation import IsaacSimulation, IsaacConfig, _RobotState

# Resolution guard: a probe script run from outside the repo silently picks up a
# different editable checkout. Print the tree so the measurement is attributable.
import strands_robots.simulation.base as _b
print("TREE:", pathlib.Path(_b.__file__).parents[2])

JOINTS = ["shoulder", "elbow", "wrist"]
HOME = [0.10, 0.20, 0.30]


class FakeArticulation:
    def __init__(self):
        self.q = list(HOME)

    def get_joint_positions(self):
        return list(self.q)

    def set_joint_positions(self, arr):
        self.q = list(np.asarray(arr, dtype=float).tolist())


def mk(queued):
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._lock = threading.RLock()
    e._world_created = True
    e._config = IsaacConfig()
    e._main_tid = -1 if queued else threading.get_ident()
    e._action_q = queue.Queue()
    art = FakeArticulation()
    e._robots = {"arm": _RobotState(name="arm", prim_path="/World/Robots/arm",
                                    joint_names=list(JOINTS), articulation=art)}
    return e, art


CASES = [
    ("positions={'shoulder': True}", {"shoulder": True}),
    ("positions={'shoulder': np.True_}", {"shoulder": np.bool_(True)}),
    ("positions={'shoulder': nan}", {"shoulder": float("nan")}),
    ("positions={'shoulder': inf}", {"shoulder": float("inf")}),
    ("positions={'shoudler': 0.9}  (typo)", {"shoudler": 0.9}),
    ("positions={'shoulder': 0.9, 'nope': 0.5}", {"shoulder": 0.9, "nope": 0.5}),
    ("positions={}", {}),
    ("positions={'shoulder': 'abc'}", {"shoulder": "abc"}),
    ("positions=[0.5, 0.6]  (3 joints)", [0.5, 0.6]),
    ("positions=[0.5, 0.6, 0.7, 0.8]", [0.5, 0.6, 0.7, 0.8]),
    ("positions='abc'", "abc"),
    ("CONTROL positions={'shoulder': 0.9}", {"shoulder": 0.9}),
    ("CONTROL positions=[0.5, 0.6, 0.7]", [0.5, 0.6, 0.7]),
]

out = []
for label, val in CASES:
    row = {"case": label}
    for queued in (False, True):
        e, art = mk(queued)
        rec = {}
        try:
            r = e.set_joint_positions(positions=val, robot_name="arm")
            rec["verdict"] = r.get("status")
            rec["text"] = " ".join(c.get("text", "") for c in r.get("content", []) if "text" in c)
        except BaseException as exc:  # noqa: BLE001 - an escape past the envelope IS the finding
            rec["verdict"] = "raised"
            rec["text"] = f"{type(exc).__name__}: {exc}"
        swallowed = None
        while not e._action_q.empty():
            fn = e._action_q.get_nowait()
            try:
                fn()
            except (RuntimeError, ValueError, AttributeError, TypeError, KeyError, IndexError) as ex:
                swallowed = f"{type(ex).__name__}: {ex}"   # production's pump swallows this
        rec["pump_swallowed"] = swallowed
        rec["joint_state"] = ["nan" if isinstance(x, float) and math.isnan(x) else x for x in art.q]
        rec["nonfinite_written"] = any(isinstance(x, float) and not math.isfinite(x) for x in art.q)
        rec["changed"] = art.q != HOME
        rec["width"] = len(art.q)
        row["queued" if queued else "main"] = rec
    out.append(row)

path = sys.argv[1]
with open(path, "w") as fh:
    json.dump({"joints": JOINTS, "home": HOME, "rows": out}, fh, indent=1)
print("wrote", path)
for r in out:
    m, q = r["main"], r["queued"]
    print(f"{r['case']:42s} main={m['verdict']:8s} q={q['verdict']:8s} "
          f"changed={m['changed']!s:5s} nonfinite={m['nonfinite_written']!s:5s} "
          f"width={m['width']} swallowed={'Y' if q['pump_swallowed'] else '-'}")
