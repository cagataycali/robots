"""Measure: which readback route each case drives, and what a substituted base costs.

Part 1 traces every route to the ``return None`` arm it reaches and marks the
arms the base coverage run reported missing.  Part 2 renders the consequence the
readback's docstring names - "a wrong base makes every world-frame target
silently wrong" - on the MuJoCo backend, since Isaac Sim is not installed here.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import numpy as np

import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
OUT = pathlib.Path(__file__).parent
RUN = os.environ["GITHUB_RUN_ID"]
facts: dict = {"tree": str(ROOT)}


def save() -> None:
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))


# --------------------------------------------------------------- part 1: routes
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
    sys.modules.setdefault(name, types.ModuleType(name))


class _AA:
    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions, self.joint_indices = joint_positions, joint_indices


sys.modules["isaacsim.core.utils.types"].ArticulationAction = _AA

sys.path.insert(0, str(ROOT / "tests"))
from simulation.isaac.test_articulation_read_write_surfaces import (  # noqa: E402
    _UNREADABLE_BASE_POSES,
    _base_pose,
    _pose_returning,
    GOOD_BASE_QUAT,
    GOOD_BASE_POS,
)

BASE_COV = json.loads(pathlib.Path(f"/tmp/cd-before-{RUN}.json").read_text())
missing_before = set(BASE_COV["files"]["strands_robots/simulation/isaac/motion_primitives.py"]["missing_lines"])


def _last_line_in(fn, art):
    """The final line executed inside *fn* - its ``return None`` arm."""
    hits: list[int] = []

    def tracer(frame, event, arg):
        if frame.f_code is fn.__code__:
            if event == "line":
                hits.append(frame.f_lineno)
            return tracer
        return None

    sys.settrace(tracer)
    try:
        result = fn(art)
    finally:
        sys.settrace(None)
    return result, hits[-1] if hits else None


routes = []
for route in sorted(_UNREADABLE_BASE_POSES):
    art = types.SimpleNamespace(get_world_pose=_UNREADABLE_BASE_POSES[route])
    result, line = _last_line_in(_base_pose, art)
    assert result is None, f"{route} did not answer None"
    routes.append({"route": route, "line": line, "driven_before": line not in missing_before})
facts["routes"] = routes
n_before = sum(1 for r in routes if r["driven_before"])
print(f"\nroutes: {len(routes)}, driven before: {n_before}, driven now: {len(routes)}")
for r in routes:
    print(f"  {r['route']:<30} line {r['line']}  before={'driven' if r['driven_before'] else 'UNREACHED'}")

# the two documented surfaces beside the routes
tensor = type("T", (), {"__init__": lambda s, a: setattr(s, "_a", a), "cpu": lambda s: s, "numpy": lambda s: s._a})
pose = _base_pose(types.SimpleNamespace(get_world_pose=_pose_returning((tensor(GOOD_BASE_POS), tensor(GOOD_BASE_QUAT)))))
facts["torch_surface_read"] = pose is not None and np.allclose(pose[0], GOOD_BASE_POS)
pose = _base_pose(types.SimpleNamespace(get_world_pose=_pose_returning((GOOD_BASE_POS, GOOD_BASE_QUAT * 7.0))))
facts["quaternion_normalized"] = float(np.linalg.norm(pose[1]))
print(f"torch surface read: {facts['torch_surface_read']}   normalized |q|: {facts['quaternion_normalized']:.6f}")
facts["missing_before"] = sorted(missing_before & {r["line"] for r in routes})
save()

# ------------------------------------------------- part 2: what a wrong base costs
from strands_robots import Simulation  # noqa: E402

BASE = [0.4, -0.2, 0.0]
# The panda is seeded from its ``home`` keyframe: at ``qpos0`` joint4 starts
# outside its own range, so every IK direction is dominated by pulling it back
# in.  This target is one the position servos hold; a panda reaching further
# forward droops short of tol, which is a fixture property, not the contract.
WORLD_TARGET = [0.4, 0.0, 0.4]
# What the readback's docstring warns about: an origin base maps the caller's
# world-frame target straight through, so the arm is aimed at target - base.
SUBSTITUTED = [round(WORLD_TARGET[i] - BASE[i], 6) for i in range(3)]
facts["base"], facts["world_target"], facts["substituted_target"] = BASE, WORLD_TARGET, SUBSTITUTED
facts["mapping_error_m"] = float(np.linalg.norm(np.array(WORLD_TARGET) - np.array(SUBSTITUTED)))
print(f"\nbase {BASE}  world target {WORLD_TARGET}  origin-mapped target {SUBSTITUTED}")
print(f"the substitution aims the arm {facts['mapping_error_m']:.4f} m from what the caller asked")


def build():
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="panda", position=BASE, keyframe="home")
    # No marker at the target: an ``add_object`` sphere there is a collision
    # geom, and the hand fights it instead of converging (measured: the same
    # target reaches 0.0196 m without it and stalls at 0.0843 m with it).
    sim.add_camera(name="look", position=[1.95, -1.65, 1.15], target=[0.6, -0.1, 0.33], fov=42)
    sim.step(120)
    return sim


def png(sim, name):
    r = sim.render(camera_name="look", width=780, height=680)
    assert r.get("status") == "success", r
    (OUT / name).write_bytes(next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c))
    return name


sim = build()
png(sim, "rest.png")
sim.cleanup()

shots = {}
for label, target in (("true_base", WORLD_TARGET), ("origin_base", SUBSTITUTED)):
    sim = build()
    r = sim.move_to(robot_name="panda", position=target, tol=0.02, max_steps=1200)
    # The primitive reports the error against the frame it discovered.
    pay = next(c["json"] for c in r["content"] if "json" in c)
    body = sim.get_body_state(body_name=pay["frame"])
    hand = next(c["json"] for c in body["content"] if "json" in c)["position"]
    shots[label] = {"status": r["status"], "target": target, "reached": pay.get("reached"),
                    "own_error_m": round(float(pay.get("position_error_m", -1)), 4),
                    "from_asked_m": round(float(np.linalg.norm(np.array(hand) - np.array(WORLD_TARGET))), 4),
                    "frame": f"{pay.get('frame')} ({pay.get('frame_type')})",
                    "png": png(sim, f"{label}.png")}
    sim.cleanup()
    print(f"{label:<12} status={r['status']:<8} reached={pay.get('reached')}  "
          f"own error {shots[label]['own_error_m']:.4f} m from {target}, "
          f"{shots[label]['from_asked_m']:.4f} m from the asked-for point")
facts["shots"] = shots

import imageio.v3 as iio  # noqa: E402

imgs = {k: iio.imread(OUT / f"{k}.png") for k in shots}
diff = float((np.abs(imgs["true_base"].astype(int) - imgs["origin_base"].astype(int)).max(2) > 8).mean())
facts["panel_diff_frac"] = round(diff, 4)
print(f"\ntrue-base vs origin-base render: {diff:.2%} of pixels differ")
assert diff > 0.10, f"panels differ on only {diff:.2%}"
assert shots["true_base"]["own_error_m"] <= 0.03, shots["true_base"]
assert shots["true_base"]["from_asked_m"] <= 0.03, shots["true_base"]
assert shots["origin_base"]["from_asked_m"] > 0.3, shots["origin_base"]
assert facts["mapping_error_m"] > 0.3, facts["mapping_error_m"]
save()
print("\nfacts.json written")
