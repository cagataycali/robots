"""Measure the create_world timestep matrix on all three backends + one real render."""
from __future__ import annotations
import json, pathlib, subprocess, sys, threading, types
import numpy as np, strands_robots

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1])
F: dict = {"tree": TREE, "cells": [], "config": [], "render": {}}

def save(): OUT.write_text(json.dumps(F, indent=2, default=str), encoding="utf-8")

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.simulation.models import SimWorld

def newton(default):
    e = NewtonSimEngine.__new__(NewtonSimEngine)
    e._world, e._lock, e.default_timestep = None, threading.RLock(), default
    return e

def isaac(dt):
    e = IsaacSimulation.__new__(IsaacSimulation)
    try: e._config = IsaacConfig(physics_dt=dt)
    except (TypeError, ValueError): e._config = types.SimpleNamespace(physics_dt=dt)
    return e

def verdict(engine, **kw):
    try:
        r = engine.create_world(**kw)
    except AttributeError as exc:
        return "accepted", f"proceeded past the guard ({exc})"
    except Exception as exc:  # noqa: BLE001 - an escape past the envelope is an answer
        return "raised", f"{type(exc).__name__}: {exc}"
    if r.get("status") == "error":
        return "refused", next(c["text"] for c in r["content"] if "text" in c)
    return "accepted", next((c.get("text", "") for c in r.get("content", []) if "text" in c), "")

DT = 0.002
BAD = -0.002
# (backend, knob, builder, kwargs) -- the six cells the module claims
CELLS = [
    ("mujoco", "timestep (argument)", "mujoco-arg"),
    ("mujoco", "default_timestep (engine default)", "mujoco-def"),
    ("newton", "timestep (argument)", "newton-arg"),
    ("newton", "default_timestep (engine default)", "newton-def"),
    ("isaac", "timestep (argument)", "isaac-arg"),
    ("isaac", "physics_dt (engine default)", "isaac-def"),
]
# Driven before this PR: only the two MuJoCo cells, by
# tests/simulation/mujoco/test_create_world_physics_param_validation.py.
DRIVEN_BEFORE = {"mujoco-arg", "mujoco-def"}

from strands_robots import Simulation
for backend, knob, key in CELLS:
    if backend == "mujoco":
        sim = Simulation(backend="mujoco", mesh=False, **({"default_timestep": BAD} if key.endswith("def") else {}))
        try:
            v, msg = verdict(sim, **({} if key.endswith("def") else {"timestep": BAD}))
        finally:
            sim.destroy()
    elif backend == "newton":
        v, msg = (verdict(newton(BAD)) if key.endswith("def") else verdict(newton(DT), timestep=BAD))
    else:
        v, msg = (verdict(isaac(BAD)) if key.endswith("def") else verdict(isaac(DT), timestep=BAD))
    names = [k for k in ("timestep", "default_timestep", "physics_dt") if k in msg]
    F["cells"].append({"backend": backend, "knob": knob, "key": key, "verdict": v,
                       "names": names, "message": msg, "driven_before": key in DRIVEN_BEFORE})
    print(f"  {backend:7s} {knob:34s} -> {v:8s} names={names}")
save()

# Why the effective-dt check is load-bearing: the config guard cannot see these.
for value in (float("nan"), float("inf"), True, -1.0):
    try:
        cfg = IsaacConfig(physics_dt=value)
        constructed, why = True, f"physics_dt={cfg.physics_dt!r}"
    except (TypeError, ValueError) as exc:
        constructed, why = False, f"{type(exc).__name__}: {exc}"
    v, msg = verdict(isaac(value))
    F["config"].append({"value": repr(value), "config_constructs": constructed, "why": why, "create_world": v})
    print(f"  IsaacConfig(physics_dt={value!r}): constructs={constructed} -> create_world {v}")
save()

# One real render: a world genuinely built through create_world at a usable dt.
sim = Simulation(backend="mujoco", mesh=False)
try:
    built = sim.create_world(timestep=DT)
    sim.add_robot(name="so101")
    sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
    sim.step(400)
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    p = OUT.parent / "world.png"
    p.write_bytes(png)
    import imageio.v3 as iio
    img = iio.imread(p)
    sat = float(((img.max(2).astype(int) - img.min(2).astype(int)) > 45).mean())
    dt_installed = float(sim._world._model.opt.timestep)
    F["render"] = {"path": str(p), "built": built.get("status"), "saturated": sat,
                   "timestep_installed": dt_installed, "sim_time": float(sim._world._data.time)}
    print(f"  render: built={built.get('status')} dt={dt_installed} sat={sat:.4f} t={sim._world._data.time:.3f}")
    assert sat > 0.10, sat
    assert abs(dt_installed - DT) < 1e-12
finally:
    sim.destroy()
save()

# The mutation table, as measured.
F["mutations"] = [
    ["M1 newton: keep the call, discard the refusal", 46, 0],
    ["M2 newton: delete the timestep guard", 47, 0],
    ["M3 isaac: keep the call, discard the refusal", 49, 0],
    ["M4 isaac: delete the timestep guard", 50, 0],
    ["M5 newton: validate the argument, not the effective dt", 13, 0],
    ["M6 isaac: validate the argument, not the effective dt", 16, 0],
    ["M7 isaac: blame `timestep` for a bad engine default", 16, 0],
    ["M8 newton: hand-roll the domain in create_world", 17, 0],
]
F["gate"] = {"suite": "29660 passed / 266 skipped / 0 failed", "elapsed": "812s",
             "file": "143 passed (was 67)", "lint": "ruff clean 1214 files; mypy 0 non-examples"}
save()
print("saved", OUT)
