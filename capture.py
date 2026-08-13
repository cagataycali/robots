"""Measure the GL-probe latch: construction count on a cleared cache, plus a real render.

Run once per tree with PYTHONPATH pinned to that tree:

    PYTHONPATH=<tree> MUJOCO_GL=egl python3 _art/capture.py <tree-label>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import tests.simulation.mujoco._gl_probe as gp

TREE = str(pathlib.Path(gp.__file__).resolve().parents[3])
print("TREE:", TREE)
LABEL = sys.argv[1]
OUT = pathlib.Path("/tmp") / f"glart-{LABEL}.json"
facts: dict[str, object] = {"tree": TREE, "label": LABEL}


def save() -> None:
    OUT.write_text(json.dumps(facts, indent=2), encoding="utf-8")


# --- how many probe renderers does a cleared cache construct? -----------------
import mujoco as mj  # noqa: E402

builds: list[str] = []
_real = mj.Renderer


def _counting(*a: object, **k: object) -> object:
    builds.append("constructed")
    return _real(*a, **k)


gp.gl_available()  # prime, as the import-time marker already did
mj.Renderer = _counting  # type: ignore[misc]
gp.gl_available.cache_clear()
answer = gp.gl_available()
gp.gl_available.cache_clear()
mj.Renderer = _real  # type: ignore[misc]
facts["cleared_cache_constructions"] = len(builds)
facts["cleared_cache_answer"] = bool(answer)
facts["latch_symbol_present"] = hasattr(gp, "_HARDWARE_PROBE_RESULT")
print(f"cleared-cache constructions={len(builds)} answer={answer} latch={facts['latch_symbol_present']}")
save()

# --- one real frame of the kind the gated tests verify ------------------------
XML = """
<mujoco>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
    <global offwidth="820" offheight="620"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.5 0.65 0.85" rgb2="0.2 0.3 0.45" width="8" height="32"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.6 0.6 0.6" width="80" height="80"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.05"/></asset>
  <worldbody>
    <light pos="0.6 -0.6 1.2" dir="-0.4 0.4 -1"/>
    <geom name="floor" type="plane" size="1.6 1.6 0.05" material="grid"/>
    <body name="crate" pos="0.05 0 0.11">
      <freejoint/><geom type="box" size="0.11 0.11 0.11" rgba="0.95 0.5 0.12 1"/></body>
    <body name="post" pos="-0.28 0.1 0.16">
      <geom type="capsule" fromto="0 0 -0.16 0 0 0.16" size="0.03" rgba="0.25 0.55 0.9 1"/></body>
    <camera name="look" pos="0.72 -0.68 0.5" mode="targetbody" target="crate" fovy="40"/>
  </worldbody>
</mujoco>
"""
model = mj.MjModel.from_xml_string(XML)
data = mj.MjData(model)
for _ in range(200):
    mj.mj_step(model, data)
renderer = mj.Renderer(model, height=620, width=820)
renderer.update_scene(data, camera="look")
frame = renderer.render()
del renderer
sat = float(((frame.max(2).astype(int) - frame.min(2).astype(int)) > 45).mean())
assert sat > 0.10, f"frame has no content: sat={sat:.4f}"
np.save(f"/tmp/glart-frame-{LABEL}.npy", frame)
facts["render_saturated_frac"] = sat
facts["crate_settled_z"] = float(data.qpos[2])
print(f"render sat={sat:.4f} crate z={data.qpos[2]:.4f}")
save()
print("wrote", OUT)
