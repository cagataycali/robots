"""Capture what the GL-gated halves verify, plus the measured host/mutation matrices."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import imageio.v3 as iio
import numpy as np

import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
OUT = ROOT / "_art"
GLBLOCK = f"/tmp/glblock-{os.environ['GITHUB_RUN_ID']}"

MODULES = [
    "tests/benchmarks/libero/test_libero_camera_config_domain.py",
    "tests/simulation/mujoco/test_entity_name_lookup_type_safety.py",
    "tests/simulation/test_unhashable_entity_name_is_reported.py",
]


def summary(args: list[str], *, no_gl: bool) -> str:
    env = {**os.environ, "MUJOCO_GL": "egl"}
    extra = ["-p", "noglhost"] if no_gl else []
    if no_gl:
        env["PYTHONPATH"] = GLBLOCK
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "--no-cov", "-p", "no:randomly", *extra],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    for line in reversed(proc.stdout.splitlines()):
        if any(w in line for w in ("passed", "failed", "error")):
            return line.strip().strip("=").strip()
    return "no summary"


def png(sim, camera: str, w: int, h: int, name: str) -> dict:
    """Render through the public envelope and save the frame the assertion checks."""
    result = sim.render(camera_name=camera, width=w, height=h)
    assert result["status"] == "success", result
    blob = next(c["image"]["source"]["bytes"] for c in result["content"] if "image" in c)
    (OUT / name).write_bytes(blob)
    arr = iio.imread(OUT / name)
    sat = float(((arr.max(axis=2).astype(int) - arr.min(axis=2).astype(int)) > 45).mean())
    return {"file": name, "shape": list(arr.shape), "saturated_frac": round(sat, 4)}


facts: dict = {"tree": str(ROOT), "frames": {}}

# --- the frame the entity-name-lookup GL half verifies -----------------------
from strands_robots.simulation import Simulation  # noqa: E402

sim = Simulation(tool_name="art_entity", mesh=False)
sim.create_world()
sim.add_object("crate", shape="box", size=[0.12, 0.12, 0.12], position=[0.3, 0.0, 0.06], is_static=False)
assert sim.get_body_state(body_name=None)["status"] == "error"  # the refused lookup
facts["frames"]["entity_name"] = png(sim, "default", 420, 340, "frame_entity_name.png")
sim.cleanup()

# --- the frame the unhashable-name GL half verifies --------------------------
ARM = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="2 2 .1" rgba=".55 .58 .62 1"/>
    <body name="base" pos="0 0 .05">
      <joint name="pan" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 .12" size=".03" rgba=".2 .5 .85 1"/>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="20" ctrlrange="-1.5 1.5"/></actuator>
</mujoco>
"""
xml = OUT / "arm.xml"
xml.write_text(ARM)
sim = Simulation(tool_name="art_unhashable", mesh=False)
sim.create_world()
sim.add_object("crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.3, 0.0, 0.05], is_static=False)
sim.add_robot(name="arm", urdf_path=str(xml))
sim.add_camera(name="look", position=[0.9, -0.9, 0.6], target=[0.3, 0.0, 0.1])
facts["frames"]["unhashable_name"] = png(sim, "look", 420, 340, "frame_unhashable_name.png")
sim.cleanup()

# --- the frame the libero camera-config GL half verifies --------------------
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from tests.benchmarks.libero.test_libero_camera_config_domain import GOOD  # noqa: E402
from tests.benchmarks.libero.test_libero_camera_install_resilience import _adapter  # noqa: E402

sim = MuJoCoSimEngine(tool_name="art_libero", mesh=False)
sim.create_world()
sim.add_robot(name="panda", data_config="panda")
adapter = _adapter()
adapter._cameras = {"image": dict(GOOD)}
adapter._install_libero_cameras(sim)
entry = sim._world.cameras["image"]
facts["installed_dims"] = [entry.width, entry.height]
facts["frames"]["libero_camera"] = png(sim, "image", 420, 340, "frame_libero_camera.png")
sim.cleanup()

# --- the host matrix, measured ----------------------------------------------
facts["host_matrix"] = {
    "gl_present": summary(MODULES, no_gl=False),
    "gl_free": summary(MODULES, no_gl=True),
}
facts["guard"] = summary(["tests/test_mujoco_render_assertions_are_gl_gated.py"], no_gl=False)

(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
