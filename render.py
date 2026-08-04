"""Render a dropped crate under the gravity each tree leaves in the world."""
import json, pathlib
import numpy as np
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

SCENE = """
<mujoco model="gravity_demo">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="{gx} {gy} {gz}"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.55 0.55 0.58" diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1"/>
    <rgba haze="0.92 0.94 0.97 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.72 0.80 0.92" rgb2="0.96 0.97 0.99" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="1.4 -1.4 3.0" dir="-0.35 0.35 -1" diffuse="0.75 0.75 0.75"/>
    <geom name="ground" type="plane" size="4 4 0.02" rgba="0.80 0.82 0.85 1"/>
    <!-- static reference marker at the release height, offset in y so it can
         never touch the crate; makes the displacement readable off the frame.
         The camera targets this static post, not the crate, so the crate's
         vertical motion is visible rather than tracked out of the frame. -->
    <body name="post" pos="0 0.46 0.39">
      <geom name="post_g" type="capsule" size="0.014 0.39" rgba="0.30 0.30 0.36 1"/>
    </body>
    <body name="marker" pos="0 0.46 0.78">
      <geom name="marker_g" type="sphere" size="0.055" rgba="0.13 0.47 0.93 1"/>
    </body>
    <body name="crate" pos="0 0 0.78">
      <freejoint/>
      <geom name="crate_g" type="box" size="0.17 0.17 0.17" rgba="0.97 0.42 0.06 1" mass="0.8"/>
    </body>
    <camera name="look" pos="2.40 -2.40 0.86" mode="targetbody" target="post" fovy="27"/>
  </worldbody>
</mujoco>
"""

def frames_for(gravity, tag, times=(0.0, 0.4, 0.8)):
    sim = MuJoCoSimEngine(tool_name=f"g_{tag}", mesh=False)
    xml = SCENE.format(gx=gravity[0], gy=gravity[1], gz=gravity[2])
    p = pathlib.Path(f"/tmp/art/{tag}.xml"); p.write_text(xml)
    sim.load_scene(str(p))
    m = sim._world._model
    assert [round(float(v), 4) for v in m.opt.gravity] == [round(float(v), 4) for v in gravity], (
        f"scene gravity is {list(m.opt.gravity)}, asked for {gravity}"
    )
    out, zs = [], []
    prev = 0.0
    for t in times:
        n = int(round((t - prev) / 0.002))
        if n:
            sim.step(n)
        prev = t
        r = sim.render(camera_name="look", width=460, height=360)
        png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
        f = pathlib.Path(f"/tmp/art/frames/{tag}_{t:.1f}.png"); f.write_bytes(png)
        out.append(str(f))
        d = sim.get_body_state(body_name="crate")
        zs.append(round(float(next(c["json"] for c in d["content"] if "json" in c)["position"][2]), 4))
    sim.cleanup()
    return out, zs

rows = {}
for tag, grav in (("main_true", [0.0, 0.0, 1.0]), ("pr_kept", [0.0, 0.0, -9.81])):
    files, zs = frames_for(grav, tag)
    rows[tag] = {"gravity": grav, "frames": files, "crate_z": zs}
    print(tag, grav, "crate z:", zs)
json.dump(rows, open("/tmp/art/render.json", "w"), indent=1)
