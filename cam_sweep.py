"""Measured camera sweep: pick the framing, do not guess it."""
import numpy as np, mujoco, itertools

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]
LIMITS = [(-3.1, 3.1), (-1.8, 1.8), (-2.4, 2.4), (-1.7, 1.7), (-0.2, 1.5)]
POSE = {"shoulder_pan": 0.0, "shoulder_lift": -0.55, "elbow": 0.95, "wrist_roll": 0.0}

def mjcf(cam):
    return f"""
<mujoco model="replay_arm">
  <compiler angle="radian"/><option gravity="0 0 0"/>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
    <global offwidth="1400" offheight="1200"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.32 0.4 0.52" rgb2="0.06 0.08 0.12" width="256" height="256"/></asset>
  <worldbody>
    <light pos="0.3 -0.3 0.7" dir="-0.3 0.3 -0.7"/>
    <body name="base"><geom type="cylinder" size="0.035 0.02" rgba="0.30 0.32 0.36 1"/>
      <body name="l1" pos="0 0 0.02">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="{LIMITS[0][0]} {LIMITS[0][1]}"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.018" rgba="0.42 0.45 0.50 1"/>
        <body name="l2" pos="0 0 0.06">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="{LIMITS[1][0]} {LIMITS[1][1]}"/>
          <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.016" rgba="0.42 0.45 0.50 1"/>
          <body name="l3" pos="0.10 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="{LIMITS[2][0]} {LIMITS[2][1]}"/>
            <geom type="capsule" fromto="0 0 0 0.085 0 0" size="0.014" rgba="0.42 0.45 0.50 1"/>
            <body name="l4" pos="0.085 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="{LIMITS[3][0]} {LIMITS[3][1]}"/>
              <geom type="capsule" fromto="0 0 0 0.030 0 0" size="0.013" rgba="0.50 0.53 0.58 1"/>
              <geom type="box" pos="0.055 0 -0.016" size="0.026 0.006 0.004" rgba="0.22 0.62 0.35 1"/>
              <body name="jaw_body" pos="0.030 0 0.010">
                <joint name="jaw" type="hinge" axis="0 1 0" range="{LIMITS[4][0]} {LIMITS[4][1]}"/>
                <geom type="box" pos="0.026 0 0" size="0.026 0.006 0.004" rgba="0.95 0.55 0.12 1"/>
              </body></body></body></body></body></body>
    {cam}
  </worldbody></mujoco>"""

def measure(dist, elev_z, fovy, target="jaw_body"):
    cam = f'<camera name="grip" pos="0 0 0" mode="targetbody" target="{target}" fovy="{fovy}"/>'
    m = mujoco.MjModel.from_xml_string(mjcf(cam))
    d = mujoco.MjData(m)
    adr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS}
    r = mujoco.Renderer(m, 620, 700)
    # aim: place the camera at target + offset by editing the model's cam pos
    d.qpos[:] = 0.0
    for j, v in POSE.items(): d.qpos[adr[j]] = v
    mujoco.mj_forward(m, d)
    tgt = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)].copy()
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "grip")
    m.cam_pos[cid] = tgt + np.array([dist * 0.6, -dist * 0.72, elev_z])
    imgs = {}
    for tag, jaw in (("closed", LIMITS[4][0]), ("opened", LIMITS[4][1])):
        d.qpos[:] = 0.0
        for j, v in POSE.items(): d.qpos[adr[j]] = v
        d.qpos[adr["jaw"]] = jaw
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="grip")
        imgs[tag] = r.render().copy()
    arm = float((imgs["closed"].mean(2) > 88).mean())
    diff = float((np.abs(imgs["closed"].astype(int) - imgs["opened"].astype(int)).sum(2) > 24).mean())
    return arm, diff, tgt, m.cam_pos[cid].copy()

print(f"{'dist':>6} {'z':>6} {'fovy':>5} {'arm%':>7} {'diff%':>7}")
best = None
for dist, elev_z, fovy in itertools.product((0.10, 0.16, 0.24, 0.34), (0.02, 0.06), (30, 42, 55)):
    arm, diff, tgt, cp = measure(dist, elev_z, fovy)
    print(f"{dist:6.2f} {elev_z:6.2f} {fovy:5d} {arm*100:6.2f}% {diff*100:6.2f}%")
    if arm > 0.06 and (best is None or diff > best[1]):
        best = (f"dist={dist} z={elev_z} fovy={fovy}", diff, arm, dist, elev_z, fovy, tgt, cp)
print("\nBEST:", best[0], f"diff={best[1]*100:.2f}% arm={best[2]*100:.2f}%")
print("target:", best[6], "cam_pos:", best[7])
