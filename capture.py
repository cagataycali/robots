"""Capture: what the Isaac primitives command through each limit source, replayed on MuJoCo.

The Isaac articulation is faked (Isaac Sim is not required to reach either
decision); the joint targets it commands are then replayed onto a MuJoCo arm
declaring the SAME joint vocabulary and the SAME limits, so the set-point the
fallback source resolved is visible rather than only tabulated.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np

import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
OUT = pathlib.Path(__file__).resolve().parent

for n in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
    sys.modules.setdefault(n, types.ModuleType(n))


class _AA:
    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions, self.joint_indices = joint_positions, joint_indices


sys.modules["isaacsim.core.utils.types"].ArticulationAction = _AA

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]
LIMITS = [(-3.1, 3.1), (-1.8, 1.8), (-2.4, 2.4), (-1.7, 1.7), (-0.2, 1.5)]


class _Art:
    """Articulation whose limit source is chosen: 'props', 'fallback' or 'none'."""

    def __init__(self, source: str):
        self.positions = np.zeros(len(JOINTS))
        self.applied: list = []
        self._targets: dict[int, float] = {}
        self.source = source
        if source == "props":
            a = np.zeros(len(JOINTS), dtype=[("hasLimits", "?"), ("lower", "f8"), ("upper", "f8")])
            for i, (lo, hi) in enumerate(LIMITS):
                a["hasLimits"][i], a["lower"][i], a["upper"][i] = True, lo, hi
            self.dof_properties = a
        elif source == "fallback":
            # (num_envs, num_dofs, 2): the shape an Isaac ArticulationView reports.
            self._table = np.array([[[lo, hi] for lo, hi in LIMITS]], dtype=np.float64)

    def __getattr__(self, name):
        if name == "get_dof_limits" and "_table" in self.__dict__:
            return lambda: self._table
        raise AttributeError(name)

    def get_joint_positions(self):
        return self.positions.copy()

    def apply_action(self, action) -> None:
        self.applied.append(action)
        for i, v in zip(np.asarray(action.joint_indices), np.asarray(action.joint_positions)):
            self._targets[int(i)] = float(v)

    def advance(self) -> None:
        for i, t in self._targets.items():
            self.positions[i] += 0.5 * (t - self.positions[i])


class _World:
    def __init__(self, art):
        self.articulation = art

    def step(self, render: bool = False) -> None:  # noqa: ARG002
        self.articulation.advance()


def _sim(art):
    s = IsaacSimulation()
    s._world = _World(art)
    s._world_created = True
    s._robots["arm"] = _RobotState(
        name="arm", prim_path="/World/Robots/arm", joint_names=list(JOINTS), articulation=art, data_config=None
    )
    return s


facts: dict = {"tree": str(ROOT), "joints": JOINTS, "limits": LIMITS, "runs": {}}


def drive(source: str, state: str):
    art = _Art(source)
    result = _sim(art).set_gripper(robot_name="arm", state=state, steps=10)
    ok = result["status"] == "success"
    row = {"source": source, "state": state, "status": result["status"]}
    if ok:
        payload = [c["json"] for c in result["content"] if "json" in c][0]
        row["targets"] = payload["targets"]
        row["setpoint_sources"] = payload["setpoint_sources"]
        row["final_positions"] = art.positions.tolist()
    else:
        row["text"] = result["content"][0]["text"]
    facts["runs"][f"{source}:{state}"] = row
    print(f"  {source:9s} {state:5s} -> {result['status']:8s} {row.get('targets', row.get('text', ''))}")
    return row


print("\n[Isaac set_gripper through each documented limit source]")
for src in ("props", "fallback", "none"):
    for st in ("close", "open"):
        drive(src, st)

# The two set-points the FALLBACK source resolved: what MuJoCo will replay.
closed = facts["runs"]["fallback:close"]
opened = facts["runs"]["fallback:open"]
assert closed["status"] == "success" and opened["status"] == "success", "the fallback source must drive"
assert closed["targets"]["jaw"] == LIMITS[4][0], closed["targets"]
assert opened["targets"]["jaw"] == LIMITS[4][1], opened["targets"]
# The fallback source must agree with the authoritative one.
assert facts["runs"]["props:open"]["targets"] == opened["targets"], "sources disagree"
assert facts["runs"]["none:open"]["status"] == "error", "no source must refuse"

# ---------------------------------------------------------------------------
# Replay the commanded targets on a MuJoCo arm with the same joint vocabulary.
# ---------------------------------------------------------------------------
import mujoco  # noqa: E402

MJCF = f"""
<mujoco model="replay_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
    <global offwidth="1400" offheight="1200"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.4 0.52" rgb2="0.06 0.08 0.12" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.7" dir="-0.3 0.3 -0.7"/>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.035 0.02" rgba="0.30 0.32 0.36 1"/>
      <body name="l1" pos="0 0 0.02">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="{LIMITS[0][0]} {LIMITS[0][1]}" damping="3"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.018" rgba="0.42 0.45 0.50 1"/>
        <body name="l2" pos="0 0 0.06">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="{LIMITS[1][0]} {LIMITS[1][1]}" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.016" rgba="0.42 0.45 0.50 1"/>
          <body name="l3" pos="0.10 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="{LIMITS[2][0]} {LIMITS[2][1]}" damping="3"/>
            <geom type="capsule" fromto="0 0 0 0.085 0 0" size="0.014" rgba="0.42 0.45 0.50 1"/>
            <body name="l4" pos="0.085 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="{LIMITS[3][0]} {LIMITS[3][1]}" damping="3"/>
              <geom type="capsule" fromto="0 0 0 0.030 0 0" size="0.013" rgba="0.50 0.53 0.58 1"/>
              <!-- fixed finger -->
              <geom type="box" pos="0.055 0 -0.016" size="0.026 0.006 0.004" rgba="0.22 0.62 0.35 1"/>
              <body name="jaw_body" pos="0.030 0 0.010">
                <joint name="jaw" type="hinge" axis="0 1 0" range="{LIMITS[4][0]} {LIMITS[4][1]}" damping="2"/>
                <geom type="box" pos="0.026 0 0" size="0.026 0.006 0.004" rgba="0.95 0.55 0.12 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <camera name="grip" pos="0 0 0" mode="targetbody" target="jaw_body" fovy="30"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 620, 700)
adr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS}
POSE = {"shoulder_pan": 0.0, "shoulder_lift": -0.55, "elbow": 0.95, "wrist_roll": 0.0}

# Framing chosen by a measured sweep (_probe/cam_sweep.py): the offset that
# maximises the closed-vs-open pixel delta while keeping the arm large in
# frame (47.4% arm pixels, 18.2% differing).
for _j, _v in POSE.items():
    data.qpos[adr[_j]] = _v
mujoco.mj_forward(model, data)
_tgt = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_body")].copy()
model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "grip")] = _tgt + np.array([0.06, -0.072, 0.06])


def replay(row, tag):
    """Set every joint to the target the Isaac primitive commanded, then render."""
    data.qpos[:] = 0.0
    for j, v in POSE.items():
        data.qpos[adr[j]] = v
    for joint, target in row["targets"].items():
        data.qpos[adr[joint]] = float(target)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="grip")
    img = renderer.render()
    np.save(OUT / f"{tag}.npy", img)
    # "the arm is in frame": the links/fingers are far brighter than the skybox.
    arm = float((img.mean(2) > 88).mean())
    print(f"  replay {tag:12s} jaw={row['targets']['jaw']:+.2f} rad  arm_pixels={arm:.3f}")
    return img, arm


print("\n[MuJoCo replay of the fallback-sourced set-points]")
img_c, sat_c = replay(closed, "closed")
img_o, sat_o = replay(opened, "opened")
diff = float((np.abs(img_c.astype(int) - img_o.astype(int)).sum(2) > 24).mean())
facts["render"] = {"arm_closed": sat_c, "arm_opened": sat_o, "differing_fraction": diff}
print(f"  closed vs opened differ on {diff * 100:.2f}% of pixels")
assert sat_c > 0.15 and sat_o > 0.15, f"the arm is not in frame: {sat_c=} {sat_o=}"
assert diff > 0.10, f"the two set-points are not visually distinguishable: {diff=}"
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print("\nOK")
