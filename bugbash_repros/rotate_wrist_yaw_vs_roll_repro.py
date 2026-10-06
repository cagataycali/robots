"""rotate_wrist(target_yaw=...) actually drives the ROLL joint - silently wrong on g1.

Bugbash fire #105, rotation: readme_quickstart.
Target version: v0.5.3.

Symptom
-------
The `rotate_wrist` motion primitive's agent-facing surface is uniformly labelled
"yaw":
  - tool_spec description: "rotate_wrist (wrist-yaw set-point holding position)"
  - parameter name:        target_yaw
  - result JSON keys:      target_yaw, final_yaw, yaw_error_rad
But the joint it actuates is picked by the hint tuple

  # strands_robots/simulation/motion_primitives_base.py:53-57
  _WRIST_HINTS = ("wrist_roll", "wrist_yaw", "wrist_rotate", "wrist")
  # Fallback: the last non-gripper hinge joint in the robot's chain
  # (the distal roll joint on most serial arms).

whose FIRST entry is "wrist_roll" and whose own documented fallback admits
it's hunting for roll joints. 'yaw' and 'roll' are orthogonal rotation axes
in standard robotics terminology (yaw = Z-axis, roll = forearm-axis); they
are not synonyms.

Severity by robot family
------------------------
* 6-DOF serial arms (so100, so101, koch, panda, ...) have ONE distal twist
  joint, named *_Roll. The API is misleadingly labelled, but at least the
  user cannot pick the wrong axis - there is only one. The result JSON
  still ships "yaw_error_rad" when the servo is a roll axis, so recorded
  datasets carry mislabelled action keys.
* g1 (and any humanoid with a 3-DOF wrist, i.e. three orthogonal joints
  roll + pitch + yaw) is SILENTLY WRONG: both wrist_roll_joint AND
  wrist_yaw_joint exist, and _WRIST_HINTS[0]='wrist_roll' makes the
  primitive pick the roll joint while the user requested yaw. The physical
  rotation is around the X axis, not the Z axis the parameter name
  promises.

Expected
--------
Either:
(a) rename the API to match reality: rotate_wrist(target_angle=...), with
    'wrist_joint' in the result naming the actual joint it chose; or
(b) make the hint priority follow the parameter name (wrist_yaw first,
    then roll as a FALLBACK with a one-line note in the result when the
    robot has no yaw joint).

Observed
--------
Below prints are from this repro on main @ 3d3a32e26 (post-v0.5.2):

  [1] tool_spec advertises 'wrist-yaw': True
  [2] parameter is 'target_yaw': confirmed
  [3] so100 picks 'Wrist_Roll' (its only distal twist, roll axis)
  [4] MJCF so100/Wrist_Roll local axis: (0.0, 1.0, 0.0)  (forearm-aligned)
  [5] 0/23 arms expose a *_yaw joint; 2/23 expose a *_roll joint
  [6] g1 has BOTH right_wrist_yaw_joint AND right_wrist_roll_joint
      - rotate_wrist(target_yaw=X) picks right_wrist_roll_joint (axis X)
      - the actual yaw joint (axis Z) is right there, unused
  [7] _WRIST_HINTS[0] = 'wrist_roll' - confession in-tree
"""

import json
import os
import re
import sys

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

from strands_robots import Robot  # noqa: E402
import mujoco  # noqa: E402

FAIL = []

# 1. The tool_spec text advertises "wrist-yaw"
r = Robot("so100")
desc = r.tool_spec["description"]
assert "wrist-yaw" in desc, "tool_spec no longer calls it wrist-yaw - test is stale"
print(f"[1] tool_spec advertises 'wrist-yaw': {'wrist-yaw' in desc}")

# 2. Parameter name is `target_yaw`
bad = r(action="rotate_wrist", not_target_yaw=0.3)
assert "'target_yaw'" in bad["content"][0]["text"], "param name moved?"
print("[2] parameter is 'target_yaw': confirmed")

# 3. so100 (6-DOF): the joint actuated is 'Wrist_Roll'
ok = r(action="rotate_wrist", target_yaw=0.3, max_steps=5)
j = None
for c in ok["content"]:
    if "json" in c:
        j = c["json"].get("wrist_joint")
print(f"[3] so100 wrist joint: {j!r}")
if "roll" in (j or "").lower() and "yaw" not in (j or "").lower():
    FAIL.append(f"so100: API label 'yaw', actual joint '{j}'")

# 4. MJCF confirms Wrist_Roll's axis is forearm-aligned (local Y)
m = r.mj_model
for i in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if "Wrist_Roll" in nm:
        axis = tuple(float(x) for x in m.jnt_axis[i])
        print(f"[4] MJCF so100/Wrist_Roll local axis: {axis}")

# 5. Scan the whole arm family
reg = json.load(open("strands_robots/registry/robots.json"))["robots"]
arms = [n for n, e in reg.items() if e.get("category") == "arm"]
yaw_arms, roll_arms = [], []
for name in arms:
    try:
        rr = Robot(name)
        res = rr(action="rotate_wrist", target_yaw=0.1, max_steps=2)
        for c in res.get("content", []):
            if "json" in c:
                jn = (c["json"].get("wrist_joint") or "").lower()
                if "yaw" in jn:
                    yaw_arms.append((name, jn))
                elif "roll" in jn:
                    roll_arms.append((name, jn))
    except Exception:
        pass

print(f"[5] arms with *_yaw joint: {len(yaw_arms)}/{len(arms)}  (API label achievable)")
print(f"[5] arms with *_roll joint: {len(roll_arms)}/{len(arms)}  (API drove the WRONG axis)")

# 6. g1 silent-wrong: both yaw and roll joints exist, roll wins
rg = Robot("g1")
rg_wrist = {}
mg = rg.mj_model
for j in range(mg.njnt):
    nm = mujoco.mj_id2name(mg, mujoco.mjtObj.mjOBJ_JOINT, j)
    if "right_wrist" in nm:
        axis = tuple(float(x) for x in mg.jnt_axis[j])
        rg_wrist[nm] = axis

res_g = rg(action="rotate_wrist", target_yaw=0.1, max_steps=5)
picked = None
for c in res_g["content"]:
    if "json" in c:
        picked = c["json"].get("wrist_joint")

print("[6] g1 right-wrist joint graph:")
for nm, ax in rg_wrist.items():
    marker = "  <-- API picked this when user asked for yaw" if picked and picked in nm else ""
    print(f"      {nm}: axis={ax}{marker}")

if picked and "roll" in picked.lower():
    FAIL.append(
        f"g1 silent-wrong: user asked target_yaw=0.1, "
        f"API picked '{picked}' (X-axis roll); "
        "actual *_wrist_yaw_joint (Z-axis) exists and was left idle."
    )

# 7. The code comment confesses
src = open("strands_robots/simulation/motion_primitives_base.py").read()
m = re.search(r"_WRIST_HINTS\s*=\s*\(([^)]+)\)", src)
if m:
    first = m.group(1).split(",")[0].strip().strip("'\"")
    print(f"[7] _WRIST_HINTS[0] = {first!r}")
    if first == "wrist_roll":
        FAIL.append(
            "motion_primitives_base.py:_WRIST_HINTS[0]='wrist_roll' - "
            "the yaw API's own first-choice hint IS a roll joint."
        )

print()
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("No defect reproduced.")
