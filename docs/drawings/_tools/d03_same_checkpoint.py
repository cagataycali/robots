"""D3: the same checkpoint, sim or real. One policy object, one call; the embodiment map speaks both dialects.

Sources: strands_robots/policies/lerobot_local/embodiment.py (state_keys, unit conversion, the
'.pos' fallback that binds hardware keys), strands_robots/policies/factory.py create_policy,
strands_robots/simulation/base.py run_policy(policy_object=), strands_robots/hardware_robot.py
run_policy(policy_object), docs/start/first-policy.md (the run this draws).
"""

from excal import MUTED, Drawing

d = Drawing(
    "d03_same_checkpoint",
    "One policy object built by create_policy from a Hub checkpoint receives observation.state in the "
    "units it was trained on and returns action in the same units; the embodiment map so101 sits "
    "between it and the robot, converting sim joints 1 to 6 in radians to degrees and back, and "
    "binding the lerobot driver's shoulder_pan.pos to gripper.pos keys, already in degrees, without "
    "converting; run_policy(policy_object=...) is the same call on Robot('so101') and "
    "Robot('so101', mode='real').",
)

ck = d.box(380, 30, 440, 78, "robotfuel/act_so101_t16b", kind="code", sub="state[6] in degrees + wrist image -> action[6] in degrees", size=16)
pol = d.box(380, 160, 440, 70, "one Policy object", kind="accent", sub='create_policy("lerobot_local", ..., embodiment="so101")', size=17)
d.arrow(ck, "b", pol, "t", "loaded once", label_dy=-4)

d.region(280, 280, 640, 150, 'embodiment map "so101"')
left = d.box(300, 320, 280, 90, "sim dialect", sub="keys 1..6 in radians: convert both ways", size=15)
right = d.box(620, 320, 280, 90, "driver dialect", sub="shoulder_pan.pos..gripper.pos in degrees: bind", size=15)
d.text(380, 240, "observation in, action out: the same tensor either way", size=13, color=MUTED, align="left")

sim = d.box(280, 500, 300, 78, 'Robot("so101")', kind="code", sub="MuJoCo, add_camera(\"wrist\")", size=16)
real = d.box(620, 500, 300, 78, 'Robot("so101", mode="real")', kind="code", sub="lerobot driver, cameras={\"wrist\": ...}", size=16)

d.path([(440, 410), (440, 500)])
d.path([(760, 410), (760, 500)])
d.path([(480, 500), (480, 410)])
d.path([(800, 500), (800, 410)])
d.text(360, 448, "actions", size=13, color=MUTED, align="right", w=70)
d.text(488, 448, "observation", size=13, color=MUTED)
d.text(680, 448, "actions", size=13, color=MUTED, align="right", w=70)
d.text(808, 448, "observation", size=13, color=MUTED)

d.path([(380, 195), (200, 195), (200, 365), (300, 365)], "run_policy(policy_object=policy, ...)", label_at=0, label_dy=-16)
d.path([(820, 195), (1000, 195), (1000, 365), (900, 365)], "the same line", label_at=0, label_dy=-16)

d.caption(280, 610, "the checkpoint never sees a robot; the map is the only thing that knows which one it is.")
d.caption(280, 630, 'obs_rename_override={"wrist": "observation.images.wrist", "default": None} names the camera, both sides.')
d.save()
