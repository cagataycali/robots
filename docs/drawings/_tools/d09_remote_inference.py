"""D9: remote inference. The robot host keeps the control loop and the gate; the GPU host runs the model.

Sources: strands_robots/inference/protocol.py (ready, set_state_keys, set_control_frequency, reset,
get_actions, actions, ok; encode_ndarray), inference/server.py PolicyServer, inference/client.py
RemotePolicy (provider "remote", create_policy("ws://host:8765")), docs/learn/policies/remote.md.
"""

from excal import MUTED, Drawing

d = Drawing(
    "d09_remote_inference",
    "Two hosts. The robot host runs Robot, the control loop, send_action and the operator gate, holding a "
    "RemotePolicy built by create_policy from a ws:// URL. The GPU host runs PolicyServer wrapping any "
    "Policy. On connect the server sends ready with the policy's metadata; the client forwards "
    "set_state_keys, set_control_frequency and reset; each control step sends get_actions with the encoded "
    "observation, the instruction and the RTC delay, and receives actions, a JSON list of action dicts.",
)

d.region(40, 30, 460, 420, "robot host (laptop, Jetson)")
robot = d.box(60, 70, 420, 64, 'Robot("so101") or Robot("so101", mode="real")', kind="code", sub="owns the control loop and send_action", size=14)
gate = d.box(60, 170, 190, 56, "operator gate", kind="accent", sub="stays on this host", size=15)
loop = d.box(270, 170, 210, 56, "control loop", sub="frequency, chunk consumption", size=15, sub_size=12)
rp = d.box(60, 270, 420, 70, "RemotePolicy", kind="code", sub='create_policy("ws://gpu:8765"), provider "remote"', size=16, sub_size=12)
d.path([(155, 134), (155, 170)])
d.path([(375, 134), (375, 170)])
d.path([(375, 226), (375, 270)])
d.text(60, 360, "a proxy with the same Policy contract:", size=13, color=MUTED)
d.text(60, 380, "get_actions crosses the wire, nothing else changes", size=13, color=MUTED)

d.region(780, 30, 400, 420, "GPU host")
ps = d.box(800, 70, 360, 64, "PolicyServer", kind="code", sub="wraps any Policy, [inference] extra", size=15)
pol = d.box(800, 270, 360, 70, "the Policy", sub="lerobot_local, wbc, ... loaded once", size=16)
d.arrow(ps, "b", pol, "t", "get_actions", label_dy=-4)

d.path([(780, 96), (500, 96)], "ready: metadata", label_dy=-16)
d.path([(500, 120), (780, 120)], "set_state_keys, reset", label_dy=8)
d.path([(500, 290), (780, 290)], "get_actions: observation, instruction", label_dy=-16)
d.path([(780, 320), (500, 320)], "actions: a JSON list", label_dy=8)

d.caption(40, 480, "what crosses the wire is an observation and a chunk of actions; the robot, the gate and the audit never do.")
d.save()
