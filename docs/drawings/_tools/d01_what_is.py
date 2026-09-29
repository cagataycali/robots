"""D1: What is Strands Robots. Agent -> Robot -> policy runtime -> backend, observations back.

Sources: strands_robots/robot.py (factory), strands_robots/policies/base.py (Policy),
strands_robots/policies/factory.py (create_policy), strands_robots/_command_gate.py (gate_motion),
strands_robots/hardware_robot.py run_policy, strands_robots/simulation/base.py run_policy,
strands_robots/inference (PolicyServer, RemotePolicy).
"""

from excal import MUTED, Drawing

d = Drawing(
    "d01_what_is",
    "An agent calls the robot tool; the robot runs a policy through the operator gate; the policy "
    "runtime maps observations and actions to one backend, simulation or a hardware driver, and "
    "can ask a remote server for the action chunk.",
)

# row 1: agent and robot
agent = d.box(40, 40, 240, 78, "Agent", sub="strands Agent(tools=[robot])", size=18)
robot = d.box(560, 40, 380, 78, "Robot", sub='Robot("so101") or Robot("so101", mode="real")', size=18)
d.arrow(agent, "r", robot, "l", "tool call: run_policy, send_action", off_a=-0.45, off_b=-0.45, label_dy=-14)
d.arrow(robot, "l", agent, "r", "result: status, state, render", off_a=0.45, off_b=0.45, label_dy=8)

d.caption(960, 52, "one object: the agent tool")
d.caption(960, 72, "and the control loop owner")

# row 2: the gate (the one accent)
gate = d.box(600, 170, 300, 60, "operator gate", kind="accent", sub="a real arm waits for a yes", size=17)
d.arrow(robot, "b", gate, "t")

# row 3: the policy runtime
d.region(300, 300, 560, 140, "policy runtime")
p1 = d.box(320, 330, 160, 50, "create_policy", kind="code", size=14)
p2 = d.box(510, 330, 160, 50, "embodiment map", kind="chip", size=14)
p3 = d.box(700, 330, 140, 50, "action chunk", kind="chip", size=14)
d.arrow(p1, "r", p2, "l")
d.arrow(p2, "r", p3, "l")
d.text(320, 400, "checkpoint + key map + units + chunking = one Policy object", size=13)
d.path([(750, 230), (750, 270), (590, 270), (590, 330)], "instruction", label_at=1, label_dy=-14)

# remote inference sits beside the runtime: the policy runs elsewhere, the robot stays here
remote = d.box(960, 310, 220, 90, "remote inference", sub="PolicyServer on a GPU host", size=16)
d.path([(860, 335), (960, 335)], "observation", label_dy=-16)
d.path([(960, 385), (860, 385)], "action chunk", label_dy=8)

# row 4: backends, joined to the runtime by an actions-down and an observation-up pair each
sim = d.box(300, 540, 250, 78, "simulation", sub="MuJoCo (default), Newton, Isaac Sim", size=17)
hw = d.box(610, 540, 250, 78, "hardware driver", sub="lerobot or a native driver", size=17)
for bx in (sim, hw):
    cx = bx["x"] + bx["width"] / 2
    d.path([(cx - 16, 440), (cx - 16, 540)])
    d.path([(cx + 16, 540), (cx + 16, 440)])
    d.text(cx - 16 - 8 - 50, 482, "actions", size=13, color=MUTED, align="right", w=50)
    d.text(cx + 16 + 8, 482, "observation", size=13, color=MUTED)

d.caption(300, 650, "one interface: get_observation, send_action, run_policy.")
d.caption(300, 670, "the backend changes; the call does not.")

d.save()
