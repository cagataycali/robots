"""D4: the operator gate. The decision order of gate_motion, and the one path that skips it.

Sources: strands_robots/_command_gate.py gate_motion (allowlist -> BYPASS_TOOL_CONSENT -> no
tool_context: refuse -> interrupt <tool>-command-approval; every outcome to the audit log),
strands_robots/hardware_robot.py (execute/start call it), strands_robots/tools (ros, serial, pose,
unitree call it), strands_robots/drivers/feetech (move_to dispatches without it).
"""

from excal import MUTED, Drawing

d = Drawing(
    "d04_gate_chain",
    "A motion command enters gate_motion and is decided in order: the tool's allowlist variable "
    "allows silently; BYPASS_TOOL_CONSENT=true allows with a warning; no operator reachable "
    "refuses, naming the variable that would pre-approve; otherwise a Strands interrupt asks the "
    "operator and y dispatches, anything else declines. Every outcome is written to the audit log. "
    "The native drivers' move_to is the one command that does not pass through the gate today.",
)

cmd = d.box(40, 60, 260, 70, "motion command", sub="execute, start; ros, serial, pose, unitree", size=17)
gate = d.box(380, 60, 260, 70, "gate_motion", kind="code", sub="strands_robots._command_gate", size=17)
d.arrow(cmd, "r", gate, "l")

steps = [
    ("1  allowlist variable names it", "STRANDS_ROBOT_COMMAND_ALLOW=execute", "allow, silently"),
    ("2  BYPASS_TOOL_CONSENT=true", "lifts every gate", "allow, WARNING logged"),
    ("3  nobody to ask", "no tool_context, no interrupt", "refuse, naming the variable"),
    ("4  ask the operator", "interrupt <tool>-command-approval", "y dispatches, anything else declines"),
]
y = 210
boxes = []
for title, sub, outcome in steps:
    b = d.box(380, y, 360, 62, title, sub=sub, size=15)
    boxes.append(b)
    d.text(780, y + 12, outcome, size=15)
    y += 90
for a, b in zip(boxes, boxes[1:], strict=False):
    d.arrow(a, "b", b, "t", "else", label_dy=-4)
d.arrow(gate, "b", boxes[0], "t")

d.region(360, 170, 700, 420, "decided in this order, once per call")
audit = d.box(780, 620, 260, 50, "audit log", kind="chip", sub="every answer, in order", size=15)
d.path([(1030, 240), (1080, 240), (1080, 645), (1040, 645)], "each outcome", label_at=1, label_dy=-14)

gap = d.box(40, 420, 280, 70, "native move_to", kind="accent", sub="drivers.feetech, not gated today", size=15)
d.text(40, 510, "reading and stopping are never gated.", size=14, color=MUTED)
d.caption(40, 680, "the operator's reply goes to the audit log, never to the model.")
d.save()
