"""D10: the agent and the robot. The LLM emits tool calls; the robot tool runs them; text comes back.

Sources: strands_robots/robot.py (factory returns an AgentTool), strands_robots/simulation/base.py
(tool_spec, stream: 77 actions in sim), strands_robots/hardware_robot.py (execute/start gated),
docs/assets/transcripts/talk-to-it.txt (the captured run the labels quote).
"""

from excal import MUTED, Drawing

d = Drawing(
    "d10_agent_and_robot",
    "A person types a sentence to a Strands Agent; the model answers with tool calls on the robot "
    "tool (add_object, run_policy, execute); each call returns a status envelope the model reads; "
    "on a real arm the execute call passes the operator gate first.",
)

person = d.box(40, 70, 190, 70, "you", sub="types a sentence", size=18)
agent = d.box(360, 50, 300, 110, "Strands Agent", sub="the model reasons, then emits tool calls", size=18)
d.arrow(person, "r", agent, "l", "a sentence", off_a=-0.3, off_b=-0.3, label_dy=-16)
d.arrow(agent, "l", person, "r", "words back", off_a=0.3, off_b=0.3, label_dy=10)

d.region(800, 20, 420, 340, "the robot tool")
sim = d.box(820, 60, 380, 64, "so101_sim", kind="code", sub='Robot("so101"): 77 actions, never gated', size=16)
real = d.box(820, 150, 380, 64, "so101", kind="code", sub='Robot("so101", mode="real"): execute waits for a yes', size=16)
gate = d.box(910, 270, 200, 56, "operator gate", kind="accent", sub="n: declined. y: dispatched", size=15)
d.arrow(real, "b", gate, "t")

d.path([(660, 80), (820, 80)], "tool_use: add_object", label_dy=-16)
d.path([(660, 130), (740, 130), (740, 182), (820, 182)], "tool_use: execute", label_at=2, label_dy=-16)

env = d.box(820, 420, 380, 70, "status, content[]", kind="chip", sub="text, json, image blocks", size=15)
d.path([(1010, 360), (1010, 420)], "tool_result", label_dy=-22)
d.path([(820, 455), (510, 455), (510, 160)], "the model reads it and answers", label_at=0, label_dy=-16)
d.text(1010 + 12, 398, "every action, sim or real", size=13, color=MUTED)

d.caption(40, 520, "the model never touches an actuator; it asks the tool, and reads what the tool says.")
d.caption(40, 540, "the transcript on Start > See it is one such run, recorded.")
d.save()
