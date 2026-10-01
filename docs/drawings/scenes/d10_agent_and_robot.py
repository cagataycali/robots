"""D10: the agent and the robot. The model emits tool calls; the robot tool runs them; text comes back.

Sources: strands_robots/robot.py (the factory returns an agent tool), simulation/base.py (tool_spec,
stream: 77 actions in sim), hardware_robot.py (execute and start are gated),
docs/assets/transcripts/talk-to-it.txt (the captured run the labels quote).
"""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d10_agent_and_robot",
        "The agent asks the tool; the tool moves the robot",
        "A sentence becomes tool calls on the Robot object; every call answers with a status envelope the model reads before it speaks.",
        "Left, you, typing a sentence to a Strands Agent; a wire, a sentence, into the agent and a dashed wire, "
        "words back. Middle, the Strands Agent: the model reasons, then emits tool calls; it never writes a "
        "servo command. Right, a dashed layer, the robot tool: so101_sim, Robot(\"so101\"), 77 actions in "
        "simulation, never gated, with chips add_object, run_policy and render; so101, Robot(\"so101\", "
        "mode=\"real\"), the same verbs, where execute and run_policy wait; under it the one green element, the "
        "operator gate, n declined, y dispatched. Wires from the agent into the layer carry tool_use: add_object "
        "and tool_use: execute. Under the layer a card, the status envelope, status and content with text, json "
        "and image blocks; a wire tool_result reaches it from the layer and a dashed wire returns to the agent, "
        "the model reads it and answers. Footnote: the model never touches an actuator.",
        h=710,
    )
    # ---------------------------------------------------------------- you and the agent
    s.section(60, 122, "a person")
    s.box(60, 134, 150, 86, "you", "type a sentence; read the words back", size=14, subsize=12)
    s.section(300, 122, "the agent")
    s.box(300, 134, 260, 136, "Strands Agent",
          "the model reasons over its tools, then emits tool calls; it never writes a servo command",
          size=14, subsize=12)
    s.chips(314, 224, ["Agent(tools=[robot])"])
    s.arrow([(210, 164), (300, 164)], label="a sentence", label_dx=0, label_dy=-8, label_anchor="middle")
    s.arrow([(300, 200), (210, 200)], dashed=True, label="words back", label_dx=0, label_dy=16, label_anchor="middle")

    # ---------------------------------------------------------------- the robot tool
    s.box(680, 134, 460, 398, None, None, dashed=True)
    s.text(694, 156, "THE ROBOT TOOL: ONE OBJECT, TWO BODIES", cls="mono muted", size=10.5, spacing="0.05em")
    s.box(700, 170, 420, 110, "so101_sim",
          'Robot("so101"): 77 actions in simulation, never gated; a scene the agent grows', size=14, subsize=12)
    s.chips(714, 244, ["add_object", "run_policy", "render", "get_state"])
    s.box(700, 300, 420, 110, "so101",
          'Robot("so101", mode="real"): the same verbs; execute and run_policy wait for a yes', size=14, subsize=12)
    s.chips(714, 374, ["execute", "start", "stop"])
    s.down(800, 410, 436)
    s.box(700, 436, 420, 76, "operator gate",
          "an interrupt on the real arm: n declined, y dispatched; the reply is audited, not shown to the model",
          accent=True, size=14, subsize=12)
    s.arrow([(560, 190), (700, 190)])
    s.text(630, 182, "tool_use: add_object", cls="mono muted", size=10, anchor="middle")
    s.arrow([(560, 240), (620, 240), (620, 350), (700, 350)])
    s.text(612, 300, "tool_use: execute", cls="mono muted", size=10.5, anchor="end")

    # ---------------------------------------------------------------- the envelope
    s.down(910, 532, 566, label="tool_result", label_dx=10, label_dy=4)
    s.box(680, 566, 460, 80, "the status envelope",
          "status and content[]: text, json and image blocks; every action answers with one, sim or real",
          size=14, subsize=12)
    s.arrow([(680, 606), (430, 606), (430, 270)], dashed=True)
    s.text(446, 598, "the model reads it and answers", cls="mono muted", size=10.5)

    s.footnote(682, "the model never touches an actuator: it asks the tool and reads what the tool says. Start, See it is one such run, recorded.")
    return s
