"""D1: What Strands Robots is. The reference drawing for the docs revamp.

Left, the spine top to bottom: the agent, the operator gate (the one green element), the robot
object, the policy runtime as a dashed layer. Right, the backends the same calls reach.
Every name here exists on the clone (see ARCHITECTURE-MAP.md): Robot factory, run_policy,
send_action, get_observation, the tool verbs get_state / render / status, create_policy,
EmbodimentMap, PolicyServer, RemotePolicy, DRIVER_CHOICES lerobot / strands, gate_motion's
interrupt "<tool>-command-approval", fail closed, audit row.
"""
from scene import Scene

L, LW = 60, 580          # left column x and width (60..640)
R, RW = 780, 360         # right column x and width (780..1140)
BUS = 730                # vertical bus between the columns
DY = 12                  # the robot group and the runtime sit 12 px lower so THE ROBOT clears the gate


def scene() -> Scene:
    s = Scene(
        "d01_what_is",
        "One robot object, any backend, a person in the loop",
        "An agent calls a Robot as a tool; the Robot runs a policy and moves a body, in simulation or on hardware, through the same calls.",
        "Left column, top to bottom: a Strands Agent with the robot in its tools makes a tool call; it passes "
        "the operator gate, the one green element, where run_policy and send_action through the tool wait for "
        "a yes (an interrupt, fail closed without an operator, an audit row); after a yes it reaches "
        "Robot(\"so101\"), the execution target and the tool, with run_policy, send_action, get_observation, "
        "get_state, render and status; a dashed result wire returns to the agent. Under the robot a dashed "
        "policy runtime layer holds create_policy, the embodiment map and the action chunk, fed by the "
        "observation and returning actions. Right column, the backends the same calls reach: simulation "
        "with MuJoCo, Newton or Isaac Sim; a hardware driver, lerobot or native; and a PolicyServer on a GPU "
        "host that a RemotePolicy talks to over a WebSocket. Footnote: one interface, get_observation, "
        "send_action, run_policy; the backend changes, the call does not.",
        h=792,
    )

    # ---------------------------------------------------------------- the agent
    s.section(L, 122, "the agent")
    s.box(L, 134, LW, 76, "Agent(tools=[robot])",
          "a Strands Agent reasons over its tools; the Robot object is one of them, so a sentence becomes "
          "a tool call and the model never writes a servo command")
    s.down(300, 210, 240, label="tool call", label_dx=-10, label_dy=4, label_anchor="end")

    # ---------------------------------------------------------------- the operator gate (the one green element)
    s.box(L, 240, 340, 84, "operator gate",
          "run_policy and send_action through the tool wait for a yes", accent=True, size=14)
    s.text(420, 262, "WHAT THE GATE DOES", cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(420, 272, ["interrupt", "fail closed"])
    s.chips(420, 302, ["audit row"])
    s.down(300, 324, 352 + DY, label="after a yes", label_dx=-10, label_dy=4, label_anchor="end")
    s.arrow([(620, 352 + DY), (620, 210)], dashed=True, label="result", label_dx=10, label_dy=4)

    # ---------------------------------------------------------------- the robot
    s.section(L, 344 + DY, "the robot")
    s.box(L, 352 + DY, LW, 116, 'Robot("so101")',
          'the execution target: mode="sim" or mode="real" picks what sits underneath, and the object is '
          "the agent's tool either way")
    s.chips(L + 14, 434 + DY, ["run_policy", "send_action", "get_observation", "get_state", "render", "status"])
    s.down(300, 468 + DY, 518 + DY, label="observation", label_dx=-10, label_dy=4, label_anchor="end")
    s.arrow([(400, 518 + DY), (400, 468 + DY)], label="actions", label_dx=10, label_dy=4)

    # ---------------------------------------------------------------- policy runtime (dashed layer)
    s.box(L, 518 + DY, LW, 194, None, None, dashed=True)
    s.text(L + 14, 540 + DY, "POLICY RUNTIME, INSIDE run_policy", cls="mono muted", size=10.5, spacing="0.05em")
    bw, gap, y0, bh = 177, 10, 556 + DY, 140
    x = L + 14
    s.box(x, y0, bw, bh, "create_policy(...)",
          "provider, checkpoint, processor and device; mock, lerobot_local, wbc, remote and more",
          size=14, subsize=12)
    x += bw + gap
    s.box(x, y0, bw, bh, "embodiment map",
          "EmbodimentMap: robot keys, units and dimensions to the model's, and back again",
          size=14, subsize=12)
    x += bw + gap
    s.box(x, y0, bw + 1, bh, "action chunk",
          "one call returns a list of action dicts; the runtime applies them at the control frequency",
          size=14, subsize=12)

    # ---------------------------------------------------------------- backends
    s.section(R, 122, "backends")
    s.box(R, 134, RW, 164, "simulation",
          'the default: Robot("so101") builds a scene the agent grows with add_robot, add_camera and '
          "add_object; joints read in radians under bare MuJoCo names")
    s.chips(R + 14, 258, ["MuJoCo", "Newton", "Isaac Sim"])
    s.box(R, 328, RW, 164, "hardware driver",
          'mode="real": the lerobot driver on a USB port, or a native driver on a serial bus, DDS or a '
          "vendor API; the tool verbs stay the same")
    s.chips(R + 14, 452, ['driver="lerobot"', 'driver="strands"'])
    s.box(R, 548, RW, 164, "PolicyServer on a GPU host",
          'RemotePolicy, provider "remote": the observation crosses a WebSocket and an action chunk '
          "comes back; the control loop stays on the robot host")
    s.chips(R + 14, 672, ['create_policy("ws://gpu:8765")', "[inference]"])

    # robot -> sim / hardware: one stem, two branches, the same two verbs on each
    s.arrow([(L + LW, 400 + DY), (BUS, 400 + DY), (BUS, 216), (R, 216)])
    s.arrow([(BUS, 400 + DY), (R, 400 + DY)])
    s.text(BUS - 8, 392 + DY, "send_action", cls="mono muted", size=10.5, anchor="end")
    s.arrow([(BUS, 400 + DY), (BUS, 432 + DY), (L + LW, 432 + DY)])
    s.text((L + LW + R) / 2, 448 + DY, "get_observation", cls="mono muted", size=10.5, anchor="middle")
    # runtime <-> PolicyServer: observation out, action chunk back
    s.arrow([(L + LW, 616 + DY), (R, 616 + DY)])
    s.text((L + LW + R) / 2, 606 + DY, "observation", cls="mono muted", size=10.5, anchor="middle")
    s.arrow([(R, 644 + DY), (L + LW, 644 + DY)])
    s.text((L + LW + R) / 2, 662 + DY, "action chunk", cls="mono muted", size=10.5, anchor="middle")

    # ---------------------------------------------------------------- footnote
    s.footnote(752 + DY, "one interface: get_observation, send_action, run_policy. the backend changes; the call does not.")
    return s
