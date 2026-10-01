"""D4: the operator gate. The decision order of gate_motion, and the one path that skips it.

Sources: strands_robots/_command_gate.py gate_motion (allowlist -> BYPASS_TOOL_CONSENT -> no
tool_context: refuse -> interrupt <tool>-command-approval; every outcome to the audit log),
hardware_robot.py (execute and start call it), strands_robots/tools (ros, serial, pose, unitree call
it), drivers/feetech (move_to dispatches without it).
"""
from scene import Scene

L, LW = 60, 340          # what enters, and the one path around
M, MW = 460, 400         # gate_motion and the four steps
R = 920                  # the outcomes


def scene() -> Scene:
    s = Scene(
        "d04_gate_chain",
        "The operator gate, decided in order",
        "A motion command enters gate_motion once per call; four questions are asked in order, and every answer is written down.",
        "Left, the motion commands that enter: execute and start on the robot tool, the ros, serial, pose and "
        "unitree tools. Middle, gate_motion and under it a dashed layer with four steps decided in order: the "
        "allowlist variable names the command, allow silently; BYPASS_TOOL_CONSENT=true, allow with a warning; "
        "nobody to ask, refuse naming the variable; ask the operator, the one green element, an interrupt "
        "where y dispatches and anything else declines. Each step's outcome is written to the right and every "
        "outcome flows to the audit log under the layer. Bottom left, the one path around the gate: the "
        "native drivers' move_to is not gated today; reading and stopping are never gated. Footnote: the "
        "operator's reply goes to the audit log, never to the model.",
        h=790,
    )
    # ---------------------------------------------------------------- what enters
    s.section(L, 122, "what enters")
    s.box(L, 134, LW, 130, "motion command",
          "execute and start on the robot tool; the ros, serial, pose and unitree tools call the same gate "
          "before they dispatch", size=14, subsize=12)
    s.chips(L + 14, 226, ["execute", "start", "pose_tool", "use_ros"])
    s.arrow([(L + LW, 170), (M, 170)])

    s.section(L, 316, "the one path around it")
    s.box(L, 328, LW, 96, "native move_to",
          "drivers.feetech dispatches a move_to without gate_motion today; the pages that show it say so",
          size=14, subsize=12)
    s.para(L, 452, "reading and stopping are never gated: get_observation, get_state, status and stop answer "
           "under any posture, and an e-stop only de-energises.", LW, size=12, cls="grot muted")

    s.section(L, 540, "what the operator sees")
    s.box(L, 552, LW, 126, "the interrupt",
          "names the tool and carries the command it is about to dispatch; the reply is read once, "
          "y or anything else", size=14, subsize=12)
    s.chips(L + 14, 642, ["y", "n", "<tool>-command-approval"])

    # ---------------------------------------------------------------- gate_motion and the four steps
    s.section(M, 122, "the gate")
    s.box(M, 134, MW, 72, "gate_motion", "strands_robots._command_gate, once per call", size=14, subsize=12)
    s.down(M + MW / 2, 206, 240, label="in order", label_dx=10, label_dy=4)
    s.box(M, 240, MW, 384, None, None, dashed=True)
    s.text(M + 14, 262, "FOUR QUESTIONS, THE FIRST YES WINS", cls="mono muted", size=10.5, spacing="0.05em")
    steps = [
        ("1  the allowlist names it", "STRANDS_ROBOT_COMMAND_ALLOW=execute", "allow, silently", False),
        ("2  BYPASS_TOOL_CONSENT=true", "lifts every gate in the process", "allow, WARNING logged", False),
        ("3  nobody to ask", "no tool_context, so no interrupt can reach a person", "refuse, naming the variable", False),
        ("4  ask the operator", "interrupt <tool>-command-approval; y dispatches, anything else declines",
         "y dispatched, n declined", True),
    ]
    y, bh, gap = 278, 70, 16
    for title, sub, outcome, accent in steps:
        s.box(M + 14, y, MW - 28, bh, title, sub, accent=accent, size=13.5, subsize=11.5)
        s.arrow([(M + MW - 14, y + bh / 2), (R, y + bh / 2)])
        s.text(R + 8, y + bh / 2 + 4, outcome, cls="mono muted", size=10.5)
        if title[0] != "4":
            s.down(M + MW / 2, y + bh, y + bh + gap, label="else", label_dx=10, label_dy=4)
        y += bh + gap
    s.down(M + MW / 2, 624, 654, label="each outcome", label_dx=10, label_dy=4)
    s.box(M, 654, MW, 80, "audit log", "every answer in order: allow, bypass, refuse, y, n; with the tool and the command",
          size=14, subsize=12)

    s.footnote(764, "the operator's reply goes to the audit log, never to the model.")
    return s
