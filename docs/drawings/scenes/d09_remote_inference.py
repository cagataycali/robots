"""D9: remote inference. The robot host keeps the control loop and the gate; the GPU host runs the model.

Sources: strands_robots/inference/protocol.py (ready, set_state_keys, set_control_frequency, reset,
get_actions, actions), inference/server.py PolicyServer, inference/client.py RemotePolicy (provider
"remote", create_policy("ws://host:8765")), docs/learn/policies/remote.md.
"""
from scene import Scene

L, LW = 60, 440
R, RW = 780, 360
MID = (L + LW + R) / 2


def scene() -> Scene:
    s = Scene(
        "d09_remote_inference",
        "Remote inference: the loop stays home, the model does not",
        "RemotePolicy is a Policy whose get_actions crosses a WebSocket; the robot host keeps the control loop, the gate and the audit.",
        "Two dashed hosts. Left, the robot host, a laptop or a Jetson: Robot(\"so101\") or Robot(\"so101\", "
        "mode=\"real\"), which owns the control loop and send_action; the operator gate, the one green "
        "element, stays on this host; the control loop consumes the chunk at the control frequency; "
        "RemotePolicy, create_policy(\"ws://gpu:8765\"), provider remote, a proxy with the Policy contract. "
        "Right, the GPU host: PolicyServer wraps any Policy with the [inference] extra and calls its "
        "get_actions; the Policy is loaded once. Four wires between them, in order: ready with the policy's "
        "metadata from the server; set_state_keys, set_control_frequency and reset from the client; "
        "get_actions with the encoded observation, the instruction and the delay each control step; and "
        "actions back, a JSON list of action dicts, drawn dashed. Footnote: what crosses the wire is an "
        "observation and a chunk of actions; the robot, the gate and the audit never do.",
        h=704,
    )
    # ---------------------------------------------------------------- robot host
    s.box(L, 134, LW, 490, None, None, dashed=True)
    s.text(L + 14, 156, "ROBOT HOST: A LAPTOP, A JETSON", cls="mono muted", size=10.5, spacing="0.05em")
    s.box(L + 20, 170, LW - 40, 84, 'Robot("so101") or Robot("so101", mode="real")',
          "owns the control loop and send_action; the same object with or without a GPU nearby", size=13.5, subsize=12)
    s.down(L + 120, 254, 282)
    s.down(L + 320, 254, 282)
    s.box(L + 20, 282, 190, 92, "operator gate",
          "stays on this host: a real arm waits for a yes before the loop starts", accent=True, size=14, subsize=11.5)
    s.box(L + 230, 282, 190, 92, "control loop",
          "send_action at the control frequency, consuming the chunk", size=14, subsize=11.5)
    s.down(L + 320, 374, 400)
    s.box(L + 20, 400, LW - 40, 150, "RemotePolicy",
          "a Policy like any other; only get_actions crosses the wire, nothing else changes for the caller",
          size=14, subsize=12)
    s.chips(L + 34, 476, ['create_policy("ws://gpu:8765")'])
    s.chips(L + 34, 506, ['provider "remote"', "[inference]"])
    s.para(L + 20, 580, "the audit log, the lockout and the e-stop are all on this side of the wire.", LW - 40,
           size=12, cls="grot muted")

    # ---------------------------------------------------------------- GPU host
    s.box(R, 134, RW, 490, None, None, dashed=True)
    s.text(R + 14, 156, "GPU HOST", cls="mono muted", size=10.5, spacing="0.05em")
    s.box(R + 20, 170, RW - 40, 84, "PolicyServer",
          "wraps any Policy behind a WebSocket; one client at a time, the [inference] extra", size=14, subsize=12)
    s.down(R + RW / 2, 254, 400, label="get_actions, in process", label_dx=10, label_dy=4)
    s.box(R + 20, 400, RW - 40, 150, "the Policy",
          "loaded once from its checkpoint; any provider create_policy knows",
          size=14, subsize=12)
    s.chips(R + 34, 476, ["lerobot_local", "wbc"])
    s.chips(R + 34, 506, ["create_policy(...)"])

    # ---------------------------------------------------------------- the wire
    s.text(MID, 386, "THE WEBSOCKET, IN ORDER", cls="mono muted", size=10.5, anchor="middle", spacing="0.05em")
    wires = [
        (412, "1  ready: the policy's metadata", True, False),
        (448, "2  state keys, control frequency, reset", False, False),
        (484, "3  get_actions: observation, instruction", False, False),
        (520, "4  actions: a JSON list of action dicts", True, True),
    ]
    for y, label, from_server, dashed in wires:
        pts = [(R, y), (L + LW, y)] if from_server else [(L + LW, y), (R, y)]
        s.arrow(pts, dashed=dashed)
        s.text(MID, y - 7, label, cls="mono muted", size=10.5, anchor="middle")
    s.text(MID, 544, "every control step: 3 and 4", cls="grot muted", size=11.5, anchor="middle")

    s.footnote(676, "what crosses the wire is an observation and a chunk of actions; the robot, the gate and the audit never do.")
    return s
