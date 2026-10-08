"""D17: the layers a command crosses before a motor turns, in order, and the row it leaves behind.

Left to right, visited in order: the operator gate (gate_motion), the motion grant spent once, command
validation against the allowlists, the wire (mTLS and the ACL, with the transport caps), the safety
envelope, the driver gate and bus access on the robot. Reads take none of these. Under the row the one
accent element: the audit log every layer writes into. Every name is on learn/security.md.
"""
from scene import Scene

X, W = 60, 1080
STEPS = [
    ("operator gate", "may this command move a robot at all", "gate_motion"),
    ("motion grant", "a browser yes is spent once, for that exact input", "_motion_grants"),
    ("validation", "an allowed action, provider, host, checkpoint", "validate_command"),
    ("the wire", "who may publish, on which key, how much", "mTLS, ACL, caps"),
    ("safety envelope", "an e-stop or resume is fresh and unreplayed", "safety-and-estop"),
    ("driver gate", "the hardware is in a state where a write is safe", "bus_access"),
]
N = len(STEPS)
GAP = 12
BW = (W - GAP * (N - 1)) / N
ROW = 256


def scene() -> Scene:
    s = Scene(
        "d17_security_layers",
        "What a command crosses before a motor turns",
        "An agent may read anything; a command that can move a robot passes every layer below, in order, and leaves a signed row behind.",
        "Top: a command that can move a robot, from an agent's tool call, a dashboard button or a mesh peer. A "
        "row of six cards it crosses in order, lit one after another: the operator gate (gate_motion, whether "
        "this command may move a robot at all); the motion grant (a browser yes spent once, for the exact tool "
        "input it approved); command validation (validate_command, an allowed action, provider, host and "
        "checkpoint); the wire (mTLS under STRANDS_MESH_AUTH_MODE, the ACL in STRANDS_MESH_ACL_FILE, the "
        "transport caps before deserialisation); the safety envelope (an e-stop or resume that is fresh, "
        "unreplayed and, for resume, proven); the driver gate and bus access (the hardware is in a state where "
        "a write is safe, one caller owns the bus). Below the row, the one green element: the audit log, where "
        "every layer leaves its row, in order, tamper-evident. Right of it, path validation: a tool writes only "
        "where it was told. Footnote: a read crosses none of this; refusals are audited too.",
        h=614,
    )

    s.section(X, 122, "a command that can move a robot")
    s.box(X, 134, W, 60, "execute, run_policy, send_action, a task, a teleop frame",
          "from an agent's tool call, a dashboard button or a mesh peer; a read takes none of the layers below",
          size=14, subsize=12)
    s.down(X + BW / 2, 194, ROW, id="enter")

    s.section(X + BW / 2 + 16, ROW - 10, "the layers, in order")
    for i, (title, sub, where) in enumerate(STEPS):
        x = X + i * (BW + GAP)
        s.box(x, ROW, BW, 130, title, sub, size=13, subsize=11, id=f"layer{i}")
        s.chip(x + 14, ROW + 96, where, size=10.5)
        s.motion.append((f"layer{i}", "visit"))
        if i < N - 1:
            s.arrow([(x + BW, ROW + 65), (x + BW + GAP, ROW + 65)], head=False)
    s.down(X + W - BW / 2, ROW + 130, ROW + 160, label="the motor turns", label_dx=-10, label_dy=4, label_anchor="end")

    # ---------------------------------------------------------------- the row left behind (the one green element)
    s.box(X, 440, 700, 88, "audit log",
          "what happened, in order, tamper-evident: every layer above writes its row here, refusals included",
          accent=True, size=14, subsize=12, id="audit")
    s.chips(X + 14, 496, ["audit", "_hitl_audit", "emergency_stop, the issuer, every reply"])
    for i in range(N):
        x = X + i * (BW + GAP) + BW / 2
        if x < X + 700:
            s.arrow([(x, ROW + 130), (x, 440)], dashed=True, head=False, id=f"row{i}")
            s.motion.append((f"row{i}", "flow"))
    s.motion.insert(0, ("audit", "pulse"))
    s.box(X + 720, 440, 360, 88, "path validation",
          "a tool writes only where it was told", size=14, subsize=12)
    s.chips(X + 734, 496, ["_path_validation"])
    s.motion.append(("enter", "flow"))

    s.footnote(580, "a read crosses none of this; a refusal is audited like a yes.")
    return s
