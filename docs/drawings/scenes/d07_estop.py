"""D7: what an e-stop does, and the lockout it leaves behind.

Sources: strands_robots/mesh/core.py emergency_stop (local lockout, local stop through _dispatch,
broadcast stop collecting 3 s of replies, publish strands/safety/estop, audit), the lockout (only
status, resume and stop admitted), resume (STRANDS_MESH_OVERRIDE_CODE of 16 characters or more,
constant-time compare, throttle, HMAC proof on strands/safety/resume), docs/learn/mesh/safety-and-estop.md.
"""
from scene import Scene

L, LW = 60, 560
R, RW = 700, 440


def scene() -> Scene:
    s = Scene(
        "d07_estop",
        "An e-stop, in five steps, and the lockout it leaves",
        "emergency_stop runs the same five steps on every peer; afterwards a peer answers status, stop and resume, and nothing else.",
        "Left, five steps in order: engage the local lockout, recording the time; stop this robot through the "
        "same dispatch a peer would use; broadcast stop and collect replies for three seconds, a peer that "
        "cannot stop is listed, not counted; publish strands/safety/estop, the one green element, so every "
        "peer that hears it locks itself; write the audit row. Right, under lockout a peer still answers "
        "status, stop and resume; everything else is refused and the refusal audited; stopping is never "
        "gated. Resume needs STRANDS_MESH_OVERRIDE_CODE of sixteen characters or more on the resuming peer and "
        "every peer that honours it, compared in constant time, throttled after five failures; on success "
        "strands/safety/resume carries an HMAC proof over peer_id, t, lockout_elapsed_s, proof_nonce and the TLS session id, "
        "never the code. Footnote: a reply counts as stopped only if it says so.",
        h=690,
    )
    s.section(L, 122, "what emergency_stop does, in order")
    steps = [
        ("1  engage the local lockout", "records the time; from here on only status, stop and resume are admitted", False),
        ("2  stop this robot", 'the same _dispatch({"action": "stop"}) a peer would run against it', False),
        ("3  broadcast stop", "collect replies for 3 s; a peer that cannot stop is listed, not counted", False),
        ("4  publish strands/safety/estop", "every peer that hears it engages its own lockout, on every site", True),
        ("5  write the audit row", "emergency_stop, the issuer, every reply", False),
    ]
    y, bh, gap = 134, 70, 22
    for title, sub, accent in steps:
        s.box(L, y, LW, bh, title, sub, accent=accent, size=14, subsize=12)
        if not title.startswith("5"):
            s.down(L + 60, y + bh, y + bh + gap)
        y += bh + gap

    s.section(R, 122, "under lockout, a peer still answers")
    s.chips(R, 134, ["status", "stop", "resume"])
    s.para(R, 182, "everything else is refused, and the refusal is audited. stopping is never gated: it only "
           "de-energises, so a second e-stop always lands.", RW, size=12, cls="grot muted")

    s.section(R, 250, "resume")
    s.box(R, 262, RW, 130, "the override code",
          "STRANDS_MESH_OVERRIDE_CODE, 16 characters or more, on the resuming peer and every peer that "
          "honours it; compared in constant time, throttled after 5 failures for 30 s, refused when stale or future",
          size=14, subsize=12)
    s.chips(R + 14, 358, ["STRANDS_MESH_OVERRIDE_CODE"])
    s.down(R + 60, 392, 416)
    s.box(R, 416, RW, 112, "strands/safety/resume",
          "carries an HMAC proof over peer_id, t, lockout_elapsed_s, proof_nonce and the TLS session id; the code itself never travels",
          size=14, subsize=12)
    s.chips(R + 14, 494, ["override_proof", "proof_nonce", "TLS session id"])
    s.para(R, 556, "a peer with no code stays locked until it is restarted, and says so when it starts.",
           RW, size=12, cls="grot muted")

    s.footnote(662, "a reply counts as stopped only if it says so; peers_not_stopped is the list an operator reads first.")
    return s
