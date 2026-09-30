"""D7: what an e-stop does, and the lockout it leaves behind.

Sources: strands_robots/mesh/core.py emergency_stop (local lockout, local stop through _dispatch,
broadcast stop collecting 3 s of replies, publish strands/safety/estop, audit), the lockout (only
status, resume, stop admitted), resume (STRANDS_MESH_OVERRIDE_CODE >= 16 chars, constant-time
compare, throttle, HMAC proof on strands/safety/resume), docs/learn/mesh/safety-and-estop.md.
"""

from excal import MUTED, Drawing

d = Drawing(
    "d07_estop",
    "An e-stop runs five steps in order: engage the local lockout, stop the local robot through the same "
    "dispatch a peer would use, broadcast stop and collect replies for three seconds, publish "
    "strands/safety/estop so every peer locks itself, write the audit row. Under lockout a peer answers "
    "only status, resume and stop. Resume needs the override code of sixteen characters or more, is "
    "throttled after failures, and publishes an HMAC proof, never the code.",
)

steps = [
    ("1  engage local lockout", "records the time"),
    ("2  stop this robot", 'the same _dispatch({"action": "stop"}) a peer would run'),
    ("3  broadcast stop", "collect replies for 3 s; a peer that cannot stop is listed, not counted"),
    ("4  publish strands/safety/estop", "every peer that hears it engages its own lockout"),
    ("5  audit", "emergency_stop, issuer, replies"),
]
y = 40
prev = None
for title, sub in steps:
    kind = "accent" if title.startswith("4") else "plain"
    b = d.box(60, y, 520, 58, title, kind=kind, sub=sub, size=15, sub_size=12)
    if prev:
        d.arrow(prev, "b", b, "t")
    prev = b
    y += 86

d.region(680, 40, 500, 200, "lockout: what a peer still answers")
d.box(700, 80, 130, 48, "status", kind="chip", size=15)
d.box(850, 80, 130, 48, "stop", kind="chip", sub="a second e-stop must land", size=15, sub_size=10)
d.box(1000, 80, 140, 48, "resume", kind="chip", sub="second factor", size=15, sub_size=10)
d.text(700, 150, "everything else is refused, and the refusal is audited", size=13, color=MUTED)
d.text(700, 170, "stopping is never gated: it only de-energises", size=13, color=MUTED)

d.region(680, 280, 500, 190, "resume")
d.text(700, 310, "STRANDS_MESH_OVERRIDE_CODE, 16 characters or more, on the", size=13)
d.text(700, 330, "resuming peer and every peer that honours it; constant-time compare,", size=13)
d.text(700, 350, "throttled after 5 failures for 30 s, refused when stale or future.", size=13)
d.text(700, 380, "on success: strands/safety/resume carries an HMAC proof over peer_id,", size=13)
d.text(700, 400, "t, elapsed, nonce and the TLS session id; the code never travels.", size=13)
d.text(700, 430, "a peer with no code stays locked until restarted, and says so at start.", size=13, color=MUTED)

d.caption(60, 510, "a reply counts as stopped only if it says so; peers_not_stopped is the list an operator reads first.")
d.save()
