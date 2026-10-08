"""D15: the dashboard is one more peer. It holds no robot; everything it shows or commands is on the mesh.

Top: the browser and its two doors (a passkey session cookie, a static token) into the dashboard
process. Middle, the one accent element: the dashboard, a robot-less gateway on the Zenoh mesh, with
its tabs as chips. Bottom: the peers it watches and spawns, hardware and simulated, on this machine
and on others; and the two e-stops it fires. Every name is on learn/dashboard.md.
"""
from scene import Scene

X, W = 60, 1080
PEERS = [
    ('Robot("so101", mode="real")', "a USB arm, a mesh peer with joints and cameras", "this machine"),
    ("so101_sim", "spawned from Devices as a child process", "this machine"),
    ('Robot("unitree_g1", mode="real")', "another host, ZENOH_CONNECT names this one", "another host"),
]
N = len(PEERS)
GAP = 16
BW = (W - GAP * (N - 1)) / N


def scene() -> Scene:
    s = Scene(
        "d15_dashboard_gateway",
        "One page over the mesh",
        "The dashboard holds no robot of its own: every card is a mesh peer, every button a message that peer answers or refuses.",
        "Top: a browser reaches strands-robots dashboard on 127.0.0.1:8090 through one of three doors, first "
        "match wins: a passkey session cookie, the static security.auth_token, or the bootstrap token from this "
        "machine's own browser while no passkey is enrolled. Middle, the one green element: the dashboard process, "
        "a robot-less gateway that joins the Zenoh mesh under the same posture as its peers; its tabs are Fleet, "
        "Devices, Calibrate, Agent and Settings. Bottom, three peer cards the Fleet shows: a real so101 on this "
        "machine, a simulated so101 (so101_sim) spawned from Devices as a child process, and a Unitree G1 on another host that "
        "named this one in ZENOH_CONNECT. The wires between the dashboard and the peers carry presence, joints and "
        "camera frames up, and tasks, teleop and stop down. Two e-stops leave the page: POST /api/safety/estop "
        "stops this process's simulations, POST /api/mesh/safety/estop is the signed fleet stop. Footnote: anything "
        "that moves a real robot pauses on a consent card first.",
        h=660,
    )

    # ---------------------------------------------------------------- the browser
    s.section(X, 122, "who may click")
    s.box(X, 134, W, 68, "the browser, 127.0.0.1:8090",
          "three doors, first match wins: a passkey session cookie, the static security.auth_token, the bootstrap "
          "token from this machine while no passkey is enrolled", size=14, subsize=12)
    s.down(600, 202, 236, label="access.require_session", label_dx=10, label_dy=4, id="session")

    # ---------------------------------------------------------------- the gateway (the one green element)
    s.section(X, 228, "the gateway")
    s.box(X, 236, W, 96, "strands-robots dashboard",
          "joins the Zenoh mesh as a robot-less peer, under the same posture as the others "
          "(STRANDS_MESH_LOCAL_DEV=true on one machine)", accent=True, size=14, subsize=12, id="gateway")
    s.chips(X + 14, 298, ["Fleet", "Devices", "Calibrate", "Agent", "Settings", "/ws/mesh", "/ws/agent"])

    # ---------------------------------------------------------------- the peers
    s.text(X + BW / 2 + 28, 372, "UP: PRESENCE, JOINTS, CAMERA FRAMES", cls="mono muted", size=10.5, spacing="0.05em")
    s.text(X + W - BW / 2 - 28, 372, "DOWN: TASK, TELEOP, STOP", cls="mono muted", size=10.5, spacing="0.05em",
           anchor="end")
    s.section(X, 400, "the peers the fleet shows")
    for i, (title, sub, where) in enumerate(PEERS):
        x = X + i * (BW + GAP)
        s.arrow([(x + BW / 2 - 12, 408), (x + BW / 2 - 12, 332)], id=f"up{i}")
        s.arrow([(x + BW / 2 + 12, 332), (x + BW / 2 + 12, 408)], id=f"down{i}")
        s.box(x, 408, BW, 96, title, sub, size=13.5, subsize=11.5)
        s.chips(x + 14, 470, [where])
        s.motion += [(f"up{i}", "flow"), (f"down{i}", "flow")]
    s.motion.insert(0, ("gateway", "pulse"))

    # ---------------------------------------------------------------- the two e-stops
    s.section(X, 536, "two e-stops, the page fires both")
    s.box(X, 544, 532, 60, "POST /api/safety/estop",
          "stops every simulation in this process and locks; routes answer 423 until resume", size=13.5, subsize=11.5)
    s.box(X + 548, 544, 532, 60, "POST /api/mesh/safety/estop",
          "the signed fleet stop; its answer names the peers that did not reply", size=13.5, subsize=11.5)

    s.footnote(640, "anything that moves a real robot pauses on a consent card; a simulated peer never does.")
    return s
