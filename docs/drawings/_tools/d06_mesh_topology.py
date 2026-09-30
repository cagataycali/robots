"""D6: the mesh. Peers on one Zenoh LAN, a dashboard as a peer, an optional bridge to AWS IoT Core.

Sources: strands_robots/mesh/core.py (Mesh: presence, state, RPC, teleop), mesh/transport (Zenoh,
BridgeTransport), mesh/iot (IoT Core direct), docs/learn/mesh/index.md (three switches, postures,
discovery on tcp/127.0.0.1:7447, ZENOH_CONNECT across hosts), docs/learn/mesh/bridges.md.
"""

from excal import MUTED, Drawing

d = Drawing(
    "d06_mesh_topology",
    "Robots and simulations each own a Mesh peer that broadcasts presence, publishes state and answers RPC "
    "over Zenoh on the LAN; the first process on a host listens on tcp 127.0.0.1:7447 and later ones "
    "connect to it, other hosts join through ZENOH_CONNECT; the dashboard is one more peer; a bridge "
    "peer can relay the same topics to AWS IoT Core for a fleet across sites, and one safety topic, "
    "strands/safety/estop, locks every peer that hears it.",
)

d.region(40, 40, 700, 330, "one LAN, mTLS by default")
h1 = d.region(60, 80, 320, 260, "host A")
a1 = d.box(80, 120, 280, 60, "so101 (real)", kind="code", sub='Robot("so101", mode="real", mesh=True)', size=15, sub_size=12)
a2 = d.box(80, 200, 280, 60, "so101_sim", kind="code", sub='Robot("so101")   presence, state, RPC', size=15, sub_size=12)
a3 = d.box(80, 280, 280, 44, "router tcp/127.0.0.1:7447", kind="chip", sub="first process listens, the rest connect", size=13, sub_size=11)
h2 = d.region(400, 80, 320, 260, "host B")
b1 = d.box(420, 120, 280, 60, "dashboard", kind="code", sub="one more peer: fleet, consent card, e-stop", size=15, sub_size=12)
b2 = d.box(420, 200, 280, 60, "g1 (real)", kind="code", sub='Robot("unitree_g1", mode="real")', size=15, sub_size=12)
b3 = d.box(420, 280, 280, 44, "ZENOH_CONNECT=tcp/hostA:7447", kind="chip", sub="how a second host joins", size=13, sub_size=11)
d.path([(360, 302), (420, 302)], both=True)

est = d.box(220, 400, 340, 50, "strands/safety/estop", kind="accent", sub="one message locks every peer", size=14, sub_size=12)

bridge = d.box(820, 120, 320, 60, "bridge peer", kind="code", sub="STRANDS_MESH_BACKEND=bridge", size=15, sub_size=12)
iot = d.box(820, 240, 320, 60, "AWS IoT Core", sub="another site's LAN, the same topics", size=15, sub_size=12)
d.path([(740, 150), (820, 150)], both=True)
d.arrow(bridge, "b", iot, "t", both=True)
d.text(1000, 200, "mTLS, a per-thing policy", size=13, color=MUTED)
d.text(820, 330, "bridged peers are marked as such; a wire", size=13, color=MUTED)
d.text(820, 350, "claim never becomes a local fact", size=13, color=MUTED)

d.text(580, 415, "published by an e-stop on any peer, heard by all", size=13, color=MUTED)
d.caption(40, 490, "the mesh is enrichment: a session that fails to open leaves the robot working without it.")
d.save()
