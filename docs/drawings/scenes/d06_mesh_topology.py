"""D6: the mesh. Peers on one Zenoh LAN, the dashboard as a peer, an optional bridge to AWS IoT Core.

Sources: strands_robots/mesh/core.py (Mesh: presence, state, RPC, teleop), mesh/transport (Zenoh,
BridgeTransport), mesh/iot, docs/learn/mesh/index.md (discovery on tcp/127.0.0.1:7447, ZENOH_CONNECT
across hosts), docs/learn/mesh/bridges.md.
"""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d06_mesh_topology",
        "One LAN of peers, one topic that stops them all",
        "Every robot and simulation is a Mesh peer on Zenoh; the dashboard is one more; a bridge carries the same topics to another site.",
        "Left, a dashed layer, one LAN with mTLS by default, holding two dashed hosts. Host A: so101 real, "
        "Robot(\"so101\", mode=\"real\", mesh=True); so101_sim, Robot(\"so101\"); the router chip "
        "tcp/127.0.0.1:7447, the first process listens and the rest connect. Host B: the dashboard, one more "
        "peer with the fleet page, the consent card and e-stop; g1 real, Robot(\"unitree_g1\", mode=\"real\"); "
        "the chip ZENOH_CONNECT=tcp/10.0.0.1:7447, how a second host joins. A two-headed wire between the hosts "
        "carries presence, state and RPC. Under the LAN the one green element, the safety topics: strands/safety/estop "
        "locks every peer that hears it, strands/safety/resume carries a proof. Right, across sites: a bridge peer, STRANDS_MESH_BACKEND=bridge, "
        "relays the same topics to AWS IoT Core, mTLS and a policy per thing; a wire claim never becomes a "
        "local fact. Footnote: the mesh is enrichment.",
        h=700,
    )
    # ---------------------------------------------------------------- the LAN
    s.box(60, 134, 700, 344, None, None, dashed=True)
    s.text(74, 156, "ONE LAN, MTLS BY DEFAULT", cls="mono muted", size=10.5, spacing="0.05em")
    for hx, name in ((80, "HOST A"), (420, "HOST B")):
        s.box(hx, 170, 320, 290, None, None, dashed=True)
        s.text(hx + 14, 192, name, cls="mono muted", size=10.5, spacing="0.05em")
    s.box(100, 206, 280, 66, "so101, real", 'Robot("so101", mode="real", mesh=True)', size=13.5, subsize=11.5)
    s.box(100, 284, 280, 66, "so101_sim", 'Robot("so101"): presence, state, RPC', size=13.5, subsize=11.5)
    s.box(100, 362, 280, 80, "router", "the first process listens, the rest connect", size=13.5, subsize=11.5)
    s.chips(114, 410, ["tcp/127.0.0.1:7447"])
    s.box(440, 206, 280, 66, "dashboard", "one more peer: the fleet page, the consent card, e-stop", size=13.5, subsize=11.5)
    s.box(440, 284, 280, 66, "g1, real", 'Robot("unitree_g1", mode="real")', size=13.5, subsize=11.5)
    s.box(440, 362, 280, 80, "a second host joins", "one variable names host A's address", size=13.5, subsize=11.5)
    s.chips(454, 410, ["ZENOH_CONNECT=tcp/10.0.0.1:7447"])
    s.arrow([(380, 317), (440, 317)], head="both")
    s.text(410, 306, "one mesh", cls="mono muted", size=10.5, anchor="middle")

    # ---------------------------------------------------------------- the safety topic (the one green element)
    s.arrow([(410, 478), (410, 508)])
    s.box(60, 508, 700, 78, "the safety topics",
          "an e-stop on any peer publishes estop and every peer that hears it engages its own lockout; resume carries a proof",
          accent=True, size=14, subsize=12)
    s.chips(74, 556, ["strands/safety/estop", "strands/safety/resume"])

    # ---------------------------------------------------------------- across sites
    s.section(820, 122, "across sites")
    s.box(820, 134, 320, 120, "bridge peer",
          "relays the same topics both ways; bridged peers are marked as such, and a wire claim never "
          "becomes a local fact", size=14, subsize=12)
    s.chips(834, 218, ["STRANDS_MESH_BACKEND=bridge"])
    s.arrow([(760, 194), (820, 194)], head="both")
    s.arrow([(980, 254), (980, 300)], head="both")
    s.text(990, 282, "mTLS, a policy per thing", cls="mono muted", size=10.5)
    s.box(820, 300, 320, 120, "AWS IoT Core",
          "another site hears the same topics; each robot there is a Thing with a certificate",
          size=14, subsize=12)
    s.chips(834, 384, ["strands/<peer>/presence"])
    s.box(820, 448, 320, 150, "what travels",
          "presence, state and health fan out; cmd and its response go point to point; the safety topics "
          "reach every peer on every site", size=14, subsize=12)
    s.chips(834, 560, ["strands/<peer>/cmd", "strands/broadcast"])

    s.footnote(658, "the mesh is enrichment: a session that fails to open leaves the robot working without it.")
    return s
