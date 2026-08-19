"""U15: a code-defined robot is a first-class peer.

The user writes ``Robot("so101", mode="real", ..., mesh=True)`` in their own
script on their own box. PLAN.md's contract: it must appear in the dashboard
exactly as a dashboard-spawned one does -- same card, name, cameras, telemetry
-- and the dashboard must not treat its own spawns as special beyond an origin
badge.

These tests pin BOTH halves:

* the badge exists and is right (so the three managed-only capabilities can be
  explained before they are clicked rather than 404 after);
* nothing else differs. The symmetry test is the acceptance test: it compares a
  managed and an external peer field by field and fails if any difference other
  than ``origin`` appears -- including one added later by someone who never read
  U15.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from strands_robots.dashboard import mesh_bridge as mb


# --- the pure labelling ----------------------------------------------------


def test_a_peer_we_spawned_is_managed_and_a_stranger_is_external():
    out = mb.peer_origins(["ours", "theirs"], managed_ids=["ours"])
    assert out == {"ours": "managed", "theirs": "external"}


def test_every_peer_gets_a_label():
    """An absent label reads as "unknown", which is a third state the UI would
    have to invent copy for. There are only two answers and we always know
    which: we either started the process or we did not."""
    out = mb.peer_origins(["a", "b", "c"], managed_ids=[])
    assert set(out) == {"a", "b", "c"}
    assert set(out.values()) == {"external"}


def test_a_child_sim_peer_inherits_its_parents_origin():
    """"<parent>__<robot>" lives INSIDE the parent's process, so if we started
    the parent we started the child - the same rule prune_peers and
    peer_is_known already use for these ids."""
    out = mb.peer_origins(
        ["sim-a", "sim-a__so101", "wild__so101"], managed_ids=["sim-a"],
    )
    assert out["sim-a"] == "managed"
    assert out["sim-a__so101"] == "managed"
    # Not ours, and its name saying "__" does not make it ours.
    assert out["wild__so101"] == "external"


def test_a_half_formed_child_id_is_not_adopted():
    """Matching peer_is_known exactly: "sim-a__" has no child half, so it must
    not borrow the parent's protection or its origin."""
    assert mb.peer_origins(["sim-a__"], managed_ids=["sim-a"]) == {"sim-a__": "external"}
    assert mb.peer_origins(["__so101"], managed_ids=["sim-a"]) == {"__so101": "external"}


def test_a_mapping_of_peers_is_accepted_like_the_snapshot_passes_it():
    """snapshot() hands its peer dict straight in; iterating keys must be enough."""
    peers = {"ours": {"peer_id": "ours"}, "theirs": {"peer_id": "theirs"}}
    assert mb.peer_origins(peers, managed_ids={"ours"}) == {
        "ours": "managed", "theirs": "external",
    }


# --- the rail the UI actually renders from ---------------------------------


def _bridge(peers: dict, managed: set[str]) -> mb.MeshBridge:
    """A MeshBridge with no mesh and no session.

    Deliberately __new__ + hand-filled attributes: constructing a real Mesh in a
    test is the Q30 class of accident (a drill that reached the live fleet), and
    snapshot() needs nothing but the peer table, the locks and the two hooks.
    """
    b = mb.MeshBridge.__new__(mb.MeshBridge)
    b.peer_id = "dash"
    b.peers = peers
    b._peers_lock = threading.RLock()
    b._coalesce_lock = threading.RLock()
    b._coalescer = SimpleNamespace(forget=lambda pid: None)
    b.protected_peer_ids = lambda: managed
    b.peer_annotations = None
    b.mesh_info = lambda: {}
    return b


def _live(peer_id: str) -> dict:
    """A peer entry as the mesh reports one, with a fresh heartbeat."""
    import time
    return {
        "peer_id": peer_id,
        "last_seen": time.time(),
        "state": {"joints": {"shoulder_pan.pos": 1.0}},
        "cameras": ["top"],
    }


def test_the_snapshot_labels_the_origin_of_every_peer():
    bridge = _bridge({"ours": _live("ours"), "theirs": _live("theirs")}, {"ours"})
    peers = mb.MeshBridge.snapshot(bridge)["peers"]
    assert peers["ours"]["origin"] == "managed"
    assert peers["theirs"]["origin"] == "external"


def test_a_code_defined_peer_is_identical_to_a_spawned_one_except_its_origin():
    """THE U15 ACCEPTANCE TEST.

    Two peers reporting the same thing must reach the UI as the same card. The
    dashboard renders from this snapshot, so field-level sameness here IS card
    sameness: no telemetry dropped, no name rewritten, no capability flag that
    would let a component quietly render an external peer as second class.
    """
    ours, theirs = _live("ours"), _live("theirs")
    # Same reported content, different id and different origin - nothing else.
    theirs["state"] = dict(ours["state"])
    theirs["last_seen"] = ours["last_seen"]
    bridge = _bridge({"ours": ours, "theirs": theirs}, {"ours"})

    peers = mb.MeshBridge.snapshot(bridge)["peers"]
    a, b = dict(peers["ours"]), dict(peers["theirs"])
    assert a.pop("origin") == "managed"
    assert b.pop("origin") == "external"
    a.pop("peer_id"), b.pop("peer_id")
    assert a == b, "a code-defined peer must reach the UI as the same card"
    # And the telemetry a card draws really did survive on the external one.
    assert b["state"]["joints"] == {"shoulder_pan.pos": 1.0}
    assert b["cameras"] == ["top"]
    assert b["stale"] is False


def test_the_role_annotation_still_rides_along_next_to_the_origin():
    """Origin is applied first; it must not shadow the measured-role fields the
    U2 badge reads (the two enrichments are independent facts)."""
    bridge = _bridge({"ours": _live("ours")}, {"ours"})
    bridge.peer_annotations = lambda: {"ours": {"role": "follower", "role_volts": 12.6}}
    peer = mb.MeshBridge.snapshot(bridge)["peers"]["ours"]
    assert peer["origin"] == "managed"
    assert peer["role"] == "follower" and peer["role_volts"] == 12.6


def test_an_external_peer_can_still_be_commanded():
    """The badge is cosmetic by design: it must not become a permission. Q7's
    guard decides addressability, and a peer present in the mesh is known
    whether or not we started it."""
    peers = {"theirs": _live("theirs")}
    assert mb.peer_is_known("theirs", peers, managed_ids=())
