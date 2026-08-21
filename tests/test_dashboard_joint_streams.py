"""Q149: a peer that is present but publishing no joints must be visible in /api/health.

The bug this pins is a REPORTING one, and it cost days of hand work: /api/health said
``peers: 4`` while three of those four arms had published no joints since a spawn 44 hours
earlier. Every number in that payload was true; the impression was false. The only way to
tell was to walk /api/fleet counting ``state.joints`` peer by peer.
"""

from strands_robots.dashboard.mesh_bridge import silent_arms


def _p(joints=(), stale=False):
    return {"state": {"joints": {f"j{i}": 0.0 for i in range(len(joints))}}, "stale": stale}


def test_a_streaming_fleet_says_nothing():
    # The refused_handshakes law: a section that is always present is a section nobody reads.
    assert silent_arms({"arm": _p(joints=range(6))}) is None


def test_a_silent_arm_is_named_not_just_counted():
    got = silent_arms({"arm-a": _p(joints=range(6)), "arm-b": _p()})
    assert got == {"streaming": 1, "silent": ["arm-b"]}


def test_the_host_process_of_a_streaming_child_is_not_a_silent_arm():
    # lib/armHosts.ts's law, server side: a process is not an arm. The simulator parent
    # reports no joints BY DESIGN; its child publishes them.
    got = silent_arms({"sim": _p(), "sim__so101": _p(joints=range(6))})
    assert got is None


def test_a_childless_jointless_peer_IS_a_broken_arm():
    # Evidence above structure: without a child, the same shape is an arm that should be
    # streaming and is not - a different sentence with a different remedy.
    got = silent_arms({"sim": _p()})
    assert got == {"streaming": 0, "silent": ["sim"]}


def test_a_host_process_is_counted_separately_when_something_else_is_silent():
    got = silent_arms({"sim": _p(), "sim__so101": _p(joints=range(6)), "arm": _p()})
    assert got == {"streaming": 1, "silent": ["arm"], "host_processes": 1}


def test_a_stale_peers_silence_is_explained_by_its_staleness():
    # Blaming a peer that is gone for not streaming would send the operator to look at
    # hardware that is not there; staleness is already reported per peer.
    got = silent_arms({"gone": _p(stale=True), "arm": _p()})
    assert got == {"streaming": 0, "silent": ["arm"], "stale": 1}


def test_the_live_shape_from_this_fleet():
    # Measured from /api/fleet on 2026-08-22: two real arms, both present, both mute.
    got = silent_arms({"so101-follower": {"state": {}, "stale": False},
                       "so101-leader": {"state": {"joints": {}}, "stale": False}})
    assert got == {"streaming": 0, "silent": ["so101-follower", "so101-leader"]}
