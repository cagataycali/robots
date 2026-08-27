"""Q7: commanding a peer that never existed cost the full RPC timeout.

``POST /api/robots/ghost_peer_zz/stop`` waited 10 seconds and then answered
**200** with ``state: "no_answer"`` -- the same word a real robot that stopped
answering produces. So a typo and a wedged arm were indistinguishable, on the stop
path of all places, and any UI waiting on it looked wedged itself.

The guard must not overreach in the other direction: a peer inside its spawn
window has a pid but no mesh presence yet, and a stale peer is still worth
shouting at.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.mesh_bridge import peer_is_known, route_task_target


def test_a_peer_in_the_mesh_table_is_known():
    assert peer_is_known("so101-arm-1", {"so101-arm-1": {}, "sim-a": {}})


def test_a_peer_that_was_never_in_the_fleet_is_not_known():
    assert not peer_is_known("ghost_peer_zz", {"so101-arm-1": {}})


def test_a_locally_managed_peer_is_known_before_it_reaches_the_mesh():
    """A spawn returns a pid immediately; the UI drives it moments later."""
    assert peer_is_known("just-spawned", {}, {"just-spawned"})


def test_a_child_sim_peer_is_known_through_its_parent():
    assert peer_is_known("sim-a__so101", {"sim-a": {}})
    assert peer_is_known("spawning__so101", {}, {"spawning"})


def test_a_child_of_nothing_is_not_known():
    assert not peer_is_known("ghost__so101", {"so101-arm-1": {}})


@pytest.mark.parametrize("peer_id", ["", "arm-1__", "__so101", "__"])
def test_the_predicate_agrees_with_the_router_on_malformed_child_ids(peer_id: str):
    """If route_task_target would not reroute it, calling it known hands it
    straight back to the timeout this guard exists to avoid."""
    peers = {"arm-1": {}}  # type: ignore[var-annotated]
    known = peer_is_known(peer_id, peers)
    target, _cmd = route_task_target(peer_id, {"action": "stop"})
    rerouted = target != peer_id
    assert known is False
    assert rerouted is False  # neither of us treats these as a child


def test_a_well_formed_child_id_is_both_known_and_rerouted():
    peers = {"arm-1": {}}
    assert peer_is_known("arm-1__so101", peers)
    target, cmd = route_task_target("arm-1__so101", {"action": "start"})
    assert target == "arm-1" and cmd["robot_name"] == "so101"


def test_staleness_is_not_part_of_being_known():
    """'The arm went quiet, try stopping it anyway' must stay possible."""
    assert peer_is_known("so101-arm-1", {"so101-arm-1": {"stale": True}})


def test_a_plain_iterable_of_ids_works_as_well_as_the_peer_mapping():
    assert peer_is_known("arm-1", ["arm-1", "arm-2"])
    assert not peer_is_known("arm-3", ["arm-1", "arm-2"])


def test_an_empty_fleet_knows_nobody():
    assert not peer_is_known("arm-1", {})
    assert not peer_is_known("arm-1", {}, ())
