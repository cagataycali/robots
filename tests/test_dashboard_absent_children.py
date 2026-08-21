"""U22: a dead managed child is reported to the fleet view, and never as a peer.

The live defect: the twin was SIGKILLed, the mesh pruned it (correctly - it really
had left the fleet), and every fleet-side surface then said nothing at all. The
manager still held its exit status the whole time.

These tests pin the four refusals that stop the memorial becoming a ghost peer.
"""
from strands_robots.dashboard.mesh_bridge import absent_children


def _child(peer_id, alive=False, returncode=-9, **extra):
    return {"peer_id": peer_id, "robot_name": "so101", "mode": "sim",
            "alive": alive, "returncode": returncode, "started_at": 1787000000, **extra}


def test_the_defect_a_killed_and_pruned_child_is_reported():
    out = absent_children({"so101-follower": {}}, [_child("twin")])
    assert [c["peer_id"] for c in out] == ["twin"]
    assert out[0]["returncode"] == -9, "the exit status is the whole point - it names the death"
    assert out[0]["mode"] == "sim" and out[0]["robot_name"] == "so101"
    assert "log_tail" not in out[0], "a 20-line ring belongs to the devices route, not a 1Hz fleet poll"


def test_a_living_child_missing_from_the_mesh_is_never_a_death():
    # A child that has just been spawned has not published presence yet. Calling that
    # absent would raise a memorial on every single spawn.
    assert absent_children({}, [_child("starting", alive=True, returncode=None)]) == []


def test_a_dead_child_still_in_the_snapshot_is_not_duplicated():
    # Its card is on the screen and the devices drawer explains it (childDeath.ts).
    assert absent_children({"twin": {"connected": True}}, [_child("twin")]) == []


def test_a_dead_parent_whose_child_peer_still_publishes_is_not_absent():
    # parent__child can only be published BY that process, so the process lives.
    assert absent_children({"twin__so101": {}}, [_child("twin")]) == []
    # ...but an unrelated family must not shield it.
    assert [c["peer_id"] for c in absent_children({"other__so101": {}}, [_child("twin")])] == ["twin"]


def test_malformed_entries_cannot_break_the_fleet_view():
    out = absent_children({}, [None, {}, {"peer_id": ""}, _child("real")])  # type: ignore[list-item]
    assert [c["peer_id"] for c in out] == ["real"]


def test_output_is_ordered_so_a_screen_does_not_reshuffle_every_poll():
    out = absent_children({}, [_child("b"), _child("a"), _child("c")])
    assert [c["peer_id"] for c in out] == ["a", "b", "c"]


def test_no_children_no_claim():
    assert absent_children({"a": {}}, []) == []
    assert absent_children({}, []) == []
