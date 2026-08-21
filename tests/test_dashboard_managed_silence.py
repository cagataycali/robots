"""Q155b: a child WE spawned that the fleet has never heard of.

silent_arms says "present but mute". This says "ours, alive, and absent" - the
state that renders as nothing at all, which is why it went unnoticed on the real
rig for a day.
"""
from strands_robots.dashboard.mesh_bridge import managed_without_presence


def test_a_managed_id_with_no_peer_is_reported():
    assert managed_without_presence({}, ["sim-a"]) == ["sim-a"]


def test_a_managed_id_that_is_present_is_not_reported():
    assert managed_without_presence({"sim-a": {}}, ["sim-a"]) == []


def test_a_parent_answers_through_its_child():
    # The sim PROCESS publishes nothing; its robot child publishes. Reporting the
    # parent would be a false alarm on the healthiest sim there is.
    assert managed_without_presence({"sim-a__so101": {}}, ["sim-a"]) == []


def test_a_child_answers_through_its_parent():
    assert managed_without_presence({"sim-a": {}}, ["sim-a__so101"]) == []


def test_a_half_formed_child_id_is_still_reported():
    # "sim-a__" is not a family member (peer_is_known refuses it too), so the
    # parent's presence cannot vouch for it.
    assert managed_without_presence({"sim-a": {}}, ["sim-a__"]) == ["sim-a__"]


def test_a_freshly_spawned_child_is_not_yet_a_fault():
    got = managed_without_presence(
        {}, ["sim-a"], spawn_times={"sim-a": 100.0}, now=105.0, grace_s=20.0
    )
    assert got == []


def test_the_same_child_after_the_grace_window_is_a_fault():
    got = managed_without_presence(
        {}, ["sim-a"], spawn_times={"sim-a": 100.0}, now=140.0, grace_s=20.0
    )
    assert got == ["sim-a"]


def test_an_unknown_spawn_time_is_not_treated_as_youth():
    # A missing timestamp is not evidence of a recent spawn; staying quiet there
    # would hide exactly the 25h-old case that prompted this rule.
    assert managed_without_presence({}, ["sim-a"], spawn_times={}, now=1e9) == ["sim-a"]


def test_empty_and_blank_ids_are_ignored():
    assert managed_without_presence({}, ["", None]) == []  # type: ignore[list-item]


def test_output_is_sorted():
    assert managed_without_presence({}, ["b", "a"]) == ["a", "b"]
