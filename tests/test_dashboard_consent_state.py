from strands_robots.dashboard import consent

def test_all_three_kinds_are_reported():
    s = consent.granted_state({})
    assert set(s["kinds"]) == {"trust_remote_code", "hf_repo_allow", "teleop_degree_units"}
    assert s["trust_remote_code"] is False and s["hf_repo_allow"] == []
    # the bug: this key did not exist, so the permissions screen could not show or revoke it
    assert s["teleop_degree_units"]["granted"] is False

def test_degree_preset_is_recognised_as_such():
    granted = consent.env_patch(consent.build_request("teleop_degree_units", "so101-arm-2"), {})
    s = consent.granted_state(granted)
    t = s["teleop_degree_units"]
    assert t["granted"] is True and t["is_degree_preset"] is True
    assert t["value_abs"] == granted["STRANDS_MESH_INPUT_VALUE_ABS"]

def test_a_hand_tuned_bound_is_granted_but_not_called_the_preset():
    s = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "9999", "STRANDS_MESH_INPUT_SLEW_ABS": "9999"})
    t = s["teleop_degree_units"]
    assert t["granted"] is True and t["is_degree_preset"] is False and t["value_abs"] == "9999"

def test_half_a_pair_is_still_in_force():
    # reach widened, speed bound untouched: the widened half already applies, so hiding it lies
    t = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "180"})["teleop_degree_units"]
    assert t["granted"] is True and t["slew_abs"] is None and t["is_degree_preset"] is False

def test_blank_and_whitespace_are_not_grants():
    t = consent.granted_state({"STRANDS_MESH_INPUT_VALUE_ABS": "  ", "STRANDS_MESH_INPUT_SLEW_ABS": ""})["teleop_degree_units"]
    assert t["granted"] is False
    assert consent.granted_state({"STRANDS_TRUST_REMOTE_CODE": " "})["trust_remote_code"] is False
    assert consent.granted_state({"STRANDS_MESH_HF_REPO_ALLOW": " a , ,b "})["hf_repo_allow"] == ["a", "b"]
