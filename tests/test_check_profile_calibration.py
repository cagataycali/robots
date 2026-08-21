"""The pure half of scripts/check_profile_calibration.py — no hardware, no filesystem.

Guards the sentence, not just the boolean: the whole value of this check is that it names the
teleoperator-vs-robot directory mistake instead of repeating lerobot's "has no calibration registered",
which sends a person to the arm when the fix is a field in profiles.json.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_profile_calibration", Path(__file__).resolve().parents[1] / "scripts" / "check_profile_calibration.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
verdicts = mod.calibration_verdicts

FOLLOWER = {"mode": "real", "robot_name": "so101", "robot_id": "follower", "peer_id": "so101-follower"}
LEADER = {"mode": "real", "robot_name": "so101", "robot_id": "leader", "peer_id": "so101-leader"}
ON_DISK = {
    "robots/so101_follower/follower.json",
    "robots/so101_follower/follower_arm.json",
    "robots/so101_follower/leader_arm.json",
    "teleoperators/so101_leader/leader.json",
}


def test_the_real_machine_state_is_reproduced_without_hardware():
    """cagatay's actual profiles.json + calibration tree: follower fine, leader unspawnable."""
    got = verdicts({"A": FOLLOWER, "B": LEADER}, ON_DISK)
    assert [v["ok"] for v in got] == [True, False]
    assert got[0]["path"] == "robots/so101_follower/follower.json"


def test_the_teleoperator_case_is_named_not_merely_failed():
    (leader,) = [v for v in verdicts({"B": LEADER}, ON_DISK) if not v["ok"]]
    assert leader["reason"] == "calibrated_as_teleoperator"
    # The two facts a person needs: the file lerobot wants, and the file that exists instead.
    assert "robots/so101_follower/leader.json" in leader["detail"]
    assert "teleoperators/so101_leader/leader.json" in leader["detail"]
    # ...and the ids that WOULD work, so the remedy is a choice rather than a search.
    assert "follower" in leader["detail"]


def test_an_arm_nobody_calibrated_is_a_different_sentence():
    (v,) = verdicts({"B": LEADER}, {"robots/so101_follower/follower.json"})
    assert v["reason"] == "missing"
    assert "teleoperator" not in v["detail"].lower(), "must not promise a file that does not exist"


def test_a_sim_twin_is_not_asked_for_calibration():
    assert verdicts({"C": {"mode": "sim", "robot_name": "so101", "robot_id": "twin"}}, ON_DISK) == []


def test_an_unknown_family_is_reported_never_guessed():
    (v,) = verdicts({"D": {"mode": "real", "robot_name": "franka", "robot_id": "x"}}, ON_DISK)
    assert v["ok"] is False and v["reason"] == "unknown_family"


def test_a_calibrated_arm_passes_when_the_id_matches_the_robot_side():
    (v,) = verdicts({"A": FOLLOWER}, ON_DISK)
    assert v["ok"] and "robots/so101_follower/follower.json" == v["path"]
