"""Q79: a checkpoint states what it was trained on; the run form never read it.

Ground truth these cases are built from -- two real configs in this machine's HF cache:
  cagataydev/scout-act-sim-v0     state [5],  action [2], camera "front"
  ncavallo/act_so100_lerobot2_block state [6], action [6], camera "robot"
cagatay's SO-101 announces 6 joints and cameras top + wrist.
"""

from strands_robots.dashboard.policy_fit import (
    action_dim,
    camera_keys,
    policy_fit,
    state_dim,
)

SO101 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

SCOUT_IN = {
    "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
    "observation.state": {"type": "STATE", "shape": [5]},
}
SCOUT_OUT = {"action": {"type": "ACTION", "shape": [2]}}

ARM_IN = {
    "observation.state": {"type": "STATE", "shape": [6]},
    "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
    "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
}
ARM_OUT = {"action": {"type": "ACTION", "shape": [6]}}


def test_the_readers_speak_the_peers_vocabulary():
    # The peer announces bare camera names, so the comparison must happen there, not on feature keys.
    assert camera_keys(SCOUT_IN) == ["front"]
    assert camera_keys(ARM_IN) == ["top", "wrist"]
    assert state_dim(SCOUT_IN) == 5
    assert action_dim(SCOUT_OUT) == 2
    assert camera_keys(None) == [] and state_dim(None) is None and action_dim({}) is None


def test_a_matching_policy_is_quiet_and_says_what_it_verified():
    v = policy_fit(
        input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=["top", "wrist"]
    )
    assert v["ok"] is True
    assert v["blocking"] is False
    # Quiet must be readable as "verified", not as "never looked".
    assert set(v["checked"]) == {"state", "action", "cameras"}


def test_the_real_mismatch_a_5dof_2dim_policy_on_a_6_joint_arm():
    v = policy_fit(
        input_features=SCOUT_IN, output_features=SCOUT_OUT, joints=SO101, cameras=["top", "wrist"]
    )
    assert v["ok"] is False
    assert v["blocking"] is True
    kinds = {p["kind"] for p in v["problems"]}
    assert kinds == {"state_dim", "action_dim", "cameras"}
    joined = " ".join(p["detail"] for p in v["problems"])
    # Every sentence has to name the physical consequence: this decision is about metal.
    assert "energised" in joined and "torqued" in joined
    assert "front" in joined and "top, wrist" in joined


def test_a_camera_the_policy_needs_and_the_robot_does_not_have():
    v = policy_fit(
        input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=["top"]
    )
    assert [p["kind"] for p in v["problems"]] == ["cameras"]
    assert "wrist" in v["problems"][0]["detail"]
    # The failure mode is the quiet one, and it is spelled out.
    assert "blank frame" in v["problems"][0]["detail"]


def test_extra_cameras_on_the_robot_are_not_a_problem():
    v = policy_fit(
        input_features=ARM_IN, output_features=ARM_OUT, joints=SO101,
        cameras=["top", "wrist", "overhead"],
    )
    assert v["ok"] is True


def test_absence_of_evidence_never_blocks_a_run_that_was_always_allowed():
    # Unreadable/absent features: no verdict at all.
    assert policy_fit(joints=SO101, cameras=["top"]) == {
        "ok": True, "blocking": False, "problems": [], "checked": []
    }
    # A peer that has not announced joints yet: dimensions are unknowable, not wrong.
    v = policy_fit(input_features=SCOUT_IN, output_features=SCOUT_OUT, joints=[], cameras=["top"])
    assert [p["kind"] for p in v["problems"]] == ["cameras"]
    assert "state" not in v["checked"] and "action" not in v["checked"]
    # A peer with no cameras announced yet: same reasoning, no camera verdict.
    v2 = policy_fit(input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=[])
    assert v2["ok"] is True
    assert "cameras" not in v2["checked"]


def test_a_malformed_config_is_ignored_rather_than_guessed():
    for bad in ({"observation.state": {"shape": "six"}}, {"observation.state": {}},
                {"observation.state": None}, {"observation.state": {"shape": []}}):
        assert state_dim(bad) is None, bad
    assert policy_fit(input_features={"observation.images.": {"shape": [3, 1, 1]}},
                      joints=SO101, cameras=["top"])["ok"] is True


def test_sim_runs_get_the_same_verdict_in_the_right_words():
    v = policy_fit(
        input_features=SCOUT_IN, output_features=SCOUT_OUT, joints=SO101,
        cameras=["top"], physical=False,
    )
    assert v["blocking"] is True
    joined = " ".join(p["detail"] for p in v["problems"])
    assert "simulated robot" in joined and "a real arm" not in joined


# --- the read side: declared_features + the route ----------------------------------------------


def test_declared_features_reads_a_cached_snapshot(tmp_path, monkeypatch):
    import json

    from strands_robots.dashboard import checkpoints

    snap = tmp_path / "hub" / "models--org--pol" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({
        "type": "act", "input_features": ARM_IN, "output_features": ARM_OUT,
    }))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    got = checkpoints.declared_features("org/pol")
    assert got["policy_type"] == "act"
    assert got["input_features"] == ARM_IN
    # And the pairing verdict is then computable with no network and no model load.
    v = policy_fit(input_features=got["input_features"], output_features=got["output_features"],
                   joints=SO101, cameras=["top", "wrist"])
    assert v["ok"] is True


def test_declared_features_says_nothing_it_cannot_prove(tmp_path, monkeypatch):
    from strands_robots.dashboard import checkpoints

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert checkpoints.declared_features("org/missing") == {}
    assert checkpoints.declared_features("") == {}
    assert checkpoints.declared_features("   ") == {}
    # Present but unparseable: still {} rather than a guess.
    snap = tmp_path / "hub" / "models--org--broken" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{not json")
    assert checkpoints.declared_features("org/broken") == {}
    # A transformers model has no features block: nothing to say.
    snap2 = tmp_path / "hub" / "models--org--bert" / "snapshots" / "a"
    snap2.mkdir(parents=True)
    (snap2 / "config.json").write_text('{"model_type": "bert"}')
    assert checkpoints.declared_features("org/bert") == {}


def test_declared_features_reads_a_local_training_output(tmp_path):
    import json

    from strands_robots.dashboard import checkpoints

    out = tmp_path / "run"
    (out / "checkpoints" / "000100" / "pretrained_model").mkdir(parents=True)
    art = out / "checkpoints" / "000100" / "pretrained_model"
    # train_config.json nests the policy config under "policy".
    (art / "train_config.json").write_text(json.dumps({
        "policy": {"type": "act", "input_features": ARM_IN, "output_features": ARM_OUT}
    }))
    got = checkpoints.declared_features(str(out))
    assert got["input_features"] == ARM_IN
    assert got["policy_type"] == "act"


def test_an_absolute_output_dir_keeps_its_leading_slash():
    """Caught while writing this: repo-id hygiene (.strip('/')) ate the leading slash off an
    absolute path, so every local training output answered {} - the exact case a just-finished
    training run needs."""
    import json
    import tempfile
    from pathlib import Path

    from strands_robots.dashboard import checkpoints

    t = Path(tempfile.mkdtemp())
    art = t / "run"
    art.mkdir()
    (art / "config.json").write_text(json.dumps({"type": "act", "output_features": ARM_OUT}))
    assert str(art).startswith("/")
    got = checkpoints.declared_features(str(art))
    assert got["output_features"] == ARM_OUT
    assert got["repo_id"] == str(art)
