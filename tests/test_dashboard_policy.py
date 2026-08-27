"""Policy-fit surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_policy_fit_norm_tag.py, test_dashboard_policy_fit_route.py, test_dashboard_policy_fit.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv
from strands_robots.dashboard.checkpoints import _declared_norm_tags
from strands_robots.dashboard.policy_fit import (
    action_dim,
    camera_keys,
    policy_fit,
    state_dim,
)

# ============================================================================
# from tests/test_dashboard_policy_fit_norm_tag.py
# An undeclared norm_tag is refused while the form is open, not after the arm is torqued.
# ============================================================================

FEATS = {"observation.state": {"type": "STATE", "shape": [6]}}
OUT = {"action": {"type": "ACTION", "shape": [6]}}
ARM = {"joints": [f"j{i}" for i in range(6)], "cameras": []}


def _fit(**kw):
    return policy_fit(input_features=FEATS, output_features=OUT, joints=ARM["joints"], **kw)


def test_an_undeclared_tag_blocks_and_names_what_is_declared() -> None:
    v = _fit(norm_tag="mean_std", declared_norm_tags=["min_max", "q99"])
    assert v["ok"] is False and v["blocking"] is True
    p = next(x for x in v["problems"] if x["kind"] == "norm_tag")
    assert "min_max, q99" in p["detail"], "the operator needs the choices, not just a refusal"
    assert "wrong statistics" in p["detail"] and "real arm" in p["detail"], "name the consequence"


def test_a_declared_tag_is_recorded_as_CHECKED_not_silent() -> None:
    v = _fit(norm_tag="min_max", declared_norm_tags=["min_max", "q99"])
    assert [x for x in v["problems"] if x["kind"] == "norm_tag"] == []
    assert "norm_tag" in v["checked"], "a quiet answer must read as verified, not as unexamined"


def test_no_declared_tags_is_no_evidence_never_a_refusal() -> None:
    # An older checkpoint ships no norm_stats.json; treating that silence as a mismatch would block
    # runs that have always worked.
    v = _fit(norm_tag="mean_std", declared_norm_tags=[])
    assert v["ok"] is True and [x for x in v["problems"] if x["kind"] == "norm_tag"] == []
    assert "norm_tag" not in v["checked"]


def test_no_tag_requested_is_not_checked() -> None:
    v = _fit(norm_tag=None, declared_norm_tags=["min_max"])
    assert "norm_tag" not in v["checked"] and v["ok"] is True


def test_declared_tags_are_read_from_the_stats_file(tmp_path) -> None:
    (tmp_path / "norm_stats.json").write_text(json.dumps({"q99": {}, "min_max": {}}))
    assert _declared_norm_tags(tmp_path) == ["min_max", "q99"]


def test_an_unreadable_or_absent_stats_file_yields_no_evidence(tmp_path) -> None:
    assert _declared_norm_tags(tmp_path) == []
    (tmp_path / "norm_stats.json").write_text("{not json")
    assert _declared_norm_tags(tmp_path) == []
    (tmp_path / "norm_stats.json").write_text("[1, 2]")
    assert _declared_norm_tags(tmp_path) == [], "a list declares no tags"


def test_every_problem_this_module_can_emit_has_the_keys_THE_SCREEN_RENDERS() -> None:
    """RunForm renders `p.detail` keyed by `p.kind`. A problem shaped any other way blocks the run
    with a BLANK line - refused, unexplained, which is worse than the mismatch it reports. This
    caught exactly that: the norm_tag problem first shipped as {field, text, remedy}."""
    verdicts = [
        _fit(norm_tag="mean_std", declared_norm_tags=["min_max"]),
        # a state dim that disagrees, an action dim that disagrees, a camera the peer lacks
        policy_fit(
            input_features={"observation.state": {"type": "STATE", "shape": [5]}},
            output_features={"action": {"type": "ACTION", "shape": [2]}},
            joints=ARM["joints"],
            cameras=["top"],
        ),
        policy_fit(
            input_features={"observation.images.front": {"type": "VISUAL", "shape": [3, 8, 8]}},
            output_features=OUT,
            joints=ARM["joints"],
            cameras=["top", "wrist"],
        ),
    ]
    seen = [p for v in verdicts for p in v["problems"]]
    assert len(seen) >= 3, "the premise: these inputs really do produce problems to inspect"
    for p in seen:
        assert set(p) == {"kind", "detail"}, f"unrenderable problem shape: {sorted(p)}"
        assert p["kind"].strip() and p["detail"].strip(), "a blank refusal explains nothing"


# ============================================================================
# from tests/test_dashboard_policy_fit_route.py
# Q79, the wiring: the run form can ASK whether a checkpoint fits the robot, before play torques it.
# ============================================================================

SO101 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}
    # An isolated HF cache: this must not depend on what the operator happens to have downloaded.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


class _Bridge:
    """A peer shaped like cagatay's real arm: 6 joints, cameras top + wrist, hardware attached."""

    def __init__(self) -> None:
        self.peers = {
            "so101-arm-2": {
                "presence": {"hw": "so_follower"},
                "state": {"joints": {j: 0.0 for j in SO101}},
                "cameras": {"top": {}, "wrist": {}},
            },
            # A peer that has announced nothing yet -- the timing case.
            "just-joined": {"presence": {}},
        }

    def snapshot(self):
        return {"peers": self.peers}

    def live_peers(self):
        return dict(self.peers)

    def start(self):
        pass

    def stop(self):
        pass


def _cache(tmp_path, repo: str, inp: dict, out: dict) -> None:
    snap = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"type": "act", "input_features": inp, "output_features": out}))


def _client() -> TestClient:
    return TestClient(srv.create_app(bridge=_Bridge()))  # type: ignore[arg-type]


def test_the_mismatch_that_used_to_reach_the_arm(tmp_path):
    # The real shape of cagataydev/scout-act-sim-v0: 5-value state, 2-value action, camera "front".
    _cache(
        tmp_path,
        "org/scout",
        {"observation.images.front": {"shape": [3, 480, 640]}, "observation.state": {"shape": [5]}},
        {"action": {"shape": [2]}},
    )
    r = _client().get("/api/robots/so101-arm-2/policy-fit?repo_id=org/scout")
    assert r.status_code == 200
    v = r.json()
    assert v["blocking"] is True
    assert v["evidence"] is True
    kinds = {p["kind"] for p in v["problems"]}
    assert kinds == {"state_dim", "action_dim", "cameras"}
    # It is a REAL arm, and the sentence has to say so.
    assert "a real arm" in " ".join(p["detail"] for p in v["problems"])
    # And it reports the robot it compared against, so the verdict is auditable.
    assert v["robot"]["joints"] == SO101
    assert v["robot"]["cameras"] == ["top", "wrist"]


def test_a_matching_checkpoint_passes_and_names_what_it_checked(tmp_path):
    _cache(
        tmp_path,
        "org/arm",
        {
            "observation.state": {"shape": [6]},
            "observation.images.top": {"shape": [3, 1, 1]},
            "observation.images.wrist": {"shape": [3, 1, 1]},
        },
        {"action": {"shape": [6]}},
    )
    v = _client().get("/api/robots/so101-arm-2/policy-fit?repo_id=org/arm").json()
    assert v["blocking"] is False
    assert v["evidence"] is True
    assert set(v["checked"]) == {"state", "action", "cameras"}
    assert v["policy_type"] == "act"


def test_no_evidence_is_never_a_refusal(tmp_path):
    c = _client()
    # An unknown checkpoint: nothing to compare.
    v = c.get("/api/robots/so101-arm-2/policy-fit?repo_id=org/never-downloaded").json()
    assert v["blocking"] is False and v["evidence"] is False and v["problems"] == []
    # No checkpoint named yet (the form is still being typed).
    v2 = c.get("/api/robots/so101-arm-2/policy-fit").json()
    assert v2["blocking"] is False and v2["evidence"] is False
    # A peer that has announced nothing yet: a timing fact, not a mismatch.
    _cache(tmp_path, "org/five", {"observation.state": {"shape": [5]}}, {"action": {"shape": [5]}})
    v3 = c.get("/api/robots/just-joined/policy-fit?repo_id=org/five").json()
    assert v3["blocking"] is False and v3["evidence"] is False


def test_an_unknown_peer_is_refused_by_the_usual_gate():
    r = _client().get("/api/robots/not-a-peer/policy-fit?repo_id=org/x")
    assert r.status_code in (404, 409)


def test_validate_also_carries_the_fit_and_flips_ok(tmp_path):
    """The form's existing validate call must not answer a green tick about a policy that cannot
    drive this robot -- the fit overrides `ok`, because that flag is what arms the button."""
    _cache(
        tmp_path,
        "org/scout2",
        {"observation.state": {"shape": [5]}},
        {"action": {"shape": [2]}},
    )
    r = _client().post(
        "/api/policies/validate",
        json={
            "policy_provider": "lerobot_local",
            "policy_config": {"pretrained_name_or_path": "org/scout2"},
            "peer_id": "so101-arm-2",
        },
    )
    assert r.status_code == 200
    v = r.json()
    assert v["fit"]["blocking"] is True
    assert v["ok"] is False
    assert v["stage"] == "fit"
    assert "6 joints" in v["error"]


# ============================================================================
# from tests/test_dashboard_policy_fit.py
# Q79: a checkpoint states what it was trained on; the run form never read it.
# ============================================================================

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
    v = policy_fit(input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=["top", "wrist"])
    assert v["ok"] is True
    assert v["blocking"] is False
    # Quiet must be readable as "verified", not as "never looked".
    assert set(v["checked"]) == {"state", "action", "cameras"}


def test_the_real_mismatch_a_5dof_2dim_policy_on_a_6_joint_arm():
    v = policy_fit(input_features=SCOUT_IN, output_features=SCOUT_OUT, joints=SO101, cameras=["top", "wrist"])
    assert v["ok"] is False
    assert v["blocking"] is True
    kinds = {p["kind"] for p in v["problems"]}
    assert kinds == {"state_dim", "action_dim", "cameras"}
    joined = " ".join(p["detail"] for p in v["problems"])
    # Every sentence has to name the physical consequence: this decision is about metal.
    assert "energised" in joined and "torqued" in joined
    assert "front" in joined and "top, wrist" in joined


def test_a_camera_the_policy_needs_and_the_robot_does_not_have():
    v = policy_fit(input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=["top"])
    assert [p["kind"] for p in v["problems"]] == ["cameras"]
    assert "wrist" in v["problems"][0]["detail"]
    # The failure mode is the quiet one, and it is spelled out.
    assert "blank frame" in v["problems"][0]["detail"]


def test_extra_cameras_on_the_robot_are_not_a_problem():
    v = policy_fit(
        input_features=ARM_IN,
        output_features=ARM_OUT,
        joints=SO101,
        cameras=["top", "wrist", "overhead"],
    )
    assert v["ok"] is True


def test_absence_of_evidence_never_blocks_a_run_that_was_always_allowed():
    # Unreadable/absent features: no verdict at all.
    assert policy_fit(joints=SO101, cameras=["top"]) == {"ok": True, "blocking": False, "problems": [], "checked": []}
    # A peer that has not announced joints yet: dimensions are unknowable, not wrong.
    v = policy_fit(input_features=SCOUT_IN, output_features=SCOUT_OUT, joints=[], cameras=["top"])
    assert [p["kind"] for p in v["problems"]] == ["cameras"]
    assert "state" not in v["checked"] and "action" not in v["checked"]
    # A peer with no cameras announced yet: same reasoning, no camera verdict.
    v2 = policy_fit(input_features=ARM_IN, output_features=ARM_OUT, joints=SO101, cameras=[])
    assert v2["ok"] is True
    assert "cameras" not in v2["checked"]


def test_a_malformed_config_is_ignored_rather_than_guessed():
    for bad in (
        {"observation.state": {"shape": "six"}},
        {"observation.state": {}},
        {"observation.state": None},
        {"observation.state": {"shape": []}},
    ):
        assert state_dim(bad) is None, bad
    assert (
        policy_fit(input_features={"observation.images.": {"shape": [3, 1, 1]}}, joints=SO101, cameras=["top"])["ok"]
        is True
    )


def test_sim_runs_get_the_same_verdict_in_the_right_words():
    v = policy_fit(
        input_features=SCOUT_IN,
        output_features=SCOUT_OUT,
        joints=SO101,
        cameras=["top"],
        physical=False,
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
    (snap / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": ARM_IN,
                "output_features": ARM_OUT,
            }
        )
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    got = checkpoints.declared_features("org/pol")
    assert got["policy_type"] == "act"
    assert got["input_features"] == ARM_IN
    # And the pairing verdict is then computable with no network and no model load.
    v = policy_fit(
        input_features=got["input_features"],
        output_features=got["output_features"],
        joints=SO101,
        cameras=["top", "wrist"],
    )
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
    (art / "train_config.json").write_text(
        json.dumps({"policy": {"type": "act", "input_features": ARM_IN, "output_features": ARM_OUT}})
    )
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
