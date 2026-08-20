"""Q79, the wiring: the run form can ASK whether a checkpoint fits the robot, before play torques it.

GET /api/robots/{peer}/policy-fit compares the checkpoint's declared features (read from disk) with
what the peer announces on the mesh. Kept on the server so the rule has exactly one implementation --
the frontend renders the sentence, it does not re-derive it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv

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
    (snap / "config.json").write_text(
        json.dumps({"type": "act", "input_features": inp, "output_features": out})
    )


def _client() -> TestClient:
    return TestClient(srv.create_app(bridge=_Bridge()))


def test_the_mismatch_that_used_to_reach_the_arm(tmp_path):
    # The real shape of cagataydev/scout-act-sim-v0: 5-value state, 2-value action, camera "front".
    _cache(
        tmp_path, "org/scout",
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
        tmp_path, "org/arm",
        {"observation.state": {"shape": [6]}, "observation.images.top": {"shape": [3, 1, 1]},
         "observation.images.wrist": {"shape": [3, 1, 1]}},
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
        tmp_path, "org/scout2",
        {"observation.state": {"shape": [5]}}, {"action": {"shape": [2]}},
    )
    r = _client().post("/api/policies/validate", json={
        "policy_provider": "lerobot_local",
        "policy_config": {"pretrained_name_or_path": "org/scout2"},
        "peer_id": "so101-arm-2",
    })
    assert r.status_code == 200
    v = r.json()
    assert v["fit"]["blocking"] is True
    assert v["ok"] is False
    assert v["stage"] == "fit"
    assert "6 joints" in v["error"]
