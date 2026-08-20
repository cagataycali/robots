"""#2486: the dashboard can SHOW what an episode was judged to be -- and says when it cannot label.

The trap this pins: `episode_labels.annotate_episode` refuses an episode with no deterministic
verdict ("an annotation layered on nothing would be a verdict in disguise"), and a REAL-ARM
recording never has one. A dashboard that offered a label button would 400 on exactly the datasets
it records, and a dashboard that wrote the judge block itself would poison `filter_episodes` for
training. So the capability is reported, not faked.
"""
import json

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.episode_label_view import label_view
from strands_robots.dashboard.server import create_app

@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """This machine has an enrolled passkey + a live settings token, so an un-isolated route test
    gets 401 from its own dashboard (repo gotcha, BUGS.md). Point auth and settings at empty temp
    stores so the guard stays in open posture."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


DOC = {
    "schema_version": 1,
    "benchmark": "cube_lift",
    "episodes": {
        "0": {"episode_index": 0, "deterministic": {"success": True, "failure": False, "steps": 150},
              "judge": {"quality": "high", "failure_mode": None, "note": "clean", "model": "human",
                        "disputes_verdict": False}},
        "1": {"episode_index": 1, "deterministic": {"success": False, "failure": True, "steps": 42}},
        "2": {"episode_index": 2},
    },
}


def test_a_real_arm_recording_says_why_it_cannot_be_labelled():
    view = label_view(None, total_episodes=12)
    assert view["can_annotate"] is False
    assert "real-arm recording has no predicate verdict" in view["why"]
    assert "not a permission problem" in view["why"], "the operator must not read this as auth"
    assert view["episodes"] == [] and view["total_episodes"] == 12


def test_each_row_says_whether_IT_may_be_annotated():
    view = label_view(DOC, total_episodes=3)
    rows = {r["episode_index"]: r for r in view["episodes"]}
    assert rows[0]["annotatable"] is True and rows[0]["quality"] == "high"
    assert rows[1]["annotatable"] is True and rows[1]["quality"] is None, "verdict, awaiting a grade"
    assert rows[2]["annotatable"] is False, "no deterministic block: annotate_episode would refuse"
    assert rows[0]["verdict"] == "success" and rows[1]["verdict"] == "failure"
    assert view["with_verdict"] == 2 and view["labelled"] == 1
    assert view["can_annotate"] is True and "waiting for a quality grade" in view["why"]


def test_a_sidecar_with_verdicts_nowhere_is_refused_with_the_reason():
    doc = {"schema_version": 1, "benchmark": "b", "episodes": {"0": {"episode_index": 0}}}
    view = label_view(doc)
    assert view["can_annotate"] is False and "refuses to layer a judgement on nothing" in view["why"]


def test_a_corrupt_sidecar_is_not_reported_as_no_labels_yet():
    # Different action entirely: "record verdicts" vs "your labels may be damaged".
    view = label_view(None, sidecar_error="JSONDecodeError: line 3")
    assert view["can_annotate"] is False
    assert "could not be read" in view["why"] and "JSONDecodeError" in view["why"]
    assert "record_deterministic_verdicts" not in view["why"], "do not send them down the wrong path"


def test_a_disputing_judge_is_counted_not_hidden():
    doc = json.loads(json.dumps(DOC))
    doc["episodes"]["1"]["judge"] = {"quality": "medium", "success_opinion": True,
                                     "disputes_verdict": True, "note": "", "model": "vlm"}
    view = label_view(doc)
    assert view["disputed"] == 1
    assert [r["disputes_verdict"] for r in view["episodes"] if r["episode_index"] == 1] == [True]


def test_the_route_reads_a_real_sidecar_and_404s_on_a_missing_root(tmp_path):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3}))
    (root / "episode_labels.json").write_text(json.dumps(DOC))

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/api/datasets/labels", params={"root": str(root)})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["benchmark"] == "cube_lift" and body["labelled"] == 1
        assert body["total_episodes"] == 3 and body["can_annotate"] is True

        missing = client.get("/api/datasets/labels", params={"root": str(tmp_path / "nope")})
        assert missing.status_code == 404


def test_the_route_does_not_offer_a_write(tmp_path):
    # The doctrine is the point: no POST here until the source can hold a HUMAN verdict for a real
    # recording. If someone adds one, this test should be the argument they have to answer.
    app = create_app()
    routes = {(r.path, m) for r in app.routes for m in getattr(r, "methods", set()) or set()}
    assert ("/api/datasets/labels", "GET") in routes
    assert not any(p == "/api/datasets/labels" and m in ("POST", "PUT", "PATCH") for p, m in routes)
