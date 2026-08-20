"""A recording the dashboard died inside is not silence (Q40)."""

from __future__ import annotations

import json
import os

from strands_robots.dashboard import record_crash


def test_a_closed_session_leaves_no_trace(tmp_path) -> None:
    p = tmp_path / "crumb.json"
    record_crash.write_crumb({"dataset": "local/x", "leader": "a", "follower": "b"}, path=p, now=1000.0)
    assert record_crash.read_crumb(p) is not None
    record_crash.clear_crumb(p)
    assert record_crash.read_crumb(p) is None
    # Clearing twice is not an error: close() runs it in a finally block.
    record_crash.clear_crumb(p)


def test_the_notice_names_the_dataset_the_arms_and_the_age(tmp_path) -> None:
    p = tmp_path / "crumb.json"
    record_crash.write_crumb(
        {"dataset": "local/cubes", "task": "pick the cube", "leader": "so101-arm-1", "follower": "so101-arm-2"},
        path=p, now=1000.0,
    )
    n = record_crash.interrupted_notice(record_crash.read_crumb(p), now=1000.0 + 3 * 60)
    assert n is not None
    assert n["dataset"] == "local/cubes"
    assert "about 3 minutes ago" in n["text"]
    assert "so101-arm-1 and so101-arm-2" in n["text"]
    assert "left despawned" in n["text"], "the arms are still parked - that is the actionable part"
    # It must not claim the dataset is broken, and must not pretend the in-flight episode survived.
    assert "not flushed" in n["text"]
    assert "corrupt" not in n["text"].lower()
    # Both real next actions are NAMED, never performed.
    assert any("delete" in s for s in n["next"])
    assert any("name is taken" in s for s in n["next"])


def test_a_crumb_from_this_very_process_is_not_called_a_restart(tmp_path) -> None:
    # A crumb written by THIS pid with no live worker means the session ended without closing
    # inside a running dashboard - a different fault, and "the dashboard stopped" would be a
    # confident invention.
    crumb = {"dataset": "local/x", "opened_at": 1000.0, "pid": os.getpid()}
    same = record_crash.interrupted_notice(crumb, now=1000.0, same_process=True)
    other = record_crash.interrupted_notice(crumb, now=1000.0, same_process=False)
    assert same is not None and other is not None
    assert "opened and never closed" in same["text"]
    assert "dashboard stopped" in other["text"]


def test_no_evidence_produces_no_notice(tmp_path) -> None:
    assert record_crash.interrupted_notice(None) is None
    assert record_crash.interrupted_notice({}) is None
    assert record_crash.interrupted_notice({"dataset": "  "}) is None
    # A corrupt or half-written crumb is no evidence, not an error.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert record_crash.read_crumb(bad) is None
    bad.write_text(json.dumps({"task": "no dataset here"}))
    assert record_crash.read_crumb(bad) is None
    assert record_crash.read_crumb(tmp_path / "absent.json") is None


def test_an_unwritable_home_does_not_stop_a_recording(tmp_path) -> None:
    # write_crumb is a courtesy; a read-only home must never be why a session refuses to open.
    record_crash.write_crumb({"dataset": "local/x"}, path=tmp_path / "no" / "such" / "\0bad")
    record_crash.clear_crumb(tmp_path / "nope" / "gone.json")


def test_an_unknown_open_time_says_so(tmp_path) -> None:
    n = record_crash.interrupted_notice({"dataset": "local/x"}, now=1000.0)
    assert n is not None and "at an unknown time" in n["text"]
    assert n["opened_ago"] is None


def test_the_controller_reports_it_when_idle(tmp_path, monkeypatch) -> None:
    from strands_robots.dashboard import record_api

    crumb = tmp_path / "crumb.json"
    monkeypatch.setenv("STRANDS_DASH_RECORD_CRUMB", str(crumb))
    record_crash.write_crumb({"dataset": "local/cubes", "leader": "a", "follower": "b"}, path=crumb, now=1.0)

    ctl = record_api.RecordController(devices=object(), backend_factory=lambda **_: object())
    idle = ctl.session()
    # The idle shape is unchanged for every existing client...
    assert idle["dataset"] is None and idle["phase"] == "idle"
    # ...and the evidence rides alongside it.
    assert idle["interrupted"]["dataset"] == "local/cubes"

    record_crash.clear_crumb(crumb)
    assert "interrupted" not in record_api.RecordController(
        devices=object(), backend_factory=lambda **_: object()
    ).session()
