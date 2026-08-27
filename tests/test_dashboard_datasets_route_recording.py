"""The datasets picker tells a LIVE recording apart from an abandoned folder (Q38, route half).

Only this route can: the metadata check cannot see a recorder, and the dashboard's own advice for
an empty dataset directory ("record into it, or delete it") names the one action that would destroy
a session in progress. So the endpoint asks the record controller and re-judges that one row.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard.server import create_app


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """This machine has an enrolled passkey + live settings token: point auth and settings at
    empty temp stores so the guard stays in open posture (repo gotcha, see BUGS.md)."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


def _dataset(root, repo, episodes):
    d = root / repo
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": episodes, "total_frames": episodes * 100, "fps": 30})
    )
    if episodes:
        chunk = d / "data" / "chunk-000"
        chunk.mkdir(parents=True)
        (chunk / "file-000.parquet").write_bytes(b"PAR1")
    return d


def _rows(client, **params):
    r = client.get("/api/training/datasets", params={"hub": "false", **params})
    assert r.status_code == 200, r.text
    return {row["repo_id"]: row for row in r.json()["datasets"]}


def test_the_dataset_being_recorded_is_not_told_to_delete_itself(tmp_path, monkeypatch) -> None:
    _dataset(tmp_path, "local/sim_recording", 0)
    _dataset(tmp_path, "org/finished", 12)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("STRANDS_ROBOTS_DATA_DIRS", raising=False)

    app = create_app()
    app.state.record = mock.Mock()
    app.state.record.session.return_value = {
        "dataset": "local/sim_recording",
        "phase": "recording",
        "episodes": [{"index": 0}, {"index": 1}],
    }
    with TestClient(app) as client:
        rows = _rows(client)

    live = rows["local/sim_recording"]
    assert live["recording"] is True
    assert live["reason"] == "recording_in_progress"
    assert "2 episode(s) captured so far" in live["problem"]
    assert "do NOT delete the folder" in live["problem"]
    # The finished dataset next to it is untouched and still trainable.
    assert rows["org/finished"]["usable"] is True
    assert "recording" not in rows["org/finished"]


def test_with_no_session_the_abandoned_verdict_stands(tmp_path, monkeypatch) -> None:
    _dataset(tmp_path, "local/sim_recording", 0)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("STRANDS_ROBOTS_DATA_DIRS", raising=False)

    app = create_app()
    app.state.record = mock.Mock()
    app.state.record.session.return_value = {"dataset": None, "phase": "idle", "episodes": []}
    with TestClient(app) as client:
        rows = _rows(client)

    assert rows["local/sim_recording"]["reason"] == "no_episodes"
    assert "delete it" in rows["local/sim_recording"]["problem"]


def test_a_broken_record_controller_cannot_break_the_picker(tmp_path, monkeypatch) -> None:
    """The picker is worth more than the annotation: a controller mid-transition (or a session
    call that raises) must cost the recording marker, never the list of datasets."""
    _dataset(tmp_path, "org/finished", 3)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("STRANDS_ROBOTS_DATA_DIRS", raising=False)

    app = create_app()
    app.state.record = mock.Mock()
    app.state.record.session.side_effect = RuntimeError("worker is being replaced")
    with TestClient(app) as client:
        rows = _rows(client)

    assert rows["org/finished"]["usable"] is True
