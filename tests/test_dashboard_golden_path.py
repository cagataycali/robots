"""U20 acceptance: collect -> train -> checkpoint -> picker, one unbroken chain.

The three screens must CHAIN: a collected dataset appears in the train screen,
a submitted training's checkpoint appears in the policy picker, and the picker
row is a loadable path. Each seam has its own unit tests; this test drives the
actual routes in one story so no seam can silently detach from its neighbour
(iteration-6 recon found exactly such a detachment at seam 3: dashboard-trained
checkpoints invisible to /api/checkpoints/search).

Heavy machinery is stubbed at the SUBPROCESS boundary and nowhere shallower:
the collect backend writes a real dataset tree, the trainer writes a real
LeRobot checkpoint tree - every discovery walk, ledger write and route in
between is the production code. Run with --no-cov.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import checkpoints as ckpt_mod
from strands_robots.dashboard import training


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}
    # This machine's real jobs ledger + remembered roots must not leak in
    # (or be polluted by) the story.
    monkeypatch.setattr(training, "JOBS_FILE", tmp_path / "ledger" / "train_jobs.json")
    monkeypatch.setattr(training, "ROOTS_FILE", tmp_path / "ledger" / "dataset_roots.json")
    # Keep the walk to OUR roots: default scan paths see the real machine.
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "nowhere-lerobot"))
    monkeypatch.setenv("STRANDS_ROBOTS_DATA_DIRS", "")
    # The picker must not show this machine's HF cache or reach the hub.
    monkeypatch.setattr(ckpt_mod, "local_checkpoints", lambda q="": [])
    monkeypatch.setattr(ckpt_mod, "hub_search", lambda q, limit=15: ([], None))


def _write_dataset(root: Path, episodes: int = 3) -> None:
    """What a finished collect leaves on disk. GROUND TRUTH (run_policy.py:83):
    dataset_root IS the dataset directory - meta/info.json sits directly under
    it, not under a repo_id subfolder."""
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "total_episodes": episodes, "total_frames": episodes * 100,
        "fps": 30, "robot_type": "so101",
    }))


def _write_checkpoint(output_dir: Path, step: str = "0001000") -> None:
    """What a finished LeRobot training leaves on disk."""
    pm = output_dir / "checkpoints" / step / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text(json.dumps({"type": "act"}))


def test_collect_train_deploy_chain(tmp_path):
    from strands_robots.dashboard.server import create_app

    app = create_app()
    client = TestClient(app)
    dataset_root = tmp_path / "cubes-collected"
    output_dir = tmp_path / "runs" / "act-cubes"

    # ---- 1. collect: the backend child is stubbed; it writes a REAL dataset
    def fake_collect(*, dataset_root, dataset_repo_id, **kw):
        _write_dataset(Path(dataset_root))
        return {"ok": True, "episodes": 3, "dataset_root": dataset_root}

    app.state.devices.collect = mock.Mock(side_effect=fake_collect)
    r = client.post("/api/collect", json={
        "dataset_root": str(dataset_root),
        "dataset_repo_id": "local/cubes",
        "n_episodes": 3,
    })
    assert r.status_code == 200, r.text

    # ---- 2. the collected dataset appears in the train screen
    rows = client.get("/api/training/datasets").json()["datasets"]
    match = [d for d in rows if d["repo_id"].endswith("cubes-collected")]
    assert match, f"collected dataset invisible to the train screen: {rows}"
    assert match[0]["total_episodes"] == 3
    dataset_path = match[0]["root"]

    # ---- 3. submit training: the trainer is stubbed; it writes a REAL checkpoint
    def fake_train_policy(action: str, **kwargs):
        assert action == "train"
        assert kwargs["dataset_root"] == dataset_path, "train must receive the discovered dataset"
        _write_checkpoint(Path(kwargs["output_dir"]))
        return {"status": "success", "content": [{"json": {"job_id": "job-e2e-1"}}]}

    with mock.patch("strands_robots.tools.train_policy.train_policy", fake_train_policy):
        r = client.post("/api/training/submit", json={
            "provider": "lerobot_local",
            "dataset_root": dataset_path,
            "base_model": "lerobot/act_base",
            "output_dir": str(output_dir),
            "steps": 10,
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["job"]["job_id"] == "job-e2e-1"

    # ---- 4. the trained checkpoint appears in the policy picker, first
    results = client.get("/api/checkpoints/search", params={"q": ""}).json()["results"]
    assert results, "trained checkpoint invisible to the picker (the U20 break)"
    row = results[0]
    assert row["source"] == "trained"
    assert row["job_id"] == "job-e2e-1"
    assert row["policy_type"] == "act"

    # ---- 5. the row is the run form's contract: a loadable path
    artifact = Path(row["repo_id"])
    assert artifact.is_dir() and (artifact / "config.json").exists()
    assert json.loads((artifact / "config.json").read_text())["type"] == "act"
    # and it's searchable by what the user remembers: the dataset they used
    by_dataset = client.get("/api/checkpoints/search", params={"q": dataset_path}).json()["results"]
    assert any(r0.get("job_id") == "job-e2e-1" for r0 in by_dataset)


def test_chain_survives_a_training_that_produced_nothing(tmp_path):
    """A submitted job whose output never materialised must not poison the
    picker with an unloadable row - the chain degrades link by link."""
    from strands_robots.dashboard.server import create_app

    app = create_app()
    client = TestClient(app)

    def fake_train_policy(action: str, **kwargs):
        return {"status": "success", "content": [{"json": {"job_id": "job-vanished"}}]}
        # note: writes NOTHING into output_dir

    with mock.patch("strands_robots.tools.train_policy.train_policy", fake_train_policy):
        r = client.post("/api/training/submit", json={
            "provider": "lerobot_local",
            "dataset_root": str(tmp_path / "ds"),
            "output_dir": str(tmp_path / "never-written"),
            "steps": 10,
        })
    assert r.json()["status"] == "success"  # the ledger remembers the job...
    results = client.get("/api/checkpoints/search", params={"q": ""}).json()["results"]
    assert results == []  # ...but the picker only offers what can actually load
