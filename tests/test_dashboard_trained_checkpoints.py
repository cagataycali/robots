"""U20 seam: a policy trained BY the dashboard must be findable IN the dashboard.

submit() records every run's output_dir in the jobs ledger, but the picker
searched only the HF cache and the hub — the user could train a policy here
and then be unable to select it here. trained_checkpoints() closes the loop
via the ledger. Run with --no-cov.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from strands_robots.dashboard import checkpoints
from strands_robots.dashboard.checkpoints import _artifact_dir, search, trained_checkpoints


def _job(output_dir: str, **extra) -> dict:
    return {
        "job_id": extra.pop("job_id", "job-1"),
        "provider": "lerobot_local",
        "dataset": extra.pop("dataset", "local/collected"),
        "base_model": extra.pop("base_model", "lerobot/smolvla_base"),
        "output_dir": output_dir,
        "steps": 10,
        "submitted_at": extra.pop("submitted_at", time.time()),
        **extra,
    }


def _lerobot_run(root: Path, steps: tuple[str, ...] = ("0000500", "0001000"), last: str | None = None) -> Path:
    """Fake a LeRobot training output: checkpoints/<step>/pretrained_model/."""
    for step in steps:
        pm = root / "checkpoints" / step / "pretrained_model"
        pm.mkdir(parents=True)
        (pm / "config.json").write_text(json.dumps({"type": "smolvla"}))
    if last:
        (root / "checkpoints" / "last").symlink_to(root / "checkpoints" / last)
    return root


@pytest.fixture
def jobs(monkeypatch):
    """Script the training jobs ledger."""
    ledger: list[dict] = []
    from strands_robots.dashboard import training

    monkeypatch.setattr(training, "jobs", lambda: list(ledger))
    return ledger


def test_artifact_dir_prefers_last_then_highest_step(tmp_path):
    run = _lerobot_run(tmp_path / "run", steps=("0000500", "0001000"), last="0000500")
    # 'last' symlink wins even when a higher step exists (lerobot's own pointer)
    assert _artifact_dir(run) == run / "checkpoints" / "0000500" / "pretrained_model"
    run2 = _lerobot_run(tmp_path / "run2", steps=("900", "1000"))
    # numeric order, not lexicographic: 1000 > 900
    assert _artifact_dir(run2) == run2 / "checkpoints" / "1000" / "pretrained_model"


def test_artifact_dir_export_layout_and_empty(tmp_path):
    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "train_config.json").write_text("{}")
    assert _artifact_dir(exported) == exported
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _artifact_dir(empty) is None


def test_trained_checkpoint_appears_with_loadable_path(tmp_path, jobs):
    run = _lerobot_run(tmp_path / "smolvla-cubes")
    jobs.append(_job(str(run)))
    rows = trained_checkpoints()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "trained"
    assert row["local"] is True
    assert row["policy_type"] == "smolvla"
    assert row["job_id"] == "job-1"
    # repo_id IS the path pretrained_name_or_path can load
    assert Path(row["repo_id"]).name == "pretrained_model"
    assert (Path(row["repo_id"]) / "config.json").exists()


def test_unfinished_or_cleaned_runs_are_not_listed(tmp_path, jobs):
    still_training = tmp_path / "in-progress"
    (still_training / "checkpoints").mkdir(parents=True)  # no pretrained_model yet
    jobs.append(_job(str(still_training), job_id="j-live"))
    jobs.append(_job(str(tmp_path / "deleted-run"), job_id="j-gone"))
    jobs.append(_job("", job_id="j-empty"))
    assert trained_checkpoints() == []


def test_newest_job_first_and_query_filters(tmp_path, jobs):
    a = _lerobot_run(tmp_path / "act-sort")
    b = _lerobot_run(tmp_path / "smolvla-pick")
    jobs.append(_job(str(a), job_id="old", submitted_at=1))
    jobs.append(_job(str(b), job_id="new", submitted_at=2))
    rows = trained_checkpoints()
    assert [r["job_id"] for r in rows] == ["new", "old"]
    assert [r["job_id"] for r in trained_checkpoints("act-sort")] == ["old"]
    # the dataset name is searchable too — "what did I train on X" is a real query
    assert [r["job_id"] for r in trained_checkpoints("local/collected")] == ["new", "old"]


def test_search_ranks_trained_first_and_survives_hub_outage(tmp_path, jobs):
    run = _lerobot_run(tmp_path / "mine")
    jobs.append(_job(str(run)))
    with mock.patch.object(checkpoints, "local_checkpoints", return_value=[
        {"repo_id": "lerobot/smolvla_base", "local": True, "downloads": None, "tags": []},
    ]), mock.patch.object(checkpoints, "hub_search", return_value=(
        [{"repo_id": "lerobot/smolvla_base", "local": False},
         {"repo_id": "org/other", "local": False}],
        None,
    )):
        body = search("")
    ids = [r["repo_id"] for r in body["results"]]
    assert body["results"][0]["source"] == "trained", "own training outranks everything"
    assert ids.count("lerobot/smolvla_base") == 1, "hub duplicate of a local row is dropped"
    assert "org/other" in ids
