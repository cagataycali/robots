"""Q58: pressing "train" must never silently delete the directory the operator typed.

``LerobotTrainer.start()`` clears a pre-existing ``output_dir`` that holds no resumable checkpoint
with ``shutil.rmtree(..., ignore_errors=True)``. The training form's output_dir is FREE TEXT, so a
reused or mistyped path — a dataset dir, a notes folder, a finished run whose checkpoint was already
exported and moved — was deleted by pressing a button labelled "train", with nothing on screen
saying so and no undo.
"""

from __future__ import annotations

from pathlib import Path

from unittest import mock

import pytest

from strands_robots.dashboard import training
from strands_robots.dashboard.output_dir_check import classify_output_dir, inspect_output_dir


# ----------------------------------------------------------------- pure verdicts

def test_a_missing_path_is_free():
    v = classify_output_dir(exists=False)
    assert v["state"] == "free" and v["needs_confirm"] is False


def test_an_empty_directory_is_free():
    v = classify_output_dir(exists=True, names=[], total=0)
    assert v["state"] == "free" and v["destructive"] is False


def test_files_with_no_checkpoint_are_a_named_loss_not_a_shrug():
    v = classify_output_dir(
        exists=True, has_checkpoint=False,
        names=["notes.md", "episode_0.mp4", "meta"], total=3,
    )
    assert v["state"] == "occupied"
    assert v["destructive"] is True and v["needs_confirm"] is True
    # The operator must be able to RECOGNISE the directory from the warning.
    assert "notes.md" in v["detail"] and "3 item(s)" in v["detail"]
    assert "DELETES" in v["detail"] and "cannot be undone" in v["detail"]


def test_a_long_listing_is_sampled_but_the_count_is_honest():
    v = classify_output_dir(
        exists=True, names=[f"f{i}" for i in range(40)], total=40,
    )
    assert len(v["entries"]) == 5 and v["total"] == 40
    assert "and 35 more" in v["detail"]


def test_a_checkpoint_dir_is_not_destructive_but_still_cannot_be_trained_into():
    """The opposite failure: kept on disk, then refused by lerobot inside the job."""
    v = classify_output_dir(exists=True, has_checkpoint=True, names=["checkpoints"], total=1)
    assert v["state"] == "resumable"
    assert v["destructive"] is False and v["needs_confirm"] is False
    assert "NOT be deleted" in v["detail"] and "cannot resume" in v["detail"]
    # Says where the failure would otherwise have hidden.
    assert "log" in v["detail"]


def test_a_file_is_a_typo_not_a_decision():
    v = classify_output_dir(exists=True, is_dir=False)
    assert v["state"] == "not_a_dir" and v["needs_confirm"] is False


def test_an_unreadable_path_is_unknown_never_free():
    """Guessing the friendly answer here is guessing about a delete."""
    v = classify_output_dir(exists=True, unreadable="PermissionError")
    assert v["state"] == "unknown"
    assert v["destructive"] is False and v["needs_confirm"] is False
    assert "unknown" in v["detail"]


# ------------------------------------------------------------- real filesystem

def test_inspect_reads_a_real_directory(tmp_path: Path):
    (tmp_path / "keepme.txt").write_text("work")
    v = inspect_output_dir(str(tmp_path), has_checkpoint=lambda p: False)
    assert v["state"] == "occupied" and v["entries"] == ["keepme.txt"]
    assert v["path"] == str(tmp_path)
    # read-only: the guard must not touch what it is protecting
    assert (tmp_path / "keepme.txt").exists()


def test_inspect_defers_to_the_trainers_checkpoint_definition(tmp_path: Path):
    (tmp_path / "checkpoints").mkdir()
    v = inspect_output_dir(str(tmp_path), has_checkpoint=lambda p: True)
    assert v["state"] == "resumable"


def test_a_failing_checkpoint_probe_is_unknown_not_a_delete(tmp_path: Path):
    (tmp_path / "stuff").write_text("x")

    def boom(_p):
        raise RuntimeError("trainer import blew up")

    v = inspect_output_dir(str(tmp_path), has_checkpoint=boom)
    assert v["state"] == "unknown" and v["needs_confirm"] is False


# --------------------------------------------------------------- submit() gate

def test_submit_refuses_an_occupied_output_dir_without_touching_the_trainer(tmp_path, monkeypatch):
    (tmp_path / "my-thesis.pdf").write_text("please do not delete me")
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        res = training.submit({
            "provider": "lerobot_local",
            "dataset_root": "/tmp/ds",
            "output_dir": str(tmp_path),
            "steps": 100,
        })
        called = tp.call_args_list
    assert res["status"] == "error"
    assert res["data"]["needs_confirm"] is True
    assert "my-thesis.pdf" in res["text"] and "confirm_clear" in res["text"]
    assert not called, "the trainer (and its rmtree) must not be reached"
    assert (tmp_path / "my-thesis.pdf").exists()


def test_confirm_clear_lets_a_deliberate_operator_through(tmp_path, monkeypatch):
    (tmp_path / "old-run.log").write_text("x")
    seen = {}

    monkeypatch.setattr(training, "_load_jobs", lambda: [])
    monkeypatch.setattr(training, "_save_jobs", lambda jobs: None)
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"status": "success", "content": [{"text": "started"}],
                           "data": {"job_id": "j1"}}
        res = training.submit({
            "provider": "lerobot_local",
            "dataset_root": "/tmp/ds",
            "output_dir": str(tmp_path),
            "confirm_clear": True,
        })
        seen.update(tp.call_args.kwargs)
    assert res["status"] == "success"
    # confirm_clear is a dashboard-level consent, not part of the trainer's vocabulary:
    # leaking it into train_policy would be the Q6 unknown-kwarg crash all over again.
    assert "confirm_clear" not in seen


def test_a_free_output_dir_needs_no_confirmation(tmp_path, monkeypatch):
    target = tmp_path / "fresh"
    monkeypatch.setattr(training, "_load_jobs", lambda: [])
    monkeypatch.setattr(training, "_save_jobs", lambda jobs: None)
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"status": "success", "content": [{"text": "ok"}],
                           "data": {"job_id": "j2"}}
        res = training.submit({"provider": "lerobot_local", "dataset_root": "/tmp/ds",
                               "output_dir": str(target)})
    assert res["status"] == "success"


def test_confirm_clear_is_not_mistaken_for_an_unknown_field(tmp_path, monkeypatch):
    """SPEC_KEYS is a closed vocabulary on purpose (Q6); the consent flag rides beside it."""
    monkeypatch.setattr(training, "_load_jobs", lambda: [])
    monkeypatch.setattr(training, "_save_jobs", lambda jobs: None)
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"status": "success", "content": [{"text": "ok"}],
                           "data": {"job_id": "j3"}}
        res = training.submit({"provider": "lerobot_local", "dataset_root": "/tmp/ds",
                               "output_dir": str(tmp_path / "new"), "confirm_clear": False})
    assert res["status"] == "success", res.get("text")
