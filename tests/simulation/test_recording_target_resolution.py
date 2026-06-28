"""Behavior tests for recording-target resolution (create-vs-resume + dir prep).

``DatasetRecordingMixin._prepare_dataset_target`` decides whether a recording
session should resume an existing dataset or create a fresh one, and makes the
target directory safe for ``LeRobotDataset.create()`` - which raises
``FileExistsError`` if its target directory already exists, even when empty.
These tests pin that contract with real temp directories (no lerobot/mujoco
needed): a pre-existing empty ``root`` (e.g. from ``tempfile.mkdtemp()``) must
be accepted, a real dataset must resume, and a non-empty non-dataset dir must
fail loudly rather than be clobbered.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.recording import DatasetRecordingMixin

_prepare = DatasetRecordingMixin._prepare_dataset_target


def test_nonexistent_target_is_fresh_create(tmp_path):
    target = tmp_path / "new_dataset"
    assert _prepare(target, overwrite=False) is False
    # Nothing was created; create() will make it.
    assert not target.exists()


def test_existing_empty_dir_is_cleared_for_fresh_create(tmp_path):
    # The canonical caller pattern: tempfile.mkdtemp() returns an EXISTING
    # empty dir. Pre-fix this dead-ended LeRobotDataset.create() with
    # [Errno 17] File exists; the empty dir must be cleared so create() runs.
    target = tmp_path / "empty"
    target.mkdir()
    assert target.exists()
    assert _prepare(target, overwrite=False) is False
    assert not target.exists(), "empty target should be cleared for create()"


def test_existing_dataset_dir_resumes(tmp_path):
    target = tmp_path / "dataset"
    (target / "meta").mkdir(parents=True)
    (target / "meta" / "info.json").write_text("{}")
    assert _prepare(target, overwrite=False) is True
    # Resume must NOT delete the existing dataset.
    assert (target / "meta" / "info.json").exists()


def test_existing_nonempty_nondataset_dir_raises(tmp_path):
    target = tmp_path / "busy"
    target.mkdir()
    (target / "unrelated.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not a LeRobotDataset"):
        _prepare(target, overwrite=False)
    # The unrelated file must be preserved (no clobber).
    assert (target / "unrelated.txt").exists()


def test_overwrite_removes_existing_dataset_dir(tmp_path):
    target = tmp_path / "dataset"
    (target / "meta").mkdir(parents=True)
    (target / "meta" / "info.json").write_text("{}")
    assert _prepare(target, overwrite=True) is False
    assert not target.exists(), "overwrite must wipe the existing target"


def test_overwrite_removes_existing_file_target(tmp_path):
    target = tmp_path / "afile"
    target.write_text("stale")
    assert _prepare(target, overwrite=True) is False
    assert not target.exists()


def test_existing_file_target_without_overwrite_raises(tmp_path):
    target = tmp_path / "afile"
    target.write_text("stale")
    with pytest.raises(ValueError, match="not a directory"):
        _prepare(target, overwrite=False)
