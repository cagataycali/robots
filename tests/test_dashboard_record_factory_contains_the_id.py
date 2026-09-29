"""The dashboard's recorder factory hands ``DatasetRecorder.create`` an id read back from the contained directory.

``strands_robots.dashboard.record_api._default_recorder_factory`` is the dashboard's one
door into ``DatasetRecorder.create``. The recorder accepts a path as an id by design
(``local_dataset_dir``: an absolute, ``./``-prefixed or slash-free id names the directory
it says), while the dashboard's contract is narrower: only an ``owner/name`` id that lands
under ``$HF_LEROBOT_HOME`` records, and the record route refuses a path id before the
factory is reached. That refusal lives in a predicate a static scanner cannot read as a
path barrier, so the seven ``py/path-injection`` findings on the recorder's target check
(``_prepare_create_target``: exists / is_dir / rmtree / unlink / iterdir on the resolved
directory) stayed open as long as the body string itself travelled into the recorder.

The factory now derives the id from the directory ``hub_dataset_dir`` contained (home
joined, ``normpath``-folded, refused unless it starts with the home), relative to that
home. For every id that passed ``dataset_id_error`` this is the same text; what changes
is that the value the recorder receives is the checked one. These tests pin both halves:
the identity, and the refusal for an id that would leave the home.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from strands_robots.dashboard import record_api
from strands_robots.dataset_source import HUB_ID_OUTSIDE_HOME


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "lerobot-home"
    root.mkdir()
    monkeypatch.setattr("strands_robots.dataset_source._lerobot_home", lambda: Path(root))
    return root


@pytest.mark.parametrize("repo_id", ["owner/name", "cagataydev/so101-pick-20260928", "a/b-c.d_e"])
def test_a_hub_id_comes_back_as_the_same_text(home, repo_id):
    assert record_api._contained_repo_id(repo_id) == repo_id


@pytest.mark.parametrize("repo_id", ["owner/../../etc", "/etc/passwd", "../outside", "owner/.."])
def test_an_id_that_would_leave_the_home_is_refused_with_the_one_sentence(home, repo_id):
    with pytest.raises(ValueError, match=HUB_ID_OUTSIDE_HOME):
        record_api._contained_repo_id(repo_id)


def test_the_factory_hands_the_recorder_the_contained_id(home):
    seen: dict[str, object] = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return "recorder"

    class Backend:
        def recorder_kwargs(self):
            return {"robot_type": "so101"}

    with patch("strands_robots.dataset_recorder.DatasetRecorder.create", side_effect=fake_create):
        make = record_api._default_recorder_factory(Backend())
        assert make(repo_id="owner/name", fps=30, task="pick") == "recorder"

    assert seen == {"repo_id": "owner/name", "fps": 30, "task": "pick", "robot_type": "so101"}


def test_the_factory_never_reaches_the_recorder_with_a_path_id(home):
    with patch("strands_robots.dataset_recorder.DatasetRecorder.create") as create:
        make = record_api._default_recorder_factory(object())
        with pytest.raises(ValueError, match=HUB_ID_OUTSIDE_HOME):
            make(repo_id="/tmp/anywhere", fps=30, task="")
    create.assert_not_called()
