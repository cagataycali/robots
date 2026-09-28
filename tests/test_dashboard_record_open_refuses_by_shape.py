"""``POST /api/record/open`` refuses a dataset name by its shape, with the right status and sentence.

A blank name is a missing form field (422, the ``record_target_verdict`` sentence); a name that is a
path is a containment refusal (400 ``path_outside_dataset_home``). The two must not collapse.
"""

import pytest
from fastapi import HTTPException

from strands_robots.dashboard.record_api import RecordController


@pytest.mark.parametrize(
    ("dataset", "status", "phrase"),
    [
        ("", 422, "a dataset name is required"),
        ("   ", 422, "a dataset name is required"),
        ("../escape", 400, "dataset home"),
    ],
)
def test_open_refuses_a_dataset_name_by_its_shape(dataset, status, phrase, tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    ctl = RecordController(devices=object(), thumb_root=str(tmp_path / "thumbs"))
    with pytest.raises(HTTPException) as exc:
        ctl.open({"dataset": dataset, "task": "pick", "leader": "l", "follower": "f"})
    assert exc.value.status_code == status
    assert phrase in str(exc.value.detail)
