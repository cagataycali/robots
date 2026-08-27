"""Q56: the record panel's "upload to the Hugging Face Hub" tick had never published anything.

``RecordWorker.close()`` called ``self._recorder.push_to_hub(repo_id=...)`` and
``DatasetRecorder.push_to_hub`` accepts ``(tags, private)`` only -- so every upload raised TypeError,
was swallowed by a bare ``except``, and surfaced at the END of a recording session as "dataset saved
but upload failed: …", wording an operator reads as a Hub outage rather than a dashboard defect. The
tick was reachable, the field beside it was inert, and the failure arrived after the episodes existed.

The second half is worse and would have survived a naive fix: the recorder does not RAISE on a real
failure. It returns ``{"status": "error", "message": …}`` for an empty dataset, a 403, a missing token.
A caller guarding only with try/except therefore reports "pushed to X" for a push that was refused.

``upload_verdict`` is the honest judge, pinned here.
"""

from __future__ import annotations

from strands_robots.dashboard.record_worker import upload_verdict


def test_a_successful_push_is_reported_from_the_recorders_own_answer():
    calls = []

    def push():
        calls.append(True)
        return {"status": "success", "repo_id": "cagatay/so101-pick", "episodes": 4}

    v = upload_verdict(asked_repo_id=None, dataset="cagatay/so101-pick", push=push)
    assert v == {"ok": True, "detail": "pushed to cagatay/so101-pick"}
    assert calls, "a plain upload must actually call the recorder"


def test_a_refused_push_is_never_reported_as_pushed():
    """The half that would have survived a naive repo_id fix."""

    def push():
        return {"status": "error", "message": "refusing to push empty dataset local/x (0 frames)"}

    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=push)
    assert v["ok"] is False
    assert "REFUSED" in v["detail"]
    # The recorder's own reason travels: "upload failed" alone sends the operator nowhere.
    assert "empty dataset" in v["detail"]
    assert "pushed to" not in v["detail"]


def test_a_raising_push_says_the_dataset_is_still_on_disk():
    def push():
        raise RuntimeError("401 Client Error")

    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=push)
    assert v["ok"] is False
    assert "saved locally" in v["detail"] and "401" in v["detail"]


def test_an_unreadable_answer_is_unknown_not_success():
    """Guessing "pushed" from a shape we do not recognise is the lie being removed."""
    v = upload_verdict(asked_repo_id=None, dataset="local/x", push=lambda: None)
    assert v["ok"] is False
    assert "UNKNOWN" in v["detail"]
    assert "pushed to" not in v["detail"]


def test_a_different_repo_id_is_refused_without_touching_the_hub():
    """A dataset publishes under the name it was recorded with; there is no argument for another.

    The old code passed the asked-for name as a kwarg that does not exist. Publishing under the
    recorded name instead would put the operator's episodes in a repo they did not name, so this
    refuses and says what would have to change.
    """
    calls = []

    v = upload_verdict(
        asked_repo_id="cagatay/other-name",
        dataset="cagatay/so101-pick",
        push=lambda: calls.append(True),
    )
    assert v["ok"] is False
    assert not calls, "the Hub must not be touched when the request cannot be honoured"
    assert "cagatay/so101-pick" in v["detail"] and "cagatay/other-name" in v["detail"]
    assert "saved" in v["detail"]


def test_the_same_repo_id_typed_out_is_not_a_conflict():
    """The UI defaults that box to the dataset name, so this is the common path."""
    v = upload_verdict(
        asked_repo_id="cagatay/so101-pick",
        dataset="cagatay/so101-pick",
        push=lambda: {"status": "success", "repo_id": "cagatay/so101-pick"},
    )
    assert v["ok"] is True


def test_the_recorder_is_still_called_with_no_repo_id_kwarg():
    """The literal defect: a kwarg the recorder's signature does not have.

    Read off the real signature so a future recorder that gains a repo_id argument makes this test
    fail loudly rather than leaving the dashboard on a stale assumption.
    """
    import inspect

    from strands_robots.dataset_recorder import DatasetRecorder

    params = inspect.signature(DatasetRecorder.push_to_hub).parameters
    assert "repo_id" not in params, (
        "push_to_hub gained a repo_id argument -- upload_verdict's refusal is now wrong and the "
        "record panel could honour a different name"
    )
