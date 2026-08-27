"""``training.export`` answers with the ARTIFACT's state, not only the trainer's (Q36 part 2).

The trainer's "success" is a report about its own run. latest_checkpoint() discovers a checkpoint
directory by its CONFIG file and the default export() returns that path unchanged, so a run killed
between the config write and the weights write reports success and fails later - when a policy is
being loaded onto a robot the operator has already chosen.

These tests use a real temp directory rather than a mocked verdict: the layouts ARE the bug, and a
fake verdict would pass no matter which way the wiring was hooked up.
"""

from __future__ import annotations

from unittest import mock

from strands_robots.dashboard import training
from strands_robots.dashboard.artifact_check import MIN_WEIGHT_BYTES


def _export(exported_path, status="success"):
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {
            "status": status,
            "content": [
                {"text": f"[mock] exported loadable artifact:\n{exported_path}"},
                {"json": {"provider": "mock", "exported_model": str(exported_path)}},
            ],
        }
        return training.export("mock", str(exported_path), "/data")


def test_a_config_without_weights_is_not_deployable(tmp_path) -> None:
    (tmp_path / "train_config.json").write_text("{}")
    res = _export(tmp_path)
    assert res["status"] == "success", "the trainer's own verdict is preserved, not overwritten"
    assert res["deployable"] is False
    assert res["artifact"]["reason"] == "config_without_weights"
    assert "BY ITS CONFIG" in res["artifact"]["message"]


def test_a_complete_checkpoint_is_deployable(tmp_path) -> None:
    (tmp_path / "train_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (MIN_WEIGHT_BYTES * 4))
    res = _export(tmp_path)
    assert res["deployable"] is True
    assert res["artifact"]["ok"] is True
    assert "load" in res["artifact"]["note"], "a pass must not read as a load guarantee"


def test_an_output_dir_that_no_longer_exists_is_caught(tmp_path) -> None:
    res = _export(tmp_path / "gone" / "pretrained_model")
    assert res["deployable"] is False
    assert res["artifact"]["reason"] == "missing"


def test_a_success_with_no_artifact_path_at_all(tmp_path) -> None:
    # The tool reported success and named nothing. Before this, the frontend's deploy path
    # was the only thing that noticed - and only because it happened to check for a string.
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"status": "success", "content": [{"text": "done"}]}
        res = training.export("mock", "/out", "/data")
    assert res["deployable"] is False
    assert res["artifact"]["reason"] == "empty_path"


def test_a_failed_export_is_left_exactly_as_the_trainer_reported_it(tmp_path) -> None:
    # No artifact block on a failure: the trainer already said why it refused, and running a
    # disk check on a path it never wrote would only add a second, vaguer reason.
    res = _export(tmp_path, status="error")
    assert res["status"] == "error"
    assert "artifact" not in res and "deployable" not in res


def test_the_trainers_own_text_survives(tmp_path) -> None:
    # The export text carries the create_policy() line the operator copies.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (MIN_WEIGHT_BYTES * 4))
    res = _export(tmp_path)
    assert "exported loadable artifact" in res["text"]
    assert res["data"]["exported_model"] == str(tmp_path)
