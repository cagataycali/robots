"""``train_policy`` refuses an unknown action before it reads anything else.

The action name is the first thing the call gets wrong when it is wrong, and
no other argument is read by an action that does not exist. Answering
``action="lsit"`` with "a data source and output_dir are required" (or with a
trainer-build error once those are supplied) sends the caller to fix arguments
that would never have been read.
"""

from __future__ import annotations

from typing import Any

from strands_robots.tools.train_policy import train_policy as train_policy_tool


def _unwrap(t: Any) -> Any:
    for attr in ("_tool_func", "original_function", "__wrapped__", "func"):
        target = getattr(t, attr, None)
        if callable(target):
            return target
    return t


train_policy = _unwrap(train_policy_tool)


def test_unknown_action_without_a_data_source_names_the_action() -> None:
    result = train_policy(action="lsit")
    text = result["content"][0]["text"]
    assert result["status"] == "error"
    assert "Unknown action: lsit" in text, text
    for verb in ("train", "validate", "status", "export", "list"):
        assert verb in text


def test_unknown_action_is_refused_before_the_trainer_is_built(tmp_path: Any) -> None:
    result = train_policy(
        action="lsit",
        provider="no-such-backend",
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    text = result["content"][0]["text"]
    assert result["status"] == "error"
    assert "Unknown action: lsit" in text, text
    assert "no-such-backend" not in text
