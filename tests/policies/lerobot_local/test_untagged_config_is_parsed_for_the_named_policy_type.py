"""A checkpoint whose config.json has no draccus ``type`` tag loads when the caller names ``policy_type``.

``PreTrainedConfig.from_pretrained`` raises ``Missing 'type' field`` on such a file
(``nepyope/pi05-can-to-martino-12k`` is one, saved from a training fork). The caller
already named the type, so lerobot_local parses the config for that class and hands it
to ``from_pretrained(config=...)``. Local directories only; no Hub access.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.policies.lerobot_local.resolution import config_for_untagged_checkpoint

pytest.importorskip("lerobot")


def _act_config(tmp_path, *, tagged: bool) -> str:
    cfg = {
        "n_obs_steps": 1,
        "chunk_size": 10,
        "n_action_steps": 10,
        "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "device": "cpu",
    }
    if tagged:
        cfg["type"] = "act"
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(tmp_path)


def test_untagged_config_parses_for_the_named_type(tmp_path):
    cfg = config_for_untagged_checkpoint(_act_config(tmp_path, tagged=False), "act")
    assert cfg is not None
    assert type(cfg).__name__ == "ACTConfig"
    assert cfg.chunk_size == 10 and cfg.n_action_steps == 10


def test_tagged_config_is_left_to_the_normal_loader(tmp_path):
    assert config_for_untagged_checkpoint(_act_config(tmp_path, tagged=True), "act") is None


def test_unknown_type_or_unreadable_checkpoint_returns_none(tmp_path):
    assert config_for_untagged_checkpoint(_act_config(tmp_path, tagged=False), "no_such_policy") is None
    assert config_for_untagged_checkpoint(str(tmp_path / "missing"), "act") is None


def test_the_inference_config_falls_back_to_the_untagged_parse(tmp_path, monkeypatch):
    """``_inference_config`` (the device and compile pre-read) hands the untagged parse to ``from_pretrained``.

    The real ``PreTrainedConfig.from_pretrained`` raises on the tagless file; the
    pre-read then parses it for the named type and still applies the device choice.
    """
    from unittest.mock import patch

    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    path = _act_config(tmp_path, tagged=False)
    with patch.object(LerobotLocalPolicy, "_load_model"):
        named = LerobotLocalPolicy(pretrained_name_or_path=path, policy_type="act", device="cpu")
        unnamed = LerobotLocalPolicy(pretrained_name_or_path=path, device="cpu")
    cfg = named._inference_config()
    assert cfg is not None and type(cfg).__name__ == "ACTConfig"
    assert cfg.device == "cpu"
    assert unnamed._inference_config() is None
