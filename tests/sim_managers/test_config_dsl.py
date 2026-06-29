"""Config DSL: build_managers, YAML/JSON loading, validation."""

from __future__ import annotations

import json

import pytest

from strands_robots.sim_managers import build_managers, load_managers_config

_CONFIG = {
    "command_manager": {"terms": [{"name": "base_velocity", "func": "uniform_velocity"}]},
    "observation_manager": {"terms": [{"func": "base_lin_vel"}, {"func": "velocity_commands"}]},
    "reward_manager": {"terms": [{"name": "track", "func": "track_lin_vel_xy_exp", "weight": 1.0}]},
    "termination_manager": {"terms": [{"func": "time_out"}]},
}


def test_build_all_managers():
    ms = build_managers(_CONFIG)
    assert ms.command is not None
    assert ms.observation is not None and ms.observation.term_names == ["base_lin_vel", "velocity_commands"]
    assert ms.reward is not None and ms.reward.weights == {"track": 1.0}
    assert ms.termination is not None


def test_missing_blocks_are_none():
    ms = build_managers({"reward_manager": {"terms": [{"func": "alive", "weight": 1.0}]}})
    assert ms.reward is not None
    assert ms.observation is None and ms.command is None and ms.termination is None


def test_unknown_manager_key_rejected():
    with pytest.raises(ValueError, match="unknown manager config keys"):
        build_managers({"bogus_manager": {"terms": []}})


def test_unknown_term_func_rejected():
    with pytest.raises(ValueError, match="unknown reward term"):
        build_managers({"reward_manager": {"terms": [{"func": "no_such_term"}]}})


def test_unknown_term_key_rejected():
    with pytest.raises(ValueError, match="unknown term config keys"):
        build_managers({"reward_manager": {"terms": [{"func": "alive", "wieght": 1.0}]}})


def test_missing_func_key_rejected():
    with pytest.raises(ValueError, match="missing required 'func'"):
        build_managers({"reward_manager": {"terms": [{"weight": 1.0}]}})


def test_load_json_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_CONFIG), encoding="utf-8")
    ms = load_managers_config(path)
    assert ms.reward is not None


def test_load_yaml_file(tmp_path):
    pytest.importorskip("yaml")
    import yaml  # type: ignore[import-untyped]

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    ms = load_managers_config(path)
    assert ms.observation is not None


def test_load_non_mapping_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_managers_config(path)
