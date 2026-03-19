"""Tests for strands_robots.policies.groot.data_config — typed config system.

Covers: Gr00tDataConfig, ModalityConfig, load_data_config, create_custom_data_config,
_extends inheritance, DATA_CONFIG_MAP, and edge cases.
"""

import json

import pytest

from strands_robots.policies.groot.data_config import (
    _CONFIG_FILE,
    DATA_CONFIG_MAP,
    Gr00tDataConfig,
    ModalityConfig,
    create_custom_data_config,
    load_data_config,
)

# Load raw JSON for validation tests
_RAW = json.loads(_CONFIG_FILE.read_text())
_RAW_CONFIGS = _RAW["configs"]
_RAW_ALIASES = _RAW.get("aliases", {})

# ---------------------------------------------------------------------------
# ModalityConfig
# ---------------------------------------------------------------------------


class TestModalityConfig:
    def test_basic_construction(self):
        config = ModalityConfig(delta_indices=[0], modality_keys=["video.front"])
        assert config.delta_indices == [0]
        assert config.modality_keys == ["video.front"]

    def test_model_dump_json_roundtrip(self):
        config = ModalityConfig(delta_indices=[0, 1, 2], modality_keys=["state.arm", "state.gripper"])
        serialized = config.model_dump_json()
        parsed = json.loads(serialized)
        assert parsed["delta_indices"] == [0, 1, 2]
        assert parsed["modality_keys"] == ["state.arm", "state.gripper"]

    def test_empty_lists(self):
        config = ModalityConfig(delta_indices=[], modality_keys=[])
        parsed = json.loads(config.model_dump_json())
        assert parsed["delta_indices"] == []
        assert parsed["modality_keys"] == []


# ---------------------------------------------------------------------------
# Gr00tDataConfig
# ---------------------------------------------------------------------------


class TestGr00tDataConfig:
    def test_default_construction(self):
        config = Gr00tDataConfig()
        assert config.name == ""
        assert config.video_keys == []
        assert config.state_keys == []
        assert config.action_keys == []
        assert config.language_keys == []
        assert config.observation_indices == []
        assert config.action_indices == []

    def test_construction_with_values(self):
        config = Gr00tDataConfig(
            name="test",
            video_keys=["video.front"],
            state_keys=["state.arm"],
            action_keys=["action.arm"],
            language_keys=["annotation.human.task_description"],
            observation_indices=[0],
            action_indices=list(range(8)),
        )
        assert config.name == "test"
        assert len(config.video_keys) == 1
        assert len(config.action_indices) == 8

    def test_modality_config_returns_all_four_modalities(self):
        config = Gr00tDataConfig(
            video_keys=["video.cam"],
            state_keys=["state.arm"],
            action_keys=["action.arm"],
            language_keys=["lang"],
            observation_indices=[0],
            action_indices=[0, 1],
        )
        modality_configs = config.modality_config()
        assert set(modality_configs.keys()) == {"video", "state", "action", "language"}
        assert isinstance(modality_configs["video"], ModalityConfig)
        assert modality_configs["video"].modality_keys == ["video.cam"]
        assert modality_configs["action"].delta_indices == [0, 1]

    def test_modality_config_observation_indices_shared(self):
        """video, state, language share observation_indices; action has its own."""
        config = Gr00tDataConfig(
            observation_indices=[0],
            action_indices=[0, 1, 2],
            video_keys=["v"],
            state_keys=["s"],
            action_keys=["a"],
            language_keys=["l"],
        )
        modality_configs = config.modality_config()
        assert modality_configs["video"].delta_indices == [0]
        assert modality_configs["state"].delta_indices == [0]
        assert modality_configs["language"].delta_indices == [0]
        assert modality_configs["action"].delta_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# DATA_CONFIG_MAP + _extends inheritance
# ---------------------------------------------------------------------------


class TestDataConfigMap:
    def test_json_file_exists_and_is_valid(self):
        """data_configs.json must exist and contain valid JSON with expected structure."""
        assert _CONFIG_FILE.exists(), f"Missing {_CONFIG_FILE}"
        raw = json.loads(_CONFIG_FILE.read_text())
        assert "configs" in raw
        assert isinstance(raw["configs"], dict)
        assert len(raw["configs"]) > 0

    def test_all_defs_are_resolved(self):
        """Every key in data_configs.json must appear in DATA_CONFIG_MAP."""
        for config_name in _RAW_CONFIGS:
            assert config_name in DATA_CONFIG_MAP, f"'{config_name}' not resolved"

    def test_aliases_resolve_correctly(self):
        """Aliases should point to the same Gr00tDataConfig as their target."""
        for alias_name, target_name in _RAW_ALIASES.items():
            assert alias_name in DATA_CONFIG_MAP, f"Alias '{alias_name}' missing from map"
            assert DATA_CONFIG_MAP[alias_name] is DATA_CONFIG_MAP[target_name]

    def test_extends_inherits_parent_fields(self):
        """so100_dualcam extends so100 — should inherit state/action keys."""
        parent = DATA_CONFIG_MAP["so100"]
        child = DATA_CONFIG_MAP["so100_dualcam"]
        assert child.video_keys == ["video.front", "video.wrist"]
        assert child.state_keys == parent.state_keys
        assert child.action_keys == parent.action_keys
        assert child.action_indices == parent.action_indices

    def test_extends_chain_so100_4cam(self):
        child = DATA_CONFIG_MAP["so100_4cam"]
        assert len(child.video_keys) == 4
        assert child.state_keys == DATA_CONFIG_MAP["so100"].state_keys

    def test_so100_has_correct_keys(self):
        config = DATA_CONFIG_MAP["so100"]
        assert config.video_keys == ["video.webcam"]
        assert "state.single_arm" in config.state_keys
        assert "state.gripper" in config.state_keys
        assert "action.single_arm" in config.action_keys
        assert config.observation_indices == [0]
        assert config.action_indices == list(range(16))

    def test_unitree_g1_full_body_has_all_body_parts(self):
        config = DATA_CONFIG_MAP["unitree_g1_full_body"]
        expected_parts = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
        for part in expected_parts:
            assert f"state.{part}" in config.state_keys, f"Missing state.{part}"
            assert f"action.{part}" in config.action_keys, f"Missing action.{part}"

    def test_fourier_gr1_arms_waist_extends_arms_only(self):
        parent = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
        child = DATA_CONFIG_MAP["fourier_gr1_arms_waist"]
        assert "state.waist" in child.state_keys
        assert "action.waist" in child.action_keys
        assert child.language_keys != parent.language_keys

    def test_all_configs_have_required_fields(self):
        """Every config must have at least video, state, action, and language keys."""
        for config_name, config in DATA_CONFIG_MAP.items():
            assert len(config.video_keys) > 0, f"'{config_name}' has no video_keys"
            assert len(config.state_keys) > 0, f"'{config_name}' has no state_keys"
            assert len(config.action_keys) > 0, f"'{config_name}' has no action_keys"
            assert len(config.language_keys) > 0, f"'{config_name}' has no language_keys"
            assert len(config.observation_indices) > 0, f"'{config_name}' has no observation_indices"
            assert len(config.action_indices) > 0, f"'{config_name}' has no action_indices"

    def test_config_names_are_set(self):
        for config_name, config in DATA_CONFIG_MAP.items():
            if config_name in _RAW_ALIASES:
                assert config.name == _RAW_ALIASES[config_name]
            else:
                assert config.name == config_name, f"Config '{config_name}' has wrong .name: '{config.name}'"


# ---------------------------------------------------------------------------
# load_data_config
# ---------------------------------------------------------------------------


class TestLoadDataConfig:
    def test_load_by_string_name(self):
        config = load_data_config("so100")
        assert isinstance(config, Gr00tDataConfig)
        assert config.name == "so100"

    def test_load_passes_through_dataconfig_instance(self):
        original = Gr00tDataConfig(name="custom", video_keys=["v"])
        result = load_data_config(original)
        assert result is original

    def test_unknown_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown data_config"):
            load_data_config("nonexistent_robot_config_xyz")

    def test_wrong_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="must be str or Gr00tDataConfig"):
            load_data_config(42)

    def test_load_alias(self):
        for alias_name, target_name in _RAW_ALIASES.items():
            config = load_data_config(alias_name)
            assert config is DATA_CONFIG_MAP[target_name]


# ---------------------------------------------------------------------------
# create_custom_data_config
# ---------------------------------------------------------------------------


class TestCreateCustomDataConfig:
    def test_creates_and_registers(self):
        config = create_custom_data_config(
            name="test_custom_robot",
            video_keys=["video.top"],
            state_keys=["state.arm"],
            action_keys=["action.arm"],
        )
        assert isinstance(config, Gr00tDataConfig)
        assert config.name == "test_custom_robot"
        assert load_data_config("test_custom_robot") is config

    def test_defaults_for_optional_fields(self):
        config = create_custom_data_config(
            name="test_defaults",
            video_keys=["video.cam"],
            state_keys=["state.s"],
            action_keys=["action.a"],
        )
        assert config.language_keys == ["annotation.human.task_description"]
        assert config.observation_indices == [0]
        assert config.action_indices == list(range(16))

    def test_custom_overrides(self):
        config = create_custom_data_config(
            name="test_overrides",
            video_keys=["video.cam"],
            state_keys=["state.s"],
            action_keys=["action.a"],
            language_keys=["custom_lang"],
            observation_indices=[0, 1],
            action_indices=list(range(32)),
        )
        assert config.language_keys == ["custom_lang"]
        assert config.observation_indices == [0, 1]
        assert len(config.action_indices) == 32

    def test_overwrites_existing_name(self):
        """Creating with same name should overwrite in the map."""
        create_custom_data_config("overwrite_test", ["v"], ["s"], ["a"])
        first = load_data_config("overwrite_test")
        create_custom_data_config("overwrite_test", ["v2"], ["s2"], ["a2"])
        second = load_data_config("overwrite_test")
        assert second.video_keys == ["v2"]
        assert first is not second
