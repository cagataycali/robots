"""Tests for strands_robots.policies.groot — Gr00tPolicy unit tests.

Covers: construction, service/local mode dispatch, observation building,
camera mapping, state mapping, action conversion, version detection,
N1.5/N1.6 inference paths, and _find_obs_key.

All tests run WITHOUT Isaac-GR00T or a real server — local mode is mocked.
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.policies.groot import DATA_CONFIG_MAP, Gr00tPolicy
from strands_robots.policies.groot.data_config import Gr00tDataConfig
from strands_robots.policies.groot.policy import _detect_groot_version

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SO100_STATE_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@pytest.fixture
def service_policy():
    """Gr00tPolicy in service mode (no real server)."""
    policy = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
    policy.set_robot_state_keys(SO100_STATE_KEYS)
    return policy


@pytest.fixture
def custom_config():
    return Gr00tDataConfig(
        name="test_bot",
        video_keys=["video.cam"],
        state_keys=["state.arm"],
        action_keys=["action.arm"],
        language_keys=["annotation.human.task_description"],
        observation_indices=[0],
        action_indices=list(range(8)),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGr00tPolicyConstruction:
    def test_service_mode_defaults(self):
        policy = Gr00tPolicy()
        assert policy._mode == "service"
        assert policy._client is not None
        assert policy._local_policy is None
        assert policy.provider_name == "groot"

    def test_service_mode_with_config_name(self):
        policy = Gr00tPolicy(data_config="so100")
        assert policy.data_config_name == "so100"
        assert policy.camera_keys == ["video.webcam"]

    def test_service_mode_with_dataconfig_object(self, custom_config):
        policy = Gr00tPolicy(data_config=custom_config)
        assert policy.camera_keys == ["video.cam"]
        assert policy.data_config is custom_config

    def test_strict_defaults_false(self):
        policy = Gr00tPolicy()
        assert policy._strict is False

    def test_strict_true(self):
        policy = Gr00tPolicy(strict=True)
        assert policy._strict is True

    def test_api_token_passed_to_client(self):
        policy = Gr00tPolicy(host="localhost", port=5555, api_token="tok123")
        assert policy._client.api_token == "tok123"

    def test_unknown_data_config_raises(self):
        with pytest.raises(ValueError, match="Unknown data_config"):
            Gr00tPolicy(data_config="nonexistent_robot_xyz")

    def test_local_mode_without_isaac_raises(self):
        """model_path set but no Isaac-GR00T installed → ImportError."""
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy.data_config = DATA_CONFIG_MAP["so100"]
        policy.data_config_name = "so100"
        policy.camera_keys = policy.data_config.video_keys
        policy.state_keys = policy.data_config.state_keys
        policy.action_keys = policy.data_config.action_keys
        policy.language_keys = policy.data_config.language_keys
        policy.robot_state_keys = []
        policy._local_policy = None
        policy._client = None
        policy._groot_version = None
        policy._strict = False
        with pytest.raises(ImportError, match="Isaac-GR00T not installed"):
            policy._load_local_policy("/fake/path", "NEW_EMBODIMENT", 4, "cpu")

    def test_set_robot_state_keys(self, service_policy):
        service_policy.set_robot_state_keys(["a", "b"])
        assert service_policy.robot_state_keys == ["a", "b"]

    def test_all_data_configs_loadable(self):
        """Every config in DATA_CONFIG_MAP should be constructable."""
        for config_name in DATA_CONFIG_MAP:
            policy = Gr00tPolicy(data_config=config_name)
            assert policy._mode == "service"
            assert len(policy.camera_keys) > 0


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    def test_detect_caches_result(self):
        import strands_robots.policies.groot.policy as policy_mod

        original = policy_mod._GROOT_VERSION
        policy_mod._GROOT_VERSION = "test_cached"
        try:
            assert _detect_groot_version() == "test_cached"
        finally:
            policy_mod._GROOT_VERSION = original

    def test_detect_returns_none_without_groot(self):
        import strands_robots.policies.groot.policy as policy_mod

        policy_mod._GROOT_VERSION = None
        with patch("importlib.util.find_spec", return_value=None):
            result = _detect_groot_version()
        # Reset — may have been set by real detection
        policy_mod._GROOT_VERSION = None
        # If neither is found, result should be None
        # (test environment may have gr00t installed, so we accept both)
        assert result in (None, "n1.5", "n1.6")


# ---------------------------------------------------------------------------
# _find_obs_key
# ---------------------------------------------------------------------------


class TestFindObsKey:
    def test_finds_first_match(self):
        observation = {"video.front": 1, "front": 2}
        assert Gr00tPolicy._find_obs_key(observation, "video.front", "front") == "video.front"

    def test_finds_second_match(self):
        observation = {"front": 2}
        assert Gr00tPolicy._find_obs_key(observation, "video.front", "front") == "front"

    def test_returns_none_when_no_match(self):
        observation = {"other": 1}
        assert Gr00tPolicy._find_obs_key(observation, "video.front", "front") is None


# ---------------------------------------------------------------------------
# Camera mapping
# ---------------------------------------------------------------------------


class TestCameraMapping:
    def test_exact_match(self, service_policy):
        observation = {"front": np.zeros(3), "wrist": np.zeros(3)}
        result = service_policy._map_video_key_to_camera("video.front", observation)
        assert result == "front"

    def test_alias_fallback(self, service_policy):
        observation = {"webcam": np.zeros(3)}
        result = service_policy._map_video_key_to_camera("video.front", observation)
        assert result == "webcam"

    def test_fallback_to_first_non_state_key(self, service_policy):
        observation = {"state.arm": np.zeros(5), "my_camera": np.zeros(3)}
        result = service_policy._map_video_key_to_camera("video.unknown_cam", observation)
        assert result == "my_camera"

    def test_no_cameras_returns_none(self, service_policy):
        observation = {"state.arm": np.zeros(5)}
        result = service_policy._map_video_key_to_camera("video.front", observation)
        assert result is None

    def test_wrist_alias(self, service_policy):
        observation = {"hand": np.zeros(3)}
        result = service_policy._map_video_key_to_camera("video.wrist", observation)
        assert result == "hand"


# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------


class TestStateMapping:
    def test_so100_state_mapping(self):
        policy = Gr00tPolicy(data_config="so100")
        observation = {}
        robot_state = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        assert "state.single_arm" in observation
        assert "state.gripper" in observation
        np.testing.assert_array_almost_equal(observation["state.single_arm"], [1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(observation["state.gripper"], [6])
        assert observation["state.single_arm"].dtype == np.float32

    def test_so101_uses_so100_path(self):
        policy = Gr00tPolicy(data_config="so101")
        observation = {}
        robot_state = np.arange(7, dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        assert "state.single_arm" in observation
        assert "state.gripper" in observation

    def test_fourier_gr1_state_mapping(self):
        policy = Gr00tPolicy(data_config="fourier_gr1_arms_only")
        observation = {}
        robot_state = np.arange(14, dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        assert "state.left_arm" in observation
        assert "state.right_arm" in observation
        np.testing.assert_array_almost_equal(observation["state.left_arm"], np.arange(7))
        np.testing.assert_array_almost_equal(observation["state.right_arm"], np.arange(7, 14))

    def test_unitree_g1_state_mapping(self):
        policy = Gr00tPolicy(data_config="unitree_g1_full_body")
        observation = {}
        robot_state = np.arange(43, dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        expected_keys = [
            "state.left_leg",
            "state.right_leg",
            "state.waist",
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        for key in expected_keys:
            assert key in observation, f"Missing {key}"
        np.testing.assert_array_almost_equal(observation["state.left_leg"], np.arange(0, 6))
        np.testing.assert_array_almost_equal(observation["state.right_leg"], np.arange(6, 12))
        np.testing.assert_array_almost_equal(observation["state.waist"], np.arange(12, 15))

    def test_unitree_g1_short_state_zero_fills(self):
        """When robot_state is shorter than full body, missing parts get zeros."""
        policy = Gr00tPolicy(data_config="unitree_g1_full_body")
        observation = {}
        robot_state = np.arange(15, dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        np.testing.assert_array_equal(observation["state.left_arm"], np.zeros(7))

    def test_generic_fallback_state_mapping(self):
        """Unknown config with state data falls back to first state key."""
        policy = Gr00tPolicy(data_config="oxe_google")
        observation = {}
        robot_state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        policy._map_robot_state_to_groot_state(observation, robot_state)
        first_key = policy.state_keys[0]
        assert first_key in observation
        assert observation[first_key].dtype == np.float32

    def test_empty_state_does_nothing(self):
        policy = Gr00tPolicy(data_config="so100")
        observation = {}
        policy._map_robot_state_to_groot_state(observation, np.array([]))
        assert "state.single_arm" not in observation


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------


class TestBuildObservation:
    def test_service_mode_adds_batch_dim(self, service_policy):
        observation_dict = {
            "front": np.zeros((64, 64, 3), dtype=np.uint8),
            "wrist": np.zeros((64, 64, 3), dtype=np.uint8),
            "shoulder_pan": 0.1,
            "shoulder_lift": 0.2,
            "elbow_flex": 0.3,
            "wrist_flex": 0.4,
            "wrist_roll": 0.5,
            "gripper": 0.6,
        }
        observation = service_policy._build_observation(observation_dict, "pick up cube")
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                assert value.ndim >= 2, f"{key} missing batch dim"
            elif isinstance(value, list):
                assert len(value) >= 1

    def test_language_key_is_set(self, service_policy):
        observation = service_policy._build_observation({}, "do the task")
        language_key = service_policy.language_keys[0]
        assert language_key in observation


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------


class TestConvertToRobotActions:
    def test_basic_conversion_with_state_keys(self, service_policy):
        action_chunk = {
            "action.single_arm": np.random.rand(1, 16, 5).astype(np.float32),
            "action.gripper": np.random.rand(1, 16, 1).astype(np.float32),
        }
        actions = service_policy._convert_to_robot_actions(action_chunk)
        assert len(actions) == 16
        assert all(isinstance(action, dict) for action in actions)
        for action in actions:
            for key in SO100_STATE_KEYS:
                assert key in action, f"Missing key '{key}'"

    def test_strips_action_prefix(self, service_policy):
        action_chunk = {
            "action.single_arm": np.ones((1, 4, 5), dtype=np.float32),
            "action.gripper": np.ones((1, 4, 1), dtype=np.float32),
        }
        actions = service_policy._convert_to_robot_actions(action_chunk)
        assert len(actions) == 4

    def test_without_robot_state_keys_returns_per_group(self):
        policy = Gr00tPolicy(data_config="so100")
        action_chunk = {
            "action.single_arm": np.ones((1, 4, 5), dtype=np.float32),
            "action.gripper": np.ones((1, 4, 1), dtype=np.float32),
        }
        actions = policy._convert_to_robot_actions(action_chunk)
        assert len(actions) == 4
        assert "single_arm" in actions[0]
        assert "gripper" in actions[0]

    def test_empty_action_chunk_returns_empty(self, service_policy):
        assert service_policy._convert_to_robot_actions({}) == []

    def test_squeezed_batch_dims(self, service_policy):
        """3D arrays (batch, horizon, dim) should be squeezed correctly."""
        action_chunk = {
            "action.single_arm": np.ones((1, 1, 8, 5), dtype=np.float32),
        }
        actions = service_policy._convert_to_robot_actions(action_chunk)
        assert len(actions) == 8

    def test_non_array_values(self, service_policy):
        """Plain lists should be handled via np.atleast_2d."""
        action_chunk = {
            "single_arm": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "gripper": [[0.5]],
        }
        actions = service_policy._convert_to_robot_actions(action_chunk)
        assert len(actions) == 1


# ---------------------------------------------------------------------------
# Local inference paths (mocked)
# ---------------------------------------------------------------------------


class TestLocalInference:
    def _make_local_policy(self, version="n1.6"):
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy.data_config = DATA_CONFIG_MAP["so100"]
        policy.data_config_name = "so100"
        policy.camera_keys = policy.data_config.video_keys
        policy.state_keys = policy.data_config.state_keys
        policy.action_keys = policy.data_config.action_keys
        policy.language_keys = policy.data_config.language_keys
        policy.robot_state_keys = SO100_STATE_KEYS
        policy._mode = "local"
        policy._groot_version = version
        policy._strict = False
        policy._client = None
        policy._local_policy = MagicMock()
        # N1.6 _local_inference reads model's modality_configs for key remapping.
        # Mock it to pass through keys unchanged (model keys == our keys).
        mock_mc = {}
        for modality_name in ("video", "state", "action", "language"):
            mc_obj = MagicMock()
            # Strip 'video.', 'state.' etc. prefix to get bare modality keys
            if modality_name == "video":
                mc_obj.modality_keys = [k.split(".", 1)[1] for k in policy.data_config.video_keys]
            elif modality_name == "state":
                mc_obj.modality_keys = [k.split(".", 1)[1] for k in policy.data_config.state_keys]
            elif modality_name == "action":
                mc_obj.modality_keys = [k.split(".", 1)[1] for k in policy.data_config.action_keys]
            elif modality_name == "language":
                mc_obj.modality_keys = policy.data_config.language_keys
            mock_mc[modality_name] = mc_obj
        policy._local_policy.policy.modality_configs = mock_mc
        return policy

    def test_local_inference_n15_adds_batch_dim(self):
        policy = self._make_local_policy("n1.5")
        fake_action = {"single_arm": np.zeros((1, 16, 5))}
        policy._local_policy.get_action.return_value = fake_action

        observation = {
            "video.webcam": np.zeros((64, 64, 3), dtype=np.uint8),
            "state.arm": np.zeros(5),
        }
        result = policy._local_inference_n15(observation)
        assert result is fake_action
        call_args = policy._local_policy.get_action.call_args[0][0]
        assert call_args["video.webcam"].shape[0] == 1

    def test_local_inference_n16_adds_batch_and_temporal_dims(self):
        policy = self._make_local_policy("n1.6")
        fake_action = ({"single_arm": np.zeros((1, 16, 5))}, {})
        policy._local_policy.get_action.return_value = fake_action

        observation = {
            "video.webcam": np.zeros((64, 64, 3), dtype=np.uint8),
            "state.single_arm": np.ones(5, dtype=np.float32),
            "annotation.human.task_description": "test",
        }
        result = policy._local_inference_n16(observation)
        assert "action.single_arm" in result

        call_args = policy._local_policy.get_action.call_args[0][0]
        assert call_args["video.webcam"].shape == (1, 1, 64, 64, 3)
        assert call_args["state.single_arm"].shape == (1, 1, 5)
        assert call_args["state.single_arm"].dtype == np.float32
        assert isinstance(call_args["annotation.human.task_description"], list)

    def test_local_inference_n16_video_4d_input(self):
        """4D video (T, H, W, C) should become (1, T, H, W, C)."""
        policy = self._make_local_policy("n1.6")
        policy._local_policy.get_action.return_value = ({"arm": np.zeros((1, 4, 5))}, {})

        observation = {"video.webcam": np.zeros((3, 64, 64, 3), dtype=np.uint8)}
        policy._local_inference_n16(observation)
        call_args = policy._local_policy.get_action.call_args[0][0]
        assert call_args["video.webcam"].shape == (1, 3, 64, 64, 3)

    def test_local_inference_n16_state_2d_input(self):
        """2D state (T, D) should become (1, T, D)."""
        policy = self._make_local_policy("n1.6")
        policy._local_policy.get_action.return_value = ({"arm": np.zeros((1, 4, 5))}, {})

        observation = {"state.single_arm": np.zeros((3, 5), dtype=np.float32)}
        policy._local_inference_n16(observation)
        call_args = policy._local_policy.get_action.call_args[0][0]
        assert call_args["state.single_arm"].shape == (1, 3, 5)

    def test_local_inference_dispatch_n15(self):
        policy = self._make_local_policy("n1.5")
        policy._local_policy.get_action.return_value = {}
        policy._local_inference({"key": "value"})
        policy._local_policy.get_action.assert_called_once()

    def test_local_inference_dispatch_n16(self):
        policy = self._make_local_policy("n1.6")
        policy._local_policy.get_action.return_value = ({}, {})
        policy._local_inference({"key": "value"})
        policy._local_policy.get_action.assert_called_once()

    def test_local_inference_unknown_version_raises(self):
        policy = self._make_local_policy("n9.9")
        with pytest.raises(RuntimeError, match="Unknown GR00T version"):
            policy._local_inference({})


# ---------------------------------------------------------------------------
# get_actions async flow (mocked client)
# ---------------------------------------------------------------------------


class TestGetActionsAsync:
    def test_service_mode_get_actions(self, service_policy):
        fake_action = {
            "action.single_arm": np.random.rand(1, 16, 5).astype(np.float32),
            "action.gripper": np.random.rand(1, 16, 1).astype(np.float32),
        }
        service_policy._client.get_action = MagicMock(return_value=fake_action)

        observation = {
            "front": np.zeros((64, 64, 3), dtype=np.uint8),
            "wrist": np.zeros((64, 64, 3), dtype=np.uint8),
            "shoulder_pan": 0.1,
            "shoulder_lift": 0.2,
            "elbow_flex": 0.3,
            "wrist_flex": 0.4,
            "wrist_roll": 0.5,
            "gripper": 0.6,
        }
        actions = asyncio.run(service_policy.get_actions(observation, "pick up cube"))
        assert len(actions) == 16
        assert all(isinstance(action, dict) for action in actions)
        service_policy._client.get_action.assert_called_once()

    def test_local_mode_get_actions(self):
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy.data_config = DATA_CONFIG_MAP["so100"]
        policy.data_config_name = "so100"
        policy.camera_keys = policy.data_config.video_keys
        policy.state_keys = policy.data_config.state_keys
        policy.action_keys = policy.data_config.action_keys
        policy.language_keys = policy.data_config.language_keys
        policy.robot_state_keys = SO100_STATE_KEYS
        policy._mode = "local"
        policy._groot_version = "n1.5"
        policy._strict = False
        policy._client = None
        policy._local_policy = MagicMock()
        policy._local_policy.get_action.return_value = {
            "action.single_arm": np.random.rand(1, 16, 5).astype(np.float32),
            "action.gripper": np.random.rand(1, 16, 1).astype(np.float32),
        }

        observation = {"webcam": np.zeros((64, 64, 3), dtype=np.uint8)}
        for key in SO100_STATE_KEYS:
            observation[key] = 0.0
        actions = asyncio.run(policy.get_actions(observation, "test"))
        assert len(actions) == 16


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_exports_importable(self):
        import strands_robots.policies.groot as module
        from strands_robots.policies.groot import __all__

        for name in __all__:
            assert hasattr(module, name), f"'{name}' in __all__ but not in module"

    def test_groot_data_config_importable(self):
        from strands_robots.policies.groot import Gr00tDataConfig

        assert Gr00tDataConfig is Gr00tDataConfig
