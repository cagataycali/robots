"""Unit tests for QwenVlaPolicy mapping, observation build, and action unpack.

These tests construct the policy in SERVICE mode with a fake client so no
network, no model, and no torch are required - the focus is the pure
robot<->model translation machinery.
"""

import numpy as np
import pytest

from strands_robots.policies.qwen_vla import QwenVlaPolicy
from strands_robots.policies.qwen_vla.data_config import load_data_config
from strands_robots.policies.qwen_vla.policy import (
    _auto_infer_action_mapping,
    _auto_infer_observation_mapping,
    _parse_action_mapping,
    _parse_observation_mapping,
)


class _FakeClient:
    """Stand-in for QwenVlaInferenceClient that records the last observation."""

    def __init__(self, action_response):
        self.action_response = action_response
        self.last_observation = None
        self.reset_calls = []

    def get_action(self, observation):
        self.last_observation = observation
        return self.action_response

    def call_endpoint(self, endpoint, data=None):
        if endpoint == "reset":
            self.reset_calls.append(data)
            return {}
        raise RuntimeError(f"unexpected endpoint {endpoint}")


def _make_service_policy(data_config, action_response, **kwargs):
    """Build a SERVICE-mode policy then swap in a fake client (no real ZMQ)."""
    # Avoid constructing the real ZMQ client: monkeypatch via subclass.
    policy = QwenVlaPolicy.__new__(QwenVlaPolicy)
    cfg = load_data_config(data_config)
    policy.data_config = cfg
    policy.data_config_name = data_config
    policy.device = "cpu"
    policy.denoising_steps = 4
    policy._language_key = "task"
    policy._default_instruction = None
    policy._local_model = None
    policy._mode = "service"
    policy._flatten_to_joints = False
    policy._robot_state_keys = []
    obs_map = kwargs.get("observation_mapping")
    act_map = kwargs.get("action_mapping")
    policy._obs_mapping = (
        _parse_observation_mapping(obs_map) if obs_map else _auto_infer_observation_mapping(cfg, "task")
    )
    policy._action_mapping = _parse_action_mapping(act_map) if act_map else _auto_infer_action_mapping(cfg)
    policy._client = _FakeClient(action_response)
    return policy


class TestMappingParsers:
    def test_parse_observation_mapping(self):
        m = _parse_observation_mapping({"front": "video.front", "joints": "state.single_arm"})
        assert m.video == {"front": "front"}
        assert m.state == {"joints": "single_arm"}

    def test_parse_observation_bad_prefix(self):
        with pytest.raises(ValueError, match="video.' or 'state."):
            _parse_observation_mapping({"x": "action.foo"})

    def test_parse_action_mapping_order_preserved(self):
        m = _parse_action_mapping({"action.b": "rb", "action.a": "ra"})
        assert list(m.actions.keys()) == ["b", "a"]

    def test_auto_infer_from_config(self):
        cfg = load_data_config("so100")
        obs = _auto_infer_observation_mapping(cfg, "task")
        act = _auto_infer_action_mapping(cfg)
        assert "webcam" in obs.video
        assert obs.state == {"single_arm": "single_arm", "gripper": "gripper"}
        assert act.actions == {"single_arm": "single_arm", "gripper": "gripper"}


class TestObservationBuild:
    def test_video_uses_view_tags(self):
        policy = _make_service_policy("so100_dualcam", {"action.single_arm": np.zeros((16, 6))})
        obs = {
            "front": np.zeros((224, 224, 3), np.uint8),
            "wrist": np.zeros((224, 224, 3), np.uint8),
            "single_arm": np.zeros(6, np.float32),
            "gripper": np.zeros(1, np.float32),
        }
        built = policy._build_observation(obs, "pick up the cube")
        # View tags become the model video keys
        assert set(built["video"].keys()) == {"ego", "cam_right_wrist"}
        # Video promoted to (B, T, H, W, C)
        assert built["video"]["ego"].shape == (1, 1, 224, 224, 3)
        # State promoted to (B, T, D)
        assert built["state"]["single_arm"].shape == (1, 1, 6)
        # Language carries the embodiment prompt
        prompt = built["language"]["task"][0][0]
        assert "so100" in prompt and "pick up the cube" in prompt

    def test_missing_camera_warns_not_raises(self, caplog):
        policy = _make_service_policy("so100", {"action.single_arm": np.zeros((16, 6))})
        built = policy._build_observation({"single_arm": np.zeros(6)}, "do x")
        assert built["video"] == {}


class TestActionUnpack:
    def test_unpack_bare_keys(self):
        resp = {
            "action.single_arm": np.arange(16 * 6).reshape(16, 6).astype(np.float32),
            "action.gripper": np.ones((16, 1), np.float32),
        }
        policy = _make_service_policy("so100", resp)
        actions = policy.get_actions_sync({"single_arm": np.zeros(6), "gripper": np.zeros(1)}, "x")
        assert len(actions) == 16
        assert "single_arm" in actions[0]
        assert "gripper" in actions[0]
        # First timestep, first 6 channels = 0..5
        assert actions[0]["single_arm"] == [0, 1, 2, 3, 4, 5]

    def test_unpack_unified_chunk_split(self):
        # Single "action" tensor split across two families evenly.
        resp = {"action": np.tile(np.arange(8, dtype=np.float32), (16, 1))}  # (16, 8)
        policy = _make_service_policy("aloha_bimanual", resp)
        actions = policy.get_actions_sync(
            {
                "left_arm": np.zeros(2),
                "right_arm": np.zeros(2),
                "left_gripper": np.zeros(1),
                "right_gripper": np.zeros(1),
            },
            "fold",
        )
        assert len(actions) == 16
        # 4 families, 8 channels => width 2 each
        assert "left_arm" in actions[0]
        assert actions[0]["left_arm"] == [0, 1]
        assert actions[0]["right_arm"] == [2, 3]

    def test_unpack_empty(self):
        policy = _make_service_policy("so100", {})
        actions = policy.get_actions_sync({"single_arm": np.zeros(6)}, "x")
        assert actions == []

    def test_unmapped_keys_prefixed(self):
        resp = {
            "action.single_arm": np.zeros((4, 6), np.float32),
            "action.gripper": np.zeros((4, 1), np.float32),
            "action.mystery": np.ones((4, 2), np.float32),
        }
        policy = _make_service_policy("so100", resp)
        actions = policy.get_actions_sync({"single_arm": np.zeros(6), "gripper": np.zeros(1)}, "x")
        assert "unmapped.mystery" in actions[0]


class TestReset:
    def test_service_reset_forwards_seed(self):
        policy = _make_service_policy("so100", {})
        policy.reset(seed=42)
        assert policy._client.reset_calls == [{"options": {"seed": 42}}]

    def test_service_reset_no_seed(self):
        policy = _make_service_policy("so100", {})
        policy.reset()
        assert policy._client.reset_calls == [None]


class TestPolicyMeta:
    def test_provider_name(self):
        policy = _make_service_policy("so100", {})
        assert policy.provider_name == "qwen_vla"

    def test_requires_images(self):
        policy = _make_service_policy("so100", {})
        assert policy.requires_images is True

    def test_get_actions_requires_instruction(self):
        policy = _make_service_policy("so100", {})
        with pytest.raises(ValueError, match="instruction"):
            policy.get_actions_sync({"single_arm": np.zeros(6)}, "")


class TestFlattenToJoints:
    """The sim-compatibility path: grouped action vectors -> per-joint scalars.

    Surfaced by actually driving the MuJoCo sim, whose so100 exposes 6 per-joint
    actuators (Rotation..Jaw) and rejects grouped vector ctrl values.
    """

    def test_set_robot_state_keys_enables_flatten(self):
        policy = _make_service_policy("so100", {})
        assert policy._flatten_to_joints is False
        policy.set_robot_state_keys(["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        assert policy._flatten_to_joints is True
        assert len(policy._robot_state_keys) == 6

    def test_flatten_grouped_to_scalars(self):
        # so100 model emits single_arm(6) + gripper(1) = 7 channels.
        resp = {
            "action.single_arm": np.tile(np.arange(6, dtype=np.float32), (4, 1)),
            "action.gripper": np.full((4, 1), 9.0, np.float32),
        }
        policy = _make_service_policy("so100", resp)
        joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        policy.set_robot_state_keys(joints)
        actions = policy.get_actions_sync({"single_arm": np.zeros(6), "gripper": np.zeros(1)}, "x")
        # Each step must now be 6 scalar (per-joint) values, NOT grouped vectors.
        step = actions[0]
        assert set(step.keys()) == set(joints)
        for v in step.values():
            assert isinstance(v, float)
        # single_arm (0..5) maps to the first 6 joints in order.
        assert step["Rotation"] == 0.0
        assert step["Jaw"] == 5.0  # 6th channel = single_arm[5] (gripper is the 7th, dropped at 6 joints)

    def test_flatten_takes_valid_prefix_of_padded_chunk(self):
        # A unified K=32 chunk (zero-padded) -> first 6 valid channels to 6 joints.
        resp = {"action": np.tile(np.arange(32, dtype=np.float32), (4, 1))}
        policy = _make_service_policy("so100", resp)
        joints = ["j0", "j1", "j2", "j3", "j4", "j5"]
        policy.set_robot_state_keys(joints)
        actions = policy.get_actions_sync({"single_arm": np.zeros(6), "gripper": np.zeros(1)}, "x")
        assert sorted(actions[0].keys()) == sorted(joints)
        assert all(isinstance(v, float) for v in actions[0].values())

    def test_sim_obs_bridge_uses_default_camera_and_joint_state(self):
        # When obs is keyed by joint names + a 'default' RGB cam (the MuJoCo
        # schema), _build_observation must auto-bridge instead of warning.
        resp = {"action.single_arm": np.zeros((4, 6), np.float32), "action.gripper": np.zeros((4, 1), np.float32)}
        policy = _make_service_policy("so100", resp)
        joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        policy.set_robot_state_keys(joints)
        sim_obs = {**{j: 0.1 * i for i, j in enumerate(joints)}, "default": np.zeros((48, 64, 3), np.uint8)}
        built = policy._build_observation(sim_obs, "pick up the cube")
        # The 'default' camera became the embodiment's primary view tag.
        assert len(built["video"]) == 1
        assert next(iter(built["video"].values())).shape == (1, 1, 48, 64, 3)
        # The per-joint scalars were assembled into one state vector.
        assert len(built["state"]) == 1
        assert next(iter(built["state"].values())).shape[-1] == 6
