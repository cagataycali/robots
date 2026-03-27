"""Tests for Gr00tPolicy — unit tests WITHOUT Isaac-GR00T installed."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed — pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed — pip install 'strands-robots[groot-service]'")

# All tests in this file require groot-service extras
pytestmark = pytest.mark.skipif(
    not msgpack or not zmq,
    reason="groot-service extras not installed",
)

from strands_robots.policies.groot import DATA_CONFIG_MAP, ActionMapping, Gr00tPolicy, ObservationMapping  # noqa: E402
from strands_robots.policies.groot.data_config import Gr00tDataConfig  # noqa: E402
from strands_robots.policies.groot.policy import (  # noqa: E402
    _auto_infer_action_mapping,
    _auto_infer_observation_mapping,
    _detect_groot_version,
    _parse_action_mapping,
    _parse_observation_mapping,
    _reference_video_shape,
    _to_state_batch,
    _to_video_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_DOF = {
    "single_arm": 5,
    "gripper": 1,
    "webcam": None,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
    "ego_view_bg_crop_pad_res256_freq20": None,
}


def _mock_mmc(video_keys=None, state_keys=None, action_keys=None, language_keys=None):
    configs = {}
    for name, keys in [
        ("video", video_keys or ["webcam"]),
        ("state", state_keys or ["single_arm", "gripper"]),
        ("action", action_keys or ["single_arm", "gripper"]),
        ("language", language_keys or ["annotation.human.task_description"]),
    ]:
        mc = MagicMock()
        mc.modality_keys = keys
        configs[name] = mc
    return configs


SO100_MMC = _mock_mmc()
GR1_MMC = _mock_mmc(
    video_keys=["ego_view_bg_crop_pad_res256_freq20"],
    state_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    action_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    language_keys=["task"],
)


def _make_policy(data_config="so100", version="n1.6", obs_mapping=None, action_mapping=None, mmc=None):
    p = Gr00tPolicy.__new__(Gr00tPolicy)
    p.data_config = DATA_CONFIG_MAP[data_config]
    p.data_config_name = data_config
    p._mode = "local"
    p._groot_version = version
    p._strict = False
    p._client = None
    p._local_policy = MagicMock()
    p._raw_obs_mapping = None
    p._raw_action_mapping = None
    p._language_key_override = None

    mc = mmc or SO100_MMC
    p._local_policy.modality_configs = mc  # Direct N16Policy
    p._local_policy.policy.modality_configs = mc  # Wrapped
    p._local_policy.modality_config = mc  # N1.5

    # Simulate discovered DOF
    p._model_state_dof = {k: _KNOWN_DOF[k] for k in mc["state"].modality_keys if _KNOWN_DOF.get(k) is not None}
    p._obs_mapping = obs_mapping
    p._action_mapping = action_mapping
    return p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_service_mode(self):
        p = Gr00tPolicy()
        assert p._mode == "service" and p._client is not None and p.provider_name == "groot"

    def test_config_name(self):
        assert Gr00tPolicy(data_config="so100").data_config_name == "so100"

    def test_config_object(self):
        cfg = Gr00tDataConfig(
            name="t",
            video_keys=["video.c"],
            state_keys=["state.a"],
            action_keys=["action.a"],
            language_keys=["annotation.human.task_description"],
        )
        assert Gr00tPolicy(data_config=cfg).data_config is cfg

    def test_strict(self):
        assert Gr00tPolicy()._strict is False
        assert Gr00tPolicy(strict=True)._strict is True

    def test_api_token(self):
        assert Gr00tPolicy(api_token="t")._client.api_token == "t"

    def test_api_token_from_env(self, monkeypatch):
        """GROOT_API_TOKEN env var is used when api_token param is None."""
        monkeypatch.setenv("GROOT_API_TOKEN", "env-secret")
        p = Gr00tPolicy()
        assert p._client.api_token == "env-secret"

    def test_api_token_param_overrides_env(self, monkeypatch):
        """Explicit api_token param takes precedence over env var."""
        monkeypatch.setenv("GROOT_API_TOKEN", "env-secret")
        p = Gr00tPolicy(api_token="explicit")
        assert p._client.api_token == "explicit"

    def test_api_token_none_without_env(self, monkeypatch):
        """When no env var and no param, api_token is None."""
        monkeypatch.delenv("GROOT_API_TOKEN", raising=False)
        p = Gr00tPolicy()
        assert p._client.api_token is None

    def test_unknown_config(self):
        with pytest.raises(ValueError):
            Gr00tPolicy(data_config="nope")

    def test_no_isaac(self):
        p = _make_policy()
        p._groot_version = None
        with pytest.raises(ImportError):
            p._load_local_policy("/f", "NEW_EMBODIMENT", "cpu")

    def test_all_configs(self):
        for name in DATA_CONFIG_MAP:
            assert Gr00tPolicy(data_config=name)._mode == "service"

    def test_no_denoising_steps_param(self):
        """denoising_steps was removed from __init__ — kwargs swallows it."""
        p = Gr00tPolicy(denoising_steps=8)
        assert p._mode == "service"  # no error, just ignored via **kwargs

    def test_set_robot_state_keys_is_noop(self):
        """set_robot_state_keys is a documented no-op."""
        p = Gr00tPolicy()
        p.set_robot_state_keys(["a", "b"])  # should not raise


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersion:
    def test_cached(self):
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "cached"
        try:
            assert _detect_groot_version() == "cached"
        finally:
            pm._GROOT_VERSION = orig

    def test_force_redetect(self):
        """force=True should bypass cache and re-detect."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "stale"
        try:
            with patch("importlib.util.find_spec", return_value=None):
                result = _detect_groot_version(force=True)
            # Should have re-detected (None since find_spec returns None)
            assert result is None
            assert pm._GROOT_VERSION is None
        finally:
            pm._GROOT_VERSION = orig

    def test_force_false_uses_cache(self):
        """force=False (default) should use cached value."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "n1.6"
        try:
            assert _detect_groot_version(force=False) == "n1.6"
        finally:
            pm._GROOT_VERSION = orig


# ---------------------------------------------------------------------------
# ObservationMapping
# ---------------------------------------------------------------------------


class TestObsMapping:
    def test_empty(self):
        m = ObservationMapping()
        assert m.video == {} and m.state == {} and m.language_key == "task"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ObservationMapping().language_key = "x"

    def test_validate_ok(self):
        ObservationMapping(
            video={"c": "ego_view_bg_crop_pad_res256_freq20"},
            state={"j": "left_arm"},
            language_key="task",
        ).validate(GR1_MMC)

    def test_bad_video(self):
        with pytest.raises(ValueError, match="video"):
            ObservationMapping(video={"c": "nope"}).validate(GR1_MMC)

    def test_bad_state(self):
        with pytest.raises(ValueError, match="state"):
            ObservationMapping(state={"j": "nope"}, language_key="task").validate(GR1_MMC)

    def test_bad_lang(self):
        with pytest.raises(ValueError, match="language"):
            ObservationMapping(language_key="nope").validate(GR1_MMC)


# ---------------------------------------------------------------------------
# ActionMapping
# ---------------------------------------------------------------------------


class TestActionMapping:
    def test_empty(self):
        assert ActionMapping().actions == {}

    def test_validate_ok(self):
        ActionMapping(actions={"left_arm": "j"}).validate(GR1_MMC)

    def test_bad(self):
        with pytest.raises(ValueError):
            ActionMapping(actions={"nope": "j"}).validate(GR1_MMC)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_obs(self):
        m = _parse_observation_mapping(
            {"cam": "video.ego_view_bg_crop_pad_res256_freq20", "j": "state.left_arm"},
            GR1_MMC,
        )
        assert m.video == {"cam": "ego_view_bg_crop_pad_res256_freq20"}
        assert m.state == {"j": "left_arm"}
        assert m.language_key == "task"

    def test_obs_default_lang_without_mmc(self):
        """Without modality_configs, language_key defaults to 'task'."""
        m = _parse_observation_mapping({"c": "video.cam"})
        assert m.language_key == "task"

    def test_bad_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            _parse_observation_mapping({"x": "audio.mic"})

    def test_action(self):
        m = _parse_action_mapping({"action.left_arm": "j", "action.left_hand": "g"})
        assert m.actions == {"left_arm": "j", "left_hand": "g"}


# ---------------------------------------------------------------------------
# Auto-inference
# ---------------------------------------------------------------------------


class TestAutoInfer:
    def test_obs_exact(self):
        m = _auto_infer_observation_mapping(DATA_CONFIG_MAP["so100"], SO100_MMC)
        assert m.video == {"webcam": "webcam"}
        assert m.state["single_arm"] == "single_arm"

    def test_obs_positional(self):
        m = _auto_infer_observation_mapping(DATA_CONFIG_MAP["so100"], GR1_MMC)
        assert m.video.get("webcam") == "ego_view_bg_crop_pad_res256_freq20"

    def test_action_exact(self):
        m = _auto_infer_action_mapping(DATA_CONFIG_MAP["so100"], SO100_MMC)
        assert m.actions["single_arm"] == "single_arm"


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


class TestShapes:
    def test_video_3d(self):
        assert _to_video_batch(np.zeros((64, 64, 3))).shape == (1, 1, 64, 64, 3)

    def test_video_4d(self):
        assert _to_video_batch(np.zeros((3, 64, 64, 3))).shape == (1, 3, 64, 64, 3)

    def test_video_5d_passthrough(self):
        assert _to_video_batch(np.zeros((1, 1, 64, 64, 3))).shape == (1, 1, 64, 64, 3)

    def test_state_1d(self):
        r = _to_state_batch(np.zeros(5))
        assert r.shape == (1, 1, 5) and r.dtype == np.float32

    def test_state_2d(self):
        assert _to_state_batch(np.zeros((3, 5))).shape == (1, 3, 5)

    def test_state_from_list(self):
        r = _to_state_batch([1.0, 2.0, 3.0])
        assert r.shape == (1, 1, 3) and r.dtype == np.float32

    def test_ref_from_mapped_video_keys(self):
        """Should only look at keys in the video_keys set."""
        obs = {
            "cam": np.zeros((128, 128, 3)),
            "state_3d": np.zeros((10, 10, 3)),  # 3D state — should NOT match
        }
        assert _reference_video_shape(obs, video_keys={"cam"}) == (128, 128, 3)

    def test_ref_ignores_non_video_keys(self):
        """State array with shape[-1]==3 should not be picked when video_keys given."""
        obs = {"state_vals": np.zeros((10, 10, 3))}
        assert _reference_video_shape(obs, video_keys={"cam"}) == (256, 256, 3)

    def test_ref_default(self):
        assert _reference_video_shape({"j": np.zeros(5)}) == (256, 256, 3)

    def test_ref_legacy_heuristic_when_no_video_keys(self):
        """Without video_keys, falls back to heuristic scan."""
        obs = {"c": np.zeros((128, 128, 3))}
        assert _reference_video_shape(obs, video_keys=None) == (128, 128, 3)


# ---------------------------------------------------------------------------
# _prepare_observation — nested dict format
# ---------------------------------------------------------------------------


class TestPrepareObs:
    def test_nested_structure(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm", "g": "left_hand"},
                language_key="task",
            ),
            mmc=GR1_MMC,
        )
        b = p._prepare_observation(
            {
                "cam": np.zeros((256, 256, 3), dtype=np.uint8),
                "j": np.zeros(7),
                "g": np.zeros(6),
            },
            "pick cube",
        )
        assert "video" in b and "state" in b and "language" in b
        assert b["video"]["ego_view_bg_crop_pad_res256_freq20"].shape == (1, 1, 256, 256, 3)
        assert b["state"]["left_arm"].shape == (1, 1, 7)
        assert b["language"]["task"] == [["pick cube"]]

    def test_zero_fills_unmapped_state(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm"},
                language_key="task",
            ),
            mmc=GR1_MMC,
        )
        b = p._prepare_observation({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "j": np.zeros(7)}, "t")
        assert "right_arm" in b["state"]
        np.testing.assert_array_equal(b["state"]["right_arm"][0, 0], np.zeros(7))
        assert "waist" in b["state"]
        np.testing.assert_array_equal(b["state"]["waist"][0, 0], np.zeros(3))

    def test_skips_zero_fill_unknown_dof(self):
        """When DOF is not discoverable, the key should be skipped."""
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"}, state={"arm": "single_arm"}, language_key="annotation.human.task_description"
            ),
        )
        # Clear DOF for gripper — simulate unknown
        p._model_state_dof = {"single_arm": 5}
        b = p._prepare_observation({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5)}, "t")
        # gripper DOF unknown → should NOT be in state dict
        assert "gripper" not in b["state"]
        assert "single_arm" in b["state"]


# ---------------------------------------------------------------------------
# _unpack_actions
# ---------------------------------------------------------------------------


class TestUnpackActions:
    def test_maps(self):
        p = _make_policy(action_mapping=ActionMapping(actions={"left_arm": "j", "left_hand": "g"}), mmc=GR1_MMC)
        acts = p._unpack_actions({"left_arm": np.ones((1, 16, 7)), "left_hand": np.ones((1, 16, 6))})
        assert len(acts) == 16 and acts[0]["j"].shape == (7,) and acts[0]["g"].shape == (6,)

    def test_unmapped(self):
        p = _make_policy(action_mapping=ActionMapping(actions={"left_arm": "j"}), mmc=GR1_MMC)
        acts = p._unpack_actions({"left_arm": np.ones((1, 4, 7)), "waist": np.ones((1, 4, 3))})
        assert "unmapped.waist" in acts[0]

    def test_empty(self):
        assert _make_policy(action_mapping=ActionMapping())._unpack_actions({}) == []


# ---------------------------------------------------------------------------
# Full local flow
# ---------------------------------------------------------------------------


class TestLocalFlow:
    def test_n16(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm", "g": "left_hand"},
                language_key="task",
            ),
            action_mapping=ActionMapping(actions={"left_arm": "j", "left_hand": "g"}),
            mmc=GR1_MMC,
        )
        p._local_policy.get_action.return_value = (
            {"left_arm": np.ones((1, 16, 7)), "left_hand": np.ones((1, 16, 6))},
            {},
        )
        acts = p._local_get_actions(
            {"cam": np.zeros((256, 256, 3), dtype=np.uint8), "j": np.zeros(7), "g": np.zeros(6)},
            "pick",
        )
        assert len(acts) == 16
        call = p._local_policy.get_action.call_args[0][0]
        assert "video" in call and "state" in call and "language" in call

    def test_n15(self):
        p = _make_policy(
            version="n1.5",
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"},
                state={"arm": "single_arm", "grip": "gripper"},
                language_key="annotation.human.task_description",
            ),
            action_mapping=ActionMapping(actions={"single_arm": "arm", "gripper": "grip"}),
        )
        p._local_policy.get_action.return_value = {
            "single_arm": np.ones((1, 16, 5)),
            "gripper": np.ones((1, 16, 1)),
        }
        acts = p._local_get_actions(
            {"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5), "grip": np.array([0.5])},
            "t",
        )
        assert len(acts) == 16 and "arm" in acts[0]

    def test_bad_version(self):
        p = _make_policy(obs_mapping=ObservationMapping(), action_mapping=ActionMapping())
        p._groot_version = "n9.9"
        with pytest.raises(RuntimeError):
            p._local_get_actions({}, "t")


# ---------------------------------------------------------------------------
# get_actions routing
# ---------------------------------------------------------------------------


class TestGetActions:
    def test_local(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"},
                state={"arm": "single_arm"},
                language_key="annotation.human.task_description",
            ),
            action_mapping=ActionMapping(actions={"single_arm": "arm"}),
        )
        p._local_policy.get_action.return_value = ({"single_arm": np.ones((1, 4, 5))}, {})
        acts = asyncio.run(p.get_actions({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5)}, "t"))
        assert len(acts) == 4

    def test_service(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._client.get_action = MagicMock(
            return_value={
                "action.single_arm": np.ones((1, 16, 5)),
                "action.gripper": np.ones((1, 16, 1)),
            }
        )
        acts = asyncio.run(p.get_actions({"webcam": np.zeros((64, 64, 3), dtype=np.uint8)}, "t"))
        assert len(acts) == 16


# ---------------------------------------------------------------------------
# Service observation + action unpack
# ---------------------------------------------------------------------------


class TestServiceObs:
    def test_flat_keys(self):
        p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
        obs = p._build_service_observation({"front": np.zeros((64, 64, 3)), "wrist": np.zeros((64, 64, 3))}, "t")
        assert "video.front" in obs and obs["video.front"].shape[0] == 1

    def test_language(self):
        obs = Gr00tPolicy(data_config="so100")._build_service_observation({}, "t")
        assert "annotation.human.task_description" in obs


class TestServiceUnpackWithMapping:
    """_unpack_service_actions should apply _action_mapping when available."""

    def test_with_mapping(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={"single_arm": "joints", "gripper": "grip"})
        result = p._unpack_service_actions(
            {
                "action.single_arm": np.ones((1, 4, 5)),
                "action.gripper": np.ones((1, 4, 1)),
            }
        )
        assert len(result) == 4
        assert "joints" in result[0]
        assert "grip" in result[0]
        assert "single_arm" not in result[0]

    def test_with_mapping_unmapped_keys(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={"single_arm": "joints"})
        result = p._unpack_service_actions(
            {
                "action.single_arm": np.ones((1, 4, 5)),
                "action.gripper": np.ones((1, 4, 1)),
            }
        )
        assert "joints" in result[0]
        assert "unmapped.gripper" in result[0]

    def test_without_mapping(self):
        """No mapping → bare model keys."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = None
        result = p._unpack_service_actions({"action.single_arm": np.ones((1, 4, 5))})
        assert "single_arm" in result[0]

    def test_empty_mapping(self):
        """Empty mapping → bare model keys."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={})
        result = p._unpack_service_actions({"action.single_arm": np.ones((1, 2, 5))})
        assert "single_arm" in result[0]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all(self):
        import strands_robots.policies.groot as mod

        for name in mod.__all__:
            assert hasattr(mod, name)
