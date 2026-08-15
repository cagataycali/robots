"""Unit tests for KimodoPolicy using an injected motion agent.

These tests exercise the frame -> action-dict mapping, prompt-change
resampling, cursor advancement, and end-of-buffer hold semantics WITHOUT
requiring torch/diffusers/CUDA/checkpoints. The real diffusers-backed sampler
is covered by an integration test gated on the ``[kimodo]`` extra + HF access.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from strands_robots.policies.kimodo import (
    KIMODO_G1_JOINTS,
    KimodoConfig,
    KimodoPolicy,
)


class _StubAgent:
    """Deterministic stub returning a linear ramp per joint."""

    def __init__(self, num_joints: int = 29) -> None:
        self.calls = 0
        self.num_joints = num_joints

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.calls += 1
        out = np.zeros((num_frames, 7 + self.num_joints), dtype=np.float32)
        out[:, 6] = 1.0  # identity quaternion
        for t in range(num_frames):
            out[t, 7:] = np.linspace(0.0, 1.0, self.num_joints) * (t / max(num_frames - 1, 1))
        return out


def _make_policy(**cfg_kwargs):
    cfg = KimodoConfig(**cfg_kwargs)
    stub = _StubAgent()
    return KimodoPolicy(config=cfg, motion_agent=stub), stub


def _first_action(policy, instruction="walk forward", **kw):
    """Convenience: run the async get_actions and return the single dict."""
    return asyncio.run(policy.get_actions({}, instruction, **kw))[0]


def test_get_actions_returns_all_g1_joints():
    policy, _ = _make_policy(num_frames=10, native_fps=30, tracker_fps=30)
    action = _first_action(policy)
    assert set(action.keys()) == set(KIMODO_G1_JOINTS)
    assert all(isinstance(v, float) for v in action.values())


def test_get_actions_returns_single_element_list():
    policy, _ = _make_policy(num_frames=10, native_fps=30, tracker_fps=30)
    result = asyncio.run(policy.get_actions({}, "walk"))
    assert isinstance(result, list) and len(result) == 1


def test_prompt_change_triggers_resample():
    policy, stub = _make_policy(num_frames=8, native_fps=30, tracker_fps=30)
    _first_action(policy, "walk")
    _first_action(policy, "walk")
    assert stub.calls == 1
    _first_action(policy, "run")
    assert stub.calls == 2


def test_cursor_advances_frame_by_frame():
    policy, _ = _make_policy(num_frames=4, native_fps=30, tracker_fps=30)
    first = _first_action(policy)
    second = _first_action(policy)
    changed = sum(1 for k in KIMODO_G1_JOINTS if first[k] != second[k])
    assert changed >= len(KIMODO_G1_JOINTS) - 1  # index 0 stays 0


def test_end_of_buffer_holds_last_frame():
    policy, _ = _make_policy(num_frames=3, native_fps=30, tracker_fps=30)
    for _ in range(3):
        _first_action(policy)
    a = _first_action(policy)
    b = _first_action(policy)
    assert a == b  # last frame held


def test_empty_prompt_raises():
    policy, _ = _make_policy()
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(policy.get_actions({}, ""))


def test_reset_rewinds_cursor():
    policy, _ = _make_policy(num_frames=4, native_fps=30, tracker_fps=30)
    first = _first_action(policy)
    _first_action(policy)
    policy.reset()
    again = _first_action(policy)
    assert again == first


def test_slerp_upsample_widens_buffer():
    policy, _ = _make_policy(num_frames=30, native_fps=30, tracker_fps=50)
    _first_action(policy)
    assert policy._motion_buffer is not None
    assert policy._motion_buffer.shape[0] > 30


def test_requires_images_false():
    policy, _ = _make_policy()
    assert policy.requires_images is False


def test_provider_name():
    policy, _ = _make_policy()
    assert policy.provider_name == "kimodo"


def test_set_robot_state_keys_ok():
    policy, _ = _make_policy()
    policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))
    assert policy._joint_names == KIMODO_G1_JOINTS


def test_set_robot_state_keys_missing_joint_raises():
    policy, _ = _make_policy()
    bad = list(KIMODO_G1_JOINTS)[:-1]  # missing last joint
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        policy.set_robot_state_keys(bad)


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="diffusion_steps"):
        KimodoConfig(diffusion_steps=0)
    with pytest.raises(ValueError, match="guidance_scale"):
        KimodoConfig(guidance_scale=-1)
    with pytest.raises(ValueError, match="num_frames"):
        KimodoConfig(num_frames=1000)
    with pytest.raises(ValueError, match="dtype"):
        KimodoConfig(dtype="int8")


def test_config_from_dict_drops_unknown_keys():
    cfg = KimodoConfig.from_dict({"diffusion_steps": 50, "unknown": "x"})
    assert cfg.diffusion_steps == 50
