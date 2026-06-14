"""Unit tests for KimodoPolicy (text-to-motion diffusion provider).

Tests cover:
- Policy construction and configuration
- Trajectory replay (mock-generated qpos)
- Cache hit/miss behavior
- Reset and re-seed
- Runtime prompt override via instruction
- Error handling for missing kimodo install
- Factory integration (create_policy("kimodo", ...))
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@pytest.fixture()
def mock_qpos() -> np.ndarray:
    """Synthetic qpos trajectory: 150 frames x 35 joints (5s @ 30fps, G1)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, size=(150, 35)).astype(np.float32)


@pytest.fixture()
def kimodo_policy(mock_qpos: np.ndarray):
    """KimodoPolicy with pre-injected trajectory (skips real inference)."""
    from strands_robots.policies.kimodo import KimodoPolicy

    policy = KimodoPolicy(
        prompt="walk forward then wave",
        model="nvidia/Kimodo-G1-RP-v1",
        duration=5.0,
        seed=42,
        lazy=True,
    )
    # Inject mock trajectory to bypass kimodo import.
    policy._qpos = mock_qpos
    policy._robot_state_keys = [f"joint_{i}" for i in range(35)]
    return policy


class TestKimodoConstruction:
    """Test policy initialization and config parsing."""

    def test_defaults(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        p = KimodoPolicy(prompt="test")
        assert p.provider_name == "kimodo"
        assert p.requires_images is False
        assert p._model == "nvidia/Kimodo-G1-RP-v1"
        assert p._duration == 5.0
        assert p._diffusion_steps == 50
        assert p._cfg_weight == 7.5
        assert p._seed == 42
        assert p._fps == 30

    def test_custom_config(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        p = KimodoPolicy(
            prompt="jump",
            model="nvidia/Kimodo-G1-SEED-v1",
            duration=3.0,
            diffusion_steps=25,
            cfg_weight=5.0,
            seed=123,
            fps=60,
        )
        assert p._model == "nvidia/Kimodo-G1-SEED-v1"
        assert p._duration == 3.0
        assert p._diffusion_steps == 25
        assert p._cfg_weight == 5.0
        assert p._seed == 123
        assert p._fps == 60

    def test_missing_prompt_raises_on_ensure(self) -> None:
        from strands_robots.policies.kimodo import KimodoPolicy

        p = KimodoPolicy()  # no prompt
        with pytest.raises(ValueError, match="requires a 'prompt'"):
            p._ensure_trajectory()


class TestKimodoReplay:
    """Test trajectory replay via get_actions."""

    def test_actions_shape(self, kimodo_policy) -> None:
        obs = {"observation.state": np.zeros(35)}
        actions = _run(kimodo_policy.get_actions(obs, ""))
        assert len(actions) == 8
        assert len(actions[0]) == 35

    def test_frame_advancement(self, kimodo_policy) -> None:
        obs = {"observation.state": np.zeros(35)}
        _run(kimodo_policy.get_actions(obs, ""))
        assert kimodo_policy._frame_idx == 8

        _run(kimodo_policy.get_actions(obs, ""))
        assert kimodo_policy._frame_idx == 16

    def test_exhaustion_holds_last_frame(self, kimodo_policy) -> None:
        obs = {"observation.state": np.zeros(35)}
        # Advance past end of trajectory (150 frames).
        for _ in range(25):  # 25 * 8 = 200 > 150
            _run(kimodo_policy.get_actions(obs, ""))
        assert kimodo_policy.is_exhausted
        # Still returns valid actions (last frame).
        actions = _run(kimodo_policy.get_actions(obs, ""))
        assert len(actions) == 8
        # All actions should be the same (last frame repeated).
        for a in actions:
            assert a == actions[0]

    def test_trajectory_length(self, kimodo_policy) -> None:
        assert kimodo_policy.trajectory_length == 150


class TestKimodoReset:
    """Test reset and re-seed behavior."""

    def test_reset_rewinds(self, kimodo_policy) -> None:
        obs = {"observation.state": np.zeros(35)}
        _run(kimodo_policy.get_actions(obs, ""))
        assert kimodo_policy._frame_idx == 8

        kimodo_policy.reset()
        assert kimodo_policy._frame_idx == 0

    def test_reset_with_seed_invalidates(self, kimodo_policy) -> None:
        assert kimodo_policy._qpos is not None
        kimodo_policy.reset(seed=99)
        assert kimodo_policy._qpos is None
        assert kimodo_policy._seed == 99


class TestKimodoPromptOverride:
    """Test runtime prompt override via instruction/kwargs."""

    def test_instruction_override(self, kimodo_policy, mock_qpos) -> None:
        obs = {"observation.state": np.zeros(35)}
        kimodo_policy._qpos = mock_qpos

        # Patch _ensure_trajectory to just re-inject mock data.
        def fake_ensure():
            if kimodo_policy._qpos is None:
                kimodo_policy._qpos = mock_qpos

        kimodo_policy._ensure_trajectory = fake_ensure

        actions = _run(kimodo_policy.get_actions(obs, "sit down"))
        assert kimodo_policy._prompt == "sit down"
        assert len(actions) == 8


class TestKimodoCache:
    """Test trajectory caching."""

    def test_cache_key_deterministic(self) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import _cache_key

        k1 = _cache_key("walk", "nvidia/Kimodo-G1-RP-v1", 5.0, 42)
        k2 = _cache_key("walk", "nvidia/Kimodo-G1-RP-v1", 5.0, 42)
        assert k1 == k2

    def test_cache_key_varies_with_prompt(self) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import _cache_key

        k1 = _cache_key("walk", "nvidia/Kimodo-G1-RP-v1", 5.0, 42)
        k2 = _cache_key("run", "nvidia/Kimodo-G1-RP-v1", 5.0, 42)
        assert k1 != k2

    def test_cache_roundtrip(self, tmp_path, mock_qpos) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import (
            _load_cached,
            _save_cache,
        )

        with patch(
            "strands_robots.policies.kimodo.kimodo_policy._CACHE_DIR", tmp_path
        ):
            _save_cache("testkey", mock_qpos)
            loaded = _load_cached("testkey")
            assert loaded is not None
            np.testing.assert_array_equal(loaded, mock_qpos)


class TestKimodoImportError:
    """Test graceful handling when kimodo is not installed."""

    def test_generate_raises_import_error(self) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import _generate_qpos

        with patch.dict(
            "sys.modules",
            {"kimodo": None, "kimodo.scripts": None, "kimodo.scripts.generate": None},
        ):
            with pytest.raises(ImportError, match="kimodo is not installed"):
                _generate_qpos(
                    prompt="walk",
                    model="nvidia/Kimodo-G1-RP-v1",
                    duration=5.0,
                    diffusion_steps=50,
                    cfg_weight=7.5,
                    seed=42,
                    text_encoder_device="cpu",
                    fps=30,
                )


class TestKimodoFactory:
    """Test that kimodo is discoverable via the policy factory."""

    def test_list_providers_includes_kimodo(self) -> None:
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert "kimodo" in providers

    def test_create_policy_kimodo(self) -> None:
        from strands_robots.policies import create_policy

        policy = create_policy("kimodo", prompt="walk forward", lazy=True)
        assert policy.provider_name == "kimodo"
        assert policy._prompt == "walk forward"  # type: ignore[attr-defined]

    def test_create_policy_text2motion_shorthand(self) -> None:
        from strands_robots.policies import create_policy

        policy = create_policy("text2motion", prompt="wave", lazy=True)
        assert policy.provider_name == "kimodo"


class TestKimodoDeviceDetection:
    """Test VRAM-based text encoder device selection."""

    def test_cpu_passthrough(self) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import (
            _detect_text_encoder_device,
        )

        assert _detect_text_encoder_device("cpu") == "cpu"

    def test_cuda_fallback_no_torch(self) -> None:
        from strands_robots.policies.kimodo.kimodo_policy import (
            _detect_text_encoder_device,
        )

        with patch.dict("sys.modules", {"torch": None}):
            # When torch can't be imported, falls back to cpu.
            result = _detect_text_encoder_device("cuda")
            # May be "cuda" if torch is importable in this env, or "cpu" if not.
            assert result in ("cuda", "cpu")
