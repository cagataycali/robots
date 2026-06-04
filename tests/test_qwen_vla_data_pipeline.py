"""Unit tests for the Qwen-VLA training data pipeline (adapter + T2A corpus)."""

import numpy as np
import pytest

from strands_robots.policies.qwen_vla import compute_quantile_stats
from strands_robots.training.qwen_vla import get_embodiment_tag
from strands_robots.training.qwen_vla.data import (
    TASK_FAMILIES,
    LanguageActionGenerator,
    LeRobotAdapter,
)


def _so100_adapter(**kw):
    return LeRobotAdapter(
        embodiment=get_embodiment_tag("so100"),
        video_keys=["observation.images.webcam"],
        view_tags={"observation.images.webcam": "ego"},
        state_keys=["observation.state.single_arm", "observation.state.gripper"],
        **{"action_dim": 32, **kw},
    )


class TestLeRobotAdapter:
    def test_adapt_shapes(self):
        adapter = _so100_adapter()
        frame = {
            "observation.images.webcam": np.zeros((224, 224, 3), np.uint8),
            "observation.state.single_arm": np.zeros(6, np.float32),
            "observation.state.gripper": np.zeros(1, np.float32),
        }
        action_chunk = np.ones((16, 7), np.float32)  # H=16, c=7
        sample = adapter.adapt(frame, action_chunk, "pick up the cube")
        assert sample.video["ego"].shape == (224, 224, 3)
        assert sample.state.shape == (7,)  # 6 + 1 concatenated
        assert sample.action.shape == (16, 32)  # padded to K=32
        assert sample.mask.shape == (16, 32)
        assert sample.c == 7
        assert sample.h_task == 16
        # Padding region is zero + masked out
        assert sample.action[:, 7:].sum() == 0.0
        assert sample.mask[:, :7].all()
        assert sample.mask[:, 7:].sum() == 0.0
        assert "so100" in sample.language

    def test_short_horizon_padded_in_time(self):
        adapter = _so100_adapter()
        frame = {
            "observation.images.webcam": np.zeros((8, 8, 3), np.uint8),
            "observation.state.single_arm": np.zeros(6),
            "observation.state.gripper": np.zeros(1),
        }
        # Navigation-style 8-step chunk into an H=16 embodiment.
        sample = adapter.adapt(frame, np.ones((8, 7), np.float32), "go")
        assert sample.action.shape == (16, 32)
        assert sample.h_task == 8
        assert sample.mask[8:, :].sum() == 0.0  # later timesteps masked

    def test_normalization_applied(self):
        rng = np.random.default_rng(0)
        stats = compute_quantile_stats(rng.uniform(-3, 3, (500, 7)).astype(np.float32))
        adapter = _so100_adapter(quantile_stats=stats)
        frame = {
            "observation.images.webcam": np.zeros((8, 8, 3), np.uint8),
            "observation.state.single_arm": np.zeros(6),
            "observation.state.gripper": np.zeros(1),
        }
        sample = adapter.adapt(frame, np.full((16, 7), 100.0, np.float32), "x")
        # Clipped to [-1, 1] by quantile norm before padding.
        assert sample.action[:, :7].max() <= 1.0 + 1e-6

    def test_missing_video_key_raises(self):
        adapter = _so100_adapter()
        with pytest.raises(ValueError, match="video key"):
            adapter.adapt(
                {"observation.state.single_arm": np.zeros(6), "observation.state.gripper": np.zeros(1)},
                np.ones((16, 7)),
                "x",
            )

    def test_horizon_too_long_raises(self):
        adapter = _so100_adapter()
        frame = {
            "observation.images.webcam": np.zeros((8, 8, 3), np.uint8),
            "observation.state.single_arm": np.zeros(6),
            "observation.state.gripper": np.zeros(1),
        }
        with pytest.raises(ValueError, match="exceeds embodiment"):
            adapter.adapt(frame, np.ones((20, 7)), "x")

    def test_bad_action_dim_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            _so100_adapter(action_dim=0)


class TestLanguageActionGenerator:
    def test_generates_n(self):
        g = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7, seed=2)
        ex = g.generate(20)
        assert len(ex) == 20
        for e in ex:
            assert e.family in TASK_FAMILIES
            assert e.action.shape == (16, 7)
            assert "so100" in e.prompt
            assert e.instruction in e.prompt

    def test_family_restriction(self):
        g = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7, seed=2)
        ex = g.generate(10, family="pick")
        assert all(e.family == "pick" for e in ex)
        assert all("pick up" in e.instruction for e in ex)

    def test_deterministic_with_seed(self):
        a = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7, seed=9).generate(5)
        b = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7, seed=9).generate(5)
        assert [e.instruction for e in a] == [e.instruction for e in b]
        for ea, eb in zip(a, b, strict=True):
            np.testing.assert_array_equal(ea.action, eb.action)

    def test_grasp_gripper_closes(self):
        g = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7, seed=3)
        ex = g.generate(1, family="pick")[0]
        # Grasp motion: gripper channel rises from ~0 to ~1.
        assert ex.action[0, -1] <= ex.action[-1, -1]

    def test_unknown_family_raises(self):
        g = LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=7)
        with pytest.raises(ValueError, match="unknown family"):
            g.generate(1, family="teleport")

    def test_bad_channels_raises(self):
        with pytest.raises(ValueError, match="action_channels"):
            LanguageActionGenerator(embodiment=get_embodiment_tag("so100"), action_channels=0)
