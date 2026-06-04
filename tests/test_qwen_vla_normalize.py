"""Unit tests for Qwen-VLA quantile normalization + channel masking."""

import numpy as np
import pytest

from strands_robots.policies.qwen_vla import (
    build_channel_mask,
    compute_quantile_stats,
    normalize,
    pad_to_width,
    unnormalize,
    unpad_from_width,
)


class TestQuantileNorm:
    def test_roundtrip_within_band(self):
        rng = np.random.default_rng(0)
        actions = rng.uniform(-2, 3, size=(1000, 4)).astype(np.float32)
        stats = compute_quantile_stats(actions)
        # Sample points strictly inside the [q_low, q_high] band round-trip.
        sample = np.clip(actions[:10], stats["q_low"] + 1e-3, stats["q_high"] - 1e-3)
        norm = normalize(sample, stats)
        back = unnormalize(norm, stats)
        np.testing.assert_allclose(back, sample, rtol=1e-3, atol=1e-3)

    def test_normalize_range(self):
        actions = np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32)
        stats = compute_quantile_stats(actions)
        norm = normalize(actions, stats)
        assert norm.min() >= -1.0 - 1e-6
        assert norm.max() <= 1.0 + 1e-6

    def test_constant_channel_no_nan(self):
        actions = np.full((50, 2), 0.7, dtype=np.float32)
        stats = compute_quantile_stats(actions)
        norm = normalize(actions, stats)
        assert not np.isnan(norm).any()

    def test_compute_stats_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_quantile_stats(np.zeros(10))

    def test_compute_stats_quantile_order(self):
        with pytest.raises(ValueError, match="q_low"):
            compute_quantile_stats(np.zeros((10, 2)), q_low=0.9, q_high=0.1)

    def test_normalize_dim_mismatch(self):
        stats = {"q_low": np.zeros(3, np.float32), "q_high": np.ones(3, np.float32)}
        with pytest.raises(ValueError, match="channel dim"):
            normalize(np.zeros((5, 4)), stats)


class TestChannelMask:
    def test_shape_and_values(self):
        mask = build_channel_mask(c=7, k=32, h_task=16, h=16)
        assert mask.shape == (16, 32)
        assert mask[:16, :7].all()
        assert mask[:, 7:].sum() == 0.0

    def test_partial_horizon(self):
        mask = build_channel_mask(c=3, k=10, h_task=8, h=16)
        assert mask[:8, :3].all()
        assert mask[8:, :].sum() == 0.0

    def test_c_exceeds_k_raises(self):
        with pytest.raises(ValueError, match="cannot exceed model channel"):
            build_channel_mask(c=33, k=32, h_task=16, h=16)

    def test_htask_exceeds_h_raises(self):
        with pytest.raises(ValueError, match="cannot exceed model horizon"):
            build_channel_mask(c=7, k=32, h_task=20, h=16)

    def test_nonpositive_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_channel_mask(c=0, k=32, h_task=16, h=16)


class TestPadding:
    def test_pad_unpad_roundtrip(self):
        actions = np.arange(16 * 7, dtype=np.float32).reshape(16, 7)
        padded = pad_to_width(actions, k=32)
        assert padded.shape == (16, 32)
        assert padded[:, 7:].sum() == 0.0
        back = unpad_from_width(padded, c=7)
        np.testing.assert_array_equal(back, actions)

    def test_pad_equal_width_passthrough(self):
        actions = np.ones((4, 5), dtype=np.float32)
        padded = pad_to_width(actions, k=5)
        np.testing.assert_array_equal(padded, actions)

    def test_pad_c_exceeds_k_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            pad_to_width(np.zeros((4, 10)), k=5)

    def test_unpad_c_exceeds_width_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            unpad_from_width(np.zeros((4, 5)), c=10)

    def test_pad_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            pad_to_width(np.zeros(10), k=32)
