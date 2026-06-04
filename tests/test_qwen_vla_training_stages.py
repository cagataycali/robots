"""Unit tests for Qwen-VLA training stage builders + PPO numerics.

Validates the torch-free parts: the T2A batch assembly, GAE/PPO math, value
head spec, and that the torch-gated run_* entrypoints fail fast with clear
errors when given no model (before requiring torch).
"""

import numpy as np
import pytest

from strands_robots.training.qwen_vla import (
    CPTConfig,
    RLConfig,
    SFTConfig,
    T2AConfig,
    build_t2a_batch,
    clipped_ppo_objective,
    get_embodiment_tag,
    run_cpt,
    run_rl,
    run_sft,
    run_t2a,
)
from strands_robots.training.qwen_vla.data.language_action import LanguageActionGenerator
from strands_robots.training.qwen_vla.ppo import RolloutBuffer, compute_gae
from strands_robots.training.qwen_vla.ppo.logprob import gaussian_logprob, ppo_ratio
from strands_robots.training.qwen_vla.ppo.value_head import ValueHeadSpec


class TestT2ABatch:
    def test_batch_shapes(self):
        emb = get_embodiment_tag("so100")
        cfg = T2AConfig(batch_size=4, action_dim=32)
        gen = LanguageActionGenerator(embodiment=emb, action_channels=7, seed=0)
        rng = np.random.default_rng(0)
        batch = build_t2a_batch(gen, emb, cfg, action_channels=7, rng=rng)
        h = emb.chunk_size
        assert batch["x_t"].shape == (4, h, 32)
        assert batch["target"].shape == (4, h, 32)
        assert batch["mask"].shape == (4, h, 32)
        assert batch["timesteps"].shape == (4,)
        # Mask: valid 7 channels, padded tail zero
        assert batch["mask"][:, :, :7].all()
        assert batch["mask"][:, :, 7:].sum() == 0.0

    def test_timesteps_in_range(self):
        emb = get_embodiment_tag("so100")
        cfg = T2AConfig(batch_size=8)
        gen = LanguageActionGenerator(embodiment=emb, action_channels=7, seed=0)
        batch = build_t2a_batch(gen, emb, cfg, action_channels=7, rng=np.random.default_rng(1))
        assert batch["timesteps"].min() >= 0.0
        assert batch["timesteps"].max() <= 1.0


class TestGAE:
    def test_zero_reward_zero_advantage(self):
        adv, ret = compute_gae(np.zeros(5), np.zeros(5), np.zeros(5))
        np.testing.assert_allclose(adv, 0.0)
        np.testing.assert_allclose(ret, 0.0)

    def test_single_terminal_reward(self):
        # One reward at the end, terminal there.
        rewards = np.array([0.0, 0.0, 1.0])
        values = np.zeros(3)
        dones = np.array([0.0, 0.0, 1.0])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        # Last advantage = reward (no bootstrap past terminal)
        assert adv[-1] == pytest.approx(1.0)
        # Earlier advantages discounted
        assert adv[0] < adv[2]
        assert adv[0] > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            compute_gae(np.zeros(3), np.zeros(2), np.zeros(3))

    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_gae(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))


class TestRolloutBuffer:
    def test_add_and_advantages(self):
        buf = RolloutBuffer()
        for r in (0.0, 0.0, 1.0):
            buf.add(log_prob=-1.0, value=0.0, reward=r, done=(r == 1.0))
        assert len(buf) == 3
        adv, ret = buf.compute_advantages()
        assert adv.shape == (3,)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty buffer"):
            RolloutBuffer().compute_advantages()

    def test_clear(self):
        buf = RolloutBuffer()
        buf.add(log_prob=0, value=0, reward=1, done=True)
        buf.clear()
        assert len(buf) == 0


class TestLogProb:
    def test_gaussian_logprob_peak_at_mean(self):
        mean = np.zeros(4)
        log_std = np.zeros(4)  # std=1
        at_mean = gaussian_logprob(mean, mean, log_std)
        off_mean = gaussian_logprob(np.ones(4), mean, log_std)
        assert at_mean > off_mean

    def test_gaussian_logprob_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            gaussian_logprob(np.zeros(3), np.zeros(4), np.zeros(4))

    def test_ppo_ratio_identity(self):
        assert ppo_ratio(-2.0, -2.0) == pytest.approx(1.0)
        assert ppo_ratio(0.0, -1.0) == pytest.approx(np.e)


class TestPPOObjective:
    def test_objective_clips(self):
        adv = np.array([1.0, 1.0])
        # Big positive logprob change -> ratio clipped to 1+eps for positive adv
        new = np.array([5.0, 5.0])
        old = np.array([0.0, 0.0])
        obj = clipped_ppo_objective(adv, new, old, clip_epsilon=0.2)
        assert obj == pytest.approx(1.2)  # (1+eps) * adv

    def test_objective_length_mismatch(self):
        with pytest.raises(ValueError, match="match length"):
            clipped_ppo_objective(np.zeros(2), np.zeros(3), np.zeros(2))


class TestValueHeadSpec:
    def test_value_lr(self):
        spec = ValueHeadSpec(hidden_dim=512, lr_multiplier=20.0)
        assert spec.value_lr(1e-4) == pytest.approx(2e-3)

    def test_rejects_bad_dims(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ValueHeadSpec(hidden_dim=0)
        with pytest.raises(ValueError, match="mlp_dims"):
            ValueHeadSpec(hidden_dim=8, mlp_dims=(0,))
        with pytest.raises(ValueError, match="lr_multiplier"):
            ValueHeadSpec(hidden_dim=8, lr_multiplier=0)


class TestStageRunnerGuards:
    """run_* must fail fast (clear ValueError) when required objects are None."""

    def test_t2a_requires_model(self):
        with pytest.raises(ValueError, match="requires a loaded model"):
            run_t2a(T2AConfig(), get_embodiment_tag("so100"), action_channels=7, model=None)

    def test_cpt_requires_model(self):
        with pytest.raises(ValueError, match="requires a loaded model"):
            run_cpt(CPTConfig(), model=None)

    def test_sft_requires_model(self):
        with pytest.raises(ValueError, match="requires a loaded model"):
            run_sft(SFTConfig(), model=None, dataset=[1])

    def test_sft_requires_dataset(self):
        with pytest.raises(ValueError, match="requires a dataset"):
            run_sft(SFTConfig(), model=object(), dataset=None)

    def test_rl_requires_model(self):
        with pytest.raises(ValueError, match="requires a loaded model"):
            run_rl(RLConfig(), model=None, env=object())

    def test_rl_requires_env(self):
        with pytest.raises(ValueError, match="requires a sim env"):
            run_rl(RLConfig(), model=object(), env=None)
