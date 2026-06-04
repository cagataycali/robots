"""Unit tests for Qwen-VLA training stage configs, embodiment tags, mixture."""

import numpy as np
import pytest

from strands_robots.training.qwen_vla import (
    CPTConfig,
    RLConfig,
    SFTConfig,
    T2AConfig,
    TimestepDist,
    get_embodiment_tag,
)
from strands_robots.training.qwen_vla.config import _BaseStageConfig
from strands_robots.training.qwen_vla.data import EMBODIMENT_TAGS, MixtureSampler, MixtureSource


class TestStageConfigs:
    def test_t2a_paper_defaults(self):
        c = T2AConfig()
        c.validate()
        assert c.freeze_vlm is True
        assert c.use_images is False
        assert c.full_sequence_prediction is True
        assert c.timestep_dist == TimestepDist.SIGMOID_NORMAL
        assert c.max_steps == 2000
        assert c.synthetic_fraction == pytest.approx(0.2)

    def test_cpt_paper_defaults(self):
        c = CPTConfig()
        c.validate()
        assert c.freeze_vlm is False
        assert c.use_images is True
        assert c.timestep_dist == TimestepDist.BETA
        assert c.vl_loss_weight == pytest.approx(0.1)
        assert c.action_loss_weight == pytest.approx(1.0)

    def test_sft_nav_chunk(self):
        c = SFTConfig()
        c.validate()
        assert c.nav_chunk_size == 8
        assert c.multi_task is True

    def test_rl_ppo_defaults(self):
        c = RLConfig()
        c.validate()
        assert c.num_envs == 128
        assert c.gamma == pytest.approx(0.99)
        assert c.gae_lambda == pytest.approx(0.95)
        assert c.clip_epsilon == pytest.approx(0.2)
        assert c.value_lr_multiplier == pytest.approx(20.0)

    def test_base_validate_rejects_bad_values(self):
        with pytest.raises(ValueError, match="chunk_size"):
            _BaseStageConfig(chunk_size=0).validate()
        with pytest.raises(ValueError, match="action_dim"):
            _BaseStageConfig(action_dim=-1).validate()
        with pytest.raises(ValueError, match="learning_rate"):
            _BaseStageConfig(learning_rate=0).validate()

    def test_t2a_rejects_bad_synthetic_fraction(self):
        with pytest.raises(ValueError, match="synthetic_fraction"):
            T2AConfig(synthetic_fraction=1.5).validate()

    def test_rl_rejects_bad_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            RLConfig(gamma=2.0).validate()


class TestEmbodimentTags:
    def test_registry_has_canonical_only(self):
        # Aliases (aloha, g1, libero) should NOT appear as separate entries.
        assert "aloha_bimanual" in EMBODIMENT_TAGS
        assert "aloha" not in EMBODIMENT_TAGS

    def test_lookup_via_alias(self):
        tag = get_embodiment_tag("aloha")
        assert tag.name == "aloha_bimanual"
        assert tag.arm_config == "dual"
        assert tag.fps == 50

    def test_prompt_matches_inference(self):
        # The training tag must render the exact same prompt as inference.
        from strands_robots.policies.qwen_vla import load_data_config

        tag = get_embodiment_tag("unitree_g1_mobile")
        cfg = load_data_config("unitree_g1_mobile")
        instr = "navigate to the kitchen"
        assert tag.render_prompt(instr) == cfg.embodiment_prompt(instr)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown embodiment"):
            get_embodiment_tag("no_such_robot")


class TestMixture:
    def test_default_probabilities_sum_to_one(self):
        m = MixtureSampler.from_default_mixture()
        assert sum(m.probabilities.values()) == pytest.approx(1.0)

    def test_manipulation_dominant(self):
        m = MixtureSampler.from_default_mixture()
        assert m.probabilities["manipulation"] > 0.7

    def test_sampling_is_deterministic_with_seed(self):
        a = MixtureSampler.from_default_mixture(seed=5).sample_batch(50)
        b = MixtureSampler.from_default_mixture(seed=5).sample_batch(50)
        assert a == b

    def test_override_upweights_source(self):
        m = MixtureSampler.from_default_mixture(manipulation=10.0)
        assert m.probabilities["manipulation"] > 0.9

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError, match="at least one source"):
            MixtureSampler([])

    def test_all_zero_weight_raises(self):
        with pytest.raises(ValueError, match="positive weight"):
            MixtureSampler([MixtureSource("a", 0.0), MixtureSource("b", 0.0)])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            MixtureSource("a", -1.0)


class TestFlowMatching:
    def test_sample_timesteps_in_range(self):
        from strands_robots.training.qwen_vla.flow_matching import sample_timesteps

        rng = np.random.default_rng(0)
        for dist in TimestepDist:
            ts = sample_timesteps(1000, dist, rng=rng)
            assert ts.min() >= 0.0
            assert ts.max() <= 1.0

    def test_interpolate_endpoints(self):
        from strands_robots.training.qwen_vla.flow_matching import interpolate

        x0 = np.zeros((2, 4, 8))
        x1 = np.ones((2, 4, 8))
        # t=0 -> x0, t=1 -> x1
        np.testing.assert_allclose(interpolate(x0, x1, np.zeros(2)), x0)
        np.testing.assert_allclose(interpolate(x0, x1, np.ones(2)), x1)
        np.testing.assert_allclose(interpolate(x0, x1, np.full(2, 0.5)), np.full((2, 4, 8), 0.5))

    def test_masked_loss_ignores_padding(self):
        from strands_robots.policies.qwen_vla import build_channel_mask
        from strands_robots.training.qwen_vla.flow_matching import masked_flow_matching_loss

        pred = np.zeros((1, 4, 8))
        tgt = np.zeros((1, 4, 8))
        # Put huge error only in the PADDED region (channels >= 3).
        tgt[0, :, 3:] = 1000.0
        mask = build_channel_mask(c=3, k=8, h_task=4, h=4)
        loss = masked_flow_matching_loss(pred, tgt, mask)
        assert loss == pytest.approx(0.0)

    def test_masked_loss_counts_valid(self):
        from strands_robots.training.qwen_vla.flow_matching import masked_flow_matching_loss

        pred = np.zeros((1, 2, 2))
        tgt = np.ones((1, 2, 2))
        mask = np.ones((1, 2, 2))
        # All cells valid, err=1 each -> mean = 1.0
        assert masked_flow_matching_loss(pred, tgt, mask) == pytest.approx(1.0)

    def test_masked_loss_no_valid_raises(self):
        from strands_robots.training.qwen_vla.flow_matching import masked_flow_matching_loss

        with pytest.raises(ValueError, match="no valid cells"):
            masked_flow_matching_loss(np.zeros((1, 2, 2)), np.ones((1, 2, 2)), np.zeros((1, 2, 2)))

    def test_target_velocity(self):
        from strands_robots.training.qwen_vla.flow_matching import target_velocity

        np.testing.assert_array_equal(target_velocity(np.zeros((2, 3)), np.full((2, 3), 5.0)), np.full((2, 3), 5.0))
